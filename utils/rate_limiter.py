"""
utils/rate_limiter.py :: Retry, backoff and throttling primitives.

Free data sources punish bursty clients. Three tools live here:

  @retry_with_backoff   Exponential backoff + jitter around transient failures.
  RateLimiter           Token bucket enforcing a per-host request budget.
  @throttled            Decorator binding a function to a named bucket.

All of them are process-local. Streamlit reruns the script on every
interaction, so buckets are stashed in a module-level registry that survives
reruns within the same Python process.
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, TypeVar

import config

log = logging.getLogger("openterm.rate_limiter")

F = TypeVar("F", bound=Callable[..., Any])


# ==========================================================================
# Exceptions
# ==========================================================================
class RateLimitExceeded(RuntimeError):
    """Raised when a provider explicitly says 429 / quota exhausted."""


class UpstreamUnavailable(RuntimeError):
    """Raised after all retries are spent. Callers should fall back."""


# HTTP statuses that are worth retrying. 4xx other than these are permanent.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 521, 522, 524})


# ==========================================================================
# retry_with_backoff
# ==========================================================================
def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_giveup: Optional[Callable[[BaseException], Any]] = None,
    respect_retry_after: bool = True,
) -> Callable[[F], F]:
    """
    Retry a callable with exponential backoff and full jitter.

    Args:
        max_retries: Attempts *after* the first. Defaults to config.NET.
        base_delay:  Multiplier for the exponential curve, in seconds.
        max_delay:   Hard ceiling on any single sleep.
        exceptions:  Exception types considered transient.
        on_giveup:   Called with the final exception instead of re-raising.
                     Return its value to the caller - use this for graceful
                     degradation (e.g. `on_giveup=lambda e: pd.DataFrame()`).
        respect_retry_after:
                     If the exception carries a `.response` with a
                     `Retry-After` header, honour it over the computed delay.

    Sleep schedule (base=1.5): ~1.5s, 3s, 6s, 12s, capped at max_delay, each
    randomised in [50%, 100%] of the nominal value to avoid thundering herds.
    """
    retries = config.NET.max_retries if max_retries is None else max_retries
    base = config.NET.backoff_base if base_delay is None else base_delay
    ceiling = config.NET.backoff_max if max_delay is None else max_delay

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as exc:  # noqa: PERF203 - retry is the point
                    last_exc = exc

                    if not _is_retryable(exc):
                        log.debug("%s: permanent error, not retrying: %s",
                                  func.__name__, exc)
                        break

                    if attempt >= retries:
                        break

                    delay = _compute_delay(exc, attempt, base, ceiling,
                                           respect_retry_after)
                    log.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        func.__name__, attempt + 1, retries + 1, exc, delay,
                    )
                    time.sleep(delay)

            # Every attempt is spent.
            log.error("%s exhausted %d attempts: %s",
                      func.__name__, retries + 1, last_exc)

            if on_giveup is not None:
                return on_giveup(last_exc)  # type: ignore[arg-type]

            raise UpstreamUnavailable(
                f"{func.__name__} failed after {retries + 1} attempts: {last_exc}"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception represents a transient condition."""
    # An open circuit refuses the next attempt without making it, so sleeping
    # before that attempt is pure delay. This covers the failure that tripped
    # the circuit as well as CircuitOpen itself.
    if isinstance(exc, CircuitOpen) or getattr(exc, "circuit_tripped", False):
        return False

    # Explicit signal from our own code.
    if isinstance(exc, RateLimitExceeded):
        return True

    # requests.HTTPError and friends expose the response object. Only a real
    # HTTP status decides, though: curl_cffi (yfinance's transport) attaches
    # a response to a refused connection too, with no status, and reading
    # that as a permanent answer meant Yahoo network errors were never
    # retried and never counted toward a circuit breaker.
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 100:
        return status in RETRYABLE_STATUS

    # Network-layer failures: match on type name so we don't need to import
    # requests/urllib3/httpx here.
    transient_names = {
        "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
        "ChunkedEncodingError", "ProtocolError", "RemoteDisconnected",
        "SSLError", "TooManyRedirects", "IncompleteRead", "socket.timeout",
        "TimeoutError", "ConnectionResetError", "PlaywrightTimeoutError",
        "TimeoutException",
    }
    for klass in type(exc).__mro__:
        if klass.__name__ in transient_names:
            return True

    # yfinance and several scrapers raise bare Exceptions with useful text.
    text = str(exc).lower()
    return any(
        token in text
        for token in ("timed out", "timeout", "temporarily", "rate limit",
                      "too many requests", "429", "503", "502", "connection reset",
                      "connection aborted", "max retries exceeded")
    )


def _compute_delay(
    exc: BaseException,
    attempt: int,
    base: float,
    ceiling: float,
    respect_retry_after: bool,
) -> float:
    """Exponential backoff with full jitter, optionally overridden by server."""
    if respect_retry_after:
        server_delay = _retry_after_seconds(exc)
        if server_delay is not None:
            return min(server_delay, ceiling)

    nominal = min(base * (2 ** attempt), ceiling)
    # Full jitter in [nominal * (1 - jitter), nominal].
    low = nominal * (1.0 - config.NET.jitter)
    return random.uniform(low, nominal)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Parse a Retry-After header off an exception's response, if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        raw = response.headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return float(raw)  # delta-seconds form
    except ValueError:
        # HTTP-date form. Rare enough that a fixed nudge is fine.
        return 5.0


# ==========================================================================
# Token bucket
# ==========================================================================
class RateLimiter:
    """
    Thread-safe token bucket.

    Example - OpenSky allows 400 anonymous credits/day; we throttle to a
    comfortable 1 request per 5 seconds:

        limiter = RateLimiter(rate=0.2, capacity=3, name="opensky")
        limiter.acquire()   # blocks until a token is available
    """

    __slots__ = ("_rate", "_capacity", "_tokens", "_last", "_lock", "name")

    def __init__(self, rate: float, capacity: Optional[float] = None,
                 name: str = "default") -> None:
        """
        Args:
            rate:     Tokens replenished per second (i.e. sustained req/s).
            capacity: Burst size. Defaults to max(1, rate) so a burst of one
                      request is always allowed immediately.
            name:     Label for logging.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._capacity = float(capacity if capacity is not None else max(1.0, rate))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.name = name

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking. True if a token was taken."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Block until `tokens` are available.

        Returns False if `timeout` elapsed first (the caller should then fall
        back rather than hammering the endpoint).
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                deficit = tokens - self._tokens
                wait = deficit / self._rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("RateLimiter[%s] acquire timed out", self.name)
                    return False
                wait = min(wait, remaining)

            time.sleep(min(wait, 1.0))

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


# --------------------------------------------------------------------------
# Named bucket registry
# --------------------------------------------------------------------------
_BUCKETS: Dict[str, RateLimiter] = {}
_BUCKETS_LOCK = threading.Lock()

# Conservative defaults per provider. Tuned to sit far inside published free
# limits so a long Streamlit session never trips a block.
_DEFAULT_BUDGETS: Dict[str, Tuple[float, float]] = {
    # name:          (requests/sec, burst)
    "yfinance":      (2.0, 5.0),
    "sec":           (8.0, 10.0),   # SEC allows 10 req/s; stay just under.
    "opensky":       (0.2, 2.0),    # ~1 per 5s.
    "fred":          (2.0, 5.0),
    "worldbank":     (2.0, 4.0),
    "gdelt":         (0.5, 2.0),
    "rss":           (3.0, 6.0),
    # Reddit answers RSS but 429s readily - three quick probes were enough to
    # get blocked during development. This is deliberately the slowest bucket
    # in the table.
    "reddit":        (0.15, 1.0),   # ~1 per 7s.
    "stocktwits":    (0.5, 2.0),    # Cloudflare-fronted; no published cap.
    "scrape":        (0.3, 2.0),    # Headless browser targets: be gentle.
    "coingecko":     (0.4, 2.0),    # Free tier ~10-30/min.
    "wikidata":      (1.0, 3.0),    # No published cap; their docs ask for
                                    # serial-ish access and a real UA.
    "default":       (2.0, 4.0),
}


def get_limiter(name: str = "default") -> RateLimiter:
    """Fetch (or lazily create) the shared bucket for a provider."""
    with _BUCKETS_LOCK:
        limiter = _BUCKETS.get(name)
        if limiter is None:
            rate, burst = _DEFAULT_BUDGETS.get(name, _DEFAULT_BUDGETS["default"])
            limiter = RateLimiter(rate=rate, capacity=burst, name=name)
            _BUCKETS[name] = limiter
        return limiter


def throttled(bucket: str = "default", tokens: float = 1.0) -> Callable[[F], F]:
    """
    Decorator form of the token bucket.

        @throttled("opensky")
        @retry_with_backoff(on_giveup=lambda e: [])
        def fetch_states(...): ...

    Order matters: put @throttled outermost so each *retry* also pays a token.

    A call made while the same-named circuit is open skips the bucket: the
    breaker will refuse it without touching the network, and queueing for a
    token first turned "fail fast" into "fail after the queue drains".
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = _CIRCUITS.get(bucket)
            if breaker is None or breaker.allow():
                get_limiter(bucket).acquire(tokens)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ==========================================================================
# Circuit breaker
# ==========================================================================
class CircuitOpen(RuntimeError):
    """Raised immediately when a provider's circuit is open (fail fast)."""


class CircuitBreaker:
    """
    Per-provider circuit breaker.

    Why this exists: retry-with-backoff is correct for a *flaky* endpoint but
    pathological for a *dead* one. The yield curve fetches 11 tenors; if FRED
    is unreachable, each pays 5 attempts x 20s timeout, so one chart takes 11
    minutes to render nothing. The breaker trips after a few consecutive
    failures and then fails instantly until a probe succeeds.

    States:
        CLOSED    Normal. Failures counted.
        OPEN      Failing fast. Flips to HALF_OPEN after recovery_timeout.
        HALF_OPEN One trial call allowed through; success closes the circuit,
                  failure re-opens it.
    """

    __slots__ = ("name", "_threshold", "_recovery", "_failures",
                 "_opened_at", "_state", "_lock")

    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 120.0) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._failures = 0
        self._opened_at = 0.0
        self._state = "CLOSED"
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Promote OPEN -> HALF_OPEN once the recovery window has elapsed."""
        if self._state == "OPEN" and (time.monotonic() - self._opened_at) >= self._recovery:
            self._state = "HALF_OPEN"
            log.info("Circuit[%s] entering HALF_OPEN - allowing a probe", self.name)

    def allow(self) -> bool:
        """True if a call may proceed."""
        with self._lock:
            self._maybe_half_open()
            return self._state in ("CLOSED", "HALF_OPEN")

    def record_success(self) -> None:
        with self._lock:
            if self._state != "CLOSED":
                log.info("Circuit[%s] closing after successful probe", self.name)
            self._failures = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= self._threshold:
                if self._state != "OPEN":
                    log.warning(
                        "Circuit[%s] OPEN after %d failures - failing fast for %.0fs",
                        self.name, self._failures, self._recovery,
                    )
                self._state = "OPEN"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._opened_at = 0.0


_CIRCUITS: Dict[str, CircuitBreaker] = {}
_CIRCUITS_LOCK = threading.Lock()


def get_circuit(name: str, failure_threshold: int = 3,
                recovery_timeout: float = 120.0) -> CircuitBreaker:
    with _CIRCUITS_LOCK:
        breaker = _CIRCUITS.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name, failure_threshold, recovery_timeout)
            _CIRCUITS[name] = breaker
        return breaker


def circuit_breaker(
    name: str,
    failure_threshold: int = 3,
    recovery_timeout: float = 120.0,
    on_open: Optional[Callable[[], Any]] = None,
    trip_on: Optional[Callable[[BaseException], bool]] = None,
    count_empty: bool = True,
) -> Callable[[F], F]:
    """
    Fail fast once a provider has proven itself down.

    Usually applied OUTSIDE @retry_with_backoff so an open circuit skips the
    retry budget entirely:

        @circuit_breaker("fred", on_open=lambda: pd.Series(dtype=float))
        @retry_with_backoff(on_giveup=lambda e: pd.Series(dtype=float))
        def _fred_series_csv(...): ...

    Applied INSIDE the retry loop instead, every failed attempt counts toward
    the threshold, and the failure that trips the circuit ends the loop
    rather than sleeping before an attempt that would be refused. That is the
    right shape when on_giveup hides exceptions from anything outside it.

    Args:
        on_open:     Returned instead of raising when the circuit is open. Give
                     this the same empty value your on_giveup returns so callers
                     see one consistent "no data" shape.
        trip_on:     Whether an exception counts as a failure. Default: every
                     exception. Pass `is_transient_error` for an upstream where
                     a bad request - a 404 for a mistyped symbol - says nothing
                     about whether the service is up.
        count_empty: Whether an empty return counts as a failure. Right for a
                     fetcher whose on_giveup turns errors into empty values;
                     wrong where "nothing listed" is a legitimate answer. An
                     empty return never counts as a success either way.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = get_circuit(name, failure_threshold, recovery_timeout)

            if not breaker.allow():
                log.debug("Circuit[%s] open - skipping %s", name, func.__name__)
                if on_open is not None:
                    return on_open()
                raise CircuitOpen(f"Circuit '{name}' is open; upstream is down")

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if trip_on is None or trip_on(exc):
                    breaker.record_failure()
                    if not breaker.allow():
                        _mark_tripped(exc)
                raise

            # A fetcher with on_giveup returns an empty value instead of
            # raising, so an exception-only breaker would never trip. Treat an
            # empty return as a failure signal too, unless told not to.
            if looks_empty(result):
                if count_empty:
                    breaker.record_failure()
            else:
                breaker.record_success()
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _mark_tripped(exc: BaseException) -> None:
    """Flag the failure that opened a circuit, so a retry loop stops on it."""
    try:
        exc.circuit_tripped = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - exception types with __slots__
        pass


def is_transient_error(exc: Optional[BaseException]) -> bool:
    """
    True when a failure says the upstream is unreachable, not that the
    request was wrong.

    Follows the `raise ... from` chain: retry_with_backoff reports a spent
    budget as UpstreamUnavailable wrapping the real error, and the wrapper's
    own message quotes that error's text, so the wrapper is skipped rather
    than matched on.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if not isinstance(exc, UpstreamUnavailable) and _is_retryable(exc):
            return True
        exc = exc.__cause__
    return False


def looks_empty(value: Any) -> bool:
    """Best-effort 'this fetch produced nothing' check across return types."""
    if value is None:
        return True
    # pandas objects: .empty, but avoid importing pandas here.
    empty_attr = getattr(value, "empty", None)
    if isinstance(empty_attr, bool):
        return empty_attr
    if isinstance(value, (list, dict, tuple, set, str)):
        return len(value) == 0
    return False


def circuit_status() -> Dict[str, str]:
    """Snapshot of every breaker, for the diagnostics panel."""
    with _CIRCUITS_LOCK:
        return {name: breaker.state for name, breaker in _CIRCUITS.items()}


def reset_all_circuits() -> None:
    """Wired to the sidebar's REFRESH button - give dead sources another go."""
    with _CIRCUITS_LOCK:
        for breaker in _CIRCUITS.values():
            breaker.reset()


def bucket_status() -> Dict[str, Dict[str, float]]:
    """Snapshot of every live bucket, for the diagnostics panel."""
    with _BUCKETS_LOCK:
        out = {}
        for name, lim in _BUCKETS.items():
            with lim._lock:  # noqa: SLF001 - internal diagnostics
                lim._refill()
                out[name] = {
                    "tokens": round(lim._tokens, 2),
                    "capacity": lim._capacity,
                    "rate_per_sec": lim._rate,
                }
        return out


__all__ = [
    "RateLimitExceeded",
    "UpstreamUnavailable",
    "CircuitOpen",
    "retry_with_backoff",
    "RateLimiter",
    "CircuitBreaker",
    "get_limiter",
    "get_circuit",
    "circuit_breaker",
    "is_transient_error",
    "looks_empty",
    "circuit_status",
    "reset_all_circuits",
    "throttled",
    "bucket_status",
]

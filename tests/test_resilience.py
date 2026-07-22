"""
Caching, backoff and circuit-breaker behaviour.

REGRESSION CONTEXT
------------------
The circuit breaker exists because retry-with-backoff is correct for a flaky
endpoint and pathological for a dead one. get_yield_curve() requests 11
tenors; when FRED was unreachable each paid 5 attempts x 20s, so a single
chart took ~11 minutes to render nothing. The fail-fast timing test below is
the guard on that.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from utils.cache import CacheEntry, SQLiteCache, make_key
from utils.rate_limiter import (
    CircuitBreaker,
    CircuitOpen,
    RateLimiter,
    UpstreamUnavailable,
    circuit_breaker,
    circuit_status,
    retry_with_backoff,
)


# ==========================================================================
# Circuit breaker
# ==========================================================================
class TestCircuitBreaker:
    def test_starts_closed(self):
        assert CircuitBreaker("t", failure_threshold=3).state == "CLOSED"

    def test_opens_only_at_threshold(self):
        breaker = CircuitBreaker("t", failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "CLOSED"
        breaker.record_failure()
        assert breaker.state == "OPEN"

    def test_open_blocks_calls(self):
        breaker = CircuitBreaker("t", failure_threshold=1)
        breaker.record_failure()
        assert not breaker.allow()

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker("t", failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "CLOSED", "counter did not reset on success"

    def test_half_opens_after_recovery_timeout(self):
        breaker = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.3)
        breaker.record_failure()
        assert breaker.state == "OPEN"
        time.sleep(0.35)
        assert breaker.state == "HALF_OPEN"
        assert breaker.allow()

    def test_half_open_success_closes(self):
        breaker = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.2)
        breaker.record_failure()
        time.sleep(0.25)
        assert breaker.state == "HALF_OPEN"
        breaker.record_success()
        assert breaker.state == "CLOSED"

    def test_half_open_failure_reopens_immediately(self):
        """One failed probe must re-open, not decrement toward the threshold."""
        breaker = CircuitBreaker("t", failure_threshold=5, recovery_timeout=0.2)
        for _ in range(5):
            breaker.record_failure()
        time.sleep(0.25)
        assert breaker.state == "HALF_OPEN"
        breaker.record_failure()
        assert breaker.state == "OPEN"

    def test_reset(self):
        breaker = CircuitBreaker("t", failure_threshold=1)
        breaker.record_failure()
        breaker.reset()
        assert breaker.state == "CLOSED"


class TestCircuitBreakerDecorator:
    def test_fails_fast_without_calling_upstream(self):
        """THE guard on the 11-minute yield-curve stall."""
        calls = {"n": 0}

        @circuit_breaker("ff_test", failure_threshold=2, recovery_timeout=60,
                         on_open=lambda: "SHORTED")
        def dead():
            calls["n"] += 1
            raise ConnectionError("upstream down")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                dead()

        calls_when_tripped = calls["n"]

        start = time.monotonic()
        results = [dead() for _ in range(50)]
        elapsed = time.monotonic() - start

        assert all(r == "SHORTED" for r in results)
        assert calls["n"] == calls_when_tripped, "upstream called while open"
        assert elapsed < 0.5, f"50 shorted calls took {elapsed:.2f}s"

    def test_raises_when_no_on_open_provided(self):
        @circuit_breaker("raise_test", failure_threshold=1, recovery_timeout=60)
        def dead():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            dead()
        with pytest.raises(CircuitOpen):
            dead()

    def test_empty_return_counts_as_failure(self):
        """
        Fetchers use on_giveup to return an empty frame instead of raising,
        so an exception-only breaker would never trip on a dead upstream.
        """
        calls = {"n": 0}

        @circuit_breaker("empty_test", failure_threshold=2, recovery_timeout=60,
                         on_open=lambda: pd.DataFrame())
        def returns_empty():
            calls["n"] += 1
            return pd.DataFrame()

        returns_empty()
        returns_empty()
        calls_when_tripped = calls["n"]
        returns_empty()
        assert calls["n"] == calls_when_tripped

    def test_success_keeps_circuit_closed(self):
        @circuit_breaker("ok_test", failure_threshold=2, recovery_timeout=60,
                         on_open=lambda: None)
        def healthy():
            return pd.DataFrame({"a": [1]})

        for _ in range(10):
            assert not healthy().empty
        assert circuit_status().get("ok_test") == "CLOSED"


# ==========================================================================
# Retry / backoff
# ==========================================================================
class TestRetryWithBackoff:
    def test_returns_on_first_success(self):
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def ok():
            calls["n"] += 1
            return "fine"

        assert ok() == "fine"
        assert calls["n"] == 1

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("transient")
            return "recovered"

        assert flaky() == "recovered"
        assert calls["n"] == 3

    def test_on_giveup_value_returned(self):
        @retry_with_backoff(max_retries=1, base_delay=0.01,
                            on_giveup=lambda exc: "fallback")
        def dead():
            raise TimeoutError("down")

        assert dead() == "fallback"

    def test_raises_upstream_unavailable_without_on_giveup(self):
        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def dead():
            raise TimeoutError("down")

        with pytest.raises(UpstreamUnavailable):
            dead()

    def test_permanent_error_not_retried(self):
        """A 404 is not transient; retrying it just wastes the budget."""
        calls = {"n": 0}

        class Response:
            status_code = 404
            headers: dict = {}

        class HTTPError(Exception):
            response = Response()

        @retry_with_backoff(max_retries=5, base_delay=0.01,
                            on_giveup=lambda exc: "gave_up")
        def not_found():
            calls["n"] += 1
            raise HTTPError("404")

        assert not_found() == "gave_up"
        assert calls["n"] == 1, "a permanent 404 was retried"

    def test_retryable_status_is_retried(self):
        calls = {"n": 0}

        class Response:
            status_code = 503
            headers: dict = {}

        class HTTPError(Exception):
            response = Response()

        @retry_with_backoff(max_retries=2, base_delay=0.01,
                            on_giveup=lambda exc: "gave_up")
        def unavailable():
            calls["n"] += 1
            raise HTTPError("503")

        assert unavailable() == "gave_up"
        assert calls["n"] == 3


# ==========================================================================
# Token bucket
# ==========================================================================
class TestRateLimiter:
    def test_burst_allowed_up_to_capacity(self):
        limiter = RateLimiter(rate=1000, capacity=5, name="t")
        assert all(limiter.try_acquire() for _ in range(5))

    def test_exhausted_bucket_refuses(self):
        limiter = RateLimiter(rate=0.001, capacity=2, name="t")
        limiter.try_acquire()
        limiter.try_acquire()
        assert not limiter.try_acquire()

    def test_refills_over_time(self):
        limiter = RateLimiter(rate=50, capacity=1, name="t")
        assert limiter.try_acquire()
        assert not limiter.try_acquire()
        time.sleep(0.1)
        assert limiter.try_acquire()

    def test_acquire_timeout_returns_false(self):
        limiter = RateLimiter(rate=0.001, capacity=1, name="t")
        limiter.acquire()
        assert limiter.acquire(timeout=0.1) is False

    def test_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(rate=0, name="t")


# ==========================================================================
# Cache
# ==========================================================================
class TestSQLiteCache:
    def test_roundtrip(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("k", {"a": 1}, ttl=60)
        entry = cache.get("k")
        assert entry is not None and entry.value == {"a": 1}
        assert entry.stale is False

    def test_dataframe_roundtrip(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        df = pd.DataFrame({"x": [1, 2, 3]})
        cache.set("df", df, ttl=60)
        pd.testing.assert_frame_equal(cache.get("df").value, df)

    def test_miss_returns_none(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        assert cache.get("absent") is None

    def test_expiry(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("k", "v", ttl=0)
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_stale_read_opt_in(self, tmp_path):
        """The mechanism behind serving old data when an upstream dies."""
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("k", "v", ttl=0)
        time.sleep(0.01)

        assert cache.get("k") is None
        entry = cache.get("k", allow_stale=True)
        assert entry is not None
        assert entry.value == "v"
        assert entry.stale is True

    def test_overwrite(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("k", "first", ttl=60)
        cache.set("k", "second", ttl=60)
        assert cache.get("k").value == "second"

    def test_delete(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("k", "v", ttl=60)
        cache.delete("k")
        assert cache.get("k") is None

    def test_clear_namespace_only(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("a", 1, ttl=60, namespace="one")
        cache.set("b", 2, ttl=60, namespace="two")
        cache.clear(namespace="one")
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_purge_expired_keeps_live(self, tmp_path):
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("dead", 1, ttl=0)
        cache.set("live", 2, ttl=60)
        time.sleep(0.01)
        cache.purge_expired()
        assert cache.get("dead") is None
        assert cache.get("live") is not None

    def test_unpicklable_value_does_not_raise(self, tmp_path):
        """A cache write must never break a render."""
        cache = SQLiteCache(path=str(tmp_path / "c.sqlite"))
        cache.set("lambda", lambda x: x, ttl=60)
        assert cache.get("lambda") is None


class TestCachedDecorator:
    def test_second_call_hits_cache(self, temp_cache):
        from utils.cache import cached

        calls = {"n": 0}

        @cached(ttl=60, namespace="dec_test")
        def fetch(x):
            calls["n"] += 1
            return x * 2

        assert fetch(5) == 10
        assert fetch(5) == 10
        assert calls["n"] == 1

    def test_distinct_args_cached_separately(self, temp_cache):
        from utils.cache import cached

        calls = {"n": 0}

        @cached(ttl=60, namespace="dec_args")
        def fetch(x):
            calls["n"] += 1
            return x

        fetch(1)
        fetch(2)
        assert calls["n"] == 2

    def test_stale_served_when_upstream_fails(self, temp_cache):
        """The behaviour that keeps the terminal readable during an outage."""
        from utils.cache import cached

        state = {"fail": False}

        @cached(ttl=0, namespace="dec_stale", allow_stale_on_error=True)
        def fetch():
            if state["fail"]:
                raise ConnectionError("upstream down")
            return "good value"

        assert fetch() == "good value"
        state["fail"] = True
        time.sleep(0.01)
        assert fetch() == "good value", "stale value was not served"

    def test_error_propagates_without_stale_entry(self, temp_cache):
        from utils.cache import cached

        @cached(ttl=60, namespace="dec_noentry", allow_stale_on_error=True)
        def fetch():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            fetch()


class TestMakeKey:
    def test_deterministic(self):
        assert make_key("ns", 1, "a", z=2) == make_key("ns", 1, "a", z=2)

    def test_kwarg_order_irrelevant(self):
        assert make_key("ns", a=1, b=2) == make_key("ns", b=2, a=1)

    def test_different_args_differ(self):
        assert make_key("ns", 1) != make_key("ns", 2)

    def test_unhashable_args_supported(self):
        assert make_key("ns", [1, 2], {"a": 1})


class TestCacheEntry:
    def test_age_label_formats(self):
        now = time.time()
        assert CacheEntry("v", now - 30, now + 60, False).age_label().endswith("s ago")
        assert CacheEntry("v", now - 300, now + 60, False).age_label().endswith("m ago")
        assert CacheEntry("v", now - 7200, now + 60, False).age_label().endswith("h ago")
        assert CacheEntry("v", now - 200000, now + 60, False).age_label().endswith("d ago")

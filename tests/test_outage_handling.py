"""
Market data during an outage: empty results and dead networks.

REGRESSION CONTEXT
------------------
  * B1. During an outage get_quote's on_giveup returned {}, @cached stored
    it over the last good quote, and the page went blank. The stale-serving
    path never ran, because nothing raised.

  * B2. yfinance had no circuit breaker. By default it also hid network
    errors behind an empty frame - the same thing a mistyped symbol gets -
    so no breaker could have told the two apart. Every call during a network
    drop paid its full timeout and retry budget: 17-20 second page stalls.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from utils.rate_limiter import (
    CircuitOpen,
    UpstreamUnavailable,
    circuit_breaker,
    circuit_status,
    is_transient_error,
    retry_with_backoff,
)


class _Response:
    def __init__(self, status) -> None:
        self.status_code = status
        self.headers: dict = {}


class HTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP Error {status}")
        self.response = _Response(status)


class CurlConnectionError(Exception):
    """
    Shaped like curl_cffi's: a response object rides along on a refused
    connection even though no HTTP exchange happened. Builtin ConnectionError
    has no `.response`, which is how the classification bug hid from tests.
    """

    def __init__(self, status=0) -> None:
        super().__init__("Failed to perform, curl: (7) Failed to connect")
        self.response = _Response(status)


CurlConnectionError.__name__ = "ConnectionError"


# ==========================================================================
# B1 / I20 - an empty result never replaces a good one
# ==========================================================================
class TestCachedKeepsLastGood:
    def test_empty_dict_does_not_overwrite_quote(self, temp_cache):
        from utils.cache import cached

        upstream = {"value": {"price": 101.5}}

        @cached(ttl=0, namespace="klg_quote")
        def fetch():
            return upstream["value"]

        assert fetch() == {"price": 101.5}
        upstream["value"] = {}
        time.sleep(0.01)

        assert fetch() == {"price": 101.5}
        assert fetch.last_entry.stale is True
        assert fetch() == {"price": 101.5}, "the empty result replaced the entry"

    def test_empty_frame_does_not_overwrite_chain(self, temp_cache):
        from utils.cache import cached

        chain = pd.DataFrame({"strike": [100.0, 105.0]})
        upstream = {"value": chain}

        @cached(ttl=0, namespace="klg_frame")
        def fetch():
            return upstream["value"]

        fetch()
        upstream["value"] = pd.DataFrame()
        time.sleep(0.01)
        pd.testing.assert_frame_equal(fetch(), chain)

    def test_good_value_still_replaces_old_one(self, temp_cache):
        from utils.cache import cached

        upstream = {"value": {"price": 1.0}}

        @cached(ttl=0, namespace="klg_replace")
        def fetch():
            return upstream["value"]

        fetch()
        upstream["value"] = {"price": 2.0}
        time.sleep(0.01)
        assert fetch() == {"price": 2.0}
        assert fetch.last_entry.stale is False

    def test_empty_without_history_is_cached_briefly(self, temp_cache):
        """A first fetch that lands in an outage must not blank a daily TTL."""
        from utils import cache as cache_mod
        from utils.cache import cached, make_key

        @cached(ttl=86400, namespace="klg_first")
        def fetch():
            return {}

        assert fetch() == {}
        entry = temp_cache.get(make_key("klg_first"))
        assert entry is not None
        assert entry.expires_at - entry.created_at <= cache_mod.EMPTY_RESULT_TTL

    def test_refresh_during_outage_keeps_value(self, temp_cache):
        """_refresh used to delete the entry before fetching."""
        from utils.cache import cached

        upstream = {"down": False}

        @cached(ttl=3600, namespace="klg_refresh")
        def fetch():
            if upstream["down"]:
                raise ConnectionError("down")
            return "good"

        fetch()
        upstream["down"] = True
        assert fetch(_refresh=True) == "good"

    def test_opt_out(self, temp_cache):
        from utils.cache import cached

        upstream = {"value": [1, 2]}

        @cached(ttl=0, namespace="klg_optout", keep_last_good=False)
        def fetch():
            return upstream["value"]

        fetch()
        upstream["value"] = []
        time.sleep(0.01)
        assert fetch() == []


# ==========================================================================
# B2 / I19 - breakers that trip on outages, not on typos
# ==========================================================================
class TestTransientOnlyBreaker:
    def test_permanent_errors_do_not_trip(self):
        @circuit_breaker("perm_test", failure_threshold=2, recovery_timeout=60,
                         trip_on=is_transient_error)
        def not_found():
            raise HTTPError(404)

        for _ in range(5):
            with pytest.raises(HTTPError):
                not_found()
        assert circuit_status()["perm_test"] == "CLOSED"

    def test_transient_errors_trip(self):
        @circuit_breaker("transient_test", failure_threshold=2,
                         recovery_timeout=60, trip_on=is_transient_error)
        def refused():
            raise ConnectionError("curl: (7) Failed to connect")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                refused()
        with pytest.raises(CircuitOpen):
            refused()

    def test_uncounted_empty_result_is_neutral(self):
        @circuit_breaker("neutral_test", failure_threshold=2,
                         recovery_timeout=60, count_empty=False)
        def no_options():
            return {}

        for _ in range(5):
            assert no_options() == {}
        assert circuit_status()["neutral_test"] == "CLOSED"

    def test_is_transient_error_reads_through_retry_wrapper(self):
        @retry_with_backoff(max_retries=0)
        def refused():
            raise ConnectionError("refused")

        @retry_with_backoff(max_retries=0)
        def missing():
            raise HTTPError(404)

        with pytest.raises(UpstreamUnavailable) as dead:
            refused()
        with pytest.raises(UpstreamUnavailable) as typo:
            missing()

        assert is_transient_error(dead.value)
        assert not is_transient_error(typo.value)
        assert not is_transient_error(ValueError("No price history"))

    @pytest.mark.parametrize("status", [0, None])
    def test_curl_refusal_with_empty_response_is_transient(self, status):
        """curl_cffi attaches a status-less response; that is not a 4xx."""
        assert is_transient_error(CurlConnectionError(status))

    def test_tripping_failure_ends_the_retry_loop(self, monkeypatch):
        calls = {"n": 0}
        sleeps: list = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        @retry_with_backoff(max_retries=6, base_delay=1.0,
                            on_giveup=lambda exc: {})
        @circuit_breaker("loop_test", failure_threshold=3, recovery_timeout=60,
                         trip_on=is_transient_error, count_empty=False)
        def refused():
            calls["n"] += 1
            raise ConnectionError("refused")

        assert refused() == {}
        assert calls["n"] == 3, "kept retrying after the circuit opened"
        assert len(sleeps) == 2, "slept before an attempt the circuit refuses"

        assert refused() == {}
        assert calls["n"] == 3, "an open circuit still reached the upstream"


class TestYahooStack:
    """The real equities decorators, with yfinance swapped for a fake."""

    @staticmethod
    def _install(monkeypatch, error):
        from data_fetchers import equities

        calls = {"n": 0}

        class FakeTicker:
            def __init__(self, symbol):
                pass

            @property
            def fast_info(self):
                calls["n"] += 1
                raise error

            def history(self, *args, **kwargs):
                calls["n"] += 1
                raise error

        class FakeYF:
            Ticker = FakeTicker

        monkeypatch.setattr(equities, "yf", FakeYF)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        return equities, calls

    def test_dead_network_trips_and_then_fails_fast(self, monkeypatch):
        equities, calls = self._install(monkeypatch, CurlConnectionError())

        assert equities.get_quote.uncached("AAPL") == {}
        assert circuit_status()["yfinance"] == "OPEN"

        after_trip = calls["n"]
        start = time.monotonic()
        for symbol in ("MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA"):
            assert equities.get_quote.uncached(symbol) == {}
        assert calls["n"] == after_trip, "an open circuit still called Yahoo"
        assert time.monotonic() - start < 1.0

    def test_unknown_symbol_does_not_trip(self, monkeypatch):
        equities, _ = self._install(monkeypatch, HTTPError(404))

        for _ in range(6):
            assert equities.get_quote.uncached("NOTAREALTICKER") == {}
        assert circuit_status().get("yfinance", "CLOSED") == "CLOSED"

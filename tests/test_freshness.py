"""
Data-age badges (I1/I2, B7) and the quote timestamp behind them (B5).

REGRESSION CONTEXT
------------------
The cache has always recorded when a value was fetched and whether it was
served past its TTL. Nothing rendered it, so a dead upstream looked exactly
like a live market - the terminal kept showing Friday's price with no hint
that it was Friday's.

The quote's own timestamp made that worse: it recorded the moment we asked
Yahoo, so a badge built on it would have read "3s ago" against a price that
last traded three days earlier.
"""

from __future__ import annotations

import time
import types
from datetime import datetime, timedelta, timezone

import pytest

from ui import components as ui


def _provenance(served_stale: bool = False, expired: bool = False,
                minutes_old: float = 0.0):
    """Stand-in for a CacheEntry, which is all the badge reads."""
    return types.SimpleNamespace(
        stale=expired or served_stale,
        served_stale=served_stale,
        as_of=datetime.now(timezone.utc) - timedelta(minutes=minutes_old),
    )


# ==========================================================================
# Provenance lookup
# ==========================================================================
class TestProvenanceLookup:
    def test_none_before_anything_is_fetched(self, temp_cache):
        from utils.cache import cached

        @cached(ttl=60, namespace="prov_absent")
        def fetch(symbol):
            return {"price": 1.0}

        assert fetch.provenance("AAPL") is None

    def test_records_age_after_a_fetch(self, temp_cache):
        from utils.cache import cached

        @cached(ttl=60, namespace="prov_age")
        def fetch(symbol):
            return {"price": 1.0}

        fetch("AAPL")
        provenance = fetch.provenance("AAPL")
        assert provenance is not None
        assert not provenance.stale
        assert provenance.age_seconds < 5
        assert (datetime.now(timezone.utc) - provenance.as_of).total_seconds() < 5
        # Rendering a badge must not cost a DataFrame unpickle.
        assert provenance.value is None

    def test_keyed_by_arguments(self, temp_cache):
        from utils.cache import cached

        @cached(ttl=60, namespace="prov_args")
        def fetch(symbol):
            return {"price": 1.0}

        fetch("AAPL")
        assert fetch.provenance("AAPL") is not None
        assert fetch.provenance("MSFT") is None, "badge would describe another symbol"

    def test_expiry_is_not_a_failed_refresh(self, temp_cache):
        from utils.cache import cached

        @cached(ttl=0, namespace="prov_stale")
        def fetch():
            return {"price": 1.0}

        fetch()
        time.sleep(0.01)
        provenance = fetch.provenance()
        assert provenance.stale, "TTL has passed"
        assert not provenance.served_stale, "nothing has failed yet"

    def test_failed_refresh_marks_the_entry(self, temp_cache):
        from utils.cache import cached

        upstream = {"down": False}

        @cached(ttl=0, namespace="prov_failed")
        def fetch():
            if upstream["down"]:
                raise ConnectionError("upstream down")
            return {"price": 1.0}

        fetch()
        assert not fetch.provenance().served_stale

        upstream["down"] = True
        time.sleep(0.01)
        assert fetch() == {"price": 1.0}
        assert fetch.provenance().served_stale, "an outage looked like a live market"

        upstream["down"] = False
        assert fetch() == {"price": 1.0}
        assert not fetch.provenance().served_stale, "recovery left the badge on"

    def test_describes_the_value_actually_served(self, temp_cache):
        """
        When an empty refetch is refused, the served value is the old one -
        so the badge has to date the old one, not the attempt.
        """
        from utils.cache import cached

        upstream = {"value": {"price": 1.0}}

        @cached(ttl=0, namespace="prov_keep")
        def fetch():
            return upstream["value"]

        fetch()
        first_fetched = fetch.provenance().created_at

        upstream["value"] = {}
        time.sleep(0.01)
        assert fetch() == {"price": 1.0}

        provenance = fetch.provenance()
        assert provenance.served_stale
        assert provenance.created_at == first_fetched


# ==========================================================================
# The badge
# ==========================================================================
class TestDataAgeBadge:
    def test_nothing_known_renders_nothing(self):
        assert ui.data_age(None) == ""

    def test_live(self):
        chip = ui.data_age(_provenance(), source="FRED")
        assert "LIVE" in chip and "FRED" in chip
        assert "STALE" not in chip

    def test_current_but_not_recent_is_cached_not_live(self):
        """A 24h TTL keeps day-old data valid; "LIVE" would oversell it."""
        chip = ui.data_age(_provenance(minutes_old=22 * 60), source="YAHOO PROFILE")
        assert "CACHED" in chip and "LIVE" not in chip
        assert "22h ago" in chip

    def test_delayed_feed(self):
        chip = ui.data_age(_provenance(), source="YAHOO", delayed_minutes=15)
        assert "DELAYED 15M" in chip

    def test_stale_outranks_delayed(self):
        """A delayed feed that also failed to refresh is stale first."""
        chip = ui.data_age(_provenance(served_stale=True), source="YAHOO",
                           delayed_minutes=15)
        assert "STALE" in chip
        assert "DELAYED" not in chip

    def test_expiry_alone_is_not_stale(self):
        """
        A 60-second quote is past its TTL a minute after every fetch. Calling
        that STALE would cry wolf until the badge meant nothing.
        """
        chip = ui.data_age(_provenance(expired=True, minutes_old=3),
                           source="YAHOO", delayed_minutes=15)
        assert "STALE" not in chip
        assert "DELAYED 15M" in chip and "3m ago" in chip

    def test_fetch_time_is_labelled_as_a_fetch_time(self):
        chip = ui.data_age(_provenance(minutes_old=4))
        assert "FETCHED" in chip and "AS OF" not in chip
        assert "4m ago" in chip

    def test_source_timestamp_is_labelled_as_of(self):
        traded = datetime.now(timezone.utc) - timedelta(hours=3)
        chip = ui.data_age(_provenance(minutes_old=0), as_of=traded)
        assert "AS OF" in chip and "3h ago" in chip

    def test_source_text_escaped(self):
        chip = ui.data_age(_provenance(), source="<b>YAHOO</b>")
        assert "<b>" not in chip and "&lt;b&gt;" in chip

    def test_row_skips_empty_chips(self, monkeypatch):
        captured: list = []
        monkeypatch.setattr(ui.st, "markdown",
                            lambda body, **_kwargs: captured.append(body))
        ui.provenance_row(["", ""])
        assert captured == []
        ui.provenance_row(["", ui.data_age(_provenance())])
        assert len(captured) == 1 and "LIVE" in captured[0]


class TestAgeChip:
    def test_fetcher_without_a_cache_renders_nothing(self):
        import app

        assert app._age_chip(lambda ticker: {}, "AAPL", source="YAHOO") == ""

    def test_broken_provenance_lookup_is_not_fatal(self):
        import app

        def explode(*_args):
            raise RuntimeError("cache down")

        fetcher = types.SimpleNamespace(provenance=explode, __name__="fetcher")
        assert app._age_chip(fetcher, "AAPL", source="YAHOO") == ""

    def test_stale_entry_reaches_the_badge(self):
        import app

        fetcher = types.SimpleNamespace(
            provenance=lambda *_a: _provenance(served_stale=True, minutes_old=90))
        chip = app._age_chip(fetcher, "AAPL", source="YAHOO", delayed_minutes=15)
        assert "STALE" in chip and "1h ago" in chip

    def test_market_timestamp_preferred_over_fetch_time(self):
        import app

        traded = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        fetcher = types.SimpleNamespace(provenance=lambda *_a: _provenance())
        chip = app._age_chip(fetcher, "AAPL", source="YAHOO", as_of=traded)
        assert "AS OF" in chip and "3d ago" in chip


# ==========================================================================
# B5: the quote's own timestamp
# ==========================================================================
class TestQuoteTimestamp:
    @staticmethod
    def _install(monkeypatch, metadata):
        from data_fetchers import equities

        class FakeTicker:
            history_metadata = metadata

            def __init__(self, symbol):
                pass

            @property
            def fast_info(self):
                return {"lastPrice": 190.0, "previousClose": 188.0,
                        "lastVolume": 1_000_000, "dayHigh": 191.0,
                        "dayLow": 187.0, "marketCap": 3e12}

            def history(self, *_args, **_kwargs):
                raise AssertionError("the fallback should not be needed")

        monkeypatch.setattr(equities, "yf", types.SimpleNamespace(Ticker=FakeTicker))
        return equities

    def test_market_time_read_from_chart_metadata(self, monkeypatch):
        equities = self._install(
            monkeypatch, {"regularMarketTime": 1_789_502_401})
        quote = equities.get_quote.uncached("AAPL")

        assert quote["market_time"] == "2026-09-15T20:00:01+00:00"
        # `timestamp` is what older callers read: the trade time when known.
        assert quote["timestamp"] == quote["market_time"]
        assert quote["fetched_at"] != quote["market_time"]

    def test_absent_metadata_leaves_market_time_unset(self, monkeypatch):
        equities = self._install(monkeypatch, {})
        quote = equities.get_quote.uncached("AAPL")

        assert quote["market_time"] is None
        assert quote["timestamp"] == quote["fetched_at"]
        assert quote["price"] == 190.0

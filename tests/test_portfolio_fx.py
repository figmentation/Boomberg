"""
Valuing a book that spans currencies.

REGRESSION CONTEXT
------------------
The book this was built against holds US listings (VOO, NVDA, MSFT) next to
SGX listings (D05.SI, 9CI.SI, U96.SI). Yahoo quotes the SGX lines in SGD, and
every figure was being summed as though it were USD:

  * SGD market values added straight onto USD ones. The total, every weight,
    the sector split and the brief's "$" paragraph were all a number in no
    currency - at SGDUSD ~0.79 each SGX line was overstated by about a
    quarter.

  * A missing FX rate is not a rate of 1.0. Falling back to one recreates the
    original bug silently on exactly the day the cross fails to load, so the
    position is left unpriced and the reason is named.

  * `financialCurrency` is not the quote currency. PDD reports in CNY and
    trades in USD.

  * Minor units. London quotes in GBp, so 250 there is 2.50 pounds.

  * Editions summarised before conversion existed. The stored paragraph had
    already printed a mixed total behind a "$", so a book with no currency
    recorded is re-derived rather than trusted.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from data_fetchers import equities, portfolio
from utils.rate_limiter import UpstreamUnavailable

SGDUSD = 0.79

BOOK = [
    {"ticker": "VOO", "quantity": 1, "cost_basis": 500.0},
    {"ticker": "D05.SI", "quantity": 10, "cost_basis": 40.0},
]


def _holdings(rows):
    """A frame shaped like portfolio.holdings() output."""
    return pd.DataFrame(rows).reindex(columns=portfolio.HOLDING_COLUMNS)


def _row(valued, ticker):
    return valued.set_index("ticker").loc[ticker]


@pytest.fixture
def market(monkeypatch):
    """
    Stub every upstream value_positions touches, and record the calls.

    Tests edit `quotes`, `currencies` and `rates` before valuing. A symbol
    missing from `currencies` raises, as an unanswerable lookup does; `rates`
    set to None makes the FX fetch itself fail.
    """
    state = {
        "quotes": {
            "VOO": {"price": 600.0, "change": 6.0, "change_pct": 1.0},
            "D05.SI": {"price": 50.0, "change": -0.5, "change_pct": -1.0},
        },
        "currencies": {"VOO": "USD", "D05.SI": "SGD"},
        "rates": {"SGD": SGDUSD},
        "currency_calls": [],
        "fx_calls": [],
    }

    def quotes(symbols):
        return {s: {"ticker": s, **state["quotes"][s]}
                for s in symbols if s in state["quotes"]}

    def quote_currency(symbol):
        state["currency_calls"].append(symbol)
        if symbol not in state["currencies"]:
            raise ValueError(f"No quote currency for {symbol}")
        return state["currencies"][symbol]

    def fx_rates(currencies, base="USD"):
        state["fx_calls"].append((tuple(currencies), base))
        if state["rates"] is None:
            raise RuntimeError("FX upstream down")
        rates = {base: 1.0}
        rates.update({code: state["rates"][code] for code in currencies
                      if code in state["rates"]})
        return rates

    monkeypatch.setattr(config, "BASE_CURRENCY", "USD")
    monkeypatch.setattr(equities, "get_quotes_batch", quotes)
    monkeypatch.setattr(equities, "get_quote_currency", quote_currency)
    monkeypatch.setattr(equities, "get_fx_rates", fx_rates)
    return state


# ==========================================================================
# Valuation
# ==========================================================================
class TestValuePositions:
    def test_sgx_listing_converted_into_base(self, market):
        dbs = _row(portfolio.value_positions(_holdings(BOOK)), "D05.SI")

        # Native figures stay native, so they still match the statement.
        assert dbs["price"] == pytest.approx(50.0)
        assert dbs["cost_basis"] == pytest.approx(40.0)
        assert dbs["currency"] == "SGD"
        assert dbs["fx_rate"] == pytest.approx(SGDUSD)

        # Everything that gets summed is in USD.
        assert dbs["market_value"] == pytest.approx(500.0 * SGDUSD)
        assert dbs["cost"] == pytest.approx(400.0 * SGDUSD)
        assert dbs["pnl"] == pytest.approx(100.0 * SGDUSD)
        assert dbs["day_pnl"] == pytest.approx(-5.0 * SGDUSD)
        assert pd.isna(dbs["unpriced_reason"])

        # A ratio has no currency.
        assert dbs["pnl_pct"] == pytest.approx(25.0)

    def test_base_currency_listing_is_untouched(self, market):
        voo = _row(portfolio.value_positions(_holdings(BOOK)), "VOO")
        assert voo["fx_rate"] == pytest.approx(1.0)
        assert voo["market_value"] == pytest.approx(600.0)
        assert voo["day_pnl"] == pytest.approx(6.0)

    def test_totals_and_weights_are_single_currency(self, market):
        valued = portfolio.value_positions(_holdings(BOOK))
        summary = portfolio.portfolio_summary(valued)
        total = 600.0 + 500.0 * SGDUSD

        assert summary["currency"] == "USD"
        assert summary["market_value"] == pytest.approx(total)
        assert summary["day_pnl"] == pytest.approx(6.0 - 5.0 * SGDUSD)
        assert summary["pnl"] == pytest.approx(100.0 + 100.0 * SGDUSD)
        assert summary["priced"] == 2
        assert _row(valued, "D05.SI")["weight"] == pytest.approx(
            500.0 * SGDUSD / total * 100.0)

    def test_missing_rate_leaves_position_unpriced(self, market):
        market["rates"] = {}
        valued = portfolio.value_positions(_holdings(BOOK))
        dbs = _row(valued, "D05.SI")

        # The quote still shows; it just is not added to anything.
        assert dbs["price"] == pytest.approx(50.0)
        for column in ("fx_rate", "market_value", "cost", "pnl", "day_pnl",
                       "weight"):
            assert pd.isna(dbs[column]), column
        assert dbs["unpriced_reason"] == "no SGD to USD rate"

        summary = portfolio.portfolio_summary(valued)
        assert summary["market_value"] == pytest.approx(600.0)
        assert summary["priced"] == 1
        assert _row(valued, "VOO")["weight"] == pytest.approx(100.0)

    def test_fx_outage_is_not_a_rate_of_one(self, market):
        market["rates"] = None
        valued = portfolio.value_positions(_holdings(BOOK))
        assert pd.isna(_row(valued, "D05.SI")["market_value"])
        assert _row(valued, "D05.SI")["unpriced_reason"] == "no SGD to USD rate"
        assert _row(valued, "VOO")["market_value"] == pytest.approx(600.0)

    def test_unknown_currency_is_not_assumed_to_be_base(self, market):
        del market["currencies"]["D05.SI"]
        dbs = _row(portfolio.value_positions(_holdings(BOOK)), "D05.SI")
        assert pd.isna(dbs["market_value"])
        assert dbs["unpriced_reason"] == "no quote currency"

    def test_dead_symbol_skips_currency_lookup(self, market):
        valued = portfolio.value_positions(_holdings(
            BOOK + [{"ticker": "DEAD", "quantity": 5, "cost_basis": 10.0}]))
        assert "DEAD" not in market["currency_calls"]
        assert _row(valued, "DEAD")["unpriced_reason"] == "no quote"
        # Without a currency the basis cannot be converted either, so it is
        # not summed into a cost total the P&L never covered.
        assert pd.isna(_row(valued, "DEAD")["cost"])

    def test_single_currency_book_fetches_no_rates(self, market):
        portfolio.value_positions(_holdings(BOOK[:1]))
        assert market["fx_calls"] == []

    def test_base_currency_is_configurable(self, market, monkeypatch):
        monkeypatch.setattr(config, "BASE_CURRENCY", "SGD")
        market["rates"] = {"USD": 1.25}
        valued = portfolio.value_positions(_holdings(BOOK))

        assert _row(valued, "D05.SI")["market_value"] == pytest.approx(500.0)
        assert _row(valued, "VOO")["market_value"] == pytest.approx(750.0)
        assert market["fx_calls"] == [(("USD",), "SGD")]
        assert portfolio.portfolio_summary(valued)["currency"] == "SGD"

    def test_pence_listing_brought_to_pounds(self, market):
        market["quotes"]["VOD.L"] = {"price": 250.0, "change": 5.0,
                                     "change_pct": 2.0}
        market["currencies"]["VOD.L"] = "GBp"
        market["rates"] = {"GBP": 1.25}
        valued = portfolio.value_positions(_holdings(
            [{"ticker": "VOD.L", "quantity": 100, "cost_basis": 200.0}]))
        vod = _row(valued, "VOD.L")

        assert vod["price"] == pytest.approx(250.0)
        assert vod["market_value"] == pytest.approx(100 * 2.50 * 1.25)
        assert vod["cost"] == pytest.approx(100 * 2.00 * 1.25)
        assert market["fx_calls"] == [(("GBP",), "USD")]
        assert portfolio.fx_applied(valued) == pytest.approx({"GBP": 1.25})


# ==========================================================================
# Upstream lookups
# ==========================================================================
class TestFxRates:
    def test_reads_crosses_and_maps_base_to_one(self, monkeypatch):
        requested = []

        def batch(symbols):
            requested.append(symbols)
            return {"SGDUSD=X": {"price": 0.7881}, "HKDUSD=X": {"price": 0.128}}

        monkeypatch.setattr(equities, "get_quotes_batch", batch)
        rates = equities.get_fx_rates.uncached(("SGD", "HKD", "USD"), "USD")
        assert rates == pytest.approx({"USD": 1.0, "SGD": 0.7881, "HKD": 0.128})
        assert requested == [("HKDUSD=X", "SGDUSD=X")]

    def test_missing_cross_is_omitted_not_filled(self, monkeypatch):
        monkeypatch.setattr(equities, "get_quotes_batch",
                            lambda symbols: {"SGDUSD=X": {"price": 0.7881}})
        rates = equities.get_fx_rates.uncached(("SGD", "MYR"), "USD")
        assert "MYR" not in rates

    def test_nothing_back_raises_so_stale_rates_are_served(self, monkeypatch):
        monkeypatch.setattr(equities, "get_quotes_batch", lambda symbols: {})
        with pytest.raises(ValueError):
            equities.get_fx_rates.uncached(("SGD",), "USD")

    def test_base_only_makes_no_request(self, monkeypatch):
        def batch(symbols):
            raise AssertionError(f"requested {symbols}")

        monkeypatch.setattr(equities, "get_quotes_batch", batch)
        assert equities.get_fx_rates.uncached(("USD",), "USD") == {"USD": 1.0}


class _FakeTicker:
    def __init__(self, currency):
        self.fast_info = type("FastInfo", (), {"currency": currency})()


class TestQuoteCurrency:
    def test_fast_info_answer(self, monkeypatch):
        monkeypatch.setattr(equities.yf, "Ticker", lambda s: _FakeTicker("SGD"))
        monkeypatch.setattr(equities, "get_company_info", lambda s: {})
        assert equities.get_quote_currency.uncached("D05.SI") == "SGD"

    def test_falls_back_to_quote_currency_in_company_info(self, monkeypatch):
        monkeypatch.setattr(equities.yf, "Ticker", lambda s: _FakeTicker(None))
        monkeypatch.setattr(equities, "get_company_info", lambda s: {
            "currency": "USD", "financialCurrency": "CNY"})
        assert equities.get_quote_currency.uncached("PDD") == "USD"

    def test_pence_code_keeps_its_case(self, monkeypatch):
        monkeypatch.setattr(equities.yf, "Ticker", lambda s: _FakeTicker("GBp"))
        assert equities.get_quote_currency.uncached("VOD.L") == "GBp"

    def test_unknown_raises_rather_than_defaulting(self, monkeypatch):
        monkeypatch.setattr(equities.yf, "Ticker", lambda s: _FakeTicker(None))
        monkeypatch.setattr(equities, "get_company_info", lambda s: {})
        with pytest.raises(UpstreamUnavailable):
            equities.get_quote_currency.uncached("DEAD")


class TestCurrencyUnits:
    @pytest.mark.parametrize("code,expected", [
        ("SGD", ("SGD", 1.0)), ("usd", ("USD", 1.0)),
        ("GBp", ("GBP", 100.0)), ("GBP", ("GBP", 1.0)), ("GBX", ("GBP", 100.0)),
        ("ZAc", ("ZAR", 100.0)), ("ILA", ("ILS", 100.0)),
        (None, (None, 1.0)), ("", (None, 1.0)),
    ])
    def test_unit(self, code, expected):
        assert equities.currency_unit(code) == expected

    @pytest.mark.parametrize("code,expected", [
        ("USD", "$"), ("SGD", "S$"), ("eur", "EUR "), (None, ""),
    ])
    def test_prefix(self, code, expected):
        assert equities.currency_prefix(code) == expected


# ==========================================================================
# The brief
# ==========================================================================
class TestBookInBaseCurrency:
    def test_paragraph_states_currency_and_rate(self, market):
        book = portfolio.book_snapshot(
            portfolio.value_positions(_holdings(BOOK)))
        assert book["currency"] == "USD"
        assert book["fx"] == pytest.approx({"SGD": SGDUSD})

        text = portfolio.compose_narrative(book)
        assert f"is worth ${600.0 + 500.0 * SGDUSD:,.0f}" in text
        assert "Figures are in USD, converting SGD at 0.7900." in text

    def test_paragraph_names_positions_missing_a_rate(self, market):
        market["rates"] = {}
        market["quotes"]["U96.SI"] = {"price": 7.0, "change": 0.1,
                                      "change_pct": 1.4}
        market["currencies"]["U96.SI"] = "SGD"
        book = portfolio.book_snapshot(portfolio.value_positions(_holdings(
            BOOK + [{"ticker": "U96.SI", "quantity": 2, "cost_basis": 6.0}])))

        assert book["priced"] == 1
        assert book["fx"] == {}
        assert book["unpriced_reasons"] == {"D05.SI": "no SGD to USD rate",
                                            "U96.SI": "no SGD to USD rate"}
        # Excluded positions do not headline the day's moves either.
        assert book["day_best"]["ticker"] == "VOO"

        text = portfolio.compose_narrative(book)
        assert "is worth $600" in text
        assert ("No SGD to USD rate came back for D05.SI and U96.SI, so they "
                "are left out of every figure above.") in text
        assert "No quote came back" not in text
        assert "converting" not in text

    def test_whole_book_unconvertible(self, market):
        market["rates"] = {}
        book = portfolio.book_snapshot(portfolio.value_positions(
            _holdings(BOOK[1:])))
        assert portfolio.compose_narrative(book).startswith(
            "None of your 1 position could be valued in USD this edition "
            "(no SGD to USD rate).")

    def test_money_uses_the_base_prefix(self, market, monkeypatch):
        monkeypatch.setattr(config, "BASE_CURRENCY", "SGD")
        book = portfolio.book_snapshot(portfolio.value_positions(
            _holdings(BOOK[1:])))
        text = portfolio.compose_narrative(book)
        assert text.startswith("Your book of 1 position is worth S$500,")
        assert "down S$5" in text
        assert "sits S$100 (+25.0%) above cost" in text
        assert "$" not in text.replace("S$", "")

    def test_json_serialisable_with_gaps(self, market):
        market["rates"] = {}
        book = portfolio.book_snapshot(
            portfolio.value_positions(_holdings(BOOK)))
        json.dumps(book, allow_nan=False)


class TestStaleEdition:
    def _write(self, tmp_path, monkeypatch, book):
        monkeypatch.setattr(config, "BRIEF_DIR", tmp_path)
        edition = portfolio.edition_date()
        stored = {"edition": edition.isoformat(), "empty": False,
                  "book": book,
                  "narrative": "Your book of 2 positions is worth $1,100.",
                  "positions": [], "top_stories": [],
                  "summary": {"stories": 0, "net_sentiment": None}}
        path = tmp_path / f"{edition.isoformat()}.json"
        path.write_text(json.dumps(stored), encoding="utf-8")

        def no_rebuild(*_args, **_kwargs):
            raise AssertionError("edition was rebuilt")

        monkeypatch.setattr(portfolio, "build_brief", no_rebuild)
        return path

    def test_book_summed_before_conversion_is_rederived(
            self, tmp_path, monkeypatch, market):
        path = self._write(tmp_path, monkeypatch,
                           {"positions": 2, "market_value": 1100.0})
        monkeypatch.setattr(portfolio, "holdings", lambda *_args: _holdings(BOOK))
        monkeypatch.setattr(portfolio, "_sector_mix",
                            lambda _frame: pd.DataFrame())

        brief = portfolio.get_brief()
        assert brief["book"]["currency"] == "USD"
        assert brief["book"]["market_value"] == pytest.approx(
            600.0 + 500.0 * SGDUSD)
        assert "$1,100" not in brief["narrative"]
        assert json.loads(path.read_text(encoding="utf-8"))["book"]["currency"] == "USD"

    def test_converted_book_is_left_alone(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch,
                    {"currency": "USD", "positions": 2, "market_value": 1100.0})

        def no_revalue(_frame):
            raise AssertionError("a converted book was re-derived")

        monkeypatch.setattr(portfolio, "value_positions", no_revalue)
        brief = portfolio.get_brief()
        assert brief["narrative"] == "Your book of 2 positions is worth $1,100."

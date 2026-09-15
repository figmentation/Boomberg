"""
The morning brief's written summary of the whole book.

REGRESSION CONTEXT
------------------
The paragraph is prose built from numbers, and prose hides bad inputs better
than a table does:

  * Summed-with-skipna totals. A book with no cost basis entered totals to a
    P&L of 0.0, and "sits $0 above cost" reads exactly like a real result.
    Missing inputs must drop the clause, not print a zero.

  * A partial basis presented as the whole book. A return measured over four
    of fourteen positions has to say so.

  * Unpriced positions vanishing silently from every figure.

  * Editions written before the summary existed. They must gain one without
    refetching a book's worth of headlines.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from data_fetchers import allocation, portfolio


def _positions(rows):
    """A frame shaped like portfolio.value_positions output."""
    frame = pd.DataFrame(rows)
    for column in ("price", "change_pct", "day_pnl", "market_value", "cost",
                   "pnl", "pnl_pct"):
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    total = frame["market_value"].sum(skipna=True)
    frame["weight"] = frame["market_value"] / total * 100.0 if total else None
    return frame


@pytest.fixture
def book_rows():
    return [
        {"ticker": "VOO", "price": 500.0, "change_pct": 1.0, "day_pnl": 60.0,
         "market_value": 6000.0, "cost": 5000.0, "pnl": 1000.0, "pnl_pct": 20.0},
        {"ticker": "NVDA", "price": 100.0, "change_pct": 3.0, "day_pnl": 60.0,
         "market_value": 2000.0, "cost": 2500.0, "pnl": -500.0, "pnl_pct": -20.0},
        {"ticker": "PATH", "price": 10.0, "change_pct": -2.0, "day_pnl": -20.0,
         "market_value": 1000.0, "cost": 800.0, "pnl": 200.0, "pnl_pct": 25.0},
        {"ticker": "MSFT", "price": 400.0, "change_pct": 0.5, "day_pnl": 5.0,
         "market_value": 1000.0, "cost": None, "pnl": None, "pnl_pct": None},
    ]


@pytest.fixture
def sectors():
    return pd.DataFrame([
        {"sector": "Information Technology", "market_value": 5000.0, "weight_pct": 50.0},
        {"sector": "Financials", "market_value": 2000.0, "weight_pct": 20.0},
        {"sector": allocation.UNCLASSIFIED, "market_value": 1000.0, "weight_pct": 10.0},
        {"sector": "Health Care", "market_value": 1000.0, "weight_pct": 10.0},
        {"sector": "Energy", "market_value": 1000.0, "weight_pct": 10.0},
    ])


class TestMoodLabel:
    @pytest.mark.parametrize("net,expected", [
        (0.5, "RISK-ON"), (-0.5, "RISK-OFF"), (0.1, "MIXED"),
        (portfolio.MOOD_THRESHOLD, "MIXED"), (None, "MIXED"),
    ])
    def test_thresholds(self, net, expected):
        assert portfolio.mood_label(net) == expected


class TestBookSnapshot:
    def test_totals_and_day_percent(self, book_rows, sectors):
        book = portfolio.book_snapshot(_positions(book_rows), sectors)
        assert book["market_value"] == pytest.approx(10000.0)
        assert book["day_pnl"] == pytest.approx(105.0)
        # Measured against yesterday's value, not today's.
        assert book["day_pct"] == pytest.approx(105.0 / 9895.0 * 100.0)
        assert book["pnl"] == pytest.approx(700.0)
        assert book["with_basis"] == 3

    def test_extremes(self, book_rows):
        book = portfolio.book_snapshot(_positions(book_rows))
        assert book["day_best"]["ticker"] == "NVDA"
        assert book["day_worst"]["ticker"] == "PATH"
        assert book["return_best"]["ticker"] == "PATH"
        assert book["return_worst"]["ticker"] == "NVDA"

    def test_sectors_exclude_unclassified(self, book_rows, sectors):
        book = portfolio.book_snapshot(_positions(book_rows), sectors)
        names = [s["sector"] for s in book["sectors"]]
        assert allocation.UNCLASSIFIED not in names
        assert names == ["Information Technology", "Financials", "Health Care"]
        assert book["unclassified_pct"] == pytest.approx(10.0)

    def test_no_basis_means_no_pnl_not_zero(self):
        rows = [{"ticker": "VOO", "price": 500.0, "change_pct": 1.0,
                 "day_pnl": 10.0, "market_value": 1000.0}]
        book = portfolio.book_snapshot(_positions(rows))
        assert book["pnl"] is None
        assert book["pnl_pct"] is None

    def test_no_day_moves_means_no_day_pnl(self):
        rows = [{"ticker": "VOO", "price": 500.0, "market_value": 1000.0}]
        assert portfolio.book_snapshot(_positions(rows))["day_pnl"] is None

    def test_json_serialisable(self, book_rows, sectors):
        book = portfolio.book_snapshot(_positions(book_rows), sectors)
        json.dumps(book, allow_nan=False)

    def test_empty(self):
        assert portfolio.book_snapshot(pd.DataFrame()) == {}


class TestComposeNarrative:
    def test_covers_the_whole_book(self, book_rows, sectors):
        book = portfolio.book_snapshot(_positions(book_rows), sectors)
        summary = {"stories": 16, "net_sentiment": 0.2}
        digests = [{"ticker": "NVDA", "net_sentiment": 0.6},
                   {"ticker": "PATH", "net_sentiment": -0.4}]
        text = portfolio.compose_narrative(book, summary, digests)

        assert "Your book of 4 positions is worth $10,000" in text
        assert "up $105" in text
        assert "$700 (+8.4%) above cost" in text
        assert "3 of 4 priced positions with a cost basis" in text
        assert "VOO is the largest holding at 60.0%" in text
        assert "roughly 2.4 equally weighted positions" in text
        assert "VOO and NVDA are above the 20% single-position marker" in text
        assert "Information Technology (50.0%)" in text
        assert "10.0% that could not be placed" in text
        assert "NVDA led at +3.00%" in text
        assert "PATH was weakest at -2.00%" in text
        assert "16 stories lean positive" in text
        assert "most constructive on NVDA" in text
        assert "most negative on PATH" in text

    def test_drops_clauses_without_inputs(self):
        rows = [{"ticker": "VOO", "price": 500.0, "market_value": 1000.0}]
        text = portfolio.compose_narrative(
            portfolio.book_snapshot(_positions(rows)))
        assert text == "Your book of 1 position is worth $1,000."

    def test_losses_read_as_losses(self):
        rows = [{"ticker": "A", "price": 1.0, "change_pct": -5.0,
                 "day_pnl": -50.0, "market_value": 950.0, "cost": 1200.0,
                 "pnl": -250.0, "pnl_pct": -20.8}]
        text = portfolio.compose_narrative(
            portfolio.book_snapshot(_positions(rows)))
        assert "down $50" in text
        assert "$250 (-20.8%) below cost" in text

    def test_names_unpriced_positions(self, book_rows):
        rows = book_rows + [{"ticker": "DEAD", "price": None,
                             "market_value": None}]
        text = portfolio.compose_narrative(
            portfolio.book_snapshot(_positions(rows)))
        assert "No quote came back for DEAD" in text

    def test_nothing_priced(self):
        rows = [{"ticker": "DEAD", "price": None, "market_value": None}]
        text = portfolio.compose_narrative(
            portfolio.book_snapshot(_positions(rows)))
        assert text.startswith("None of your 1 position returned a quote")

    def test_empty_book(self):
        assert portfolio.compose_narrative({}) == ""


class TestLegacyEditionBackfill:
    def test_adds_summary_without_rebuilding(self, tmp_path, monkeypatch,
                                             book_rows, sectors):
        monkeypatch.setattr(config, "BRIEF_DIR", tmp_path)
        edition = portfolio.edition_date()
        legacy = {"edition": edition.isoformat(), "empty": False,
                  "positions": [], "top_stories": [],
                  "summary": {"stories": 0, "net_sentiment": None}}
        path = tmp_path / f"{edition.isoformat()}.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")

        def no_rebuild(*_args, **_kwargs):
            raise AssertionError("legacy edition was rebuilt")

        monkeypatch.setattr(portfolio, "build_brief", no_rebuild)
        monkeypatch.setattr(portfolio, "holdings", lambda *_args: pd.DataFrame())
        monkeypatch.setattr(portfolio, "value_positions",
                            lambda _frame: _positions(book_rows))
        monkeypatch.setattr(portfolio, "_sector_mix", lambda _frame: sectors)

        brief = portfolio.get_brief()
        assert brief["narrative"].startswith("Your book of 4 positions")
        assert "narrative" in json.loads(path.read_text(encoding="utf-8"))

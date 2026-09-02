"""
Sector allocation, benchmark drift and rebalancing arithmetic.

REGRESSION CONTEXT
------------------
Every failure guarded here renders a chart that looks entirely reasonable:

  * A fund filed as one sector. VOO is not "a technology position" - it is
    37% technology and ten other things. Booking it under its largest sector
    overstates that sector by the whole position; dropping it understates the
    entire book. The real portfolio this was built against is 55% VOO, so the
    difference is not marginal.

  * `quoteType` missing from the company-info whitelist made the fund
    look-through branch unreachable. Every ETF filed as UNCLASSIFIED while
    its sector weights sat one call away, and the exposure chart showed a
    diversified book as 55% unknown.

  * An unclassified slug left in the drift denominator. It understates every
    sector weight by the same amount, so the whole book reads underweight
    against any benchmark and eleven spurious ADD actions appear.

  * Hardcoded benchmark targets. Index sector weights move constantly, so a
    typed table is wrong the day after it is written and keeps rendering a
    confident drift column while it rots.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from data_fetchers import allocation


def _valued(rows):
    """A frame shaped like portfolio.value_positions output."""
    frame = pd.DataFrame(rows)
    total = frame["market_value"].sum(skipna=True)
    frame["weight"] = frame["market_value"] / total * 100.0 if total else None
    return frame


# ==========================================================================
# Sector normalisation
# ==========================================================================
class TestNormaliseSector:
    @pytest.mark.parametrize("raw,expected", [
        ("Technology", "Information Technology"),
        ("technology", "Information Technology"),
        ("Financial Services", "Financials"),
        ("financial_services", "Financials"),
        ("Healthcare", "Health Care"),
        ("healthcare", "Health Care"),
        ("Consumer Cyclical", "Consumer Discretionary"),
        ("consumer_cyclical", "Consumer Discretionary"),
        ("Consumer Defensive", "Consumer Staples"),
        ("consumer_defensive", "Consumer Staples"),
        ("Basic Materials", "Materials"),
        ("basic_materials", "Materials"),
        ("Real Estate", "Real Estate"),
        ("realestate", "Real Estate"),
    ])
    def test_both_yahoo_spellings_resolve(self, raw, expected):
        """
        Yahoo spells its sectors two ways - title case in `info["sector"]`
        and snake_case in an ETF's weightings. Treating them as different
        sectors splits one exposure across two rows, and each row on its own
        looks perfectly reasonable.
        """
        assert allocation.normalise_sector(raw) == expected

    def test_unknown_returns_none_rather_than_guessing(self):
        assert allocation.normalise_sector("Crypto") is None
        assert allocation.normalise_sector("") is None
        assert allocation.normalise_sector(None) is None

    def test_every_gics_name_maps_to_itself(self):
        for sector in config.GICS_SECTORS:
            assert allocation.normalise_sector(sector) == sector, sector


# ==========================================================================
# Exposure
# ==========================================================================
class TestSectorExposure:
    def _patch(self, monkeypatch, mapping):
        monkeypatch.setattr(allocation, "classify", lambda t: mapping[t])

    def test_single_sector_equities(self, monkeypatch):
        self._patch(monkeypatch, {
            "AAA": {"kind": "EQUITY", "sectors": {"Information Technology": 1.0},
                    "note": ""},
            "BBB": {"kind": "EQUITY", "sectors": {"Health Care": 1.0}, "note": ""},
        })
        frame = allocation.sector_exposure(_valued([
            {"ticker": "AAA", "market_value": 750.0},
            {"ticker": "BBB", "market_value": 250.0},
        ]))

        weights = dict(zip(frame["sector"], frame["weight_pct"]))
        assert weights["Information Technology"] == pytest.approx(75.0)
        assert weights["Health Care"] == pytest.approx(25.0)

    def test_fund_is_distributed_not_filed_under_one_sector(self, monkeypatch):
        """
        The core case. A 55% VOO position booked under technology would show
        technology at 55%+ when the truth is nearer 30%.
        """
        self._patch(monkeypatch, {
            "VOO": {"kind": "FUND", "note": "",
                    "sectors": {"Information Technology": 0.37,
                                "Financials": 0.12, "Health Care": 0.09,
                                "Energy": 0.42}},
        })
        frame = allocation.sector_exposure(
            _valued([{"ticker": "VOO", "market_value": 1000.0}]))

        weights = dict(zip(frame["sector"], frame["weight_pct"]))
        assert weights["Information Technology"] == pytest.approx(37.0)
        assert len(frame) == 4
        assert frame.attrs["lookthrough"][0]["ticker"] == "VOO"

    def test_weights_sum_to_one_hundred(self, monkeypatch):
        self._patch(monkeypatch, {
            "AAA": {"kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""},
            "VOO": {"kind": "FUND", "note": "",
                    "sectors": {"Information Technology": 0.6,
                                "Financials": 0.4}},
        })
        frame = allocation.sector_exposure(_valued([
            {"ticker": "AAA", "market_value": 400.0},
            {"ticker": "VOO", "market_value": 600.0},
        ]))
        assert frame["weight_pct"].sum() == pytest.approx(100.0)

    def test_unclassified_is_reported_not_spread(self, monkeypatch):
        """
        Spreading an unplaceable holding across sectors to make the
        percentages tidy would be inventing exposure. A book that is 30%
        unclassified should look 30% unclassified.
        """
        self._patch(monkeypatch, {
            "AAA": {"kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""},
            "???": {"kind": allocation.UNCLASSIFIED, "sectors": {},
                    "note": "No sector reported."},
        })
        frame = allocation.sector_exposure(_valued([
            {"ticker": "AAA", "market_value": 700.0},
            {"ticker": "???", "market_value": 300.0},
        ]))

        weights = dict(zip(frame["sector"], frame["weight_pct"]))
        assert weights[allocation.UNCLASSIFIED] == pytest.approx(30.0)
        assert frame.attrs["unclassified"][0]["ticker"] == "???"

    def test_unpriced_positions_are_excluded(self, monkeypatch):
        self._patch(monkeypatch, {
            "AAA": {"kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""},
            "DEAD": {"kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""},
        })
        frame = allocation.sector_exposure(pd.DataFrame([
            {"ticker": "AAA", "market_value": 500.0, "weight": 100.0},
            {"ticker": "DEAD", "market_value": None, "weight": None},
        ]))
        assert frame.attrs["total"] == pytest.approx(500.0)

    def test_empty_portfolio(self):
        frame = allocation.sector_exposure(pd.DataFrame())
        assert frame.empty
        assert frame.attrs["total"] == 0.0


# ==========================================================================
# Rebalancing
# ==========================================================================
class TestRebalance:
    def _setup(self, monkeypatch, holdings, targets):
        monkeypatch.setattr(allocation, "classify", lambda t: holdings[t])
        monkeypatch.setattr(
            allocation, "benchmark_weights",
            lambda key=config.DEFAULT_BENCHMARK: {
                "key": key, "label": "Test", "proxy": "TEST",
                "description": "", "weights": targets, "note": ""})

    def test_drift_and_dollar_adjustment(self, monkeypatch):
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Information Technology": 1.0}},
                     "BBB": {"kind": "EQUITY", "note": "",
                             "sectors": {"Health Care": 1.0}}},
                    {"Information Technology": 40.0, "Health Care": 60.0})

        report = allocation.rebalance(_valued([
            {"ticker": "AAA", "market_value": 600.0},
            {"ticker": "BBB", "market_value": 400.0},
        ]))
        rows = report["rows"].set_index("Sector")

        assert rows.loc["Information Technology", "Drift %"] == pytest.approx(20.0)
        assert rows.loc["Information Technology", "Adjust $"] == pytest.approx(-200.0)
        assert rows.loc["Health Care", "Adjust $"] == pytest.approx(200.0)

    def test_adjustments_net_to_zero(self, monkeypatch):
        """Rebalancing moves money between sectors; it does not create any."""
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Energy": 1.0}},
                     "BBB": {"kind": "EQUITY", "note": "",
                             "sectors": {"Utilities": 1.0}}},
                    {"Energy": 30.0, "Utilities": 20.0, "Financials": 50.0})

        report = allocation.rebalance(_valued([
            {"ticker": "AAA", "market_value": 500.0},
            {"ticker": "BBB", "market_value": 500.0},
        ]))
        assert report["rows"]["Adjust $"].sum() == pytest.approx(0.0, abs=1e-6)

    def test_unclassified_is_out_of_the_drift_denominator(self, monkeypatch):
        """
        Leaving it in understates every sector weight by the same amount, so
        the whole book reads underweight and eleven spurious ADDs appear.
        """
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Energy": 1.0}},
                     "???": {"kind": allocation.UNCLASSIFIED, "sectors": {},
                             "note": "unknown"}},
                    {"Energy": 100.0})

        report = allocation.rebalance(_valued([
            {"ticker": "AAA", "market_value": 500.0},
            {"ticker": "???", "market_value": 500.0},
        ]))
        rows = report["rows"].set_index("Sector")

        # Energy is 100% of the CLASSIFIED book, so no drift at all.
        assert rows.loc["Energy", "Current %"] == pytest.approx(100.0)
        assert rows.loc["Energy", "Drift %"] == pytest.approx(0.0)
        assert report["unclassified_pct"] == pytest.approx(50.0)

    def test_small_drift_produces_no_action(self, monkeypatch):
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Energy": 1.0}},
                     "BBB": {"kind": "EQUITY", "note": "",
                             "sectors": {"Utilities": 1.0}}},
                    {"Energy": 51.0, "Utilities": 49.0})

        report = allocation.rebalance(_valued([
            {"ticker": "AAA", "market_value": 500.0},
            {"ticker": "BBB", "market_value": 500.0},
        ]))
        assert report["actions"] == []

    def test_zero_exposure_sector_is_an_open_not_an_add(self, monkeypatch):
        """
        "You hold none of this" is a different decision from "you hold a
        little less than the index", and the action label says which.
        """
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Energy": 1.0}}},
                    {"Energy": 70.0, "Financials": 30.0})

        report = allocation.rebalance(
            _valued([{"ticker": "AAA", "market_value": 1000.0}]))
        actions = {a["sector"]: a["action"] for a in report["actions"]}

        assert actions["Financials"] == "OPEN"
        assert actions["Energy"] == "TRIM"

    def test_actions_are_ordered_by_size_of_gap(self, monkeypatch):
        self._setup(monkeypatch,
                    {"AAA": {"kind": "EQUITY", "note": "",
                             "sectors": {"Energy": 1.0}}},
                    {"Energy": 20.0, "Financials": 50.0, "Utilities": 30.0})

        report = allocation.rebalance(
            _valued([{"ticker": "AAA", "market_value": 1000.0}]))
        gaps = [abs(a["drift_pct"]) for a in report["actions"]]
        assert gaps == sorted(gaps, reverse=True)

    def test_missing_benchmark_declines_rather_than_defaulting(self, monkeypatch):
        """
        With no live targets the honest output is no drift column, not a
        fallback to a stale hardcoded table.
        """
        monkeypatch.setattr(allocation, "classify", lambda t: {
            "kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""})
        monkeypatch.setattr(
            allocation, "benchmark_weights",
            lambda key=config.DEFAULT_BENCHMARK: {
                "key": key, "weights": {}, "note": "proxy published nothing"})

        report = allocation.rebalance(
            _valued([{"ticker": "AAA", "market_value": 100.0}]))
        assert report["rows"].empty
        assert report["actions"] == []
        assert report["note"]

    def test_empty_portfolio_is_a_note_not_a_crash(self, monkeypatch):
        report = allocation.rebalance(pd.DataFrame())
        assert report["actions"] == []
        assert report["note"]


# ==========================================================================
# Concentration
# ==========================================================================
class TestConcentration:
    def test_effective_positions_is_one_over_hhi(self):
        stats = allocation.concentration(_valued([
            {"ticker": "A", "market_value": 250.0},
            {"ticker": "B", "market_value": 250.0},
            {"ticker": "C", "market_value": 250.0},
            {"ticker": "D", "market_value": 250.0},
        ]))
        assert stats["effective_positions"] == pytest.approx(4.0)

    def test_one_dominant_name_collapses_the_effective_count(self):
        """
        The number worth knowing: twenty names where one is 60% behaves like
        about three, and a plain weight list does not say so.
        """
        rows = [{"ticker": "BIG", "market_value": 600.0}]
        rows += [{"ticker": f"S{i}", "market_value": 400.0 / 19}
                 for i in range(19)]
        stats = allocation.concentration(_valued(rows))

        assert stats["positions"] == 20
        assert stats["effective_positions"] < 4.0

    def test_flags_a_position_over_the_marker(self):
        stats = allocation.concentration(_valued([
            {"ticker": "BIG", "market_value": 900.0},
            {"ticker": "SMALL", "market_value": 100.0},
        ]))
        assert any("BIG" in flag for flag in stats["flags"])

    def test_unpriced_positions_are_called_out(self):
        """
        They were vanishing from the weight chart silently, so the
        percentages described a smaller book than the one being held.
        """
        frame = pd.DataFrame([
            {"ticker": "A", "market_value": 500.0, "weight": 100.0},
            {"ticker": "DEAD", "market_value": None, "weight": None},
        ])
        stats = allocation.concentration(frame)
        assert any("no price" in flag for flag in stats["flags"])

    def test_empty(self):
        assert allocation.concentration(pd.DataFrame())["positions"] == 0


# ==========================================================================
# Schema
# ==========================================================================
class TestSchema:
    def test_serialisable(self, monkeypatch):
        import json

        monkeypatch.setattr(allocation, "classify", lambda t: {
            "kind": "EQUITY", "sectors": {"Energy": 1.0}, "note": ""})
        monkeypatch.setattr(
            allocation, "benchmark_weights",
            lambda key=config.DEFAULT_BENCHMARK: {
                "key": key, "label": "Test", "proxy": "TEST",
                "description": "", "weights": {"Energy": 50.0,
                                               "Financials": 50.0},
                "note": ""})

        report = allocation.rebalance(
            _valued([{"ticker": "AAA", "market_value": 1000.0}]))
        payload = allocation.to_schema(report)

        json.dumps(payload)
        assert payload["sector_allocation"]
        assert payload["rebalance_rows"]
        assert payload["actions"]
        assert "exposure" not in payload

    def test_keys_are_stable_on_an_empty_report(self):
        payload = allocation.to_schema({})
        for key in ("as_of", "benchmark", "total_value", "sector_allocation",
                    "rebalance_rows", "actions", "concentration",
                    "unclassified_holdings", "disclaimer"):
            assert key in payload


# ==========================================================================
# Configuration invariants
# ==========================================================================
class TestAllocationConfig:
    def test_eleven_gics_sectors(self):
        assert len(config.GICS_SECTORS) == 11
        assert len(set(config.GICS_SECTORS)) == 11

    def test_crosswalk_only_targets_real_sectors(self):
        assert set(config.YAHOO_TO_GICS.values()) <= set(config.GICS_SECTORS)

    def test_every_gics_sector_is_reachable_from_yahoo(self):
        """A sector with no inbound mapping can never receive exposure."""
        reachable = set(config.YAHOO_TO_GICS.values())
        assert reachable == set(config.GICS_SECTORS)

    def test_benchmarks_name_a_proxy_fund_not_a_weight_table(self):
        """
        Targets are read from a live fund. A hardcoded weight table would be
        wrong the day after it was written and would keep rendering a
        confident drift column while it rotted.
        """
        for benchmark in config.ALLOCATION_BENCHMARKS.values():
            assert benchmark.proxy
            assert not hasattr(benchmark, "weights")

    def test_default_benchmark_exists(self):
        assert config.DEFAULT_BENCHMARK in config.ALLOCATION_BENCHMARKS


# ==========================================================================
# Donut rendering
# ==========================================================================
class TestDonut:
    """
    The share-of-whole view. Guards three things that are wrong in ways a
    glance would not catch: reordered slices, a pooled remainder that loses
    value, and a palette that repeats a colour on two slices.
    """

    @pytest.fixture(autouse=True)
    def _template(self):
        from ui.terminal_theme import _register_plotly_template
        _register_plotly_template()

    def _frame(self, n=5):
        return pd.DataFrame({
            "label": [f"S{i}" for i in range(n)],
            "value": [float(n - i) for i in range(n)],
        })

    def test_slice_order_is_preserved(self):
        """
        Plotly re-sorts pie slices by default, which silently breaks the
        correspondence between the donut and the table printed beside it.
        """
        from ui import components as ui

        figure = ui.donut(self._frame(5), "label", "value")
        assert figure.data[0].sort is False
        assert list(figure.data[0].labels) == ["S0", "S1", "S2", "S3", "S4"]

    def test_pooling_preserves_the_total(self):
        from ui import components as ui

        frame = self._frame(20)
        figure = ui.donut(frame, "label", "value", max_slices=6)

        assert len(figure.data[0].labels) == 6
        assert figure.data[0].labels[-1] == "OTHER (15)"
        assert sum(figure.data[0].values) == pytest.approx(frame["value"].sum())

    def test_no_pooling_below_the_limit(self):
        from ui import components as ui

        figure = ui.donut(self._frame(5), "label", "value", max_slices=12)
        assert not any("OTHER" in str(l) for l in figure.data[0].labels)

    def test_non_positive_values_are_dropped(self):
        """A zero or negative slice has no meaning on a share-of-whole chart."""
        from ui import components as ui

        frame = pd.DataFrame({"label": ["A", "B", "C"],
                              "value": [10.0, 0.0, -5.0]})
        figure = ui.donut(frame, "label", "value")
        assert list(figure.data[0].labels) == ["A"]

    def test_empty_input_renders_nothing_not_a_crash(self):
        from ui import components as ui

        assert ui.donut(pd.DataFrame(), "label", "value").data == ()
        assert ui.donut(pd.DataFrame({"label": ["A"], "value": [0]}),
                        "label", "value").data == ()

    def test_centre_annotation_carries_the_total(self):
        """The one number a donut otherwise throws away."""
        from ui import components as ui

        figure = ui.donut(self._frame(3), "label", "value",
                          center_value="$941", center_label="book value")
        text = figure.layout.annotations[0].text
        assert "$941" in text and "book value" in text

    def test_palette_covers_every_gics_sector(self):
        """
        Eleven sectors against an eight-colour cycle put two slices in the
        same colour, which reads as one category split in two.
        """
        import plotly.io as pio

        colorway = pio.templates["openterm"].layout.colorway
        assert len(colorway) >= len(config.GICS_SECTORS)
        assert len(set(colorway)) == len(colorway)

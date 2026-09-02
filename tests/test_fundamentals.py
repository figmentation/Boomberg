"""
Fundamental scoring, valuation arithmetic and the verdict gates.

REGRESSION CONTEXT
------------------
The failures guarded here all render as confident, plausible output:

  * A DCF whose discount rate does not exceed terminal growth returns a
    negative or astronomical fair value from a Gordon denominator at or below
    zero. Nothing about the number looks wrong on a tile.

  * A missing metric scored as zero rather than skipped drags an axis down
    exactly as though the company had performed badly. The report then reads
    as a judgement on the business when it is a judgement on which fetches
    succeeded.

  * Source-coupled row labels. The engine prefers SEC XBRL and falls back to
    Yahoo, which names every line differently. Before `_ROW_ALIASES` the
    fallback fetched statements perfectly and then produced an empty metric
    set, so a company with complete filings reported INSUFFICIENT DATA.

  * And the two the brief names explicitly: a low P/E must not by itself
    produce BUY, and a high P/E must not by itself produce SELL. One summed
    score cannot express "cheap but deteriorating", which is why the verdict
    is a gate on three axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data_fetchers import fundamentals as fund
from data_fetchers import valuation as val
from utils import fundamental_math as fm


# ==========================================================================
# Pure maths
# ==========================================================================
class TestCagr:
    def test_doubling_over_five_years(self):
        assert fm.cagr(100, 200, 5) == pytest.approx(0.148698, abs=1e-6)

    def test_flat(self):
        assert fm.cagr(100, 100, 3) == pytest.approx(0.0)

    @pytest.mark.parametrize("first,last", [(-5, 10), (0, 10), (10, -5), (10, 0)])
    def test_non_positive_endpoints_return_none(self, first, last):
        """
        A loss-to-profit swing has no compound rate.

        The root of a negative ratio is either complex or, worse, a real
        number that looks entirely reasonable in a table.
        """
        assert fm.cagr(first, last, 3) is None

    def test_zero_years(self):
        assert fm.cagr(100, 200, 0) is None


class TestScoreBand:
    ANCHORS = [(0, 0), (0.08, 40), (0.15, 70), (0.25, 90), (0.40, 100)]

    def test_lands_on_an_anchor(self):
        assert fm.score_band(0.15, self.ANCHORS) == pytest.approx(70)

    def test_interpolates_between_anchors(self):
        # 0.19 is 40% of the way from 0.15 to 0.25, so 70 + 0.4*(90-70).
        assert fm.score_band(0.19, self.ANCHORS) == pytest.approx(78)

    def test_clamps_outside_the_table(self):
        assert fm.score_band(3.0, self.ANCHORS) == 100
        assert fm.score_band(-1.0, self.ANCHORS) == 0

    def test_descending_points_express_lower_is_better(self):
        """Debt/EBITDA anchors descend; a low reading must score high."""
        anchors = [(0, 100), (2.5, 70), (6.0, 10)]
        assert fm.score_band(0.0, anchors) == 100
        assert fm.score_band(6.0, anchors) == 10
        assert fm.score_band(2.5, anchors) == 70

    def test_non_numeric_returns_none(self):
        assert fm.score_band(None, self.ANCHORS) is None
        assert fm.score_band(float("nan"), self.ANCHORS) is None


class TestWeightedScore:
    def test_renormalises_over_resolved_components(self):
        components = [
            fm.ScoreComponent("a", 1, "x", 80, 1.0),
            fm.ScoreComponent("b", None, "x", None, 5.0),
            fm.ScoreComponent("c", 1, "x", 60, 1.0),
        ]
        score, count = fm.weighted_score(components)
        assert count == 2
        assert score == pytest.approx(70.0)

    def test_missing_component_is_not_scored_zero(self):
        """
        The guard on the quiet failure: were the unresolved component counted
        as zero it would carry its weight of 5 and drag the axis to ~23.
        """
        with_missing = [
            fm.ScoreComponent("a", 1, "x", 80, 1.0),
            fm.ScoreComponent("b", None, "x", None, 5.0),
            fm.ScoreComponent("c", 1, "x", 60, 1.0),
        ]
        without = [
            fm.ScoreComponent("a", 1, "x", 80, 1.0),
            fm.ScoreComponent("c", 1, "x", 60, 1.0),
        ]
        assert fm.weighted_score(with_missing)[0] == fm.weighted_score(without)[0]

    def test_single_component_is_not_an_axis(self):
        assert fm.weighted_score([fm.ScoreComponent("a", 1, "x", 90, 1.0)])[0] is None


class TestDcf:
    def test_matches_a_hand_built_discounting_table(self):
        """No fade, so each year is a plain geometric term."""
        result = fm.dcf_per_share(
            fcf0=100.0, growth=0.05, fade_to=0.05, years=3,
            terminal_growth=0.02, discount_rate=0.10,
            net_cash=0.0, shares=1.0)

        flows = [100 * 1.05 ** year for year in (1, 2, 3)]
        present = sum(f / 1.10 ** (i + 1) for i, f in enumerate(flows))
        terminal = flows[-1] * 1.02 / (0.10 - 0.02)
        expected = present + terminal / 1.10 ** 3

        assert result == pytest.approx(expected, rel=1e-9)

    def test_discount_rate_at_or_below_terminal_growth_returns_none(self):
        """
        The most common way a DCF produces confident nonsense: the Gordon
        denominator goes to zero or negative and the fair value comes back
        enormous or below zero, with nothing on screen to signal it.
        """
        assert fm.dcf_per_share(100, 0.05, 0.03, 5, 0.10, 0.10, shares=1) is None
        assert fm.dcf_per_share(100, 0.05, 0.03, 5, 0.12, 0.10, shares=1) is None

    def test_negative_base_cash_flow_returns_none(self):
        assert fm.dcf_per_share(-50, 0.05, 0.03, 5, 0.02, 0.10, shares=1) is None

    def test_missing_shares_returns_none(self):
        assert fm.dcf_per_share(100, 0.05, 0.03, 5, 0.02, 0.10, shares=0) is None

    def test_net_cash_lifts_the_value_one_for_one(self):
        base = fm.dcf_per_share(100, 0.04, 0.03, 5, 0.02, 0.09,
                                net_cash=0, shares=10)
        with_cash = fm.dcf_per_share(100, 0.04, 0.03, 5, 0.02, 0.09,
                                     net_cash=500, shares=10)
        assert with_cash - base == pytest.approx(50.0)


class TestCapm:
    def test_ordinary_beta(self):
        assert fm.capm_discount_rate(0.045, 1.2, 0.05) == pytest.approx(0.105)

    def test_extreme_beta_is_clamped(self):
        """
        yfinance publishes betas below zero and above three on thin names.
        Unbounded CAPM then returns a rate under terminal growth, which makes
        the DCF explode, or over 40%, which values everything at nothing.
        """
        assert fm.capm_discount_rate(0.045, 8.0, 0.05) == 0.16
        assert fm.capm_discount_rate(0.045, -3.0, 0.05) == 0.06


class TestMarginOfSafety:
    def test_denominator_is_fair_value_not_price(self):
        """50 against a fair value of 100 is a 50% margin, not 100%."""
        assert fm.margin_of_safety(50, 100) == pytest.approx(0.5)

    def test_premium_is_negative(self):
        assert fm.margin_of_safety(150, 100) == pytest.approx(-0.5)

    def test_trigger_price(self):
        assert fm.price_for_margin_of_safety(100, 0.25) == pytest.approx(75.0)


# ==========================================================================
# Statement access
# ==========================================================================
class TestRowAliases:
    """
    The engine reads SEC XBRL first and Yahoo second, and the two name every
    line differently. Before the alias map the fallback path fetched perfect
    statements and then produced an empty metric set, so a company with
    complete filings reported INSUFFICIENT DATA.
    """

    def test_sec_labels_resolve(self):
        frame = pd.DataFrame({"FY2025": [100.0]}, index=["Revenue"])
        assert fund._series(frame, "Revenue") == [100.0]

    def test_yahoo_labels_resolve_through_the_same_call(self):
        frame = pd.DataFrame({"2025": [100.0]}, index=["Total Revenue"])
        assert fund._series(frame, "Revenue") == [100.0]

    @pytest.mark.parametrize("canonical,yahoo_label", [
        ("Revenue", "Total Revenue"),
        ("CapEx", "Capital Expenditure"),
        ("Total Equity", "Stockholders Equity"),
        ("Cash & Equivalents", "Cash And Cash Equivalents"),
        ("EPS Diluted", "Diluted EPS"),
        ("Diluted Shares", "Diluted Average Shares"),
        ("Operating Income", "EBIT"),
        ("Total Current Assets", "Current Assets"),
    ])
    def test_every_fallback_label_is_mapped(self, canonical, yahoo_label):
        frame = pd.DataFrame({"2025": [42.0]}, index=[yahoo_label])
        assert fund._series(frame, canonical) == [42.0], canonical

    def test_absent_row_is_empty_not_an_exception(self):
        frame = pd.DataFrame({"FY2025": [1.0]}, index=["Something Else"])
        assert fund._series(frame, "Revenue") == []


# ==========================================================================
# Provenance
# ==========================================================================
class TestProvenance:
    def test_missing_value_is_demoted_to_unavailable(self):
        """A Fact must never claim CALCULATED provenance with no value."""
        fact = fund._fact(None, fund.CALCULATED, "x")
        assert fact.source == fund.UNAVAILABLE
        assert fact.value is None
        assert fact.note

    def test_nan_is_treated_as_missing(self):
        assert fund._fact(float("nan"), fund.FILED).source == fund.UNAVAILABLE

    def test_known_value_keeps_its_source(self):
        fact = fund._fact(1.5, fund.CALCULATED, "x")
        assert fact.source == fund.CALCULATED and fact.known

    def test_proxies_are_labelled_proxy_not_filed(self):
        """
        Moat and capital allocation are stand-ins, and must be legible as
        such. A fabricated durability rating reads identically to a real one.
        """
        roic = [0.20, 0.22, 0.19, 0.21]
        gross = [50.0, 48.0, 47.0, 46.0]
        revenue = [100.0, 100.0, 100.0, 100.0]
        assert fund._moat_proxy(roic, gross, revenue).source == fund.PROXY

    def test_proxy_refuses_on_thin_history(self):
        assert fund._moat_proxy([0.2], [50.0], [100.0]).value is None


# ==========================================================================
# The two constraints the brief names
# ==========================================================================
def _axes(quality: float, value: float, risk: float):
    """Axis blocks scoring to the requested composites."""
    def axis(score):
        return {"score": score,
                "components": [fm.ScoreComponent("x", 1, "x", score, 1.0),
                               fm.ScoreComponent("y", 1, "x", score, 1.0)],
                "resolved": 2}
    return {
        "business_quality": axis(quality),
        "growth": axis(quality),
        "financial_health": axis(quality),
        "valuation": axis(value),
        "risk": axis(risk),
    }


class TestVerdictConstraints:
    GENERIC = config.FUNDAMENTAL_PROFILES["GENERIC"]
    NO_FLAGS = {"flags": [], "count": 0}

    def test_cheap_but_deteriorating_is_not_a_buy(self):
        """
        The explicit instruction: do NOT recommend BUY simply because the
        P/E is low. A business scoring 40 for quality at a bargain price is
        a value trap, and the quality gate is tested before the value gate.
        """
        result = fund.verdict(_axes(quality=40.0, value=95.0, risk=20.0),
                              95.0, 20.0, self.NO_FLAGS, self.GENERIC)
        assert result["verdict"] not in ("BUY", "STRONG BUY")
        assert result["verdict"] == "SELL"

    def test_expensive_but_excellent_is_not_a_sell(self):
        """
        The mirror instruction: do NOT recommend SELL simply because the P/E
        is high. Expensive is not the same as broken.
        """
        result = fund.verdict(_axes(quality=90.0, value=30.0, risk=35.0),
                              30.0, 35.0, self.NO_FLAGS, self.GENERIC)
        assert result["verdict"] not in ("SELL", "STRONG SELL")
        assert result["verdict"] == "HOLD"

    @pytest.mark.parametrize("quality,value,risk,expected", [
        (85.0, 80.0, 30.0, "STRONG BUY"),
        (65.0, 60.0, 50.0, "BUY"),
        (70.0, 40.0, 40.0, "HOLD"),
        (40.0, 60.0, 50.0, "SELL"),
        (30.0, 10.0, 80.0, "STRONG SELL"),
    ])
    def test_every_label_is_reachable(self, quality, value, risk, expected):
        result = fund.verdict(_axes(quality, value, risk), value, risk,
                              self.NO_FLAGS, self.GENERIC)
        assert result["verdict"] == expected

    def test_red_flags_override_a_cheap_weak_business(self):
        """Cheapness caused by deteriorating accounting is not an opportunity."""
        flags = {"flags": [{"flag": "a", "detail": ""},
                           {"flag": "b", "detail": ""}], "count": 2}
        result = fund.verdict(_axes(quality=48.0, value=90.0, risk=30.0),
                              90.0, 30.0, flags, self.GENERIC)
        assert result["verdict"] == "STRONG SELL"

    def test_missing_axis_fails_a_gate_rather_than_passing_it(self):
        axes = _axes(85.0, 80.0, 30.0)
        result = fund.verdict(axes, None, 30.0, self.NO_FLAGS, self.GENERIC)
        assert result["verdict"] not in ("BUY", "STRONG BUY")


# ==========================================================================
# Earnings quality
# ==========================================================================
class TestEarningsQuality:
    def _metrics(self, **values):
        return {key: fund._fact(value, fund.CALCULATED, "x")
                for key, value in values.items()}

    def test_weak_conversion_flags(self):
        result = fund.earnings_quality(self._metrics(fcf_to_net_income=0.4))
        assert result["count"] == 1
        assert "40%" in result["flags"][0]["detail"]

    def test_healthy_numbers_raise_nothing(self):
        result = fund.earnings_quality(self._metrics(
            fcf_to_net_income=1.1, accruals_ratio=-0.01,
            receivables_vs_revenue=2.0, inventory_vs_revenue=1.0,
            gaap_vs_adjusted_gap=0.05, sbc_to_revenue=0.02))
        assert result["count"] == 0

    def test_missing_metric_does_not_flag(self):
        """An absent number is not evidence of a problem."""
        assert fund.earnings_quality({})["count"] == 0

    def test_flags_quote_their_numbers(self):
        result = fund.earnings_quality(self._metrics(receivables_vs_revenue=30.0))
        assert "30" in result["flags"][0]["detail"]


# ==========================================================================
# Profiles
# ==========================================================================
class TestProfiles:
    def _facts(self, sector=None, industry=None):
        return {"info": {"sectorKey": sector, "industryKey": industry}}

    def test_bank_routes_on_its_own_classification(self):
        profile = fund.detect_profile(self._facts(industry="banks-regional"))
        assert profile.key == "BANK"
        assert profile.valuation_anchor == "book"

    def test_bank_is_not_scored_on_debt_to_ebitda(self):
        """
        Category error, not conservatism: a lender funds itself with deposits
        and debt by design, so leverage ratios built for industrials say
        nothing about it.
        """
        profile = config.FUNDAMENTAL_PROFILES["BANK"]
        assert "net_debt_to_ebitda" in profile.skip_metrics
        assert "interest_coverage" in profile.skip_metrics

        metrics = {"net_debt_to_ebitda": fund._fact(9.0, fund.CALCULATED, "x")}
        assert fund._component("net_debt_to_ebitda", "Leverage", metrics,
                               1.0, profile) is None

    def test_high_growth_tech_needs_both_sector_and_growth(self):
        facts = self._facts(sector="technology")
        assert fund.detect_profile(facts, 0.05).key == "GENERIC"
        assert fund.detect_profile(facts, 0.40).key == "HIGH_GROWTH_TECH"

    def test_unknown_classification_falls_back_to_generic(self):
        assert fund.detect_profile(self._facts()).key == "GENERIC"

    def test_every_profile_names_what_it_cannot_source(self):
        """
        The honest half of the profile system. A REIT profile that quietly
        omitted NAV and occupancy would read as complete.
        """
        for key in ("BANK", "INSURANCE", "REIT", "COMMODITY",
                    "HIGH_GROWTH_TECH"):
            assert config.FUNDAMENTAL_PROFILES[key].unavailable, key

    def test_profile_skip_lists_reference_real_metrics(self):
        """Guards a typo in skip_metrics silently skipping nothing."""
        known = set(config.SCORE_ANCHORS) | {
            "fcf_margin", "fcf_cagr_3y", "fcf_to_net_income", "fcf_yield",
            "gross_margin", "net_margin", "eps_cagr_3y", "revenue_cagr_3y",
            "cash_to_debt", "current_ratio", "interest_coverage",
            "net_debt_to_ebitda", "risk_leverage", "pe_vs_own_history",
            "pe_vs_peers", "operating_margin",
        }
        for profile in config.FUNDAMENTAL_PROFILES.values():
            for metric in profile.skip_metrics:
                assert metric in known, f"{profile.key}: {metric}"


# ==========================================================================
# Confidence
# ==========================================================================
class TestConfidence:
    PROFILE = config.FUNDAMENTAL_PROFILES["GENERIC"]

    def _call(self, facts=None, metrics=None, axes=None, valuation=None):
        return fund.confidence(
            facts or {"statement_source": "SEC XBRL",
                      "periods": ["FY%d" % y for y in range(2018, 2026)]},
            metrics or {},
            axes or {"business_quality": {"score": 80.0}},
            valuation or {"dispersion": 0.2},
            self.PROFILE)

    def test_vendor_statements_cost_confidence(self):
        sec = self._call()["score"]
        yahoo = self._call(facts={"statement_source": "yfinance",
                                  "periods": ["2025", "2024", "2023", "2022"]})["score"]
        assert yahoo < sec

    def test_wide_dispersion_costs_confidence(self):
        tight = self._call(valuation={"dispersion": 0.1})["score"]
        wide = self._call(valuation={"dispersion": 1.2})["score"]
        assert wide < tight

    def test_unscored_axis_costs_confidence(self):
        scored = self._call()["score"]
        unscored = self._call(axes={"business_quality": {"score": None}})["score"]
        assert unscored < scored

    def test_every_deduction_is_itemised(self):
        result = self._call(facts={"statement_source": "yfinance",
                                   "periods": ["2025", "2024"]})
        assert result["deductions"]
        for deduction in result["deductions"]:
            assert deduction["reason"] and deduction["points"] > 0

    def test_unsourceable_qualitative_always_deducts(self):
        """
        Market share and industry outlook are missing from every report, so
        no run should ever read as fully confident.
        """
        assert self._call()["score"] < 100.0

    def test_score_never_leaves_the_range(self):
        result = fund.confidence(
            {"statement_source": fund.UNAVAILABLE, "periods": []},
            {f"m{i}": fund._fact(None, fund.CALCULATED) for i in range(40)},
            {name: {"score": None} for name in
             ("business_quality", "growth", "financial_health", "valuation")},
            {"dispersion": None},
            config.FUNDAMENTAL_PROFILES["REIT"])
        assert 0.0 <= result["score"] <= 100.0


# ==========================================================================
# Valuation assembly
# ==========================================================================
class TestValuationAssembly:
    def test_scenarios_span_the_methods(self):
        methods = [{"method": "a", "per_share": 80.0},
                   {"method": "b", "per_share": 100.0},
                   {"method": "c", "per_share": 150.0},
                   {"method": "d", "per_share": None}]
        bands = val.scenarios(methods, {"growth": 0.05, "fade_to": 0.03,
                                        "horizon_years": 5,
                                        "terminal_growth": 0.02,
                                        "discount_rate": 0.09},
                              {"info": {}}, {})
        assert bands["bear"] == 80.0
        assert bands["base"] == 100.0
        assert bands["bull"] == 150.0

    def test_bear_base_bull_are_ordered(self):
        methods = [{"method": "a", "per_share": v}
                   for v in (12.0, 30.0, 55.0, 91.0)]
        bands = val.scenarios(methods, {"growth": 0.05, "fade_to": 0.03,
                                        "horizon_years": 5,
                                        "terminal_growth": 0.02,
                                        "discount_rate": 0.09},
                              {"info": {}}, {})
        assert bands["bear"] <= bands["base"] <= bands["bull"]

    def test_no_resolved_method_gives_no_fair_value(self):
        bands = val.scenarios([{"method": "a", "per_share": None}], {},
                              {"info": {}}, {})
        assert bands == {"bear": None, "base": None, "bull": None}

    def test_dispersion_needs_two_methods(self):
        assert val.dispersion([{"method": "a", "per_share": 10.0}]) is None
        assert val.dispersion(
            [{"method": "a", "per_share": 50.0},
             {"method": "b", "per_share": 150.0}]) == pytest.approx(1.0)

    def test_column_year_reads_both_source_formats(self):
        """SEC columns are 'FY2025' strings; Yahoo columns are Timestamps."""
        assert val._column_year("FY2025") == 2025
        assert val._column_year(pd.Timestamp("2025-12-31")) == 2025
        assert val._column_year("nonsense") is None

    def test_growth_assumption_is_a_median_not_a_first_hit(self):
        """
        A single window is fragile in a way that changes the answer: Apple's
        3-year FCF CAGR is negative purely because FY2022 was a peak, and
        feeding that to a ten-year DCF compounds one dip into a decade of
        decline. The median across windows is robust to any one of them.
        """
        metrics = {
            "fcf_cagr_3y": fund._fact(-0.04, fund.CALCULATED, "x"),
            "fcf_cagr_5y": fund._fact(0.06, fund.CALCULATED, "x"),
            "revenue_cagr_5y": fund._fact(0.09, fund.CALCULATED, "x"),
            "revenue_cagr_3y": fund._fact(0.02, fund.CALCULATED, "x"),
        }
        assumptions = val.build_assumptions(metrics, {"beta": 1.0})
        assert assumptions["growth"] == pytest.approx(0.04)

    def test_growth_assumption_is_capped(self):
        metrics = {"fcf_cagr_5y": fund._fact(0.90, fund.CALCULATED, "x"),
                   "revenue_cagr_5y": fund._fact(0.85, fund.CALCULATED, "x")}
        assumptions = val.build_assumptions(metrics, {"beta": 1.0})
        assert assumptions["growth"] == config.DCF.growth_cap

    def test_overrides_win(self):
        metrics = {"fcf_cagr_5y": fund._fact(0.05, fund.CALCULATED, "x")}
        assumptions = val.build_assumptions(metrics, {"beta": 1.0},
                                            {"growth": 0.12})
        assert assumptions["growth"] == 0.12

    def test_assumptions_carry_their_derivation(self):
        assumptions = val.build_assumptions({}, {"beta": 1.0})
        for key in ("growth", "discount_rate", "terminal_growth"):
            assert assumptions["_provenance"][key].startswith("ASSUMPTION")

    def test_justified_book_refuses_when_growth_exceeds_cost_of_equity(self):
        result = val._justified_book_value(
            1000.0, 100.0, 0.15,
            {"terminal_growth": 0.12, "discount_rate": 0.10},
            config.FUNDAMENTAL_PROFILES["BANK"])
        assert result["per_share"] is None
        assert result["note"]

    def test_justified_book_matches_the_identity(self):
        result = val._justified_book_value(
            1000.0, 100.0, 0.15,
            {"terminal_growth": 0.025, "discount_rate": 0.10},
            config.FUNDAMENTAL_PROFILES["BANK"])
        expected = (0.15 - 0.025) / (0.10 - 0.025) * 10.0
        assert result["per_share"] == pytest.approx(expected)


# ==========================================================================
# Configuration invariants
# ==========================================================================
class TestScoringConfig:
    def test_anchors_are_ascending_in_metric_value(self):
        for name, anchors in config.SCORE_ANCHORS.items():
            values = [x for x, _ in anchors]
            assert values == sorted(values), name

    def test_anchor_points_stay_in_range(self):
        for name, anchors in config.SCORE_ANCHORS.items():
            for _, points in anchors:
                assert 0 <= points <= 100, name

    def test_every_verdict_label_has_a_colour(self):
        labels = {label for label, _ in config.VERDICT_RULES}
        labels |= {config.VERDICT_DEFAULT, "INSUFFICIENT DATA"}
        for label in labels:
            assert label in config.VERDICT_COLOURS, label
            assert hasattr(config.THEME, config.VERDICT_COLOURS[label])

    def test_sell_gates_are_evaluated_before_buy_gates(self):
        """
        Order is load-bearing. If a BUY rule were tested first, a cheap
        deteriorating business would match on value before the quality gate
        ever ran - which is the exact failure the brief rules out.
        """
        labels = [label for label, _ in config.VERDICT_RULES]
        assert labels.index("STRONG SELL") < labels.index("BUY")
        assert labels.index("SELL") < labels.index("STRONG BUY")

    def test_terminal_growth_stays_below_the_discount_floor(self):
        """Otherwise every DCF returns None and the anchor silently vanishes."""
        assert config.DCF.terminal_growth < 0.06


# ==========================================================================
# XBRL unit selection
# ==========================================================================
class TestXbrlUnitSelection:
    """
    REGRESSION: `_facts_for_tag` used to pick the fact unit as
    `"USD" if "USD" in units else next(iter(units))` — first key in dict
    order. Coca-Cola tags EarningsPerShareDiluted under BOTH "pure" (four
    stray 10-Q facts) and "USD/shares" (fifty-one 10-K facts). "USD" is not
    an exact key match, so the fallback took "pure", found no annual facts,
    and returned nothing.

    KO's diluted EPS came back empty from a filing that reports it on every
    page, and both P/E-based valuations silently dropped out of the
    comparison. Nothing on screen indicated a missing input.
    """

    def _node(self, units):
        return {"units": units}

    def _annual(self, value, year=2025):
        return {"start": f"{year}-01-01", "end": f"{year}-12-31",
                "val": value, "form": "10-K", "filed": f"{year + 1}-02-20"}

    def _quarterly(self, value, year=2025):
        return {"start": f"{year}-01-01", "end": f"{year}-03-31",
                "val": value, "form": "10-Q", "filed": f"{year}-04-20"}

    def test_picks_the_unit_carrying_the_wanted_form(self):
        from data_fetchers import equities

        node = self._node({
            "pure": [self._quarterly(0.1)],
            "USD/shares": [self._annual(3.04)],
        })
        result = equities._facts_for_tag(node, annual=True)
        assert [value for _, value in result.values()] == [3.04]

    def test_dict_order_does_not_decide(self):
        """The same node with the junk unit first must give the same answer."""
        from data_fetchers import equities

        forward = equities._facts_for_tag(self._node({
            "USD/shares": [self._annual(3.04)],
            "pure": [self._quarterly(0.1)]}), annual=True)
        reverse = equities._facts_for_tag(self._node({
            "pure": [self._quarterly(0.1)],
            "USD/shares": [self._annual(3.04)]}), annual=True)
        assert forward == reverse

    def test_plain_usd_still_wins_where_it_has_the_facts(self):
        from data_fetchers import equities

        node = self._node({
            "USD": [self._annual(1_000.0)],
            "EUR": [self._annual(900.0)],
        })
        result = equities._facts_for_tag(node, annual=True)
        assert [value for _, value in result.values()] == [1_000.0]

    def test_no_matching_form_returns_empty(self):
        from data_fetchers import equities

        node = self._node({"USD/shares": [self._quarterly(0.5)]})
        assert equities._facts_for_tag(node, annual=True) == {}


# ==========================================================================
# Normalised DCF base
# ==========================================================================
class TestNormalisedFcf:
    """
    REGRESSION: the DCF used the latest filed free cash flow as its base and
    compounded it for ten years. Coca-Cola's FY2025 FCF was $5.3bn against
    $9-11bn either side, entirely from one contingent-consideration payment.
    Compounding that produced a fair value roughly half what the business
    supports, with a completely ordinary-looking tile.
    """

    def _frames(self, ocf, capex, revenue):
        columns = [f"FY{2025 - i}" for i in range(len(ocf))]
        income = pd.DataFrame([revenue], index=["Revenue"], columns=columns)
        cash = pd.DataFrame([ocf, capex],
                            index=["Operating Cash Flow", "CapEx"],
                            columns=columns)
        return {"income_statement": income, "balance_sheet": pd.DataFrame(),
                "cash_flow": cash}

    def _metrics(self, ocf, capex, revenue):
        return fund.get_metrics({
            "statements": self._frames(ocf, capex, revenue),
            "statement_source": "SEC XBRL", "periods": ["FY2025"],
            "info": {}, "quote": {}})

    def test_one_off_year_does_not_set_the_base(self):
        # A single depressed year among four healthy ones.
        metrics = self._metrics(
            ocf=[7.4, 11.6, 11.0, 12.6],
            capex=[2.1, 1.9, 1.5, 1.4],
            revenue=[47.9, 45.8, 43.0, 38.7])

        latest = metrics["free_cash_flow"].value
        normalised = metrics["normalized_fcf"].value

        assert latest == pytest.approx(5.3, abs=0.01)
        assert normalised > latest * 1.5
        assert metrics["normalized_fcf"].note

    def test_normalisation_respects_current_scale(self):
        """
        A fast-growing company must not be dragged back to what it earned
        three years ago - which is why this normalises the MARGIN and applies
        it to today's revenue, rather than averaging past dollars.
        """
        metrics = self._metrics(
            ocf=[100.0, 50.0, 25.0, 12.0],
            capex=[10.0, 5.0, 2.5, 1.2],
            revenue=[300.0, 150.0, 75.0, 36.0])

        latest = metrics["free_cash_flow"].value
        normalised = metrics["normalized_fcf"].value
        assert normalised == pytest.approx(latest, rel=0.05)

    def test_insufficient_history_reports_rather_than_guesses(self):
        metrics = fund.get_metrics({
            "statements": {"income_statement": pd.DataFrame(),
                           "balance_sheet": pd.DataFrame(),
                           "cash_flow": pd.DataFrame()},
            "statement_source": "SEC XBRL", "periods": [], "info": {},
            "quote": {}})
        assert metrics["normalized_fcf"].value is None
        assert metrics["normalized_fcf"].source == fund.UNAVAILABLE

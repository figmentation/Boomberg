"""
Macro normalisation and regime classification.

REGRESSION CONTEXT
------------------
Every bug guarded here renders as a plausible number. That is what makes them
worth a suite of their own:

  * The transform trap. Unemployment moving 4.0 -> 4.2 is +0.2 percentage
    points. Run it through pct_change and the table reports +5.0, which reads
    as a labour market falling apart. Both numbers format identically.

  * The ragged edge. On any given day core PCE is two months stale while the
    10Y breakeven is current to yesterday. Forward-filling keeps the panel
    rectangular, but if those copied cells reach the change columns then
    "has not printed yet" is displayed as "unchanged this month".

  * The unit trap. WALCL and WTREGEN are published in millions, RRPONTSYD in
    billions. Subtract them raw and the repo leg comes off a thousand times
    too small - immaterial today with the facility near empty, a $2.2trn
    error at its 2022-23 peak, and invisible in the chart either way.

  * Silent contributor loss. A dropped input must reduce the contributor
    count, never default to zero, or every axis drifts toward the boundary
    and the quadrant call follows it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data_fetchers import macro, macro_regime
from utils import macro_analytics as ma


def _monthly(values, start="2020-01-31"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="ME"),
        dtype=float)


# ==========================================================================
# Alignment
# ==========================================================================
class TestAlignPanel:
    def test_mixed_frequencies_land_on_one_index(self):
        panel = ma.align_panel({
            "monthly": _monthly([1, 2, 3, 4]),
            "weekly": pd.Series(
                range(18), index=pd.date_range("2020-01-04", periods=18,
                                               freq="W-SAT"), dtype=float),
            "daily": pd.Series(
                range(120), index=pd.date_range("2020-01-01", periods=120,
                                                freq="D"), dtype=float),
        })

        assert list(panel.columns) == ["monthly", "weekly", "daily"]
        assert (panel.index == panel.index.to_period("M").to_timestamp("M")).all()

    def test_saturday_stamped_series_survives(self):
        """
        Initial claims are stamped on Saturdays.

        An earlier draft built the intermediate grid with freq="B", which
        silently dropped every observation and left the column empty.
        """
        claims = pd.Series(
            [200.0, 210.0, 220.0],
            index=pd.to_datetime(["2020-01-04", "2020-01-11", "2020-01-18"]))
        panel = ma.align_panel({"ICSA": claims})

        assert panel["ICSA"].notna().any()
        assert panel["ICSA"].iloc[0] == 220.0

    def test_locf_carries_forward_not_backward(self):
        """A late-starting series must be NaN before it existed."""
        early = _monthly([1, 2, 3, 4, 5, 6])
        late = _monthly([9, 9], start="2020-05-31")

        panel = ma.align_panel({"early": early, "late": late})

        assert panel["late"].iloc[0] != panel["late"].iloc[0]  # NaN
        assert panel["late"].dropna().index.min() == pd.Timestamp("2020-05-31")

    def test_empty_input(self):
        assert ma.align_panel({}).empty
        assert ma.align_panel({"x": pd.Series(dtype=float)}).empty


# ==========================================================================
# Z-scores
# ==========================================================================
class TestZScore:
    def test_nan_below_min_periods(self):
        series = _monthly(list(range(10)))
        assert ma.zscore(series, 12).isna().all()

    def test_matches_hand_computation(self):
        series = _monthly([1, 2, 3, 4, 5, 6])
        result = ma.zscore(series, 6)

        window = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        expected = (6 - window.mean()) / window.std(ddof=1)
        assert result.iloc[-1] == pytest.approx(expected)

    def test_flat_series_is_nan_not_infinite(self):
        series = _monthly([5.0] * 12)
        assert not np.isinf(ma.zscore(series, 12)).any()
        assert ma.zscore(series, 12).isna().all()


# ==========================================================================
# The transform trap
# ==========================================================================
class TestTransforms:
    def test_level_series_reports_percentage_points(self):
        """Unemployment 4.0 -> 4.2 is +0.2pp, not +5%."""
        unrate = _monthly([4.0, 4.2])
        assert ma.mom(unrate, "level").iloc[-1] == pytest.approx(0.2)

    def test_pct_series_reports_percent(self):
        cpi = _monthly([100.0, 105.0])
        assert ma.mom(cpi, "pct").iloc[-1] == pytest.approx(5.0)

    def test_the_two_transforms_disagree(self):
        """
        The guard on the whole trap: if these ever coincide the test is
        no longer proving anything.
        """
        series = _monthly([4.0, 4.2])
        assert ma.mom(series, "level").iloc[-1] != pytest.approx(
            ma.mom(series, "pct").iloc[-1])

    def test_yoy_spans_twelve_months(self):
        series = _monthly([100.0] * 12 + [110.0])
        assert ma.yoy(series, "pct").iloc[-1] == pytest.approx(10.0)

    def test_momentum_3m_annualises_by_compounding(self):
        series = _monthly([100.0, 100.0, 100.0, 102.0])
        expected = ((1.02 ** 4) - 1) * 100
        assert ma.momentum_3m(series, "pct").iloc[-1] == pytest.approx(expected)

    def test_momentum_3m_level_scales_by_four(self):
        series = _monthly([1.0, 1.0, 1.0, 1.5])
        assert ma.momentum_3m(series, "level").iloc[-1] == pytest.approx(2.0)

    def test_momentum_3m_survives_a_zero_crossing(self):
        """
        CFNAI oscillates around zero. Compounding a ratio with a negative or
        zero base yields complex or explosive numbers, so those periods must
        come back NaN rather than as a spectacular reading.
        """
        series = _monthly([-0.5, 0.1, 0.2, 0.3])
        assert not np.isinf(ma.momentum_3m(series, "pct")).any()


# ==========================================================================
# Signal labels and runs
# ==========================================================================
class TestSignalLabel:
    def test_thresholds(self):
        assert ma.signal_label(1.5) == "BULLISH"
        assert ma.signal_label(-1.5) == "BEARISH"
        assert ma.signal_label(0.4) == "NEUTRAL"

    def test_invert_flips_the_reading(self):
        """Rising claims is a high z-score and a bearish signal."""
        assert ma.signal_label(1.5, invert=True) == "BEARISH"
        assert ma.signal_label(-1.5, invert=True) == "BULLISH"

    def test_missing_is_neutral_not_an_error(self):
        assert ma.signal_label(None) == "NEUTRAL"
        assert ma.signal_label(float("nan")) == "NEUTRAL"


class TestRuns:
    def test_collapses_contiguous_labels(self):
        labels = pd.Series(["A", "A", "B", "B", "B", "A"],
                           index=pd.date_range("2020-01-31", periods=6, freq="ME"))
        result = ma.runs(labels)

        assert [label for _, _, label in result] == ["A", "B", "A"]
        assert result[0][0] == labels.index[0]
        assert result[-1][1] == labels.index[-1]

    def test_single_run_and_empty(self):
        labels = pd.Series(["X"] * 4,
                           index=pd.date_range("2020-01-31", periods=4, freq="ME"))
        assert len(ma.runs(labels)) == 1
        assert ma.runs(pd.Series(dtype=object)) == []


class TestRescaling:
    def test_index_to_100_rebases_at_the_first_observation(self):
        rebased = ma.index_to_100(_monthly([50.0, 75.0, 100.0]))
        assert rebased.iloc[0] == pytest.approx(100.0)
        assert rebased.iloc[-1] == pytest.approx(200.0)

    def test_index_to_100_refuses_a_series_that_crosses_zero(self):
        """
        The bug this guards shipped briefly and was visible on the chart:
        rebasing CFNAI divided by a first value of -0.08, which inverted the
        line and multiplied it by a thousand - the overlay came out spanning
        -5,000 to 20,000 with no indication anything was wrong.
        """
        assert ma.index_to_100(_monthly([-0.08, 0.10, 0.20])).empty
        assert ma.index_to_100(_monthly([0.0, 1.0])).empty

    def test_standardize_handles_a_zero_crossing_series(self):
        oscillating = _monthly([-0.5, 0.1, -0.2, 0.4, 0.0, -0.3])
        result = ma.standardize(oscillating)

        assert len(result) == len(oscillating)
        assert result.mean() == pytest.approx(0.0, abs=1e-12)
        assert result.std() == pytest.approx(1.0)

    def test_standardize_flat_series_returns_empty(self):
        assert ma.standardize(_monthly([3.0] * 5)).empty


# ==========================================================================
# Regime classification
# ==========================================================================
def _panel_from(spec_values, months=60):
    """Build a panel keyed by series id, with `last_observed` populated."""
    index = pd.date_range("2020-01-31", periods=months, freq="ME")
    panel = pd.DataFrame(
        {sid: pd.Series(values, index=index) for sid, values in spec_values.items()})
    panel.attrs["last_observed"] = {
        sid: str(index[-1].date()) for sid in spec_values}
    return panel


class TestQuadrantMapping:
    @pytest.mark.parametrize("growth,inflation,expected", [
        (1.0, -1.0, "GOLDILOCKS"),
        (1.0, 1.0, "OVERHEATING"),
        (-1.0, 1.0, "STAGFLATION"),
        (-1.0, -1.0, "DEFLATIONARY BUST"),
    ])
    def test_all_four_quadrants_reachable(self, growth, inflation, expected):
        assert macro_regime._quadrant(growth, inflation)["regime"] == expected

    def test_every_quadrant_has_a_theme_colour(self):
        for _, _, colour in config.REGIME_QUADRANTS.values():
            assert hasattr(config.THEME, colour)


class TestAxisScores:
    @staticmethod
    def _spiking(rng_seed: int = 5) -> np.ndarray:
        """
        Steady drift for five years, then a sharp jump in the last quarter.

        Contributors measure ACCELERATION, not level, so the input has to
        accelerate for the sign to mean anything. A merely rising line is the
        wrong probe: linear growth on a growing base is decelerating in
        percentage terms, and reads as such.
        """
        rng = np.random.default_rng(rng_seed)
        base = 200.0 * np.exp(0.002 * np.arange(60)) + rng.normal(0, 1.5, 60)
        base[-3:] *= 1.20
        return base

    def test_inverted_input_lowers_growth_when_it_spikes(self):
        """
        A jobless-claims surge must WEAKEN the growth axis.

        Without the invert flag a deteriorating labour market reads as
        acceleration, which flips the regime call in precisely the conditions
        where being right matters most.
        """
        spec = next(s for s in config.REGIME_INPUTS if s.series_id == "ICSA")
        assert spec.invert is True

        panel = _panel_from({"ICSA": self._spiking()})
        contribution = macro_regime._contributor_momentum(panel, spec)

        assert contribution is not None
        assert contribution.dropna().iloc[-1] < 0

    def test_uninverted_input_raises_growth_on_the_same_shape(self):
        """
        The converse, pinning the sign convention from both directions: an
        identical surge in payrolls must STRENGTHEN growth.
        """
        spec = next(s for s in config.REGIME_INPUTS if s.series_id == "PAYEMS")
        assert spec.invert is False

        panel = _panel_from({"PAYEMS": self._spiking()})
        contribution = macro_regime._contributor_momentum(panel, spec)

        assert contribution is not None
        assert contribution.dropna().iloc[-1] > 0

    def test_missing_contributor_reduces_the_count(self):
        """A dropped input must not be silently treated as a zero vote."""
        noisy = np.random.default_rng(7).normal(100, 5, 60).cumsum()
        panel = _panel_from({"CFNAI": noisy, "PAYEMS": noisy})

        scores = macro_regime.axis_scores(panel)
        assert scores["growth_n"].iloc[-1] == 2
        assert scores["inflation_n"].iloc[-1] == 0
        assert np.isnan(scores["inflation"].iloc[-1])


class TestClassify:
    def test_refuses_to_call_without_enough_contributors(self, monkeypatch):
        noisy = np.random.default_rng(3).normal(0, 1, 60).cumsum()
        panel = _panel_from({"CFNAI": noisy})
        monkeypatch.setattr(macro_regime, "build_panel", lambda *a, **k: panel)

        verdict = macro_regime.classify.__wrapped__()
        assert verdict["regime"] is None
        assert "contributors" in verdict["reason"]

    def test_empty_panel_is_a_reason_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(macro_regime, "build_panel",
                            lambda *a, **k: pd.DataFrame())
        verdict = macro_regime.classify.__wrapped__()
        assert verdict["regime"] is None
        assert verdict["reason"]

    def test_run_length_agrees_with_the_history(self, monkeypatch):
        """
        classify() and regime_history() must share one predicate.

        When they filtered separately the run counter was measured against a
        history that stopped short of the month being reported, so a
        four-month regime could be announced as its first.
        """
        rng = np.random.default_rng(11)
        panel = _panel_from({
            sid: rng.normal(100, 3, 60).cumsum()
            for sid in ("CFNAI", "PAYEMS", "PCEPILFE", "CPILFESL")
        })
        monkeypatch.setattr(macro_regime, "build_panel", lambda *a, **k: panel)

        verdict = macro_regime.classify.__wrapped__()
        history = macro_regime.regime_history(panel)

        assert verdict["regime"] is not None
        assert history.index[-1] == verdict["as_of"]
        assert history.iloc[-1] == verdict["regime"]
        assert verdict["run_months"] >= 1


class TestRaggedEdge:
    def test_stale_series_does_not_report_zero_change(self):
        """
        The panel forward-fills a series that has not printed. That copy must
        not reach the change columns, or "no release yet" is displayed as
        "no change" - a different claim in an identical-looking cell.
        """
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        # Real prints through April; May and June are forward-filled copies.
        column = pd.Series([100.0, 101.0, 102.0, 103.0, 103.0, 103.0],
                           index=index)
        panel = pd.DataFrame({"PCEPILFE": column})
        panel.attrs["last_observed"] = {"PCEPILFE": "2020-04-30"}

        history = macro_regime._real_history(panel, "PCEPILFE")

        assert history.index[-1] == pd.Timestamp("2020-04-30")
        assert ma.mom(history, "pct").iloc[-1] != 0.0

    def test_without_truncation_the_change_would_be_zero(self):
        """Names the bug the previous test guards, so it cannot be optimised away."""
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        column = pd.Series([100.0, 101.0, 102.0, 103.0, 103.0, 103.0],
                           index=index)
        assert ma.mom(column, "pct").iloc[-1] == 0.0


# ==========================================================================
# The unit trap
# ==========================================================================
class TestNetLiquidityUnits:
    @pytest.mark.parametrize("units,expected", [
        ("Mil. of U.S. $", 1e-3),
        ("Millions of U.S. Dollars", 1e-3),
        ("Bil. of US $", 1.0),
        ("Billions of Dollars", 1.0),
        ("Thousands of Dollars", 1e-6),
    ])
    def test_units_parse_to_a_scale(self, units, expected):
        assert macro._scale_to_billions(units) == pytest.approx(expected)

    def test_unreadable_units_return_none(self):
        assert macro._scale_to_billions("Index 1982=100") is None
        assert macro._scale_to_billions("") is None
        assert macro._scale_to_billions(None) is None

    def test_millions_and_billions_reconcile(self):
        """
        WALCL is published in millions and RRPONTSYD in billions. Scaled, a
        balance sheet of 6,676,249 (mil) and a facility of 11.7 (bil) must
        both land in the same units before anything is subtracted.
        """
        walcl_bn = 6_676_249.0 * macro._scale_to_billions("Mil. of U.S. $")
        rrp_bn = 11.7 * macro._scale_to_billions("Bil. of US $")

        assert walcl_bn == pytest.approx(6676.249)
        assert rrp_bn == pytest.approx(11.7)
        assert walcl_bn - rrp_bn == pytest.approx(6664.549)

    def test_raw_subtraction_is_the_bug_being_guarded(self):
        """
        Names the failure explicitly. Subtracting unscaled, the repo facility
        removes 11.7 million from a 6.7 trillion balance sheet - so net
        liquidity comes back within 0.001% of WALCL and the chart looks fine.
        """
        raw = 6_676_249.0 - 11.7
        assert abs(raw - 6_676_249.0) / 6_676_249.0 < 1e-5

    def test_no_key_declines_rather_than_guessing(self, monkeypatch):
        """
        With no FRED key the units cannot be read, and magnitude inference is
        actively dangerous here: RRPONTSYD currently prints ~12, which any
        magnitude heuristic reads as trillions and scales up by 1000x.
        """
        monkeypatch.setattr(config, "FRED_API_KEY", "")
        result = macro.get_net_liquidity.__wrapped__(2)

        assert result.empty
        assert "FRED API key" in result.attrs.get("reason", "")


# ==========================================================================
# Configuration invariants
# ==========================================================================
class TestRegimeConfig:
    def test_axes_have_enough_contributors_configured(self):
        for axis in ("growth", "inflation"):
            members = [s for s in config.REGIME_INPUTS if s.axis == axis]
            assert len(members) >= config.REGIME_MIN_CONTRIBUTORS, axis

    def test_transforms_and_axes_are_known_values(self):
        for spec in config.REGIME_INPUTS + config.REGIME_CONTEXT:
            assert spec.transform in ("pct", "level"), spec.series_id
            assert spec.axis in ("growth", "inflation", "context"), spec.series_id
            assert isinstance(spec.invert, bool), spec.series_id

    def test_voting_inputs_carry_weight(self):
        for spec in config.REGIME_INPUTS:
            assert spec.weight > 0, spec.series_id

    def test_oscillating_indices_are_not_marked_pct(self):
        """
        CFNAI and NFCI both sit around zero. Marked "pct" their momentum
        divides by a near-zero base and returns nonsense that is occasionally
        enormous, which is exactly the kind of number that looks like a
        signal.
        """
        for spec in config.REGIME_INPUTS + config.REGIME_CONTEXT:
            if spec.series_id in ("CFNAI", "NFCI", "T10Y2Y"):
                assert spec.transform == "level", spec.series_id

    def test_deteriorating_series_are_inverted(self):
        inverted = {s.series_id for s in config.REGIME_INPUTS if s.invert}
        assert "ICSA" in inverted
        assert "NFCI" in inverted

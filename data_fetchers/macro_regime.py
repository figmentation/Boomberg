"""
data_fetchers/macro_regime.py :: Module G - Macro Regime Matrix.

Bloomberg equivalents: ECAN (economic analysis), the growth/inflation quadrant
framing that sits behind most asset-allocation commentary.

WHAT THIS ANSWERS
-----------------
`macro.py` fetches levels. Levels alone do not tell you what environment you
are in: core PCE at 2.8% is a very different world depending on whether it is
on its way up or down. This module converts the levels into standardised
acceleration - is growth speeding up or slowing, is inflation speeding up or
slowing - and reads the quadrant off the pair:

    growth accel + inflation decel   GOLDILOCKS          risk-on
    growth accel + inflation accel   OVERHEATING         hike risk
    growth decel + inflation accel   STAGFLATION         defensive
    growth decel + inflation decel   DEFLATIONARY BUST   bonds

HOW THE AXES ARE BUILT
----------------------
Each contributing series in `config.REGIME_INPUTS` is turned into its
three-month annualised momentum, that momentum is z-scored against its own
three-year history, inverted where a rising reading means a weakening axis
(claims, credit spreads), and the axis score is the weighted mean of those
z-scores. Standardising first is what makes payrolls and financial-conditions
indices addable at all - they share no units and differ by orders of magnitude.

A composite rather than one series per axis, because single-series
classification flips on revisions: one soft payrolls print should move the
growth axis, not redefine the regime. Every contributor is returned alongside
the verdict so the call can be argued with.

WHAT THIS IS NOT
----------------
Not a forecast, not a backtest, and not advice. It is a description of what
public data currently says, in the same spirit as
`macro.get_recession_indicators`. The historical strip carries a specific
caveat about revisions - see `regime_history`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data_fetchers import macro
from utils import macro_analytics as ma
from utils.cache import cached

log = logging.getLogger("openterm.macro_regime")

# Rolling windows on a monthly panel.
Z_WINDOW_1Y = 12
Z_WINDOW_3Y = 36

# Conviction bands on the weaker of the two axis scores. Descriptive labels
# for how far from the quadrant boundary the reading sits - a regime called
# off two scores of 0.02 is a coin toss wearing a verdict.
CONVICTION_HIGH = 0.75
CONVICTION_MODERATE = 0.35


def _all_specs() -> List[config.RegimeInput]:
    """Voting inputs plus the context rows shown in the same table."""
    return list(config.REGIME_INPUTS) + list(config.REGIME_CONTEXT)


# ==========================================================================
# PANEL
# ==========================================================================
@cached(ttl=config.TTL.macro, namespace="macro_panel")
def build_panel(years: int = 15) -> pd.DataFrame:
    """
    Every regime series on one month-end index.

    Fetching goes through `macro.get_fred_series`, so each series is already
    cached, throttled and circuit-breaker guarded; a dead FRED only costs the
    columns it owns. Alignment is LOCF via `macro_analytics.align_panel`.

    `attrs["last_observed"]` records each series' true last print date before
    forward-filling. The panel's own index is month-end, so a column whose
    latest real observation was in June still carries a value at the August
    bucket - reporting that bucket as the "as of" date would claim a release
    that has not happened. Callers show the recorded date instead.

    Returns an empty frame if nothing resolved - callers must handle that
    rather than assume columns exist.
    """
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).date().isoformat()

    series_map: Dict[str, pd.Series] = {}
    last_observed: Dict[str, str] = {}

    for spec in _all_specs():
        try:
            series = macro.get_fred_series(spec.series_id, start=start)
        except Exception as exc:
            log.warning("Regime series %s failed: %s", spec.series_id, exc)
            continue
        if series is not None and len(series):
            series_map[spec.series_id] = series
            last_observed[spec.series_id] = str(
                pd.Timestamp(series.index.max()).date())

    if not series_map:
        return pd.DataFrame()

    panel = ma.align_panel(series_map, freq="ME")
    panel.attrs["last_observed"] = last_observed
    return panel


# ==========================================================================
# AXIS SCORES
# ==========================================================================
def _real_history(panel: pd.DataFrame, series_id: str) -> pd.Series:
    """
    One column, truncated at its own last genuine print.

    Macro data has a ragged edge: on 26 August the 10Y breakeven is current
    to yesterday while core PCE is current to June. LOCF fills the gap so the
    panel stays rectangular, but those filled cells are copies, and letting
    them reach the change calculations is how "core PCE has not been
    published yet" renders as "core PCE was unchanged this month" - a
    completely different claim, printed in the same cell.

    Each series is therefore evaluated at its own latest real observation.
    The metric table's As Of column is what tells the reader those dates
    differ.
    """
    if series_id not in panel.columns:
        return pd.Series(dtype=float)

    column = panel[series_id]
    stamp = panel.attrs.get("last_observed", {}).get(series_id)
    if stamp:
        try:
            cutoff = pd.Timestamp(stamp) + pd.offsets.MonthEnd(0)
            column = column[column.index <= cutoff]
        except Exception:
            pass

    return column.dropna()


def _contributor_momentum(panel: pd.DataFrame,
                          spec: config.RegimeInput) -> Optional[pd.Series]:
    """
    One contributor's standardised acceleration, sign-corrected.

    Returns None when the series is absent or has too little history for a
    three-year z-score - a missing contributor is dropped from the weighted
    mean rather than defaulted to zero, which would drag the axis toward
    "no change" and quietly bias every quadrant call toward the boundary.
    """
    history = _real_history(panel, spec.series_id)
    if history.empty:
        return None

    momentum = ma.momentum_3m(history, spec.transform)
    z = ma.zscore(momentum, Z_WINDOW_3Y)
    if z.dropna().empty:
        return None

    return -z if spec.invert else z


def axis_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted-mean standardised acceleration per axis, month by month.

    Returns a frame with `growth` and `inflation` columns plus
    `growth_n` / `inflation_n` contributor counts, so a caller can tell a
    confident zero from an axis nobody voted on.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()

    weighted: Dict[str, pd.DataFrame] = {"growth": pd.DataFrame(),
                                         "inflation": pd.DataFrame()}
    weights: Dict[str, Dict[str, float]] = {"growth": {}, "inflation": {}}

    for spec in config.REGIME_INPUTS:
        if spec.axis not in weighted:
            continue
        contribution = _contributor_momentum(panel, spec)
        if contribution is None:
            continue
        weighted[spec.axis][spec.series_id] = contribution
        weights[spec.axis][spec.series_id] = spec.weight

    out = pd.DataFrame(index=panel.index)
    for axis, frame in weighted.items():
        if frame.empty:
            out[axis] = np.nan
            out[f"{axis}_n"] = 0
            continue

        axis_weights = pd.Series(weights[axis])
        present = frame.notna()
        # Renormalise per row so a month where one contributor has not
        # printed yet is still scored on the ones that have.
        denominator = present.mul(axis_weights, axis=1).sum(axis=1)
        numerator = frame.mul(axis_weights, axis=1).sum(axis=1, min_count=1)

        out[axis] = numerator / denominator.replace(0.0, np.nan)
        out[f"{axis}_n"] = present.sum(axis=1)

    return out


# ==========================================================================
# CLASSIFICATION
# ==========================================================================
def _quadrant(growth: float, inflation: float) -> Dict[str, str]:
    """Map a pair of axis scores onto the regime matrix."""
    label, posture, colour = config.REGIME_QUADRANTS[(growth > 0, inflation > 0)]
    return {"regime": label, "posture": posture, "colour": colour}


def scored_months(scores: pd.DataFrame) -> pd.DataFrame:
    """
    The rows a quadrant may legitimately be read off.

    Both axes must be scored and both must have cleared
    `config.REGIME_MIN_CONTRIBUTORS`. `classify` and `regime_history` share
    this predicate deliberately: when they filtered separately, the run-length
    counter was measured against a history that stopped a month or two short
    of the regime being reported, so "1 month" could describe a run of four.
    """
    if scores is None or scores.empty:
        return pd.DataFrame()

    minimum = config.REGIME_MIN_CONTRIBUTORS
    return scores[
        scores["growth"].notna() & scores["inflation"].notna()
        & (scores["growth_n"] >= minimum) & (scores["inflation_n"] >= minimum)
    ]


def regime_history(panel: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    The quadrant label for every month with enough contributors to score.

    REVISION CAVEAT, which the UI must repeat: FRED stamps an observation at
    the START of the period it describes, but a July CPI reading is not
    published until mid-August and is revised for months afterwards. This
    series is therefore built from today's revised data placed at the date it
    describes, not from what was knowable at the time. It will look more
    prescient than any real-time reading was. It is a description of history,
    not a backtest, and nothing here should be measured as though it were.
    """
    panel = build_panel() if panel is None else panel
    scores = axis_scores(panel)
    if scores.empty:
        return pd.Series(dtype=object)

    usable = scored_months(scores)
    if usable.empty:
        return pd.Series(dtype=object)

    return pd.Series(
        [_quadrant(g, i)["regime"]
         for g, i in zip(usable["growth"], usable["inflation"])],
        index=usable.index, dtype=object, name="regime",
    )


@cached(ttl=config.TTL.macro, namespace="macro_regime")
def classify(as_of: Optional[str] = None) -> Dict[str, Any]:
    """
    The current macro regime, with the evidence behind it.

    Args:
        as_of: "YYYY-MM-DD" to classify a past month instead of the latest.

    Returns a dict carrying the quadrant, both axis scores, the number of
    contributors each axis got, a conviction band, how many consecutive
    months this regime has held, and the per-contributor detail.

    When either axis has fewer than `config.REGIME_MIN_CONTRIBUTORS` resolved
    series, `regime` is None and `reason` explains why. That is a real answer:
    calling a quadrant off one series that happened to load would be a
    confident verdict on an accident of which fetches succeeded.
    """
    panel = build_panel()
    if panel.empty:
        return {"regime": None,
                "reason": "No macro series resolved; FRED is unreachable."}

    scores = axis_scores(panel)
    if scores.empty:
        return {"regime": None,
                "reason": "Not enough history to standardise any axis."}

    candidates = scores[scores.index <= pd.Timestamp(as_of)] if as_of else scores
    usable = scored_months(candidates)

    if usable.empty:
        minimum = config.REGIME_MIN_CONTRIBUTORS
        return {
            "regime": None,
            "reason": (
                f"No month has both axes scored by at least {minimum} "
                "contributors yet. Calling a quadrant off fewer would be a "
                "verdict on which fetches happened to succeed."
            ),
        }

    row = usable.iloc[-1]
    stamp = usable.index[-1]
    growth, inflation = float(row["growth"]), float(row["inflation"])
    verdict = _quadrant(growth, inflation)

    # How long this regime has held, counted back from the month being
    # reported rather than from the end of the full history.
    history = regime_history(panel)
    history = history[history.index <= stamp]
    run_months = 0
    for label in reversed(history.tolist()):
        if label != verdict["regime"]:
            break
        run_months += 1

    # How often the call has changed lately. Momentum-based classification
    # is genuinely unstable near the axes: a quadrant read off two scores of
    # 0.01 will flip next month on noise. Rather than smooth that away with a
    # persistence rule - which would hide the instability behind a steadier
    # looking label - the flip count is reported so the reader can discount
    # the verdict themselves.
    recent = history.tail(13)
    flips_12m = int((recent != recent.shift()).sum() - 1) if len(recent) > 1 else 0

    conviction_score = min(abs(growth), abs(inflation))
    conviction = (
        "HIGH" if conviction_score >= CONVICTION_HIGH
        else "MODERATE" if conviction_score >= CONVICTION_MODERATE
        else "LOW"
    )

    contributors: List[Dict[str, Any]] = []
    for spec in config.REGIME_INPUTS:
        contribution = _contributor_momentum(panel, spec)
        if contribution is None:
            contributors.append({
                "label": spec.label, "series_id": spec.series_id,
                "axis": spec.axis, "weight": spec.weight,
                "invert": spec.invert, "z": None,
                "note": "insufficient history",
            })
            continue
        trimmed = contribution[contribution.index <= stamp].dropna()
        contributors.append({
            "label": spec.label, "series_id": spec.series_id,
            "axis": spec.axis, "weight": spec.weight, "invert": spec.invert,
            "z": round(float(trimmed.iloc[-1]), 2) if len(trimmed) else None,
            "note": "" if len(trimmed) else "insufficient history",
        })

    return {
        "regime": verdict["regime"],
        "posture": verdict["posture"],
        "colour": verdict["colour"],
        "as_of": stamp,
        "growth_z": round(growth, 2),
        "inflation_z": round(inflation, 2),
        "growth_n": int(row["growth_n"]),
        "inflation_n": int(row["inflation_n"]),
        "conviction": conviction,
        "run_months": run_months,
        "flips_12m": max(flips_12m, 0),
        "contributors": contributors,
    }


# ==========================================================================
# METRIC TABLE
# ==========================================================================
@cached(ttl=config.TTL.macro, namespace="macro_metrics")
def metric_table() -> pd.DataFrame:
    """
    One row per tracked indicator: level, changes, z-scores and a signal chip.

    Columns: Metric, Series, Level, 1M Chg, YoY, 3M Mom, Z (1Y), Z (3Y),
    Signal, As Of.

    Change columns follow each series' declared transform, so percentage
    series report percentage-POINT moves and index series report percentage
    moves. Mixing those up is the single easiest way to publish a labour
    market collapse that did not happen - see utils/macro_analytics.
    """
    panel = build_panel()
    if panel.empty:
        return pd.DataFrame()

    last_observed = panel.attrs.get("last_observed", {})
    rows: List[Dict[str, Any]] = []

    def last(series: pd.Series) -> Optional[float]:
        values = series.dropna() if series is not None else pd.Series(dtype=float)
        return float(values.iloc[-1]) if len(values) else None

    for spec in _all_specs():
        series = _real_history(panel, spec.series_id)
        if series.empty:
            continue

        z_1y = last(ma.zscore(series, Z_WINDOW_1Y))
        rows.append({
            "Metric": spec.label,
            "Series": spec.series_id,
            "Level": last(series),
            "1M Chg": last(ma.mom(series, spec.transform)),
            "YoY": last(ma.yoy(series, spec.transform)),
            "3M Mom": last(ma.momentum_3m(series, spec.transform)),
            "Z (1Y)": z_1y,
            "Z (3Y)": last(ma.zscore(series, Z_WINDOW_3Y)),
            "Signal": ma.signal_label(z_1y, invert=spec.invert),
            "Axis": spec.axis,
            # The series' own last print, not the forward-filled panel bucket.
            "As Of": last_observed.get(
                spec.series_id, series.index[-1].date().isoformat()),
        })

    return pd.DataFrame(rows)


__all__ = [
    "build_panel", "axis_scores", "scored_months", "classify",
    "regime_history",
    "metric_table", "Z_WINDOW_1Y", "Z_WINDOW_3Y",
]

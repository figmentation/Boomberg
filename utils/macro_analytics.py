"""
utils/macro_analytics.py :: Cross-series normalisation for macro indicators.

Pure functions only - no network, no config, no cache. Everything here takes
pandas in and hands pandas back, which is what makes the regime engine
testable without an API key or an internet connection.

WHY THIS EXISTS
---------------
Raw macro levels are not comparable. Core PCE sits near 120 as an index,
initial claims near 220,000, the 10Y-2Y spread near 0.5. Reading them
side by side tells you nothing about which is unusual. Z-scoring each
against its own recent history puts them on one axis, and momentum answers
the question levels cannot: not "is inflation high" but "is it accelerating".

THE TRANSFORM ARGUMENT IS LORE, NOT DECORATION
----------------------------------------------
Every change function takes a `transform`, and passing the wrong one produces
a number that looks completely reasonable and means nothing.

  * "pct"   - the series is an index or a count (CPILFESL, PAYEMS, PPIACO).
              Change is a percentage change.
  * "level" - the series is ALREADY a percentage or a standardised index
              (UNRATE, T10Y2Y, BAMLH0A0HYM2, CFNAI). Change is a difference
              in percentage points.

Unemployment going 4.0 -> 4.2 is +0.2pp, not +5%. Quote the +5% and you have
described a labour market collapse. CFNAI is worse: it oscillates around
zero, so `pct_change` divides by something near nothing and returns garbage
that is occasionally enormous. There is no way to infer the right transform
from the values alone, which is why callers must declare it per series.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Transform = Literal["pct", "level"]

# Z-score magnitude at which a reading is called out rather than shrugged at.
# One standard deviation is deliberately unheroic: it flags "unusual", not
# "extreme", and the table shows the underlying z so nobody has to trust it.
SIGNAL_THRESHOLD = 1.0

# The panel is monthly, so a year is 12 rows and a quarter is 3.
PERIODS_PER_YEAR = 12
QUARTER = 3


# ==========================================================================
# ALIGNMENT
# ==========================================================================
def align_panel(
    series_map: Dict[str, pd.Series],
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Put mixed-frequency series on one index by LOCF, then resample.

    Monthly (CPI, payrolls), weekly (claims) and daily (yields, spreads)
    series are each forward-filled onto a daily grid and then taken at the
    period end, so every column carries the most recent value that was
    actually observed rather than an interpolated invention.

    Args:
        series_map: {column name: series indexed by date}.
        freq:       Any pandas offset alias; "ME" (month end) by default.

    Returns:
        DataFrame, one column per input, indexed at `freq`. Periods before a
        series' first real observation are NaN - forward-fill runs forward
        only. Back-filling here would manufacture a decade of Fed balance
        sheet data for years when the series did not exist, and every
        downstream z-score would inherit it.
    """
    cleaned: Dict[str, pd.Series] = {}

    for name, series in series_map.items():
        if series is None or len(series) == 0:
            continue
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue
        try:
            values.index = pd.to_datetime(values.index).normalize()
        except Exception:
            continue
        values = values[~values.index.duplicated(keep="last")].sort_index()
        cleaned[name] = values

    if not cleaned:
        return pd.DataFrame()

    last_observation = max(s.index.max() for s in cleaned.values())
    columns: Dict[str, pd.Series] = {}

    for name, series in cleaned.items():
        # Calendar days, not business days: initial claims are stamped on
        # Saturdays and a business-day grid drops every one of them.
        daily = pd.date_range(series.index.min(), last_observation, freq="D")
        columns[name] = series.reindex(daily).ffill().resample(freq).last()

    return pd.DataFrame(columns).sort_index()


# ==========================================================================
# NORMALISATION
# ==========================================================================
def zscore(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Rolling z-score: how unusual is this reading against its own recent past.

    `min_periods` defaults to the full window. A 36-month z-score computed
    off four observations is arithmetically valid and analytically worthless,
    so the early rows are NaN rather than confidently wrong.

    A flat window (zero standard deviation) yields NaN, not a divide-by-zero
    infinity - a series that has not moved has no anomaly to report.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)

    effective = window if min_periods is None else min_periods
    rolling = series.rolling(window, min_periods=effective)
    deviation = rolling.std()

    return (series - rolling.mean()) / deviation.replace(0.0, np.nan)


def _change(series: pd.Series, periods: int, transform: Transform) -> pd.Series:
    """Percentage change or percentage-point difference. See module docstring."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    if transform == "level":
        return series.diff(periods)
    return series.pct_change(periods, fill_method=None) * 100.0


def mom(series: pd.Series, transform: Transform) -> pd.Series:
    """Month-over-month change."""
    return _change(series, 1, transform)


def yoy(series: pd.Series, transform: Transform) -> pd.Series:
    """Year-over-year change."""
    return _change(series, PERIODS_PER_YEAR, transform)


def momentum_3m(series: pd.Series, transform: Transform) -> pd.Series:
    """
    Three-month change, annualised - the acceleration signal.

    For an index series this compounds the quarterly rate to a year
    ((1+r)**4 - 1); for a level series it scales the percentage-point move by
    four. Annualising makes a 3M reading directly comparable to the YoY
    column beside it, which is the whole point of showing both.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)

    if transform == "level":
        return series.diff(QUARTER) * 4.0

    ratio = series / series.shift(QUARTER)
    ratio = ratio.where(ratio > 0)          # negative bases cannot compound
    return (ratio ** 4 - 1.0) * 100.0


# ==========================================================================
# LABELLING
# ==========================================================================
def signal_label(z: Optional[float], invert: bool = False) -> str:
    """
    Turn a z-score into the chip shown in the metric table.

    Args:
        invert: True when a HIGH reading is the bearish one - initial claims
                and high-yield spreads both rise as conditions deteriorate.

    Descriptive, in the manner of `equities.summarize_technicals`: it reports
    where a series sits in its own distribution and nothing more.
    """
    if z is None:
        return "NEUTRAL"
    try:
        value = float(z)
    except (TypeError, ValueError):
        return "NEUTRAL"
    if np.isnan(value):
        return "NEUTRAL"

    if invert:
        value = -value
    if value >= SIGNAL_THRESHOLD:
        return "BULLISH"
    if value <= -SIGNAL_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def runs(labels: pd.Series) -> List[Tuple[Any, Any, str]]:
    """
    Collapse a label series into contiguous stretches.

    Returns [(start_index, end_index, label)], which is exactly the shape a
    caller needs to draw one `add_vrect` per regime rather than one per month.
    """
    if labels is None or labels.empty:
        return []

    out: List[Tuple[Any, Any, str]] = []
    start_position = 0

    for position in range(1, len(labels)):
        if labels.iloc[position] != labels.iloc[start_position]:
            out.append((labels.index[start_position],
                        labels.index[position - 1],
                        str(labels.iloc[start_position])))
            start_position = position

    out.append((labels.index[start_position], labels.index[-1],
                str(labels.iloc[start_position])))
    return out


def index_to_100(series: pd.Series) -> pd.Series:
    """
    Rebase a strictly positive series to 100 at its first observation.

    Only meaningful for series with a real zero and a constant sign - prices,
    price indices, employment counts. Anything that crosses or hovers near
    zero returns empty rather than a rebased figure: CFNAI oscillates about
    zero, so dividing by a first value of -0.08 both inverts the series and
    multiplies it by a thousand. Use `standardize` for those.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    values = series.dropna()
    if values.empty or not (values > 0).all():
        return pd.Series(dtype=float)
    return values / values.iloc[0] * 100.0


def standardize(series: pd.Series) -> pd.Series:
    """
    Centre and scale a whole series to mean 0, standard deviation 1.

    The scale-free way to put two unlike series on one axis - an activity
    index that swings either side of zero and the S&P 500 in points. Unlike
    rebasing it needs no meaningful zero, and unlike a secondary y-axis it
    cannot be slid around until the lines appear to agree.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    values = series.dropna()
    deviation = values.std()
    if values.empty or not deviation or np.isnan(deviation):
        return pd.Series(dtype=float)
    return (values - values.mean()) / deviation


__all__ = [
    "Transform", "SIGNAL_THRESHOLD", "PERIODS_PER_YEAR",
    "align_panel", "zscore", "mom", "yoy", "momentum_3m",
    "signal_label", "runs", "index_to_100", "standardize",
]

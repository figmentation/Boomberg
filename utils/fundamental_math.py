"""
utils/fundamental_math.py :: Scoring and valuation arithmetic.

Pure functions only - no network, no config, no cache. Everything takes
numbers and hands numbers back, which is what makes the fundamental engine
testable without an API key.

WHY SCORES ARE ANCHOR TABLES
----------------------------
The brief this was built for asked for scores that are explainable rather than
a black-box number, so there is no curve fitting and no tuned constant
anywhere in this module. Every score is three things a caller can render:

    a raw value, a table of named anchor points, and a linear interpolation

`score_band(0.19, [(0, 0), (0.08, 40), (0.15, 70), (0.25, 90), (0.40, 100)])`
returns 78, and the UI can print exactly that sentence: ROIC of 19% sits
between the 15% anchor worth 70 and the 25% anchor worth 90. Nothing about
the number needs to be taken on trust.

Anchors run in ascending x. y may descend, which is how "lower is better"
metrics are expressed - Debt/EBITDA anchors at [(0, 100), (2.5, 70), (6, 10)]
score a low reading highly without a separate inverted code path.

MISSING INPUTS RETURN None, NEVER A DEFAULT
-------------------------------------------
A metric that could not be computed must not be scored as zero. Zero is a
score, and a bad one: it drags an axis down exactly as though the company had
performed badly, when in fact nothing was measured. `weighted_score`
renormalises over the components that actually resolved and reports how many
those were, so a thinly-evidenced axis is visible as such rather than
silently pessimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Anchors = Sequence[Tuple[float, float]]


# ==========================================================================
# BASICS
# ==========================================================================
def safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    """Divide, or None. Guards the zero denominator and non-numeric inputs."""
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator == 0 or math.isnan(numerator) or math.isnan(denominator):
        return None
    result = numerator / denominator
    return None if math.isnan(result) or math.isinf(result) else result


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def cagr(first: Any, last: Any, years: float) -> Optional[float]:
    """
    Compound annual growth rate as a decimal, or None.

    Returns None when either endpoint is non-positive. A company that went
    from a loss to a profit has no meaningful compound rate - the root of a
    negative ratio is either complex or, worse, silently plausible. Callers
    that care about loss-to-profit transitions should test the endpoints
    themselves rather than reading a number out of here.
    """
    try:
        first = float(first)
        last = float(last)
        years = float(years)
    except (TypeError, ValueError):
        return None

    if years <= 0 or first <= 0 or last <= 0:
        return None
    if math.isnan(first) or math.isnan(last):
        return None

    return (last / first) ** (1.0 / years) - 1.0


def trend_slope(values: Sequence[Any]) -> Optional[float]:
    """
    Direction and pace of a short series, scaled by its own average level.

    Ordinary least squares slope divided by the mean absolute value, so the
    result is comparable across metrics of wildly different magnitude - a
    margin in percent and a share count in millions both come back as
    "fraction of its own size per period". Positive means rising.

    Needs three points; fewer is not a trend.
    """
    cleaned: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            cleaned.append(number)

    if len(cleaned) < 3:
        return None

    n = len(cleaned)
    mean_x = (n - 1) / 2.0
    mean_y = sum(cleaned) / n

    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return None

    slope = sum((i - mean_x) * (y - mean_y)
                for i, y in enumerate(cleaned)) / denominator

    scale = sum(abs(y) for y in cleaned) / n
    return None if scale == 0 else slope / scale


def median_of(values: Sequence[Any]) -> Optional[float]:
    """Median of whatever is numeric, or None if nothing is."""
    cleaned = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            cleaned.append(number)

    if not cleaned:
        return None
    cleaned.sort()
    middle = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[middle]
    return (cleaned[middle - 1] + cleaned[middle]) / 2.0


# ==========================================================================
# SCORING
# ==========================================================================
def score_band(value: Any, anchors: Anchors) -> Optional[float]:
    """
    Map a raw metric onto 0-100 by interpolating between named anchors.

    Args:
        anchors: [(metric_value, points)] in ascending metric order. Points
                 may ascend (higher is better) or descend (lower is better).

    Readings outside the table clamp to its end points, so an ROIC of 300%
    scores 100 rather than extrapolating off the top into a number no anchor
    justifies.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None

    points = sorted(anchors, key=lambda pair: pair[0])
    if not points:
        return None

    if value <= points[0][0]:
        return clamp(float(points[0][1]))
    if value >= points[-1][0]:
        return clamp(float(points[-1][1]))

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            span = x1 - x0
            if span == 0:
                return clamp(float(y1))
            fraction = (value - x0) / span
            return clamp(float(y0) + fraction * (float(y1) - float(y0)))

    return None


@dataclass(frozen=True)
class ScoreComponent:
    """
    One scored metric, carrying everything needed to justify its points.

    `points` of None means the metric did not resolve. Such a component is
    still returned - the UI lists it as unmeasured with its `note` - but it
    is excluded from the weighted mean rather than counted as zero.
    """

    name: str
    value: Optional[float]
    unit: str
    points: Optional[float]
    weight: float
    source: str = ""
    anchors: Anchors = field(default_factory=tuple)
    note: str = ""

    @property
    def resolved(self) -> bool:
        return self.points is not None

    @property
    def contribution(self) -> Optional[float]:
        """Weighted points, before renormalisation."""
        return None if self.points is None else self.points * self.weight

    def anchor_label(self) -> str:
        """The anchor table as a readable string, for the breakdown row."""
        if not self.anchors:
            return ""
        return " · ".join(f"{x:g}→{y:g}" for x, y in self.anchors)


def weighted_score(
    components: Sequence[ScoreComponent],
    min_components: int = 2,
) -> Tuple[Optional[float], int]:
    """
    Weighted mean of the components that resolved.

    Returns (score, resolved_count). The score is None below
    `min_components`: an axis resting on one metric is not an assessment of
    anything, and reporting it as though it were invites the reader to trust
    an accident of which fetches succeeded.
    """
    resolved = [c for c in components if c.resolved]
    if len(resolved) < min_components:
        return None, len(resolved)

    total_weight = sum(c.weight for c in resolved)
    if total_weight <= 0:
        return None, len(resolved)

    total = sum((c.points or 0.0) * c.weight for c in resolved)
    return clamp(total / total_weight), len(resolved)


# ==========================================================================
# VALUATION
# ==========================================================================
def dcf_per_share(
    fcf0: Any,
    growth: float,
    fade_to: float,
    years: int,
    terminal_growth: float,
    discount_rate: float,
    net_cash: float = 0.0,
    shares: Any = None,
) -> Optional[float]:
    """
    Two-stage discounted cash flow, per share.

    Stage one grows `fcf0` for `years`, with the growth rate fading linearly
    from `growth` to `fade_to` - no company compounds at its trailing rate
    forever, and a flat-growth DCF says it does. Stage two is a Gordon
    terminal value on the final year.

    Returns None rather than a number when the model does not apply:

      * `discount_rate <= terminal_growth` - the Gordon denominator goes zero
        or negative and the "fair value" comes back negative or astronomical.
        This is the single most common way a DCF produces confident nonsense.
      * `fcf0 <= 0` - there is no base to compound. A loss-making company
        needs a different method, not this one with a minus sign.
      * missing or non-positive share count.

    Every input here is a FORECAST ASSUMPTION, not a fact, and callers are
    expected to label it as such wherever the output is shown.
    """
    try:
        fcf0 = float(fcf0)
        shares = float(shares) if shares is not None else 0.0
        growth = float(growth)
        fade_to = float(fade_to)
        terminal_growth = float(terminal_growth)
        discount_rate = float(discount_rate)
        years = int(years)
    except (TypeError, ValueError):
        return None

    if fcf0 <= 0 or shares <= 0 or years <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None

    present_value = 0.0
    cash_flow = fcf0

    for year in range(1, years + 1):
        # Linear fade from `growth` toward `fade_to` across the horizon.
        fraction = (year - 1) / max(years - 1, 1)
        rate = growth + (fade_to - growth) * fraction
        cash_flow *= (1.0 + rate)
        present_value += cash_flow / ((1.0 + discount_rate) ** year)

    terminal = (cash_flow * (1.0 + terminal_growth)
                / (discount_rate - terminal_growth))
    present_value += terminal / ((1.0 + discount_rate) ** years)

    equity_value = present_value + float(net_cash or 0.0)
    return None if equity_value <= 0 else equity_value / shares


def capm_discount_rate(
    risk_free: Any, beta: Any, equity_risk_premium: float,
    floor: float = 0.06, cap: float = 0.16,
) -> Optional[float]:
    """
    Cost of equity, bounded.

    The bounds are not cosmetic. yfinance publishes betas below zero and
    above three for thinly traded names; unbounded CAPM then returns a
    discount rate under the terminal growth rate (which makes the DCF
    explode) or over 40% (which values every company at nearly nothing).
    Clamping keeps a bad beta from silently producing a confident fair value.
    """
    try:
        risk_free = float(risk_free)
        beta = float(beta)
    except (TypeError, ValueError):
        return None
    if math.isnan(risk_free) or math.isnan(beta):
        return None

    return clamp(risk_free + beta * equity_risk_premium, floor, cap)


def margin_of_safety(price: Any, fair_value: Any) -> Optional[float]:
    """
    Discount of price to fair value, as a decimal. Positive = trading below.

    Note the denominator is fair value, not price: a stock at 50 against a
    fair value of 100 has a 50% margin of safety, not 100%.
    """
    ratio = safe_div(fair_value, 1)
    if ratio is None or ratio <= 0:
        return None
    return safe_div(float(fair_value) - float(price), float(fair_value))


def upside(price: Any, target: Any) -> Optional[float]:
    """Return from price to target, as a decimal."""
    result = safe_div(target, price)
    return None if result is None else result - 1.0


def price_for_margin_of_safety(fair_value: Any,
                               required: float) -> Optional[float]:
    """The price at which `fair_value` offers `required` margin of safety."""
    try:
        fair_value = float(fair_value)
    except (TypeError, ValueError):
        return None
    if fair_value <= 0 or required >= 1.0:
        return None
    return fair_value * (1.0 - required)


__all__ = [
    "Anchors", "ScoreComponent",
    "safe_div", "clamp", "cagr", "trend_slope", "median_of",
    "score_band", "weighted_score",
    "dcf_per_share", "capm_discount_rate", "margin_of_safety", "upside",
    "price_for_margin_of_safety",
]

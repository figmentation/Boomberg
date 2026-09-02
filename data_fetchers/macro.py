"""
data_fetchers/macro.py :: Module D - Macroeconomics & Yield Curves.

Bloomberg equivalents: ECO (economic calendar/indicators), YCRV (yield curve),
WIRP (rate probabilities), FOMC.

Source hierarchy - each level is tried before falling through:

  1. fredapi with a free FRED key      (best: full metadata, vintages)
  2. FRED's keyless CSV endpoint       (fredgraph.csv - no key required at all)
  3. Yahoo Finance yield proxies       (^IRX ^FVX ^TNX ^TYX) as a last resort

That layering matters: a user who never registers for a FRED key still gets
a working yield curve and CPI series, just without the metadata niceties.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from utils import macro_analytics
from utils.cache import cached, get_session
from utils.rate_limiter import circuit_breaker, retry_with_backoff, throttled

log = logging.getLogger("openterm.macro")

try:
    from fredapi import Fred

    FREDAPI_AVAILABLE = True
except Exception:
    Fred = None  # type: ignore[assignment,misc]
    FREDAPI_AVAILABLE = False


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred"

_fred_client: Optional[Any] = None


def _get_fred():
    """Lazily construct the fredapi client. Returns None if unavailable."""
    global _fred_client
    if _fred_client is not None:
        return _fred_client
    if not (FREDAPI_AVAILABLE and config.FRED_API_KEY):
        return None
    try:
        _fred_client = Fred(api_key=config.FRED_API_KEY)
        return _fred_client
    except Exception as exc:
        log.warning("fredapi init failed: %s", exc)
        return None


# ==========================================================================
# FRED SERIES
# ==========================================================================
@cached(ttl=config.TTL.macro, namespace="fred_series")
@throttled("fred")
@retry_with_backoff(on_giveup=lambda exc: pd.Series(dtype=float))
def get_fred_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    One FRED time series as a float Series indexed by date.

    Args:
        series_id: e.g. "DGS10", "CPIAUCSL", "UNRATE".
        start/end: "YYYY-MM-DD". Defaults to the last 10 years.

    Returns an empty Series (not an exception) if every source fails.
    """
    if start is None:
        start = (date.today() - timedelta(days=365 * 10)).isoformat()
    if end is None:
        end = date.today().isoformat()

    # --- Path 1: authenticated fredapi ------------------------------------
    client = _get_fred()
    if client is not None:
        try:
            series = client.get_series(
                series_id, observation_start=start, observation_end=end
            )
            if series is not None and len(series) > 0:
                series = pd.to_numeric(series, errors="coerce").dropna()
                series.name = series_id
                return series
        except Exception as exc:
            log.info("fredapi failed for %s (%s), trying keyless CSV",
                     series_id, exc)

    # --- Path 2: keyless CSV ----------------------------------------------
    return _fred_series_csv(series_id, start, end)


@circuit_breaker("fred", failure_threshold=3, recovery_timeout=180.0,
                 on_open=lambda: pd.Series(dtype=float))
@throttled("fred")
@retry_with_backoff(on_giveup=lambda exc: pd.Series(dtype=float))
def _fred_series_csv(series_id: str, start: str, end: str) -> pd.Series:
    """
    FRED's public CSV download - the same URL the website's chart uses.

    No API key, no registration. Rate limits are generous but not published,
    so we still run it through the token bucket.

    IMPORTANT - User-Agent: this endpoint must be called with a plain client
    UA. Sent a Chrome User-Agent it accepts the TCP/TLS connection and then
    never replies, so every request burns the full read timeout. Verified
    empirically; see config.PLAIN_USER_AGENT.

    The circuit breaker matters here more than anywhere else: get_yield_curve
    requests 11 tenors, so without it a FRED outage costs 11 x the full retry
    budget before the page renders.
    """
    resp = get_session("fred", expire_after=config.TTL.macro,
                       user_agent=config.PLAIN_USER_AGENT).get(
        FRED_CSV_URL,
        params={"id": series_id, "cosd": start, "coed": end},
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"FRED CSV for {series_id} was empty")

    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # FRED uses "." for missing observations (holidays, non-release days).
    df[value_col] = pd.to_numeric(
        df[value_col].astype(str).str.strip().replace(".", np.nan),
        errors="coerce",
    )

    series = df.dropna(subset=[date_col]).set_index(date_col)[value_col].dropna()
    series.name = series_id
    return series


@cached(ttl=86400, namespace="fred_meta")
@throttled("fred")
@retry_with_backoff(on_giveup=lambda exc: {})
def get_fred_series_info(series_id: str) -> Dict[str, Any]:
    """Series title, units, frequency, last update. Requires a FRED key."""
    if not config.FRED_API_KEY:
        return {}

    resp = get_session("fred", expire_after=86400,
                       user_agent=config.PLAIN_USER_AGENT).get(
        f"{FRED_API_URL}/series",
        params={
            "series_id": series_id,
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
        },
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()
    items = resp.json().get("seriess", [])
    if not items:
        return {}

    meta = items[0]
    return {
        "id": meta.get("id"),
        "title": meta.get("title"),
        "units": meta.get("units_short") or meta.get("units"),
        "frequency": meta.get("frequency_short"),
        "seasonal_adjustment": meta.get("seasonal_adjustment_short"),
        "last_updated": meta.get("last_updated"),
        "observation_start": meta.get("observation_start"),
        "observation_end": meta.get("observation_end"),
        "notes": (meta.get("notes") or "")[:600],
    }


def get_macro_series_group(group: str, years: int = 10) -> Dict[str, pd.Series]:
    """
    Fetch every series in one of config.MACRO_SERIES' groups.

    Args:
        group: INFLATION | RATES | GROWTH | LABOR | LIQUIDITY | STRESS
    """
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    out: Dict[str, pd.Series] = {}

    for series_id, label in config.MACRO_SERIES.get(group, {}).items():
        try:
            series = get_fred_series(series_id, start=start)
            if series is not None and len(series):
                series.name = label
                out[series_id] = series
        except Exception as exc:
            log.warning("Macro series %s failed: %s", series_id, exc)

    return out


# ==========================================================================
# YIELD CURVE
# ==========================================================================
@cached(ttl=config.TTL.macro, namespace="yield_curve")
def get_yield_curve(as_of: Optional[str] = None) -> pd.DataFrame:
    """
    Full US Treasury constant-maturity curve.

    Args:
        as_of: "YYYY-MM-DD" to pull a historical curve. None = latest close.

    Returns:
        DataFrame: tenor | years | yield | date
        (empty if every source is down)
    """
    start = (
        (pd.Timestamp(as_of) - pd.Timedelta(days=14)).date().isoformat()
        if as_of else (date.today() - timedelta(days=21)).isoformat()
    )
    end = as_of or date.today().isoformat()

    # Fetched concurrently: 11 serial round trips is ~4s on a good day and
    # far worse on a bad one. The token bucket still enforces the overall
    # request budget, so this is polite, just not sequential.
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(get_fred_series, series_id, start, end): tenor
            for tenor, series_id in config.YIELD_CURVE_SERIES.items()
        }
        for future in as_completed(futures):
            tenor = futures[future]
            try:
                series = future.result()
                if series is None or series.empty:
                    continue
                rows.append({
                    "tenor": tenor,
                    "years": config.TENOR_YEARS[tenor],
                    "yield": float(series.iloc[-1]),
                    "date": series.index[-1],
                })
            except Exception as exc:
                log.debug("Curve point %s failed: %s", tenor, exc)

    if rows:
        return pd.DataFrame(rows).sort_values("years").reset_index(drop=True)

    # --- Fallback: Yahoo yield indices ------------------------------------
    log.warning("FRED unavailable for the curve - falling back to Yahoo proxies")
    return _yield_curve_from_yahoo()


def _yield_curve_from_yahoo() -> pd.DataFrame:
    """
    Coarse 4-point curve from Yahoo's CBOE yield indices.

    Note ^IRX/^FVX/^TNX/^TYX are quoted in percent already (e.g. 4.25 = 4.25%),
    unlike the option-style x10 convention some sources use.
    """
    try:
        from data_fetchers.equities import get_quotes_batch

        symbols = tuple(config.YAHOO_YIELD_PROXIES.values())
        quotes = get_quotes_batch(symbols)

        rows = []
        for tenor, symbol in config.YAHOO_YIELD_PROXIES.items():
            quote = quotes.get(symbol)
            if quote and quote.get("price") is not None:
                rows.append({
                    "tenor": tenor,
                    "years": config.TENOR_YEARS[tenor],
                    "yield": float(quote["price"]),
                    "date": pd.Timestamp.now().normalize(),
                })
        return pd.DataFrame(rows).sort_values("years").reset_index(drop=True)
    except Exception as exc:
        log.error("Yahoo yield fallback also failed: %s", exc)
        return pd.DataFrame()


@cached(ttl=config.TTL.macro, namespace="yield_curve_history")
def get_yield_curve_history(days: int = 730) -> pd.DataFrame:
    """
    Historical curve, wide format: rows = dates, columns = tenors.

    Powers the 3D surface / curve-animation view and the spread charts.
    """
    start = (date.today() - timedelta(days=days)).isoformat()
    frames: Dict[str, pd.Series] = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(get_fred_series, series_id, start): tenor
            for tenor, series_id in config.YIELD_CURVE_SERIES.items()
        }
        for future in as_completed(futures):
            tenor = futures[future]
            try:
                series = future.result()
                if series is not None and len(series):
                    frames[tenor] = series
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    ordered = [t for t in config.TENOR_YEARS if t in df.columns]
    return df[ordered].sort_index()


def analyze_yield_curve(curve: pd.DataFrame) -> Dict[str, Any]:
    """
    Inversion detection and curve-shape classification.

    The two spreads that matter for recession signalling:
      * 10Y-2Y  - the headline, most-cited inversion measure.
      * 10Y-3M  - the NY Fed's preferred model input; historically the
                  better predictor, with a ~12-18 month lead.

    Returns a dict of spreads, an `inversions` list, a `shape` label and a
    `recession_risk` bucket. This is descriptive signal processing on public
    rate data, not investment advice.
    """
    if curve is None or curve.empty:
        return {}

    yields = dict(zip(curve["tenor"], curve["yield"]))
    out: Dict[str, Any] = {"yields": yields, "as_of": None}

    if "date" in curve.columns and len(curve):
        try:
            out["as_of"] = pd.to_datetime(curve["date"]).max()
        except Exception:
            pass

    def spread(long_t: str, short_t: str) -> Optional[float]:
        if long_t in yields and short_t in yields:
            return float(yields[long_t] - yields[short_t])
        return None

    spreads = {
        "10Y-2Y": spread("10Y", "2Y"),
        "10Y-3M": spread("10Y", "3M"),
        "30Y-10Y": spread("30Y", "10Y"),
        "5Y-2Y": spread("5Y", "2Y"),
        "2Y-3M": spread("2Y", "3M"),
    }
    out["spreads"] = {k: v for k, v in spreads.items() if v is not None}

    # --- Inversions --------------------------------------------------------
    inversions: List[Dict[str, Any]] = []
    for name, value in out["spreads"].items():
        if value < 0:
            inversions.append({
                "pair": name,
                "spread_bps": round(value * 100, 1),
                "severity": (
                    "SEVERE" if value < -0.50
                    else "MODERATE" if value < -0.20
                    else "MILD"
                ),
            })
    out["inversions"] = inversions
    out["is_inverted"] = bool(inversions)

    # --- Recession risk bucket --------------------------------------------
    s_10_2 = out["spreads"].get("10Y-2Y")
    s_10_3m = out["spreads"].get("10Y-3M")

    if s_10_2 is not None and s_10_3m is not None:
        if s_10_2 < 0 and s_10_3m < 0:
            risk, note = "ELEVATED", (
                "Both 10Y-2Y and 10Y-3M inverted. Every US recession since "
                "1969 was preceded by this configuration, though lead times "
                "have ranged from roughly 6 to 24 months."
            )
        elif s_10_2 < 0 or s_10_3m < 0:
            risk, note = "MODERATE", (
                "One of the two headline spreads is inverted. Partial "
                "inversions have historically produced false positives."
            )
        elif s_10_2 < 0.25 or s_10_3m < 0.25:
            risk, note = "WATCH", "Curve is flat. Small moves could invert it."
        else:
            risk, note = "LOW", "Curve is positively sloped across key tenors."
    else:
        risk, note = "UNKNOWN", "Insufficient curve data to assess."

    out["recession_risk"] = risk
    out["recession_note"] = note

    # --- Shape classification ---------------------------------------------
    if len(curve) >= 3:
        short_end = curve[curve["years"] <= 2]["yield"].mean()
        belly = curve[(curve["years"] > 2) & (curve["years"] <= 10)]["yield"].mean()
        long_end = curve[curve["years"] > 10]["yield"].mean()

        if pd.notna(short_end) and pd.notna(long_end):
            if short_end > long_end + 0.10:
                shape = "INVERTED"
            elif abs(long_end - short_end) <= 0.25:
                shape = "FLAT"
            elif pd.notna(belly) and belly > max(short_end, long_end) + 0.10:
                shape = "HUMPED"
            else:
                shape = "NORMAL"
            out["shape"] = shape
            out["short_end_avg"] = round(float(short_end), 3)
            out["long_end_avg"] = round(float(long_end), 3)

    # --- Steepening / flattening over 30 days ------------------------------
    try:
        history = get_yield_curve_history(days=120)
        if not history.empty and {"10Y", "2Y"}.issubset(history.columns):
            hist_spread = (history["10Y"] - history["2Y"]).dropna()
            if len(hist_spread) > 21:
                change = float(hist_spread.iloc[-1] - hist_spread.iloc[-22])
                out["spread_change_30d_bps"] = round(change * 100, 1)
                out["curve_direction"] = (
                    "STEEPENING" if change > 0.05
                    else "FLATTENING" if change < -0.05
                    else "STABLE"
                )
    except Exception as exc:
        log.debug("Curve direction calc failed: %s", exc)

    return out


# ==========================================================================
# DASHBOARD SNAPSHOT
# ==========================================================================
def get_macro_dashboard() -> Dict[str, Dict[str, Any]]:
    """
    The headline macro tiles: latest value, prior value, and change.

    Each entry: {label, value, previous, change, change_pct, date, units}
    """
    indicators = [
        ("UNRATE", "Unemployment Rate", "%"),
        ("CPIAUCSL", "CPI (YoY)", "%", "yoy"),
        ("CPILFESL", "Core CPI (YoY)", "%", "yoy"),
        ("PCEPILFE", "Core PCE (YoY)", "%", "yoy"),
        ("FEDFUNDS", "Fed Funds Rate", "%"),
        ("PAYEMS", "Nonfarm Payrolls (chg)", "K", "diff"),
        ("ICSA", "Initial Claims", ""),
        ("M2SL", "M2 Money Supply (YoY)", "%", "yoy"),
        ("WALCL", "Fed Balance Sheet", "$M"),
        ("T10Y2Y", "10Y-2Y Spread", "%"),
        ("T10YIE", "10Y Breakeven Inflation", "%"),
        ("BAMLH0A0HYM2", "HY Credit Spread (OAS)", "%"),
    ]

    out: Dict[str, Dict[str, Any]] = {}

    for spec in indicators:
        series_id, label, units = spec[0], spec[1], spec[2]
        transform = spec[3] if len(spec) > 3 else None

        try:
            raw = get_fred_series(series_id, start=(date.today() - timedelta(days=1200)).isoformat())
            if raw is None or raw.empty:
                continue

            series = _apply_transform(raw, transform)
            if series.empty:
                continue

            latest = float(series.iloc[-1])
            previous = float(series.iloc[-2]) if len(series) >= 2 else None
            change = latest - previous if previous is not None else None

            out[series_id] = {
                "label": label,
                "value": latest,
                "previous": previous,
                "change": change,
                "change_pct": (change / abs(previous) * 100) if change is not None and previous else None,
                "date": series.index[-1],
                "units": units,
                "series": series.tail(60),   # sparkline data
            }
        except Exception as exc:
            log.warning("Macro dashboard item %s failed: %s", series_id, exc)

    return out


def _apply_transform(series: pd.Series, transform: Optional[str]) -> pd.Series:
    """Convert a raw index level into the reading people actually quote."""
    if transform == "yoy":
        # Infer periods-per-year from the observed index spacing.
        freq = _infer_periods_per_year(series)
        return (series.pct_change(freq) * 100).dropna()
    if transform == "diff":
        return series.diff().dropna()
    return series


def _infer_periods_per_year(series: pd.Series) -> int:
    """Monthly -> 12, weekly -> 52, quarterly -> 4, daily -> 252."""
    if len(series) < 3:
        return 12
    median_gap = pd.Series(series.index).diff().dt.days.median()
    if pd.isna(median_gap):
        return 12
    if median_gap <= 3:
        return 252
    if median_gap <= 10:
        return 52
    if median_gap <= 45:
        return 12
    return 4


# ==========================================================================
# FED POLICY EXPECTATIONS
# ==========================================================================
def get_rate_expectations() -> Dict[str, Any]:
    """
    Market-implied policy direction from 30-Day Fed Funds futures.

    CME publishes FedWatch probabilities but gates the data. We derive the
    same signal from free Yahoo quotes on the ZQ contract: implied rate =
    100 - price. Comparing that to the current effective rate tells you how
    many cuts or hikes the market has priced.

    This is a directional read from public futures prices, not CME FedWatch's
    exact probability distribution.
    """
    out: Dict[str, Any] = {}

    try:
        current = get_fred_series("DFEDTARU", start=(date.today() - timedelta(days=120)).isoformat())
        if current is not None and len(current):
            out["current_target_upper"] = float(current.iloc[-1])

        effective = get_fred_series("FEDFUNDS", start=(date.today() - timedelta(days=200)).isoformat())
        if effective is not None and len(effective):
            out["effective_rate"] = float(effective.iloc[-1])
    except Exception as exc:
        log.debug("Current policy rate fetch failed: %s", exc)

    # Front Fed Funds future.
    try:
        from data_fetchers.equities import get_quote

        quote = get_quote("ZQ=F")
        price = quote.get("price") if quote else None
        if price:
            implied = 100.0 - float(price)
            out["implied_rate_front_contract"] = round(implied, 3)

            reference = out.get("effective_rate")
            if reference is not None:
                delta = implied - reference
                out["implied_change_bps"] = round(delta * 100, 1)
                # One standard Fed move is 25bp.
                moves = delta / 0.25
                out["implied_moves"] = round(moves, 2)
                out["policy_bias"] = (
                    "CUTS PRICED" if moves <= -0.25
                    else "HIKES PRICED" if moves >= 0.25
                    else "ON HOLD"
                )
    except Exception as exc:
        log.debug("Fed funds futures fetch failed: %s", exc)

    # A 2s10s-based directional cross-check.
    try:
        two_year = get_fred_series("DGS2", start=(date.today() - timedelta(days=200)).isoformat())
        if two_year is not None and len(two_year) > 60:
            recent = float(two_year.iloc[-1])
            prior = float(two_year.iloc[-60])
            out["ust2y_3m_change_bps"] = round((recent - prior) * 100, 1)
            out["ust2y_signal"] = (
                "MARKET EASING BIAS" if recent < prior - 0.15
                else "MARKET TIGHTENING BIAS" if recent > prior + 0.15
                else "MARKET NEUTRAL"
            )
    except Exception:
        pass

    return out


# ==========================================================================
# FED NET LIQUIDITY
# ==========================================================================
# Multiplier to reach $ billions, keyed on the magnitude word FRED uses.
# It writes units as "Mil. of U.S. $" or "Millions of U.S. Dollars", so match
# the first three letters rather than enumerating every phrasing.
_UNIT_TO_BILLIONS: Dict[str, float] = {
    "bil": 1.0,
    "mil": 1e-3,
    "tri": 1e3,
    "tho": 1e-6,
}


def _scale_to_billions(units: Optional[str]) -> Optional[float]:
    """Multiplier taking a series in `units` to $bn, or None if unreadable."""
    if not units:
        return None
    lowered = str(units).lower()
    for token, factor in _UNIT_TO_BILLIONS.items():
        if token in lowered:
            return factor
    return None


@cached(ttl=config.TTL.macro, namespace="net_liquidity")
def get_net_liquidity(years: int = 5) -> pd.DataFrame:
    """
    Fed net liquidity: balance sheet minus the TGA minus the reverse repo.

    Total assets are only part of the story. Cash parked in the Treasury
    General Account or the overnight reverse repo facility is drained out of
    the financial system, so subtracting both gives the read that actually
    tracks risk assets.

    THE UNIT TRAP - the whole reason this function exists rather than being
    three inline subtractions. FRED does not publish these on one scale:

        WALCL      Mil. of U.S. $     ~6,676,000
        WTREGEN    Mil. of U.S. $       ~800,500
        RRPONTSYD  Bil. of US $              ~12

    Subtract them raw and the repo leg comes off a thousand times too small.
    Today that is a rounding error because the facility is nearly empty; in
    2022-23 it held roughly $2,200bn, where the same bug overstates net
    liquidity by $2.2 trillion. Nothing about the resulting chart looks wrong.

    WHY THERE IS NO KEYLESS FALLBACK: units live in FRED's series metadata,
    which needs an API key - the keyless CSV endpoint returns bare
    observations. Inferring the scale from magnitude was considered and
    rejected: RRPONTSYD currently reads ~12, which any magnitude heuristic
    reads as trillions and scales up by 1000x. A wrong scale here is
    invisible, so with no key this returns empty and says why rather than
    publishing a number it cannot stand behind.

    Returns:
        DataFrame indexed daily with walcl_bn, tga_bn, rrp_bn and
        net_liquidity_bn, plus `attrs["unit_source"]` mapping each leg to the
        units it was scaled from and `attrs["reason"]` when empty.
    """
    start = (date.today() - timedelta(days=int(365.25 * years))).isoformat()

    empty = pd.DataFrame()
    if not config.FRED_API_KEY:
        empty.attrs["reason"] = (
            "Net liquidity needs a FRED API key. The three legs are published "
            "on different scales (WALCL and WTREGEN in millions, RRPONTSYD in "
            "billions) and only the keyed metadata endpoint reports units. "
            "Free key at fredaccount.stlouisfed.org/apikeys."
        )
        return empty

    legs: Dict[str, pd.Series] = {}
    provenance: Dict[str, str] = {}

    for column, series_id in config.NET_LIQUIDITY_SERIES.items():
        try:
            raw = get_fred_series(series_id, start=start)
        except Exception as exc:
            log.warning("Net liquidity leg %s failed: %s", series_id, exc)
            raw = None

        if raw is None or raw.empty:
            provenance[column] = "unavailable"
            continue

        units = (get_fred_series_info(series_id) or {}).get("units")
        factor = _scale_to_billions(units)
        if factor is None:
            provenance[column] = f"unreadable units ({units!r})"
            continue

        legs[column] = raw * factor
        provenance[column] = str(units)

    missing = [c for c in config.NET_LIQUIDITY_SERIES if c not in legs]
    if missing:
        empty.attrs["unit_source"] = provenance
        empty.attrs["reason"] = (
            "Could not place every leg on a common scale; missing or "
            f"unreadable: {', '.join(missing)}. Net liquidity is not shown "
            "rather than shown with a leg on the wrong scale."
        )
        return empty

    # Weekly balance sheet, weekly TGA, daily repo - LOCF onto a daily grid so
    # the subtraction uses the latest known value of each.
    panel = macro_analytics.align_panel(legs, freq="D").ffill()
    panel = panel.rename(columns={c: f"{c}_bn" for c in panel.columns})
    panel = panel.dropna(subset=["walcl_bn", "tga_bn", "rrp_bn"])

    panel["net_liquidity_bn"] = (
        panel["walcl_bn"] - panel["tga_bn"] - panel["rrp_bn"]
    )

    panel.attrs["unit_source"] = provenance
    return panel

# ==========================================================================
# WORLD BANK
# ==========================================================================
@cached(ttl=86400 * 7, namespace="worldbank")
@throttled("worldbank")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_world_bank_indicator(
    indicator: str,
    countries: str = "US;CN;JP;DE;GB;IN;FR;BR",
    years: int = 15,
) -> pd.DataFrame:
    """
    Cross-country macro from the World Bank Open Data API (free, no key).

    Args:
        indicator: e.g. "NY.GDP.MKTP.KD.ZG" (see config.WORLD_BANK_INDICATORS)
        countries: Semicolon-separated ISO2 codes.

    Returns: DataFrame with country | iso3 | year | value
    """
    end_year = date.today().year
    url = f"{config.WORLD_BANK_BASE}/country/{countries}/indicator/{indicator}"

    resp = get_session("worldbank", expire_after=86400 * 7).get(
        url,
        params={
            "format": "json",
            "per_page": "2000",
            "date": f"{end_year - years}:{end_year}",
        },
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()
    payload = resp.json()

    # Response shape: [metadata, [observations]]
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError(f"World Bank returned no data for {indicator}")

    rows = [
        {
            "country": item["country"]["value"],
            "iso3": item.get("countryiso3code"),
            "year": int(item["date"]),
            "value": item["value"],
        }
        for item in payload[1]
        if item.get("value") is not None
    ]

    if not rows:
        raise ValueError(f"All World Bank observations null for {indicator}")

    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)


# ==========================================================================
# COMPOSITE RISK READOUT
# ==========================================================================
def get_recession_indicators() -> Dict[str, Any]:
    """
    Composite recession dashboard combining several independent signals.

    Signals scored:
      * 10Y-3M inversion            (NY Fed model input)
      * 10Y-2Y inversion            (headline measure)
      * Sahm Rule                   (real-time unemployment trigger)
      * High-yield credit spreads   (corporate stress)
      * Financial stress index      (St. Louis Fed)

    Each contributes to a 0-100 composite. This aggregates public series;
    it is not a forecast and carries no confidence interval.
    """
    signals: List[Dict[str, Any]] = []
    start = (date.today() - timedelta(days=900)).isoformat()

    def add(name: str, value: Optional[float], triggered: bool,
            detail: str, weight: float) -> None:
        signals.append({
            "name": name, "value": value, "triggered": triggered,
            "detail": detail, "weight": weight,
        })

    # 1. Yield curve inversions
    for series_id, name, weight in (
        ("T10Y3M", "10Y-3M Curve Inversion", 30.0),
        ("T10Y2Y", "10Y-2Y Curve Inversion", 25.0),
    ):
        try:
            series = get_fred_series(series_id, start=start)
            if series is not None and len(series):
                latest = float(series.iloc[-1])
                add(name, latest, latest < 0,
                    f"Spread at {latest * 100:.0f}bp", weight)
        except Exception:
            pass

    # 2. Sahm Rule: 3m avg unemployment minus its trailing 12m low >= 0.50
    try:
        unrate = get_fred_series("UNRATE", start=(date.today() - timedelta(days=1100)).isoformat())
        if unrate is not None and len(unrate) >= 15:
            three_month = unrate.rolling(3).mean()
            trailing_low = three_month.rolling(12).min()
            sahm = float((three_month - trailing_low).iloc[-1])
            add("Sahm Rule", round(sahm, 2), sahm >= 0.50,
                f"3m avg unemployment {sahm:.2f}pp above 12m low "
                f"(trigger at 0.50)", 25.0)
    except Exception:
        pass

    # 3. High-yield credit spreads
    try:
        hy = get_fred_series("BAMLH0A0HYM2", start=start)
        if hy is not None and len(hy):
            latest = float(hy.iloc[-1])
            # >6% OAS has historically coincided with meaningful risk-off.
            add("HY Credit Spread", round(latest, 2), latest > 6.0,
                f"OAS at {latest:.2f}% (stress threshold ~6%)", 10.0)
    except Exception:
        pass

    # 4. Financial stress index (0 = normal by construction)
    try:
        stress = get_fred_series("STLFSI4", start=start)
        if stress is not None and len(stress):
            latest = float(stress.iloc[-1])
            add("Financial Stress Index", round(latest, 2), latest > 1.0,
                f"STLFSI4 at {latest:.2f} (0 = historical normal)", 10.0)
    except Exception:
        pass

    total_weight = sum(s["weight"] for s in signals) or 1.0
    triggered_weight = sum(s["weight"] for s in signals if s["triggered"])
    score = round(triggered_weight / total_weight * 100, 1)

    return {
        "signals": signals,
        "score": score,
        "triggered_count": sum(1 for s in signals if s["triggered"]),
        "total_count": len(signals),
        "level": (
            "HIGH" if score >= 60 else
            "ELEVATED" if score >= 35 else
            "MODERATE" if score >= 15 else
            "LOW"
        ),
    }


__all__ = [
    "get_fred_series", "get_fred_series_info", "get_macro_series_group",
    "get_yield_curve", "get_yield_curve_history", "analyze_yield_curve",
    "get_macro_dashboard", "get_rate_expectations", "get_net_liquidity",
    "get_world_bank_indicator", "get_recession_indicators",
    "FREDAPI_AVAILABLE",
]

"""
data_fetchers/allocation.py :: Module J - Sector exposure and rebalancing.

Bloomberg equivalents: PORT (portfolio analytics), the attribution and
allocation tabs.

WHAT THIS ANSWERS
-----------------
Where is the book actually invested, how far is that from a chosen
benchmark, and what would it take in dollars to close the gap.

TWO THINGS IT REFUSES TO GUESS
------------------------------
1. **A fund is not a sector.** VOO is not "a technology position" and it is
   not unclassifiable either - it is 37% technology, 12% financials and nine
   other things. Every holding whose `quoteType` is a fund is looked through
   to its published sector weights and distributed across sectors
   accordingly. Assigning an S&P 500 ETF to one sector, or dropping it, both
   produce an exposure chart that is confidently wrong.

2. **Benchmark weights are read, never typed.** Each benchmark names a real
   fund and its CURRENT published sector weights define the target. Index
   sector weights move constantly - technology has gone from roughly a
   quarter of the S&P 500 to over a third in a few years - so a hardcoded
   target table is wrong the day after it is written and keeps rendering a
   confident "drift vs benchmark" column while it rots.

Anything that cannot be classified is reported as UNCLASSIFIED with its
dollar value, and is excluded from the drift maths rather than being spread
across sectors or silently dropped. A portfolio that is 30% unclassified
should look 30% unclassified.

NOT ADVICE
----------
This computes the arithmetic difference between two weight vectors and
converts it to dollars. It has no view on whether the benchmark is the right
one for you, on tax, on transaction costs, or on your circumstances.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_fetchers import equities
from utils.cache import cached

log = logging.getLogger("openterm.allocation")

UNCLASSIFIED = "UNCLASSIFIED"


def normalise_sector(raw: Any) -> Optional[str]:
    """
    Map a Yahoo sector label onto its GICS name.

    Handles both spellings Yahoo uses: title case from `info["sector"]`
    ("Consumer Cyclical") and snake_case from an ETF's sector weightings
    ("consumer_cyclical"). Returns None for anything unrecognised rather than
    inventing a sector.
    """
    if not raw:
        return None
    key = str(raw).lower().replace("_", "").replace(" ", "").replace("-", "")
    return config.YAHOO_TO_GICS.get(key)


@cached(ttl=config.TTL.fundamentals, namespace="alloc_classify")
def classify(ticker: str) -> Dict[str, Any]:
    """
    Sector breakdown for one holding.

    Returns:
        {kind, sectors: {gics: fraction}, source, note}

        kind is EQUITY (a single sector at 100%), FUND (looked through to its
        published sector weights) or UNCLASSIFIED.

    A fund's weights are used as published even when they do not sum to
    exactly 1.0 - they are renormalised, and the shortfall is reported rather
    than assigned somewhere convenient.
    """
    symbol = equities.normalize_ticker(ticker).upper()
    out: Dict[str, Any] = {"ticker": symbol, "kind": UNCLASSIFIED,
                           "sectors": {}, "source": "", "note": ""}

    try:
        info = equities.get_company_info(symbol) or {}
    except Exception as exc:
        log.warning("Classification failed for %s: %s", symbol, exc)
        out["note"] = f"Company profile unavailable: {exc}"
        return out

    quote_type = str(info.get("quoteType") or "").upper()

    # --- Single-sector equity --------------------------------------------
    sector = normalise_sector(info.get("sector"))
    if sector:
        out.update({"kind": "EQUITY", "sectors": {sector: 1.0},
                    "source": "Yahoo sector, mapped to GICS"})
        return out

    # --- Fund: look through to its published weights ----------------------
    # Attempted whenever quoteType says fund OR no sector was reported at
    # all. The absence of a sector is itself the tell, and relying on
    # quoteType alone made this branch unreachable when the field was not
    # being surfaced - every ETF filed as unclassified while its weights sat
    # one call away.
    if quote_type in ("ETF", "MUTUALFUND", "INDEX") or not info.get("sector"):
        weights = _fund_sector_weights(symbol)
        if weights:
            out.update({
                "kind": "FUND", "sectors": weights,
                "source": f"{symbol} published sector weightings, mapped to GICS",
                "note": "Looked through to constituent sectors - a fund is "
                        "not a single-sector position.",
            })
            return out
        out["note"] = (f"{symbol} is a {quote_type.lower()} but publishes no "
                       "sector weightings, so it cannot be allocated.")
        return out

    if info.get("sector"):
        out["note"] = (f"Sector '{info['sector']}' is not in the GICS "
                       "crosswalk.")
    else:
        out["note"] = "No sector reported for this symbol."
    return out


def _fund_sector_weights(symbol: str) -> Dict[str, float]:
    """A fund's sector weightings, mapped to GICS and renormalised to 1.0."""
    if not equities.YFINANCE_AVAILABLE:
        return {}
    try:
        raw = equities.yf.Ticker(symbol).funds_data.sector_weightings or {}
    except Exception as exc:
        log.info("No fund sector weightings for %s: %s", symbol, exc)
        return {}

    mapped: Dict[str, float] = {}
    for label, weight in raw.items():
        sector = normalise_sector(label)
        if sector is None:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if value > 0:
            mapped[sector] = mapped.get(sector, 0.0) + value

    total = sum(mapped.values())
    if total <= 0:
        return {}
    return {sector: weight / total for sector, weight in mapped.items()}


# ==========================================================================
# EXPOSURE
# ==========================================================================
def sector_exposure(valued: pd.DataFrame) -> pd.DataFrame:
    """
    Dollar and percentage exposure per GICS sector, with funds looked through.

    Args:
        valued: the frame from `portfolio.value_positions`, which carries
                ticker and market_value.

    Returns:
        DataFrame [sector, market_value, weight_pct] covering every sector
        with exposure plus an UNCLASSIFIED row when anything could not be
        placed. `df.attrs` carries `unclassified` (the tickers) and
        `lookthrough` (the funds that were decomposed).
    """
    empty = pd.DataFrame(columns=["sector", "market_value", "weight_pct"])
    if valued is None or valued.empty or "market_value" not in valued.columns:
        empty.attrs.update({"unclassified": [], "lookthrough": [], "total": 0.0})
        return empty

    priced = valued.dropna(subset=["market_value"])
    total = float(priced["market_value"].sum())
    if total <= 0:
        empty.attrs.update({"unclassified": [], "lookthrough": [], "total": 0.0})
        return empty

    buckets: Dict[str, float] = {}
    unclassified: List[Dict[str, Any]] = []
    lookthrough: List[Dict[str, Any]] = []

    for _, row in priced.iterrows():
        symbol = row["ticker"]
        value = float(row["market_value"])
        result = classify(symbol)

        if not result["sectors"]:
            buckets[UNCLASSIFIED] = buckets.get(UNCLASSIFIED, 0.0) + value
            unclassified.append({"ticker": symbol, "market_value": value,
                                 "reason": result.get("note", "")})
            continue

        if result["kind"] == "FUND":
            lookthrough.append({
                "ticker": symbol, "market_value": value,
                "sectors": len(result["sectors"]),
                "largest": max(result["sectors"].items(),
                               key=lambda kv: kv[1])[0],
            })

        for sector, fraction in result["sectors"].items():
            buckets[sector] = buckets.get(sector, 0.0) + value * fraction

    rows = [{"sector": sector, "market_value": value,
             "weight_pct": value / total * 100.0}
            for sector, value in buckets.items()]

    frame = pd.DataFrame(rows).sort_values(
        "market_value", ascending=False).reset_index(drop=True)
    frame.attrs.update({"unclassified": unclassified,
                        "lookthrough": lookthrough, "total": total})
    return frame


# ==========================================================================
# BENCHMARK
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="alloc_benchmark")
def benchmark_weights(key: str = config.DEFAULT_BENCHMARK) -> Dict[str, Any]:
    """
    Target sector weights, read live from the benchmark's proxy fund.

    Returns {key, label, proxy, weights: {gics: pct}, note}. `weights` is
    empty when the fund did not publish - the caller must then decline to
    compute drift rather than falling back to a stale table.
    """
    benchmark = config.ALLOCATION_BENCHMARKS.get(key)
    if benchmark is None:
        return {"key": key, "weights": {},
                "note": f"Unknown benchmark '{key}'."}

    weights = _fund_sector_weights(benchmark.proxy)
    return {
        "key": benchmark.key,
        "label": benchmark.label,
        "proxy": benchmark.proxy,
        "description": benchmark.description,
        "weights": {sector: fraction * 100.0
                    for sector, fraction in weights.items()},
        "note": ("" if weights else
                 f"{benchmark.proxy} published no sector weightings, so this "
                 "benchmark has no targets and no drift can be computed."),
    }


# ==========================================================================
# REBALANCING
# ==========================================================================
def rebalance(valued: pd.DataFrame,
              benchmark_key: str = config.DEFAULT_BENCHMARK) -> Dict[str, Any]:
    """
    Sector exposure against a benchmark, with the dollar moves to close it.

    Actions are only produced for drift beyond
    `config.REBALANCE_MIN_DRIFT_PCT`. Below that the gap costs more in spread
    and tax to fix than it represents, and a list of eleven trivial "actions"
    buries the two that matter.

    Unclassified holdings are excluded from the drift arithmetic and reported
    separately, with their weight stated. Spreading them across sectors to
    make the percentages tidy would be inventing exposure.
    """
    exposure = sector_exposure(valued)
    benchmark = benchmark_weights(benchmark_key)

    total = float(exposure.attrs.get("total", 0.0) or 0.0)
    unclassified_rows = exposure.attrs.get("unclassified", [])
    unclassified_value = sum(row["market_value"] for row in unclassified_rows)

    report: Dict[str, Any] = {
        "as_of": datetime.now(timezone.utc),
        "benchmark": benchmark,
        "total_value": total,
        "exposure": exposure,
        "unclassified": unclassified_rows,
        "unclassified_value": unclassified_value,
        "unclassified_pct": (unclassified_value / total * 100.0
                             if total else 0.0),
        "lookthrough": exposure.attrs.get("lookthrough", []),
        "rows": pd.DataFrame(),
        "actions": [],
        "disclaimer": (
            "Arithmetic difference between two weight vectors, converted to "
            "dollars. No view on tax, transaction costs, or whether this "
            "benchmark suits you. Not advice."
        ),
    }

    if total <= 0:
        report["note"] = "No priced positions to allocate."
        return report

    if not benchmark.get("weights"):
        report["note"] = benchmark.get("note", "Benchmark unavailable.")
        return report

    # Drift is measured over the classified book only. Including an
    # unclassified slug in the denominator would understate every sector
    # weight by the same amount and make the whole book look underweight.
    classified = exposure[exposure["sector"] != UNCLASSIFIED]
    classified_total = float(classified["market_value"].sum())
    if classified_total <= 0:
        report["note"] = "Nothing in the book could be classified by sector."
        return report

    current = {row["sector"]: row["market_value"] / classified_total * 100.0
               for _, row in classified.iterrows()}

    rows: List[Dict[str, Any]] = []
    for sector in config.GICS_SECTORS:
        current_pct = current.get(sector, 0.0)
        target_pct = benchmark["weights"].get(sector, 0.0)
        drift = current_pct - target_pct
        rows.append({
            "Sector": sector,
            "Current $": current_pct / 100.0 * classified_total,
            "Current %": current_pct,
            "Target %": target_pct,
            "Drift %": drift,
            "Adjust $": -drift / 100.0 * classified_total,
        })

    frame = pd.DataFrame(rows).sort_values("Drift %", ascending=False)
    report["rows"] = frame.reset_index(drop=True)
    report["classified_total"] = classified_total
    report["actions"] = _actions(frame, valued)
    report["concentration"] = concentration(valued)
    return report


def _actions(frame: pd.DataFrame, valued: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Concrete trims and adds, largest gap first.

    A sector at zero weight is called out distinctly from one that is merely
    light: "you hold none of this" is a different decision from "you hold a
    little less than the index".
    """
    threshold = config.REBALANCE_MIN_DRIFT_PCT
    actions: List[Dict[str, Any]] = []

    holders = _sector_holders(valued)

    for _, row in frame.iterrows():
        drift = row["Drift %"]
        if abs(drift) < threshold:
            continue

        sector = row["Sector"]
        if drift > 0:
            names = holders.get(sector, [])
            actions.append({
                "action": "TRIM",
                "sector": sector,
                "drift_pct": drift,
                "amount": abs(row["Adjust $"]),
                "detail": (
                    f"Overweight by {drift:.1f}pp "
                    f"({row['Current %']:.1f}% held vs {row['Target %']:.1f}% "
                    f"target). Reducing by ${abs(row['Adjust $']):,.0f} would "
                    f"align it."
                    + (f" Held via {', '.join(names)}." if names else "")),
            })
        elif row["Current %"] == 0:
            actions.append({
                "action": "OPEN",
                "sector": sector,
                "drift_pct": drift,
                "amount": abs(row["Adjust $"]),
                "detail": (
                    f"No exposure at all against a {row['Target %']:.1f}% "
                    f"benchmark weight. ${abs(row['Adjust $']):,.0f} would "
                    "bring it to target."),
            })
        else:
            actions.append({
                "action": "ADD",
                "sector": sector,
                "drift_pct": drift,
                "amount": abs(row["Adjust $"]),
                "detail": (
                    f"Underweight by {abs(drift):.1f}pp "
                    f"({row['Current %']:.1f}% held vs {row['Target %']:.1f}% "
                    f"target). Adding ${abs(row['Adjust $']):,.0f} would "
                    "align it."),
            })

    return sorted(actions, key=lambda a: abs(a["drift_pct"]), reverse=True)


def _sector_holders(valued: pd.DataFrame) -> Dict[str, List[str]]:
    """Which single-sector positions sit in each sector, for the trim note."""
    holders: Dict[str, List[str]] = {}
    if valued is None or valued.empty:
        return holders

    for _, row in valued.dropna(subset=["market_value"]).iterrows():
        result = classify(row["ticker"])
        if result["kind"] != "EQUITY":
            continue
        for sector in result["sectors"]:
            holders.setdefault(sector, []).append(row["ticker"])
    return holders


# ==========================================================================
# CONCENTRATION
# ==========================================================================
def concentration(valued: pd.DataFrame) -> Dict[str, Any]:
    """
    How concentrated the book is, by position.

    Concentration is a property of the position, not the sector: a book that
    is perfectly sector-neutral can still be one name away from ruin. The
    effective position count is 1/HHI - a portfolio of twenty names where one
    is 60% has an effective count near three, which is the number worth
    knowing.
    """
    out: Dict[str, Any] = {"positions": 0, "largest": None,
                           "largest_ticker": None, "top3_pct": None,
                           "effective_positions": None, "flags": []}
    if valued is None or valued.empty or "weight" not in valued.columns:
        return out

    weights = valued.dropna(subset=["weight"]).sort_values(
        "weight", ascending=False)
    if weights.empty:
        return out

    fractions = weights["weight"] / 100.0
    hhi = float((fractions ** 2).sum())

    out.update({
        "positions": len(weights),
        "largest": float(weights["weight"].iloc[0]),
        "largest_ticker": weights["ticker"].iloc[0],
        "top3_pct": float(weights["weight"].head(3).sum()),
        "effective_positions": (1.0 / hhi) if hhi > 0 else None,
    })

    limit = config.POSITION_CONCENTRATION_PCT
    for _, row in weights.iterrows():
        if row["weight"] >= limit:
            out["flags"].append(
                f"{row['ticker']} is {row['weight']:.1f}% of the book, above "
                f"the {limit:.0f}% single-position marker.")

    unpriced = int(valued["market_value"].isna().sum())
    if unpriced:
        out["flags"].append(
            f"{unpriced} position(s) have no price, so they are absent from "
            "these weights entirely - the percentages describe the priced "
            "book only.")

    return out


def to_schema(report: Dict[str, Any]) -> Dict[str, Any]:
    """The report as a plain JSON-serialisable object, DataFrames flattened."""
    as_of = report.get("as_of")
    rows = report.get("rows")
    exposure = report.get("exposure")

    return {
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else None,
        "benchmark": {k: v for k, v in (report.get("benchmark") or {}).items()},
        "total_value": report.get("total_value"),
        "classified_total": report.get("classified_total"),
        "unclassified_value": report.get("unclassified_value"),
        "unclassified_pct": report.get("unclassified_pct"),
        "unclassified_holdings": report.get("unclassified", []),
        "lookthrough": report.get("lookthrough", []),
        "sector_allocation": (
            exposure.to_dict("records") if isinstance(exposure, pd.DataFrame)
            and not exposure.empty else []),
        "rebalance_rows": (rows.to_dict("records")
                           if isinstance(rows, pd.DataFrame) and not rows.empty
                           else []),
        "actions": report.get("actions", []),
        "concentration": report.get("concentration", {}),
        "note": report.get("note", ""),
        "disclaimer": report.get("disclaimer"),
    }


__all__ = [
    "normalise_sector", "classify", "sector_exposure", "benchmark_weights",
    "rebalance", "concentration", "to_schema", "UNCLASSIFIED",
]

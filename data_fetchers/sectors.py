"""
data_fetchers/sectors.py :: Module K - Sector drill-down.

Bloomberg equivalents: the member list behind a sector tile on IMAP / BI.

WHAT THIS ANSWERS
-----------------
Which companies make up a sector, how much of it each one is, and how they
are trading today - narrowed by name, industry, analyst rating and size.

WHAT IT DOES NOT CLAIM
----------------------
Yahoo publishes up to fifty of the largest companies per industry, not every
listed issuer. Technology counts 853 companies and publishes about 350 across
its twelve industries. The list is therefore always reported against Yahoo's
own count, with the share of sector market cap it covers. A partial list
presented as "the sector" would look complete and not be.

Weights are read, never typed: a company's share of the sector is its
published share of its industry times the industry's published share of the
sector.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

import config
from data_fetchers import equities
from utils.cache import cached

log = logging.getLogger("openterm.sectors")

UNRATED = "Unrated"

# Yahoo's analyst consensus labels, most positive first. Orders the rating
# filter; a label Yahoo adds later is listed after these rather than dropped.
RATING_ORDER = ("Strong Buy", "Buy", "Hold", "Underperform", "Sell")

TABLE_COLUMNS = ["ticker", "name", "industry", "rating", "industry_weight_pct",
                 "sector_weight_pct", "market_cap"]


def _rating(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() != "nan" else UNRATED


def _member_row(symbol: Any, member: pd.Series, industry: Optional[str],
                within: Optional[float],
                sector_share: Optional[float]) -> Dict[str, Any]:
    return {
        "ticker": str(symbol).upper(),
        "name": str(member.get("name") or ""),
        "industry": industry,
        "rating": _rating(member.get("rating")),
        "industry_weight_pct": within * 100.0 if within is not None else None,
        "sector_weight_pct": (sector_share * 100.0
                              if sector_share is not None else None),
    }


# ==========================================================================
# CONSTITUENTS
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="sector_constituents")
def get_sector_constituents(sector_key: str) -> Dict[str, Any]:
    """
    Every company Yahoo publishes for one sector, gathered industry by industry.

    Args:
        sector_key: Yahoo's sector slug, e.g. "technology", "real-estate".

    Returns:
        {key, name, companies_count, market_cap, industries,
         failed_industries, coverage_pct, table}

        `table` has TABLE_COLUMNS, largest share of the sector first.
        `market_cap` is the company's sector weight times the sector's total
        cap - within a few percent of the company's own figure (NVDA: 5,067B
        derived against 5,271B reported), so it is shown as approximate.
        `coverage_pct` is the share of sector cap the listed companies make
        up, which is what says how much the fifty-per-industry limit left out.

    An industry that fails to load is named in `failed_industries` rather than
    silently shrinking the list. The sector's own leaders table backfills its
    largest names with no industry attached, so a failed call never hides the
    companies that matter most.

    Raises when nothing at all comes back, so `@cached` serves the last good
    list instead of caching an empty one for a day.
    """
    if not equities.YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance unavailable")

    sector = equities.yf.Sector(sector_key)
    overview = sector.overview or {}
    industries = sector.industries
    if industries is None or industries.empty:
        raise ValueError(f"No industries published for sector '{sector_key}'")

    keys = [str(key) for key in industries.index]
    # Same pattern as build_brief: twenty-odd serial requests make a click
    # feel dead. Each industry is one small JSON call.
    with ThreadPoolExecutor(max_workers=4) as pool:
        tables = dict(zip(keys, pool.map(
            lambda key: equities._constituents("Industry", key), keys)))

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    for key in keys:
        label = (str(industries.at[key, "name"])
                 if "name" in industries.columns else key)
        table = tables.get(key)
        if table is None:
            failed.append(label)
            continue
        share = (equities._safe_float(industries.at[key, "market weight"])
                 if "market weight" in industries.columns else None)
        for symbol, member in table.iterrows():
            within = equities._safe_float(member.get("market weight"))
            # A zero-weight constituent is delisted or untraded - the same
            # rule suggest_peers applies - not a company to browse.
            if within is not None and within <= 0:
                continue
            rows.append(_member_row(
                symbol, member, label, within,
                within * share if within is not None and share is not None
                else None))

    listed = {row["ticker"] for row in rows}
    leaders = equities._constituents("Sector", sector_key)
    if leaders is not None:
        for symbol, member in leaders.iterrows():
            weight = equities._safe_float(member.get("market weight"))
            if str(symbol).upper() in listed or (weight is not None and weight <= 0):
                continue
            rows.append(_member_row(symbol, member, None, None, weight))
            listed.add(str(symbol).upper())

    if not rows:
        raise ValueError(f"No constituents returned for sector '{sector_key}'")

    frame = pd.DataFrame(rows, columns=TABLE_COLUMNS[:-1])
    for column in ("industry_weight_pct", "sector_weight_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (frame.sort_values("sector_weight_pct", ascending=False,
                               na_position="last")
             .drop_duplicates(subset=["ticker"], keep="first")
             .reset_index(drop=True))

    cap = equities._safe_float(overview.get("market_cap"))
    frame["market_cap"] = (frame["sector_weight_pct"] * (cap / 100.0)
                           if cap else float("nan"))

    count = equities._safe_float(overview.get("companies_count"))
    return {
        "key": sector_key,
        "name": getattr(sector, "name", None) or sector_key,
        "companies_count": int(count) if count else None,
        "market_cap": cap,
        "industries": len(keys),
        "failed_industries": failed,
        "coverage_pct": float(frame["sector_weight_pct"].sum(skipna=True)),
        "table": frame,
    }


# ==========================================================================
# FILTERS
# ==========================================================================
def filter_constituents(table: pd.DataFrame, query: str = "",
                        industries: Iterable[str] = (),
                        ratings: Iterable[str] = (),
                        min_market_cap: Optional[float] = None) -> pd.DataFrame:
    """
    Rows matching every filter given. An empty filter matches everything.

    `query` is a case-insensitive plain substring of the ticker or the name -
    not a regex, so "c.o" or "S&P (A)" mean what they say. A minimum market
    cap excludes companies whose cap is unknown: a size filter cannot vouch
    for a size nobody published.
    """
    if table is None or table.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    mask = pd.Series(True, index=table.index)

    text = str(query or "").strip().lower()
    if text:
        mask &= (table["ticker"].astype(str).str.lower()
                 .str.contains(text, regex=False)
                 | table["name"].fillna("").astype(str).str.lower()
                 .str.contains(text, regex=False))

    chosen = [str(item) for item in (industries or ())]
    if chosen:
        mask &= table["industry"].isin(chosen)

    wanted = [str(item) for item in (ratings or ())]
    if wanted:
        mask &= table["rating"].isin(wanted)

    if min_market_cap:
        mask &= pd.to_numeric(table["market_cap"], errors="coerce").fillna(-1.0) \
            >= float(min_market_cap)

    return table.loc[mask].reset_index(drop=True)


def industry_options(table: pd.DataFrame) -> List[str]:
    """Industries present in the table, largest share of the sector first."""
    if table is None or table.empty:
        return []
    totals = (table.dropna(subset=["industry"])
              .groupby("industry")["sector_weight_pct"].sum(min_count=1))
    return [str(name) for name in
            totals.sort_values(ascending=False, na_position="last").index]


def rating_options(table: pd.DataFrame) -> List[str]:
    """Ratings present, in consensus order, with Unrated last."""
    if table is None or table.empty:
        return []
    present = set(table["rating"].dropna().astype(str))
    ordered = [rating for rating in RATING_ORDER if rating in present]
    extra = sorted(present - set(RATING_ORDER) - {UNRATED})
    return ordered + extra + ([UNRATED] if UNRATED in present else [])


def attach_quotes(table: pd.DataFrame,
                  quotes: Optional[Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """The table with `price` and `change_pct` for whichever symbols were quoted."""
    out = table.copy()
    quotes = quotes or {}
    for column, field in (("price", "price"), ("change_pct", "change_pct")):
        out[column] = pd.to_numeric(pd.Series(
            [(quotes.get(ticker) or {}).get(field) for ticker in out["ticker"]],
            index=out.index, dtype=object), errors="coerce")
    return out


__all__ = [
    "UNRATED", "RATING_ORDER", "TABLE_COLUMNS",
    "get_sector_constituents", "filter_constituents",
    "industry_options", "rating_options", "attach_quotes",
]

"""
Portfolio, watchlist, and the 08:00 SGT holdings brief.

Positions are the one thing in this terminal the user authored, so they live
in a plain JSON file under .openterm/ rather than in the SQLite cache. The
cache is disposable by design - the sidebar's PURGE button empties it - and
holdings are not.

The brief is an *edition*, not a background job. Asking a Streamlit app to run
something at 08:00 assumes a process is alive at 08:00, which for a local
terminal is exactly when it is not. So each edition is stamped with the SGT
trading day it belongs to: before 08:00 SGT you are still reading yesterday's,
and the first page load at or after 08:00 builds today's and persists it.
Open the terminal at 07:00, at 08:00 or at noon and you read the same edition
you would have read at 08:00 - which is the property that actually matters.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from data_fetchers import equities, news

log = logging.getLogger("openterm.portfolio")

# Singapore has been UTC+8 with no daylight saving since 1982, so a fixed
# offset is exact here. It also keeps the brief working on machines with no
# IANA tz database installed, which is the common case for Python on Windows.
SGT = timezone(timedelta(hours=8), "SGT")

HOLDING_COLUMNS = ["ticker", "quantity", "cost_basis", "note"]
WATCHLIST_COLUMNS = ["ticker", "note"]


# ==========================================================================
# 1. STORE
# ==========================================================================
def _empty_store() -> Dict[str, List[Dict[str, Any]]]:
    return {"holdings": [], "watchlist": []}


def load() -> Dict[str, List[Dict[str, Any]]]:
    """
    Read the position file. A missing or corrupt file is not an error.

    A terminal that refuses to open because its state file has a stray comma
    is worse than one that opens empty and lets you retype four rows, so a
    parse failure is logged and the file is left on disk untouched.
    """
    path = config.PORTFOLIO_FILE
    if not path.exists():
        return _empty_store()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Portfolio file unreadable (%s) - starting empty: %s", path, exc)
        return _empty_store()

    store = _empty_store()
    for key in store:
        rows = raw.get(key)
        if isinstance(rows, list):
            store[key] = [row for row in rows if isinstance(row, dict)]
    return store


def _records(frame: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    """
    Rows as plain JSON types.

    An empty cost basis arrives here as a float NaN, and json.dumps writes
    that as a bare `NaN` token - which Python reads back happily and every
    other JSON parser rejects. Positions are the user's data; the file has to
    stay readable by something other than this app.
    """
    cleaned = _clean(frame, columns)
    if cleaned.empty:
        return []
    return [
        {key: (None if isinstance(value, float) and pd.isna(value) else value)
         for key, value in row.items()}
        for row in cleaned.to_dict("records")
    ]


def save(holdings: pd.DataFrame, watchlist: pd.DataFrame) -> None:
    """Write both tables back, normalised and with empty rows dropped."""
    payload = {
        "holdings": _records(holdings, HOLDING_COLUMNS),
        "watchlist": _records(watchlist, WATCHLIST_COLUMNS),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so a future regression fails loudly here rather than
    # writing a file that only Python can read.
    config.PORTFOLIO_FILE.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    log.info("Portfolio saved: %d holdings, %d watchlist",
             len(payload["holdings"]), len(payload["watchlist"]))


def _clean(df: Optional[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    """Normalise tickers, coerce numerics, drop rows with no symbol."""
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    out = out[columns]

    out["ticker"] = (out["ticker"].astype(str).str.strip().str.upper()
                     .replace({"": None, "NONE": None, "NAN": None}))
    out = out[out["ticker"].notna()]
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["ticker"] = out["ticker"].map(equities.normalize_ticker)

    for column in ("quantity", "cost_basis"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "note" in out.columns:
        out["note"] = out["note"].fillna("").astype(str).str.slice(0, 120)

    # Last row wins on a duplicated symbol - the same position typed twice is
    # an edit, not two positions.
    return out.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def holdings() -> pd.DataFrame:
    """Stored holdings, exactly as entered."""
    return _clean(pd.DataFrame(load()["holdings"]), HOLDING_COLUMNS)


def watchlist() -> pd.DataFrame:
    """Stored watchlist, exactly as entered."""
    return _clean(pd.DataFrame(load()["watchlist"]), WATCHLIST_COLUMNS)


def add_to_watchlist(ticker: str, note: str = "") -> bool:
    """
    Append one symbol. Returns False if it was already there.

    Backs the `NVDA WATCH` command form, so it has to be idempotent - typing
    the same command twice should not produce two rows.
    """
    symbol = equities.normalize_ticker(str(ticker).strip().upper())
    if not symbol:
        return False

    current = watchlist()
    if not current.empty and symbol in set(current["ticker"]):
        return False

    row = pd.DataFrame([{"ticker": symbol, "note": note}])
    save(holdings(), pd.concat([current, row], ignore_index=True))
    return True


# ==========================================================================
# 2. VALUATION
# ==========================================================================
def _quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}
    try:
        return equities.get_quotes_batch(tuple(sorted(set(symbols))))
    except Exception as exc:
        log.warning("Position quotes failed: %s", exc)
        return {}


def value_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Mark holdings to market.

    Cost basis is optional: a row with a quantity but no basis still gets a
    market value and a day move, it just cannot show a return. Reporting a
    P&L of -100% because the basis field was left blank would be worse than
    reporting nothing.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    quotes = _quotes(list(frame["ticker"]))
    rows: List[Dict[str, Any]] = []

    for _, holding in frame.iterrows():
        symbol = holding["ticker"]
        quote = quotes.get(symbol, {})
        price = quote.get("price")
        quantity = holding.get("quantity")
        basis = holding.get("cost_basis")

        market_value = (price * quantity
                        if price is not None and pd.notna(quantity) else None)
        cost = (basis * quantity
                if pd.notna(basis) and pd.notna(quantity) else None)
        pnl = (market_value - cost
               if market_value is not None and cost is not None else None)
        day_pnl = (quote.get("change") * quantity
                   if quote.get("change") is not None and pd.notna(quantity)
                   else None)

        rows.append({
            "ticker": symbol,
            "quantity": quantity,
            "cost_basis": basis,
            "price": price,
            "change_pct": quote.get("change_pct"),
            "day_pnl": day_pnl,
            "market_value": market_value,
            "cost": cost,
            "pnl": pnl,
            "pnl_pct": (pnl / cost * 100.0) if pnl is not None and cost else None,
            "note": holding.get("note", ""),
        })

    valued = pd.DataFrame(rows)
    total = valued["market_value"].sum(skipna=True)
    valued["weight"] = (valued["market_value"] / total * 100.0
                        if total else None)
    return valued.sort_values("market_value", ascending=False,
                              na_position="last").reset_index(drop=True)


def portfolio_summary(valued: pd.DataFrame) -> Dict[str, Any]:
    """Headline totals for the metric row."""
    if valued is None or valued.empty:
        return {"positions": 0, "market_value": None, "day_pnl": None,
                "pnl": None, "pnl_pct": None, "priced": 0}

    market_value = valued["market_value"].sum(skipna=True)
    cost = valued["cost"].sum(skipna=True)
    pnl = valued["pnl"].sum(skipna=True)
    day_pnl = valued["day_pnl"].sum(skipna=True)

    return {
        "positions": len(valued),
        "priced": int(valued["price"].notna().sum()),
        "market_value": float(market_value) if market_value else None,
        "cost": float(cost) if cost else None,
        "day_pnl": float(day_pnl) if pd.notna(day_pnl) else None,
        "pnl": float(pnl) if pd.notna(pnl) else None,
        "pnl_pct": float(pnl / cost * 100.0) if cost else None,
    }


def watchlist_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Watchlist with live prices attached."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    quotes = _quotes(list(frame["ticker"]))
    rows = []
    for _, item in frame.iterrows():
        quote = quotes.get(item["ticker"], {})
        rows.append({
            "ticker": item["ticker"],
            "price": quote.get("price"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
            "volume": quote.get("volume"),
            "note": item.get("note", ""),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 3. EDITIONS
# ==========================================================================
def sgt_now() -> datetime:
    """Wall clock in Singapore, whatever the host machine is set to."""
    return datetime.now(SGT)


def edition_date(now: Optional[datetime] = None) -> date:
    """
    The trading day the current brief belongs to.

    Before the cutoff you are still reading yesterday's edition, which is what
    makes an 07:55 read and an 08:05 read different documents rather than the
    same one built twice.
    """
    moment = now or sgt_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=SGT)
    moment = moment.astimezone(SGT)

    if moment.hour < config.BRIEF_HOUR:
        return (moment - timedelta(days=1)).date()
    return moment.date()


def next_edition_at(now: Optional[datetime] = None) -> datetime:
    """When the following edition is due, for the 'next build' readout."""
    moment = (now or sgt_now()).astimezone(SGT)
    today = moment.replace(hour=config.BRIEF_HOUR, minute=0, second=0,
                           microsecond=0)
    return today if moment < today else today + timedelta(days=1)


def _brief_path(edition: date):
    return config.BRIEF_DIR / f"{edition.isoformat()}.json"


def stored_editions(limit: int = 14) -> List[date]:
    """Editions already on disk, newest first."""
    if not config.BRIEF_DIR.exists():
        return []
    found = []
    for path in config.BRIEF_DIR.glob("*.json"):
        try:
            found.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(found, reverse=True)[:limit]


# ==========================================================================
# 4. THE BRIEF
# ==========================================================================
def _position_digest(symbol: str, weight: Optional[float],
                     change_pct: Optional[float]) -> Dict[str, Any]:
    """Headlines and mood for one holding."""
    try:
        articles = news.get_ticker_news(
            symbol, limit=config.BRIEF_STORIES_PER_POSITION * 3, score=True)
    except Exception as exc:
        log.warning("Brief: news failed for %s: %s", symbol, exc)
        articles = pd.DataFrame()

    digest: Dict[str, Any] = {
        "ticker": symbol,
        "weight": weight,
        "change_pct": change_pct,
        "stories": [],
        "bullish": 0, "bearish": 0, "neutral": 0,
        "net_sentiment": None,
    }
    if articles.empty:
        return digest

    summary = news.sentiment_summary(articles)
    digest.update({
        "bullish": int(summary.get("bullish", 0) or 0),
        "bearish": int(summary.get("bearish", 0) or 0),
        "neutral": int(summary.get("neutral", 0) or 0),
        "net_sentiment": summary.get("net_sentiment"),
    })

    top = articles.head(config.BRIEF_STORIES_PER_POSITION)
    for _, article in top.iterrows():
        published = article.get("published")
        digest["stories"].append({
            "title": str(article.get("title", ""))[:300],
            "link": str(article.get("link", "") or ""),
            "source": str(article.get("source", ""))[:40],
            "published": published.isoformat() if hasattr(published, "isoformat")
            else str(published or ""),
            "sentiment": _float_or_none(article.get("sentiment")),
            "label": str(article.get("label", "NEUTRAL")),
        })
    return digest


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def build_brief(edition: Optional[date] = None) -> Dict[str, Any]:
    """
    Assemble the edition from whatever the portfolio holds right now.

    Stories are ranked by |sentiment| x position weight, so a mildly negative
    story about a third of the book outranks a screaming headline about a 2%
    position. That is the ordering a holder actually wants; ranking on
    sentiment alone just surfaces the loudest writing.
    """
    edition = edition or edition_date()
    positions = value_positions(holdings())

    if positions.empty:
        return {
            "edition": edition.isoformat(),
            "built_at": sgt_now().isoformat(timespec="seconds"),
            "empty": True,
            "positions": [], "top_stories": [],
            "summary": {"symbols": 0, "stories": 0, "net_sentiment": None},
        }

    workload = positions.head(config.BRIEF_MAX_SYMBOLS)
    digests: List[Dict[str, Any]] = []

    # Same pattern as news.get_news: a serial pass over a 20-name book takes
    # long enough that the page looks hung.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_position_digest, row["ticker"],
                        _float_or_none(row.get("weight")),
                        _float_or_none(row.get("change_pct"))): row["ticker"]
            for _, row in workload.iterrows()
        }
        for future in as_completed(futures):
            try:
                digests.append(future.result())
            except Exception as exc:
                log.warning("Brief: digest failed for %s: %s",
                            futures[future], exc)

    digests.sort(key=lambda d: (d.get("weight") or 0), reverse=True)

    top_stories: List[Dict[str, Any]] = []
    for digest in digests:
        weight = (digest.get("weight") or 0) / 100.0
        for story in digest["stories"]:
            sentiment = story.get("sentiment")
            if sentiment is None:
                continue
            top_stories.append({
                **story,
                "ticker": digest["ticker"],
                # Weight floors at a small value so an unpriced position still
                # competes rather than vanishing from the ranking entirely.
                "impact": abs(sentiment) * max(weight, 0.01),
            })
    top_stories.sort(key=lambda s: s["impact"], reverse=True)

    scored = [d["net_sentiment"] for d in digests if d["net_sentiment"] is not None]
    movers = sorted(
        [d for d in digests if d.get("change_pct") is not None],
        key=lambda d: abs(d["change_pct"]), reverse=True)[:3]

    return {
        "edition": edition.isoformat(),
        "built_at": sgt_now().isoformat(timespec="seconds"),
        "empty": False,
        "positions": digests,
        "top_stories": top_stories[:12],
        "summary": {
            "symbols": len(digests),
            "stories": sum(len(d["stories"]) for d in digests),
            "net_sentiment": (sum(scored) / len(scored)) if scored else None,
            "movers": [{"ticker": m["ticker"], "change_pct": m["change_pct"]}
                       for m in movers],
        },
    }


def get_brief(force: bool = False) -> Dict[str, Any]:
    """
    Today's edition, built once and reused.

    Reads from disk when the current edition already exists, so reruns - and
    Streamlit reruns constantly - never refetch a book's worth of headlines.
    """
    edition = edition_date()
    path = _brief_path(edition)

    if path.exists() and not force:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Brief %s unreadable, rebuilding: %s", edition, exc)

    brief = build_brief(edition)
    try:
        config.BRIEF_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not persist brief %s: %s", edition, exc)
    return brief


def load_edition(edition: date) -> Optional[Dict[str, Any]]:
    """An earlier edition, for the archive selector. None if not on disk."""
    path = _brief_path(edition)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Edition %s unreadable: %s", edition, exc)
        return None


__all__ = [
    "SGT", "HOLDING_COLUMNS", "WATCHLIST_COLUMNS",
    "load", "save", "holdings", "watchlist", "add_to_watchlist",
    "value_positions", "portfolio_summary", "watchlist_quotes",
    "sgt_now", "edition_date", "next_edition_at", "stored_editions",
    "build_brief", "get_brief", "load_edition",
]

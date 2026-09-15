"""
Portfolio, watchlist, and the 08:00 SGT holdings brief.

Positions are the one thing in this terminal the user authored, so they live
in a plain JSON file under .openterm/ rather than in the SQLite cache. The
cache is disposable by design - the sidebar's PURGE button empties it - and
holdings are not.

Every store function takes the owner's key from utils.identity. The local
user reads and writes the original .openterm/portfolio.json and briefs/;
each signed-in user gets .openterm/users/<key>/ and can reach nothing else.

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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from data_fetchers import allocation, equities, news
from utils.identity import LOCAL_USER

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
# utils.identity hashes account identifiers, so a real key always has this
# shape. Checking anyway guarantees a key cannot walk a path out of USERS_DIR.
_USER_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _user_dir(user: str) -> Path:
    if not isinstance(user, str) or not _USER_KEY.fullmatch(user):
        raise ValueError("Invalid portfolio owner key")
    return config.USERS_DIR / user


def portfolio_path(user: str = LOCAL_USER) -> Path:
    """Where `user`'s holdings and watchlist are stored."""
    if user == LOCAL_USER:
        return config.PORTFOLIO_FILE
    return _user_dir(user) / "portfolio.json"


def brief_dir(user: str = LOCAL_USER) -> Path:
    """Where `user`'s morning-brief editions are stored."""
    if user == LOCAL_USER:
        return config.BRIEF_DIR
    return _user_dir(user) / "briefs"


def _empty_store() -> Dict[str, List[Dict[str, Any]]]:
    return {"holdings": [], "watchlist": []}


def load(user: str = LOCAL_USER) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read the position file. A missing or corrupt file is not an error.

    A terminal that refuses to open because its state file has a stray comma
    is worse than one that opens empty and lets you retype four rows, so a
    parse failure is logged and the file is left on disk untouched.
    """
    path = portfolio_path(user)
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


def save(holdings: pd.DataFrame, watchlist: pd.DataFrame,
         user: str = LOCAL_USER) -> None:
    """Write both tables back, normalised and with empty rows dropped."""
    payload = {
        "holdings": _records(holdings, HOLDING_COLUMNS),
        "watchlist": _records(watchlist, WATCHLIST_COLUMNS),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = portfolio_path(user)
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so a future regression fails loudly here rather than
    # writing a file that only Python can read.
    path.write_text(
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


def holdings(user: str = LOCAL_USER) -> pd.DataFrame:
    """Stored holdings, exactly as entered."""
    return _clean(pd.DataFrame(load(user)["holdings"]), HOLDING_COLUMNS)


def watchlist(user: str = LOCAL_USER) -> pd.DataFrame:
    """Stored watchlist, exactly as entered."""
    return _clean(pd.DataFrame(load(user)["watchlist"]), WATCHLIST_COLUMNS)


def add_to_watchlist(ticker: str, note: str = "",
                     user: str = LOCAL_USER) -> bool:
    """
    Append one symbol. Returns False if it was already there.

    Backs the `NVDA WATCH` command form, so it has to be idempotent - typing
    the same command twice should not produce two rows.
    """
    symbol = equities.normalize_ticker(str(ticker).strip().upper())
    if not symbol:
        return False

    current = watchlist(user)
    if not current.empty and symbol in set(current["ticker"]):
        return False

    row = pd.DataFrame([{"ticker": symbol, "note": note}])
    save(holdings(user), pd.concat([current, row], ignore_index=True), user)
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


def _listing_currencies(symbols: List[str]) -> Dict[str, Optional[str]]:
    """
    Quote currency per symbol, None where it could not be established.

    Cached for a week per listing, so this costs a request the first time a
    symbol is held and a cache read after that.
    """
    unique = sorted(set(symbols))
    if not unique:
        return {}

    def lookup(symbol: str) -> Optional[str]:
        try:
            return equities.get_quote_currency(symbol)
        except Exception as exc:
            log.warning("Quote currency unknown for %s: %s", symbol, exc)
            return None

    # Same pattern as build_brief: a first visit with a dozen new listings
    # would otherwise hold the holdings page on a dozen serial lookups.
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(unique, pool.map(lookup, unique)))


def _fx_rates(currencies: List[Optional[str]], base: str) -> Dict[str, float]:
    """Rates into `base`, or only the base itself if the fetch failed."""
    wanted = tuple(sorted({code for code in currencies if code and code != base}))
    if not wanted:
        return {base: 1.0}
    try:
        return equities.get_fx_rates(wanted, base)
    except Exception as exc:
        log.warning("FX into %s failed for %s: %s", base, ", ".join(wanted), exc)
        return {base: 1.0}


def _valued_mask(frame: pd.DataFrame) -> pd.Series:
    """
    Rows that made it into the base-currency totals.

    A native price alone is not enough: an SGX listing with a quote but no SGD
    rate shows its price and is still absent from every sum.
    """
    mask = frame["price"].notna()
    if "unpriced_reason" in frame.columns:
        mask &= frame["unpriced_reason"].isna()
    return mask


def fx_applied(valued: pd.DataFrame) -> Dict[str, float]:
    """
    The crosses actually used on the valued book, per major currency.

    Excludes the base currency and anything left unpriced, so a caption built
    from it never cites a rate that did not touch a figure.
    """
    if (valued is None or valued.empty or "currency" not in valued.columns
            or "fx_rate" not in valued.columns):
        return {}
    used: Dict[str, float] = {}
    priced = valued.loc[_valued_mask(valued)]
    for code, rate in zip(priced["currency"], priced["fx_rate"]):
        major, units = equities.currency_unit(code if pd.notna(code) else None)
        if major and major != config.BASE_CURRENCY and pd.notna(rate):
            used[major] = float(rate) * units
    return used


def value_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Mark holdings to market in `config.BASE_CURRENCY`.

    Yahoo prices each listing in its exchange's currency - D05.SI in SGD, VOO
    in USD - and cost basis is entered in that same listing currency, because
    that is what the broker statement shows. `price` and `cost_basis` stay
    native so they still match the statement; `day_pnl`, `market_value`,
    `cost` and `pnl` are converted into the base currency so they can be
    summed. Adding SGD to USD directly gives a total in no currency at all.

    Both legs of the return are converted at today's rate, so `pnl` is the
    local-market return translated at today's rate. The currency move since
    purchase is not in it: the rate on the purchase date was never recorded.
    `pnl_pct` is a ratio and is unaffected by the conversion.

    A position that cannot be converted is left unpriced, and
    `unpriced_reason` says why: no quote, no quote currency, or no rate for
    its currency. Falling back to a rate of 1.0 would be a guess that looks
    exactly like a figure, and it would flow into every total, weight and
    sector.

    Cost basis is optional: a row with a quantity but no basis still gets a
    market value and a day move, it just cannot show a return. Reporting a
    P&L of -100% because the basis field was left blank would be worse than
    reporting nothing.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    base = config.BASE_CURRENCY
    quotes = _quotes(list(frame["ticker"]))
    # Only quoted symbols are worth a currency lookup. A dead symbol has no
    # price to convert, and asking after its currency adds a failing call to
    # every page load.
    currencies = _listing_currencies(
        [symbol for symbol in frame["ticker"]
         if quotes.get(symbol, {}).get("price") is not None])
    rates = _fx_rates([equities.currency_unit(code)[0]
                       for code in currencies.values()], base)
    rows: List[Dict[str, Any]] = []

    for _, holding in frame.iterrows():
        symbol = holding["ticker"]
        quote = quotes.get(symbol, {})
        price = quote.get("price")
        change = quote.get("change")
        quantity = holding.get("quantity")
        basis = holding.get("cost_basis")

        currency = currencies.get(symbol)
        major, units = equities.currency_unit(currency)
        rate = rates.get(major) if major else None

        if price is None:
            reason = "no quote"
        elif currency is None:
            reason = "no quote currency"
        elif rate is None:
            reason = f"no {major} to {base} rate"
        else:
            reason = None

        # Base currency per one quote unit, so a pence listing is brought
        # down to pounds before the GBP cross is applied.
        fx = rate / units if reason is None else None
        sized = fx is not None and pd.notna(quantity)

        market_value = price * quantity * fx if sized else None
        cost = basis * quantity * fx if sized and pd.notna(basis) else None
        pnl = (market_value - cost
               if market_value is not None and cost is not None else None)
        day_pnl = change * quantity * fx if sized and change is not None else None

        rows.append({
            "ticker": symbol,
            "quantity": quantity,
            "cost_basis": basis,
            "price": price,
            "currency": currency,
            "fx_rate": fx,
            "change_pct": quote.get("change_pct"),
            "day_pnl": day_pnl,
            "market_value": market_value,
            "cost": cost,
            "pnl": pnl,
            "pnl_pct": (pnl / cost * 100.0) if pnl is not None and cost else None,
            "unpriced_reason": reason,
            "note": holding.get("note", ""),
        })

    valued = pd.DataFrame(rows)
    total = valued["market_value"].sum(skipna=True)
    valued["weight"] = (valued["market_value"] / total * 100.0
                        if total else None)
    return valued.sort_values("market_value", ascending=False,
                              na_position="last").reset_index(drop=True)


def portfolio_summary(valued: pd.DataFrame) -> Dict[str, Any]:
    """Headline totals for the metric row, in `config.BASE_CURRENCY`."""
    if valued is None or valued.empty:
        return {"positions": 0, "market_value": None, "day_pnl": None,
                "pnl": None, "pnl_pct": None, "priced": 0,
                "currency": config.BASE_CURRENCY}

    market_value = valued["market_value"].sum(skipna=True)
    cost = valued["cost"].sum(skipna=True)
    pnl = valued["pnl"].sum(skipna=True)
    day_pnl = valued["day_pnl"].sum(skipna=True)

    return {
        "positions": len(valued),
        "priced": int(_valued_mask(valued).sum()),
        "currency": config.BASE_CURRENCY,
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


def _brief_path(edition: date, user: str = LOCAL_USER) -> Path:
    return brief_dir(user) / f"{edition.isoformat()}.json"


def stored_editions(limit: int = 14, user: str = LOCAL_USER) -> List[date]:
    """Editions already on disk, newest first."""
    folder = brief_dir(user)
    if not folder.exists():
        return []
    found = []
    for path in folder.glob("*.json"):
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


# ==========================================================================
# 5. THE BOOK SUMMARY
# ==========================================================================
# Net sentiment beyond this reads as a lean rather than noise. Shared with the
# brief's PORTFOLIO MOOD tile so the paragraph and the tile never disagree.
MOOD_THRESHOLD = 0.15


def mood_label(net: Optional[float]) -> str:
    """RISK-ON, RISK-OFF or MIXED for a net sentiment score."""
    if net is not None and net > MOOD_THRESHOLD:
        return "RISK-ON"
    if net is not None and net < -MOOD_THRESHOLD:
        return "RISK-OFF"
    return "MIXED"


def _sector_mix(positions: pd.DataFrame) -> pd.DataFrame:
    """Sector exposure, or nothing - a failed lookup must not sink the brief."""
    try:
        return allocation.sector_exposure(positions)
    except Exception as exc:
        log.warning("Brief: sector exposure failed: %s", exc)
        return pd.DataFrame()


def _extreme(frame: pd.DataFrame, column: str,
             largest: bool) -> Optional[Dict[str, Any]]:
    if column not in frame.columns:
        return None
    rows = frame.dropna(subset=[column])
    if rows.empty:
        return None
    row = rows.loc[rows[column].idxmax() if largest else rows[column].idxmin()]
    return {"ticker": str(row["ticker"]), column: float(row[column])}


def book_snapshot(positions: pd.DataFrame,
                  sectors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    The whole book in plain JSON types, as the paragraph needs it.

    Stored with the edition so an archived brief describes the book as it was
    that morning, not as it is when someone reopens it a week later.
    """
    if positions is None or positions.empty:
        return {}

    totals = portfolio_summary(positions)
    stats = allocation.concentration(positions)
    priced = _valued_mask(positions)
    valued, unpriced = positions.loc[priced], positions.loc[~priced]

    # portfolio_summary sums with skipna, so a book with no day moves or no
    # cost basis at all totals to 0.0. Zero is a claim; absent is honest.
    day_pnl = (totals.get("day_pnl")
               if positions["day_pnl"].notna().any() else None)
    cost = totals.get("cost")
    pnl = totals.get("pnl") if cost else None
    market_value = totals.get("market_value")
    prior = (market_value - day_pnl
             if market_value is not None and day_pnl is not None else None)

    limit = config.POSITION_CONCENTRATION_PCT
    weights = positions.dropna(subset=["weight"])
    reasons = ({str(ticker): str(reason) for ticker, reason
                in zip(unpriced["ticker"], unpriced["unpriced_reason"])
                if pd.notna(reason)}
               if "unpriced_reason" in unpriced.columns else {})

    book: Dict[str, Any] = {
        "as_of": sgt_now().isoformat(timespec="seconds"),
        # Recorded so an archived paragraph keeps the currency it was written
        # in, and so a book summed before conversion existed can be told apart.
        "currency": totals.get("currency") or config.BASE_CURRENCY,
        "fx": fx_applied(positions),
        "positions": len(positions),
        "priced": int(priced.sum()),
        "with_basis": int(positions["cost"].notna().sum()),
        "market_value": market_value,
        "day_pnl": day_pnl,
        "day_pct": (day_pnl / prior * 100.0) if prior else None,
        "cost": cost,
        "pnl": pnl,
        "pnl_pct": totals.get("pnl_pct") if pnl is not None else None,
        "largest_ticker": stats.get("largest_ticker"),
        "largest": _float_or_none(stats.get("largest")),
        "top3_pct": _float_or_none(stats.get("top3_pct")),
        "effective_positions": _float_or_none(stats.get("effective_positions")),
        "over_limit": [str(t) for t in
                       weights.loc[weights["weight"] >= limit, "ticker"]],
        # Over the valued book only, so the paragraph never leads with the
        # move of a position it then says was left out of every figure.
        "day_best": _extreme(valued, "change_pct", largest=True),
        "day_worst": _extreme(valued, "change_pct", largest=False),
        "return_best": _extreme(valued, "pnl_pct", largest=True),
        "return_worst": _extreme(valued, "pnl_pct", largest=False),
        "unpriced": [str(t) for t in unpriced["ticker"]],
        "unpriced_reasons": reasons,
        "sectors": [],
        "unclassified_pct": None,
    }

    if sectors is not None and not sectors.empty:
        unclassified = sectors["sector"] == allocation.UNCLASSIFIED
        book["sectors"] = [
            {"sector": str(row["sector"]), "weight_pct": float(row["weight_pct"])}
            for _, row in sectors[~unclassified].head(3).iterrows()]
        if unclassified.any():
            book["unclassified_pct"] = float(
                sectors.loc[unclassified, "weight_pct"].sum())

    return book


def _money(value: float, currency: Optional[str] = None) -> str:
    prefix = equities.currency_prefix(currency or config.BASE_CURRENCY)
    return f"{prefix}{abs(value):,.0f}"


def _join(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def compose_narrative(book: Optional[Dict[str, Any]],
                      summary: Optional[Dict[str, Any]] = None,
                      digests: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    One paragraph on the whole book, written from the numbers.

    Every clause is conditional on the figure behind it existing. A sentence
    that says "0.0% above cost" for a book with no cost basis entered reads
    exactly like a real result, so a missing input drops the clause instead.

    Money is written in the currency the book was converted into, and the
    rates applied are named: a total for a book holding SGX listings only
    means something once it says what the SGD was converted at.
    """
    if not book or not book.get("positions"):
        return ""
    summary = summary or {}
    digests = digests or []
    sentences: List[str] = []

    count = book["positions"]
    noun = "position" if count == 1 else "positions"
    priced = book.get("priced", 0)
    currency = book.get("currency") or config.BASE_CURRENCY
    reasons = book.get("unpriced_reasons") or {}

    # --- Value and P&L ----------------------------------------------------
    value = book.get("market_value")
    if value is None:
        # A book that could not be converted did return quotes. Saying it did
        # not sends the reader off to re-check symbols that are fine.
        if set(reasons.values()) - {"no quote"}:
            sentences.append(
                f"None of your {count} {noun} could be valued in {currency} "
                f"this edition ({'; '.join(sorted(set(reasons.values())))}).")
        else:
            sentences.append(f"None of your {count} {noun} returned a quote, "
                             "so the book cannot be valued this edition.")
    else:
        text = (f"Your book of {count} {noun} is worth "
                f"{_money(value, currency)}")
        day = book.get("day_pnl")
        if day is not None:
            text += f", {'up' if day >= 0 else 'down'} {_money(day, currency)}"
            if book.get("day_pct") is not None:
                text += f" ({book['day_pct']:+.2f}%)"
            text += " on the last session"
        pnl = book.get("pnl")
        if pnl is not None:
            text += f", and sits {_money(pnl, currency)}"
            if book.get("pnl_pct") is not None:
                text += f" ({book['pnl_pct']:+.1f}%)"
            text += f" {'above' if pnl >= 0 else 'below'} cost"
            with_basis = book.get("with_basis", 0)
            if with_basis < priced:
                text += (f" on the {with_basis} of {priced} priced positions "
                         "with a cost basis entered")
        sentences.append(text + ".")

        fx = book.get("fx") or {}
        if fx:
            crosses = [f"{code} at {rate:.4f}" for code, rate in sorted(fx.items())]
            sentences.append(
                f"Figures are in {currency}, converting {_join(crosses)}.")

    # --- Concentration ----------------------------------------------------
    largest = book.get("largest_ticker")
    if largest and book.get("largest") is not None and priced > 1:
        text = (f"{largest} is the largest holding at {book['largest']:.1f}% "
                "of the book")
        if priced >= 3 and book.get("top3_pct") is not None:
            text += f" and the top three make up {book['top3_pct']:.1f}%"
        effective = book.get("effective_positions")
        if effective is not None:
            text += (f", so the {priced} priced names behave like roughly "
                     f"{effective:.1f} equally weighted positions")
            if effective < 5:
                text += ", which is a concentrated book"
        sentences.append(text + ".")

    # A one-name book is trivially 100% of itself; flagging it is noise.
    over = book.get("over_limit") or []
    if over and priced > 1:
        sentences.append(
            f"{_join(over)} {'is' if len(over) == 1 else 'are'} above the "
            f"{config.POSITION_CONCENTRATION_PCT:.0f}% single-position marker.")

    # --- Sectors ----------------------------------------------------------
    sectors = book.get("sectors") or []
    if sectors:
        parts = [f"{s['sector']} ({s['weight_pct']:.1f}%)" for s in sectors]
        text = f"By sector it leans towards {_join(parts)}"
        unclassified = book.get("unclassified_pct")
        if unclassified is not None and unclassified >= 0.5:
            text += (f", with {unclassified:.1f}% that could not be placed "
                     "in a sector")
        sentences.append(text + ".")

    # --- Movers -----------------------------------------------------------
    best, worst = book.get("day_best"), book.get("day_worst")
    if best and worst and best["ticker"] != worst["ticker"]:
        sentences.append(
            f"On the day {best['ticker']} led at {best['change_pct']:+.2f}% "
            f"while {worst['ticker']} was weakest at "
            f"{worst['change_pct']:+.2f}%.")
    elif best:
        sentences.append(
            f"On the day {best['ticker']} moved {best['change_pct']:+.2f}%.")

    best, worst = book.get("return_best"), book.get("return_worst")
    if best and worst and best["ticker"] != worst["ticker"]:
        sentences.append(
            f"Since purchase {best['ticker']} is the best performer at "
            f"{best['pnl_pct']:+.1f}% and {worst['ticker']} the weakest at "
            f"{worst['pnl_pct']:+.1f}%.")
    elif best:
        sentences.append(
            f"Since purchase {best['ticker']} is at {best['pnl_pct']:+.1f}%.")

    # --- News -------------------------------------------------------------
    stories = summary.get("stories") or 0
    net = summary.get("net_sentiment")
    if stories and net is not None:
        tone = {"RISK-ON": "lean positive", "RISK-OFF": "lean negative",
                "MIXED": "are mixed"}[mood_label(net)]
        text = (f"Headlines across {stories} stories {tone} "
                f"(net sentiment {net:+.2f})")
        scored = [d for d in digests if d.get("net_sentiment") is not None]
        leans = []
        if scored:
            high = max(scored, key=lambda d: d["net_sentiment"])
            low = min(scored, key=lambda d: d["net_sentiment"])
            if high["net_sentiment"] > MOOD_THRESHOLD:
                leans.append(f"most constructive on {high['ticker']}")
            if low["net_sentiment"] < -MOOD_THRESHOLD:
                leans.append(f"most negative on {low['ticker']}")
        if leans:
            text += ", " + " and ".join(leans)
        sentences.append(text + ".")

    # --- Gaps -------------------------------------------------------------
    # Grouped by what was missing, since "no quote" and "no SGD to USD rate"
    # call for different fixes. Books stored before reasons were recorded
    # only ever had the first.
    unpriced = book.get("unpriced") or []
    if unpriced and value is not None:
        groups: Dict[str, List[str]] = {}
        for ticker in unpriced:
            groups.setdefault(reasons.get(ticker, "no quote"), []).append(ticker)
        for reason, tickers in groups.items():
            sentences.append(
                f"{reason[0].upper()}{reason[1:]} came back for "
                f"{_join(tickers)}, so "
                f"{'it is' if len(tickers) == 1 else 'they are'} left out of "
                "every figure above.")

    return " ".join(sentences)


def build_brief(edition: Optional[date] = None,
                user: str = LOCAL_USER) -> Dict[str, Any]:
    """
    Assemble the edition from whatever the portfolio holds right now.

    Stories are ranked by |sentiment| x position weight, so a mildly negative
    story about a third of the book outranks a screaming headline about a 2%
    position. That is the ordering a holder actually wants; ranking on
    sentiment alone just surfaces the loudest writing.
    """
    edition = edition or edition_date()
    positions = value_positions(holdings(user))

    if positions.empty:
        return {
            "edition": edition.isoformat(),
            "built_at": sgt_now().isoformat(timespec="seconds"),
            "empty": True,
            "book": {}, "narrative": "",
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

    summary = {
        "symbols": len(digests),
        "stories": sum(len(d["stories"]) for d in digests),
        "net_sentiment": (sum(scored) / len(scored)) if scored else None,
        "movers": [{"ticker": m["ticker"], "change_pct": m["change_pct"]}
                   for m in movers],
    }
    book = book_snapshot(positions, _sector_mix(positions))

    return {
        "edition": edition.isoformat(),
        "built_at": sgt_now().isoformat(timespec="seconds"),
        "empty": False,
        "book": book,
        "narrative": compose_narrative(book, summary, digests),
        "positions": digests,
        "top_stories": top_stories[:12],
        "summary": summary,
    }


def _persist(path: Path, brief: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not persist brief %s: %s", path.stem, exc)


def get_brief(force: bool = False, user: str = LOCAL_USER) -> Dict[str, Any]:
    """
    Today's edition, built once and reused.

    Reads from disk when the current edition already exists, so reruns - and
    Streamlit reruns constantly - never refetch a book's worth of headlines.

    An edition written before the book summary existed gets one added rather
    than being rebuilt: the headlines are the expensive half, and they are
    already on disk. So does one whose summary has no currency recorded -
    that book was summed before listings were converted, and its paragraph
    printed SGD and USD added together behind a "$".
    """
    edition = edition_date()
    path = _brief_path(edition, user)

    if path.exists() and not force:
        try:
            brief = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Brief %s unreadable, rebuilding: %s", edition, exc)
        else:
            if (not brief.get("empty")
                    and "currency" not in (brief.get("book") or {})):
                positions = value_positions(holdings(user))
                brief["book"] = book_snapshot(positions, _sector_mix(positions))
                brief["narrative"] = compose_narrative(
                    brief["book"], brief.get("summary"), brief.get("positions"))
                _persist(path, brief)
            return brief

    brief = build_brief(edition, user)
    _persist(path, brief)
    return brief


def load_edition(edition: date, user: str = LOCAL_USER) -> Optional[Dict[str, Any]]:
    """An earlier edition, for the archive selector. None if not on disk."""
    path = _brief_path(edition, user)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Edition %s unreadable: %s", edition, exc)
        return None


__all__ = [
    "SGT", "HOLDING_COLUMNS", "WATCHLIST_COLUMNS", "LOCAL_USER",
    "portfolio_path", "brief_dir",
    "load", "save", "holdings", "watchlist", "add_to_watchlist",
    "value_positions", "portfolio_summary", "fx_applied", "watchlist_quotes",
    "sgt_now", "edition_date", "next_edition_at", "stored_editions",
    "build_brief", "get_brief", "load_edition",
    "MOOD_THRESHOLD", "mood_label", "book_snapshot", "compose_narrative",
]

"""
data_fetchers/news.py :: Module E - OSINT News & Intelligence.

Bloomberg equivalents: TOP, N, NI.

Sources (all free):
  * RSS/Atom feeds        - CNBC, MarketWatch, Yahoo Finance, SEC, Fed,
                            Treasury, gCaptain, EIA, and Google News queries
                            for outlets that killed their public feeds.
  * GDELT 2.0 Doc API     - Global event/tone index across ~65 languages.
                            No key, no registration, genuinely open.

Sentiment:
  * VADER (default)       - Lexicon + rule based, extended here with a
                            finance-specific lexicon. Fast, deterministic,
                            zero model download. Good enough for headlines,
                            which is all we score.
  * FinBERT (optional)    - If `transformers` and `torch` are installed,
                            ProsusAI/finbert gives materially better results
                            on financial text. First run downloads ~440MB.

Scoring caveat: headline sentiment measures the *tone of the writing*, not
whether the news is good for a position. "Company beats estimates, stock
falls" scores positive. Treat it as a triage signal for what to read, not a
trading input.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config
from utils.cache import cached, get_session
from utils.rate_limiter import retry_with_backoff, throttled

log = logging.getLogger("openterm.news")

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    feedparser = None  # type: ignore[assignment]
    FEEDPARSER_AVAILABLE = False


# ==========================================================================
# SENTIMENT ENGINES
# ==========================================================================
_vader_analyzer: Optional[Any] = None
_vader_checked = False
_finbert_pipeline: Optional[Any] = None
_finbert_checked = False


def _get_vader():
    """
    Lazily build the VADER analyser with the finance lexicon merged in.

    Downloads the ~1MB lexicon on first use if NLTK doesn't already have it.

    The failure is cached as well as the success. Without that, a corrupt or
    unreachable lexicon made every scored headline re-attempt the download -
    and with NLTK's server returning 429 that turned one news panel into
    dozens of sequential network timeouts, hanging the whole page.
    """
    global _vader_analyzer, _vader_checked
    if _vader_checked:
        return _vader_analyzer
    _vader_checked = True

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        log.warning("nltk not installed - sentiment scoring disabled")
        return None

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        # Lexicon not downloaded yet.
        try:
            import nltk

            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

            analyzer = SIA()
        except Exception as exc:
            log.error("Could not obtain VADER lexicon: %s", exc)
            return None
    except Exception as exc:
        log.error("VADER init failed: %s", exc)
        return None

    # Teach VADER what "beat", "downgrade" and "hawkish" mean.
    analyzer.lexicon.update(config.FINANCE_LEXICON)
    _vader_analyzer = analyzer
    return analyzer


def _get_finbert():
    """Load FinBERT if the optional heavy deps are present. Cached, once."""
    global _finbert_pipeline, _finbert_checked
    if _finbert_checked:
        return _finbert_pipeline

    _finbert_checked = True
    try:
        from transformers import pipeline

        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        log.info("FinBERT loaded - using transformer sentiment")
    except Exception as exc:
        log.info("FinBERT unavailable (%s) - using VADER", type(exc).__name__)
        _finbert_pipeline = None

    return _finbert_pipeline


def score_sentiment(text: str, engine: str = "auto") -> Dict[str, Any]:
    """
    Score one piece of text.

    Args:
        engine: "auto" (FinBERT if available, else VADER) | "vader" | "finbert"

    Returns:
        {score: -1..1, label: BULLISH|BEARISH|NEUTRAL, confidence, engine}
    """
    if not text or not text.strip():
        return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0, "engine": "none"}

    text = text.strip()

    if engine in ("auto", "finbert"):
        finbert = _get_finbert()
        if finbert is not None:
            try:
                result = finbert(text[:512])[0]
                label = result["label"].lower()
                confidence = float(result["score"])
                score = (
                    confidence if label == "positive"
                    else -confidence if label == "negative"
                    else 0.0
                )
                return {
                    "score": round(score, 4),
                    "label": _bucket(score),
                    "confidence": round(confidence, 4),
                    "engine": "finbert",
                }
            except Exception as exc:
                log.debug("FinBERT scoring failed, falling back: %s", exc)

    analyzer = _get_vader()
    if analyzer is None:
        return {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0, "engine": "none"}

    scores = analyzer.polarity_scores(text)
    compound = float(scores["compound"])
    return {
        "score": round(compound, 4),
        "label": _bucket(compound),
        "confidence": round(abs(compound), 4),
        "engine": "vader",
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
    }


def _bucket(score: float) -> str:
    """VADER's conventional +/-0.05 thresholds, relabelled for markets."""
    if score >= 0.05:
        return "BULLISH"
    if score <= -0.05:
        return "BEARISH"
    return "NEUTRAL"


# ==========================================================================
# RSS
# ==========================================================================
@cached(ttl=config.TTL.news, namespace="rss_feed")
@throttled("rss")
@retry_with_backoff(max_retries=2, on_giveup=lambda exc: [])
def fetch_rss_feed(url: str, source_name: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Parse one RSS/Atom feed into normalised article dicts.

    Fetched through `requests` (so it goes through the HTTP cache and carries
    a real User-Agent) rather than letting feedparser fetch it - several
    publishers 403 feedparser's default agent.
    """
    if not FEEDPARSER_AVAILABLE:
        return []

    # Government publishers gate on User-Agent. sec.gov in particular returns
    # 403 to the browser UA we use elsewhere and expects the descriptive
    # contact string from its fair-access policy; bls.gov behaves similarly.
    host = url.split("/")[2].lower() if "//" in url else ""
    if host.endswith(("sec.gov", "bls.gov", "federalreserve.gov", "treasury.gov")):
        session = get_session("rss_gov", expire_after=config.TTL.news,
                              user_agent=config.SEC_USER_AGENT)
    else:
        session = get_session("rss", expire_after=config.TTL.news)

    resp = session.get(url, timeout=config.NET.request_timeout)
    resp.raise_for_status()

    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Malformed feed: {source_name}")

    articles: List[Dict[str, Any]] = []
    for entry in parsed.entries[:limit]:
        title = _clean_text(entry.get("title", ""))
        if not title:
            continue

        summary = _clean_text(
            entry.get("summary") or entry.get("description") or ""
        )[:500]

        articles.append({
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "published": _parse_entry_time(entry),
            "source": source_name,
            "author": entry.get("author", ""),
            "id": _article_id(title, entry.get("link", "")),
        })

    return articles


def get_news(
    categories: Optional[List[str]] = None,
    limit_per_feed: int = 20,
    max_articles: int = 150,
    score: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Aggregate every configured feed into one scored, deduplicated frame.

    Feeds are fetched concurrently - a serial pass over 20 feeds takes 30+
    seconds and makes the terminal feel dead.

    Args:
        categories: Keys of config.RSS_FEEDS. None = all of them.
        score:      Run sentiment analysis. Set False for a fast headline dump.

    Returns:
        DataFrame sorted newest-first with sentiment/label/confidence columns.
    """
    categories = categories or list(config.RSS_FEEDS.keys())

    jobs: List[Tuple[str, str, str]] = []
    for category in categories:
        for source_name, url in config.RSS_FEEDS.get(category, []):
            jobs.append((category, source_name, url))

    if not jobs:
        return pd.DataFrame()

    articles: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_rss_feed, url, source_name, limit_per_feed):
                (category, source_name)
            for category, source_name, url in jobs
        }
        for future in as_completed(futures):
            category, source_name = futures[future]
            try:
                for article in future.result() or []:
                    article["category"] = category
                    articles.append(article)
            except Exception as exc:
                log.warning("Feed %s failed: %s", source_name, exc)

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["id"], keep="first")

    # Also drop near-duplicate headlines syndicated across outlets.
    df["_norm_title"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.drop_duplicates(subset=["_norm_title"], keep="first").drop(columns=["_norm_title"])

    if score:
        df = add_sentiment(df)

    df = df.sort_values("published", ascending=False, na_position="last")
    return df.head(max_articles).reset_index(drop=True)


def add_sentiment(df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
    """
    Attach sentiment columns to a news frame.

    Scores title + summary together when a summary exists: headlines are
    short and VADER does better with a little more context.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    results: List[Dict[str, Any]] = []

    for _, row in out.iterrows():
        text = str(row.get(text_column, ""))
        summary = str(row.get("summary", "") or "")
        if summary:
            text = f"{text}. {summary[:300]}"
        results.append(score_sentiment(text))

    out["sentiment"] = [r["score"] for r in results]
    out["label"] = [r["label"] for r in results]
    out["confidence"] = [r["confidence"] for r in results]
    out["engine"] = [r["engine"] for r in results]
    return out


# ==========================================================================
# TICKER-SPECIFIC NEWS
# ==========================================================================
def get_ticker_news(ticker: str, limit: int = 30, score: bool = True) -> pd.DataFrame:
    """
    News for one symbol: Yahoo's per-ticker feed plus a Google News query.

    Two sources because Yahoo's coverage is inconsistent for smaller names
    and Google News catches the trade press Yahoo doesn't syndicate.
    """
    from data_fetchers.equities import normalize_ticker

    symbol = normalize_ticker(ticker)
    articles: List[Dict[str, Any]] = []

    # Yahoo per-ticker RSS.
    try:
        yahoo_url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={symbol}&region=US&lang=en-US"
        )
        articles.extend(fetch_rss_feed(yahoo_url, f"Yahoo:{symbol}", limit))
    except Exception as exc:
        log.debug("Yahoo ticker feed failed for %s: %s", symbol, exc)

    # Google News query, restricted to the last 7 days.
    try:
        google_url = (
            "https://news.google.com/rss/search?"
            f"q={symbol}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        )
        articles.extend(fetch_rss_feed(google_url, f"Google:{symbol}", limit))
    except Exception as exc:
        log.debug("Google News failed for %s: %s", symbol, exc)

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles).drop_duplicates(subset=["id"])
    df["ticker"] = symbol
    df["category"] = "TICKER"

    if score:
        df = add_sentiment(df)

    return df.sort_values("published", ascending=False).head(limit).reset_index(drop=True)


# ==========================================================================
# GDELT
# ==========================================================================
@cached(ttl=config.TTL.news, namespace="gdelt")
@throttled("gdelt")
@retry_with_backoff(max_retries=2, on_giveup=lambda exc: pd.DataFrame())
def query_gdelt(
    query: str,
    timespan: str = "24h",
    max_records: int = 75,
    mode: str = "artlist",
) -> pd.DataFrame:
    """
    Query GDELT 2.0's document index.

    GDELT monitors global news in 65+ languages and publishes a genuinely
    open API - no key, no quota page, no signup. It's the single best free
    source for geopolitical and supply-chain OSINT.

    Args:
        query:    GDELT query syntax. Supports quoted phrases, OR, and
                  operators like `domain:reuters.com`, `sourcecountry:china`,
                  `theme:ECON_STOCKMARKET`.
        timespan: "15min" "1h" "24h" "7d" "1w" "1m"
        mode:     "artlist" (articles) | "timelinevol" (volume over time)

    Returns:
        DataFrame with title/url/domain/language/country/seendate (+sentiment).
    """
    resp = get_session("gdelt", expire_after=config.TTL.news).get(
        config.GDELT_DOC_API,
        params={
            "query": query,
            "mode": mode,
            "maxrecords": min(max_records, 250),
            "timespan": timespan,
            "format": "json",
            "sort": "datedesc",
        },
        # GDELT scans a very large index and routinely takes 15-25s for a
        # broad query - measured 16s on a simple one. The default 20s timeout
        # would fail most real queries, so this one gets its own budget.
        timeout=max(config.NET.request_timeout, 45),
    )
    resp.raise_for_status()

    # GDELT returns HTML error pages with a 200 status on malformed queries.
    try:
        payload = resp.json()
    except ValueError:
        raise ValueError(f"GDELT rejected the query: {query!r}")

    records = payload.get("articles", [])
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    rename = {
        "seendate": "published", "socialimage": "image",
        "sourcecountry": "country", "domain": "source",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "published" in df.columns:
        # GDELT format: 20250722T143000Z
        df["published"] = pd.to_datetime(
            df["published"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
        )

    if "title" in df.columns:
        df["title"] = df["title"].map(_clean_text)
        df["id"] = df.apply(
            lambda r: _article_id(str(r.get("title", "")), str(r.get("url", ""))),
            axis=1,
        )
        df["link"] = df.get("url", "")
        df["summary"] = ""
        df["category"] = "GDELT"
        df = add_sentiment(df)

    return df.reset_index(drop=True)


def get_geopolitical_osint(timespan: str = "24h") -> Dict[str, pd.DataFrame]:
    """
    Preset GDELT sweeps across the themes that move commodity and trade flows.

    Runs the queries concurrently since each is an independent HTTP call.
    """
    queries = {
        "SHIPPING DISRUPTION": (
            '("red sea" OR "suez canal" OR "panama canal" OR "strait of hormuz") '
            '(shipping OR vessel OR tanker OR attack OR blocked OR delay)'
        ),
        "ENERGY SUPPLY": (
            '(opec OR "oil production" OR "crude output" OR "gas pipeline" '
            'OR lng) (cut OR increase OR disruption OR sanctions)'
        ),
        "TRADE POLICY": (
            '(tariff OR "export control" OR "trade war" OR sanctions OR embargo) '
            '(china OR "united states" OR "european union")'
        ),
        "CENTRAL BANKS": (
            '("federal reserve" OR "european central bank" OR "bank of japan" '
            'OR "bank of england") (rate OR policy OR inflation)'
        ),
        "SUPPLY CHAIN": (
            '("supply chain" OR "port congestion" OR "semiconductor shortage" '
            'OR "freight rates") (disruption OR shortage OR surge OR delay)'
        ),
    }

    out: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(query_gdelt, query, timespan, 40): name
            for name, query in queries.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    out[name] = df
            except Exception as exc:
                log.warning("GDELT query '%s' failed: %s", name, exc)

    return out


# ==========================================================================
# AGGREGATE ANALYTICS
# ==========================================================================
def sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate mood across a news frame - the header gauge on the news page.

    `net_sentiment` is (bullish - bearish) / total, which is more legible than
    a mean compound score because it ignores intensity and just counts sides.
    """
    if df is None or df.empty or "sentiment" not in df.columns:
        return {"total": 0}

    scores = pd.to_numeric(df["sentiment"], errors="coerce").dropna()
    if scores.empty:
        return {"total": len(df)}

    counts = df["label"].value_counts().to_dict() if "label" in df else {}
    bullish = counts.get("BULLISH", 0)
    bearish = counts.get("BEARISH", 0)
    total = len(df)

    net = (bullish - bearish) / total if total else 0.0

    return {
        "total": total,
        "mean_score": round(float(scores.mean()), 4),
        "median_score": round(float(scores.median()), 4),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": counts.get("NEUTRAL", 0),
        "net_sentiment": round(net, 4),
        "mood": (
            "RISK-ON" if net > 0.15
            else "RISK-OFF" if net < -0.15
            else "MIXED"
        ),
        "most_bullish": df.loc[scores.idxmax()].to_dict() if len(scores) else None,
        "most_bearish": df.loc[scores.idxmin()].to_dict() if len(scores) else None,
        "engine": df["engine"].mode().iloc[0] if "engine" in df and len(df) else "none",
    }


# Words that dominate any financial corpus and carry no signal.
_STOPWORDS = frozenset("""
a an and are as at be by for from has have how in is it its of on or that the
to was were will with what when where who why this these those there their
they them he she his her you your we our us i but not no if then than so such
after before over under up down out about into more most other some can could
would should may might new says say said report reports year years week weeks
day days time market markets stock stocks share shares company companies
""".split())


def extract_trending_terms(df: pd.DataFrame, top_n: int = 20,
                           min_length: int = 3) -> pd.DataFrame:
    """
    Frequency-ranked terms across headlines, with the average sentiment of the
    articles each term appears in.

    Crude but effective: it surfaces "tariff" spiking before you'd notice it
    scrolling the feed. Not TF-IDF - a stopword list is enough at this corpus
    size and keeps the result interpretable.
    """
    if df is None or df.empty or "title" not in df.columns:
        return pd.DataFrame()

    rows: List[Tuple[str, float]] = []
    has_sentiment = "sentiment" in df.columns

    for _, row in df.iterrows():
        title = str(row.get("title", "")).lower()
        score = float(row.get("sentiment", 0.0)) if has_sentiment else 0.0
        seen_in_row = set()
        for token in re.findall(r"[a-z][a-z0-9\-']+", title):
            if len(token) < min_length or token in _STOPWORDS:
                continue
            if token in seen_in_row:
                continue
            seen_in_row.add(token)
            rows.append((token, score))

    if not rows:
        return pd.DataFrame()

    terms = pd.DataFrame(rows, columns=["term", "sentiment"])
    agg = terms.groupby("term").agg(
        mentions=("term", "size"),
        avg_sentiment=("sentiment", "mean"),
    ).reset_index()

    agg = agg[agg["mentions"] >= 2]
    agg["avg_sentiment"] = agg["avg_sentiment"].round(3)
    return agg.sort_values("mentions", ascending=False).head(top_n).reset_index(drop=True)


def filter_news(
    df: pd.DataFrame,
    query: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    sources: Optional[List[str]] = None,
    hours_back: Optional[int] = None,
) -> pd.DataFrame:
    """Apply the news page's filter controls to a frame."""
    if df is None or df.empty:
        return df

    out = df

    if query:
        pattern = re.escape(query.strip())
        mask = out["title"].str.contains(pattern, case=False, na=False)
        if "summary" in out.columns:
            mask |= out["summary"].str.contains(pattern, case=False, na=False)
        out = out[mask]

    if sentiment_filter and sentiment_filter != "ALL" and "label" in out.columns:
        out = out[out["label"] == sentiment_filter]

    if sources and "source" in out.columns:
        out = out[out["source"].isin(sources)]

    if hours_back and "published" in out.columns:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        published = pd.to_datetime(out["published"], utc=True, errors="coerce")
        out = out[published >= cutoff]

    return out.reset_index(drop=True)


# ==========================================================================
# HELPERS
# ==========================================================================
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(text: Any) -> str:
    """Strip HTML tags and entities out of feed content."""
    if not text:
        return ""
    out = _TAG_RE.sub(" ", str(text))
    out = html.unescape(out)
    return _WS_RE.sub(" ", out).strip()


def _article_id(title: str, link: str) -> str:
    """Stable hash for deduplication across feeds that syndicate each other."""
    basis = f"{title.lower().strip()}|{link.split('?')[0]}"
    return hashlib.md5(basis.encode("utf-8", "replace")).hexdigest()[:16]


def _parse_entry_time(entry: Any) -> Optional[datetime]:
    """Extract a tz-aware timestamp from a feedparser entry."""
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                import calendar

                return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            except Exception:
                continue

    for field in ("published", "updated"):
        raw = entry.get(field)
        if raw:
            try:
                stamp = pd.to_datetime(raw, utc=True, errors="coerce")
                if pd.notna(stamp):
                    return stamp.to_pydatetime()
            except Exception:
                continue

    return None


def time_ago(timestamp: Any) -> str:
    """'3m ago' / '2h ago' - the age stamp on each news row."""
    if timestamp is None or pd.isna(timestamp):
        return "—"
    try:
        stamp = pd.to_datetime(timestamp, utc=True)
        delta = datetime.now(timezone.utc) - stamp.to_pydatetime()
        seconds = delta.total_seconds()

        if seconds < 0:
            return "just now"
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"
    except Exception:
        return "—"


def sentiment_engine_status() -> Dict[str, Any]:
    """Which scorer is live - shown in the news page footer."""
    return {
        "vader": _get_vader() is not None,
        "finbert": _get_finbert() is not None,
        "active": "finbert" if _get_finbert() is not None else
                  "vader" if _get_vader() is not None else "none",
    }


__all__ = [
    "get_news", "get_ticker_news", "query_gdelt", "get_geopolitical_osint",
    "score_sentiment", "add_sentiment", "sentiment_summary",
    "extract_trending_terms", "filter_news", "fetch_rss_feed",
    "time_ago", "sentiment_engine_status",
]

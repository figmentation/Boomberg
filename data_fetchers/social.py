"""
data_fetchers/social.py :: Module I - Multi-platform sentiment fusion.

Bloomberg equivalents: <EQUITY> TWTR / SENT (social velocity and tone).

WHAT THIS ANSWERS
-----------------
Three crowds look at the same ticker and frequently disagree. This fuses
them into one score on -1.00 (extreme bearish) to +1.00 (extreme bullish):

    Institutional  Yahoo Finance headlines, via the existing news module
    Retail         StockTwits, where users tag their own posts Bull/Bear
    Speculative    Reddit - r/wallstreetbets, r/stocks, r/investing

THE DISAGREEMENT IS THE POINT
-----------------------------
A fused score alone hides the only genuinely interesting reading: when
headlines are negative and retail is buying, or the reverse. That case is
detected explicitly and reported in `signal_divergence`, and it pulls
confidence down rather than being averaged into a comfortable middle.

ENDPOINT REALITIES, ESTABLISHED BY TESTING
------------------------------------------
  * StockTwits answers without a key and returns explicit user Bull/Bear
    tags - genuinely good data. It sits behind Cloudflare and REJECTS
    `config.PLAIN_USER_AGENT` with a challenge page, so it needs a
    descriptive or browser UA. This is the exact opposite of FRED's CSV
    endpoint, which hangs when sent a browser UA. Both are load-bearing.

  * Reddit's `.json` API returns 403 to everything without OAuth. The `.rss`
    endpoints still answer with a browser UA, but rate-limit hard: three
    quick probes during development earned a 429. Hence the slowest token
    bucket in the codebase and a full news TTL on the cache.

WHAT THIS IS NOT
----------------
Sentiment is a measure of what people are saying, not of whether they are
right. Crowds are loudest at turning points and most confident at the top.
Nothing here is advice, and a bullish reading is not a reason to buy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config
from data_fetchers import news
from utils.cache import cached, get_session
from utils.rate_limiter import circuit_breaker, retry_with_backoff, throttled

log = logging.getLogger("openterm.social")

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    log.error("feedparser import failed: %s", exc)
    feedparser = None  # type: ignore[assignment]
    FEEDPARSER_AVAILABLE = False


# ==========================================================================
# STOCKTWITS
# ==========================================================================
@cached(ttl=config.TTL.news, namespace="stocktwits")
@throttled("stocktwits")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_stocktwits(ticker: str) -> pd.DataFrame:
    """
    Recent StockTwits messages for one symbol, with their user sentiment tags.

    The tag is the valuable part. Roughly half of posts carry a self-declared
    Bullish or Bearish label, which is a statement of the author's actual
    position rather than a lexicon's guess at their tone. Untagged posts fall
    back to VADER on the message body.

    Returns:
        DataFrame with body, tag, sentiment, label, created_at, followers.
        Empty on any failure - the caller reports the gap rather than
        substituting a neutral reading, because "nobody is posting" and
        "the API refused us" are different facts.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        return pd.DataFrame()

    # Browser UA, not PLAIN: Cloudflare serves a challenge page to the plain
    # client string. See the module docstring.
    session = get_session("stocktwits", expire_after=config.TTL.news,
                          user_agent=config.BROWSER_USER_AGENT)
    response = session.get(
        config.STOCKTWITS_STREAM_URL.format(symbol=symbol),
        timeout=config.NET.request_timeout,
    )

    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()

    messages = (response.json() or {}).get("messages") or []
    rows: List[Dict[str, Any]] = []

    for message in messages:
        body = news._clean_text(message.get("body"))
        if not body:
            continue

        entities = message.get("entities") or {}
        sentiment_block = entities.get("sentiment") or {}
        tag = sentiment_block.get("basic")           # "Bullish" | "Bearish" | None

        scored = news.score_sentiment(body)
        user = message.get("user") or {}

        rows.append({
            "body": body[:400],
            "tag": tag,
            "sentiment": scored["score"],
            # Direction the author declared, where they declared one.
            # VADER reads "$NVDA sell it bro. Best salesman of the century"
            # as +0.79 bullish - lexicons were not built for retail slang,
            # sarcasm or cash-tag shorthand. A self-applied Bear tag on that
            # post is simply better evidence, so it wins for display and
            # driver selection. The lexicon score is kept alongside rather
            # than overwritten, because the platform score blends the two
            # deliberately and because their disagreement is itself
            # measurable (see `tag_lexicon_conflicts`).
            "driver_sentiment": _tag_direction(tag, scored["score"]),
            "label": scored["label"],
            "engine": scored["engine"],
            "created_at": _parse_timestamp(message.get("created_at")),
            "followers": user.get("followers"),
            "user": user.get("username"),
            "link": f"https://stocktwits.com/message/{message.get('id')}",
            "platform": "StockTwits",
        })

    return pd.DataFrame(rows)


# ==========================================================================
# REDDIT
# ==========================================================================
@cached(ttl=config.TTL.news, namespace="reddit_social")
@circuit_breaker("reddit", failure_threshold=2, recovery_timeout=300.0,
                 on_open=lambda: pd.DataFrame())
@throttled("reddit")
@retry_with_backoff(max_retries=1, on_giveup=lambda exc: pd.DataFrame())
def _fetch_subreddit(subreddit: str, query: str, limit: int) -> pd.DataFrame:
    """
    One subreddit's recent posts mentioning `query`, via RSS.

    RSS rather than JSON because Reddit's `.json` endpoints now return 403
    without OAuth. The feed carries titles and timestamps but no score or
    comment count, so post popularity is not available here - only tone and
    volume.

    ONE RETRY, THEN A BREAKER. The default retry budget is five attempts,
    which across three subreddits means up to fifteen requests fired at an
    endpoint that has already said "too many requests" - the retry storm
    makes the rate limiting worse and takes half a minute to fail. Reddit
    returns 429 because we are asking too often, and asking again faster is
    not a fix. Two failures open the circuit for five minutes and the
    remaining subreddits fail instantly, which is both faster and politer.
    """
    if not FEEDPARSER_AVAILABLE:
        return pd.DataFrame()

    session = get_session("reddit", expire_after=config.TTL.news,
                          user_agent=config.BROWSER_USER_AGENT)
    response = session.get(
        config.REDDIT_SEARCH_URL.format(subreddit=subreddit, query=query),
        timeout=config.NET.request_timeout,
    )
    response.raise_for_status()

    parsed = feedparser.parse(response.text)
    rows: List[Dict[str, Any]] = []

    for entry in parsed.entries[:limit]:
        title = news._clean_text(entry.get("title"))
        if not title:
            continue

        scored = news.score_sentiment(title)
        rows.append({
            "body": title[:400],
            "tag": None,
            "sentiment": scored["score"],
            "label": scored["label"],
            "engine": scored["engine"],
            "created_at": news._parse_entry_time(entry),
            "user": entry.get("author"),
            "subreddit": subreddit,
            "link": entry.get("link", ""),
            "platform": f"r/{subreddit}",
        })

    return pd.DataFrame(rows)


def get_reddit(ticker: str, subreddits: Optional[Tuple[str, ...]] = None,
               limit: int = 25) -> pd.DataFrame:
    """
    Posts mentioning the ticker across the configured subreddits.

    Each subreddit is fetched separately and a failure in one does not lose
    the others - Reddit 429s readily, and a partial read is worth more than
    an exception. `df.attrs["failed"]` names the subreddits that did not
    answer so the caller can say so rather than quietly reporting less data.
    """
    subreddits = subreddits or config.SOCIAL_SUBREDDITS
    query = ticker.strip().upper()

    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    for subreddit in subreddits:
        try:
            frame = _fetch_subreddit(subreddit, query, limit)
        except Exception as exc:
            log.warning("Reddit %s failed for %s: %s", subreddit, query, exc)
            failed.append(subreddit)
            continue
        if frame is None or frame.empty:
            failed.append(subreddit)
            continue
        frames.append(frame)

    combined = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame())
    combined.attrs["failed"] = failed
    return combined


# ==========================================================================
# PLATFORM SCORING
# ==========================================================================
def _net_sentiment(frame: pd.DataFrame) -> Optional[float]:
    """
    Share bullish minus share bearish, on -1..+1.

    Counting sides rather than averaging intensity, the same statistic
    `news.sentiment_summary` already uses. One incandescent post should not
    outvote ten measured ones.
    """
    if frame is None or frame.empty or "label" not in frame.columns:
        return None
    counts = frame["label"].value_counts()
    total = int(counts.sum())
    if not total:
        return None
    return (int(counts.get("BULLISH", 0)) - int(counts.get("BEARISH", 0))) / total


def score_stocktwits(frame: pd.DataFrame) -> Dict[str, Any]:
    """
    Retail score, weighting explicit user tags above lexicon readings.

    A user who tags their own post Bullish has told you their position. That
    is better evidence than VADER's opinion of their wording, so tags carry
    `config.STOCKTWITS_TAG_WEIGHT` of the platform score - but not all of it,
    because only about half of posts are tagged and ignoring the rest throws
    away half the sample.
    """
    out: Dict[str, Any] = {
        "platform": "StockTwits", "score": None, "samples": 0,
        "tagged": 0, "bullish_tags": 0, "bearish_tags": 0, "note": "",
    }
    if frame is None or frame.empty:
        out["note"] = "No StockTwits messages returned for this symbol."
        return out

    out["samples"] = len(frame)
    text_score = _net_sentiment(frame)

    tags = frame["tag"].dropna() if "tag" in frame.columns else pd.Series(dtype=object)
    bullish = int((tags == "Bullish").sum())
    bearish = int((tags == "Bearish").sum())
    tagged = bullish + bearish

    conflicts = 0
    if tagged and "label" in frame.columns:
        tagged_rows = frame[frame["tag"].notna()]
        conflicts = int(
            ((tagged_rows["tag"] == "Bullish") & (tagged_rows["label"] == "BEARISH")).sum()
            + ((tagged_rows["tag"] == "Bearish") & (tagged_rows["label"] == "BULLISH")).sum())

    out.update({"tagged": tagged, "bullish_tags": bullish,
                "bearish_tags": bearish, "tag_lexicon_conflicts": conflicts})

    if tagged:
        tag_score = (bullish - bearish) / tagged
        if text_score is None:
            out["score"] = tag_score
            out["note"] = f"{tagged} explicit user tags; no lexicon reading."
        else:
            weight = config.STOCKTWITS_TAG_WEIGHT
            out["score"] = tag_score * weight + text_score * (1 - weight)
            out["note"] = (
                f"{tagged} of {len(frame)} posts carry an explicit user tag "
                f"({bullish} bull / {bearish} bear), weighted {weight:.0%} "
                "against the lexicon read of all message bodies.")
    else:
        out["score"] = text_score
        out["note"] = ("No posts carried an explicit Bull/Bear tag; score is "
                       "the lexicon reading of message bodies alone.")

    return out


def score_platform(frame: pd.DataFrame, platform: str,
                   empty_note: str) -> Dict[str, Any]:
    """Generic net-sentiment score for a text platform."""
    if frame is None or frame.empty:
        return {"platform": platform, "score": None, "samples": 0,
                "note": empty_note}
    return {
        "platform": platform,
        "score": _net_sentiment(frame),
        "samples": len(frame),
        "note": "",
    }


# ==========================================================================
# FUSION
# ==========================================================================
def band_for(score: Optional[float]) -> str:
    """Map a fused score onto its band label."""
    if score is None:
        return "NO SIGNAL"
    for lower, label in config.SENTIMENT_BANDS:
        if score >= lower:
            return label
    return config.SENTIMENT_BANDS[-1][1]


def _confidence(platforms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Confidence 0.00-1.00, from platform coverage, sample volume and agreement.

    Each component is returned separately, because "0.42" tells you nothing
    while "0.42, because only one platform answered and it had nine posts"
    tells you what to do about it.
    """
    resolved = {name: block for name, block in platforms.items()
                if block.get("score") is not None}

    if not resolved:
        return {"score": 0.0, "coverage": 0.0, "volume": 0.0,
                "agreement": 0.0,
                "components": ["No platform returned usable data."]}

    components: List[str] = []

    coverage = len(resolved) / len(platforms)
    components.append(
        f"Coverage {coverage:.0%} - {len(resolved)} of {len(platforms)} "
        "platforms returned data.")

    total_samples = sum(block.get("samples", 0) for block in resolved.values())
    target = config.SENTIMENT_VOLUME_TARGET * len(platforms)
    volume = min(total_samples / target, 1.0) if target else 0.0
    components.append(
        f"Volume {volume:.0%} - {total_samples} posts against a target of "
        f"{target}.")

    scores = [block["score"] for block in resolved.values()]
    if len(scores) < 2:
        agreement = 0.5
        components.append(
            "Agreement not measurable - a single platform cannot corroborate "
            "itself, so this is held at 0.50.")
    else:
        spread = max(scores) - min(scores)
        # Spread runs 0..2 on a -1..+1 scale; halve it to get 0..1.
        agreement = max(0.0, 1.0 - spread / 2.0)
        components.append(
            f"Agreement {agreement:.0%} - platform scores span {spread:.2f} "
            "on a -1 to +1 scale.")

    # Coverage and volume are about how much was heard; agreement is about
    # whether it was consistent. Weighted so a loud but contradictory read
    # cannot present as confident.
    score = coverage * 0.35 + volume * 0.25 + agreement * 0.40

    return {"score": round(score, 2), "coverage": round(coverage, 2),
            "volume": round(volume, 2), "agreement": round(agreement, 2),
            "components": components}


def _divergence(platforms: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Where the crowds disagree, and where a stream went silent.

    Both are reported here because both mean the fused number is less
    informative than it looks, and for opposite reasons.
    """
    notes: List[str] = []

    for name, block in platforms.items():
        if block.get("score") is None:
            notes.append(
                f"{block['platform']}: no usable data. {block.get('note', '')}".strip())

    resolved = [(name, block) for name, block in platforms.items()
                if block.get("score") is not None]

    for index, (name_a, block_a) in enumerate(resolved):
        for name_b, block_b in resolved[index + 1:]:
            gap = abs(block_a["score"] - block_b["score"])
            opposed = block_a["score"] * block_b["score"] < 0

            # Opposite directions are reported even below the gap threshold,
            # provided both readings are big enough to mean something.
            # Headlines at +0.20 against retail at -0.24 is a 0.44 gap - under
            # the 0.50 bar - but it is precisely the case this module exists
            # to surface, and gating it on magnitude alone hid it. The floor
            # keeps ±0.02 noise from being dressed up as a disagreement.
            meaningful_opposition = opposed and min(
                abs(block_a["score"]), abs(block_b["score"])) >= 0.15

            if gap < config.SENTIMENT_DIVERGENCE_THRESHOLD and not meaningful_opposition:
                continue
            notes.append(
                f"{block_a['platform']} ({block_a['score']:+.2f}) and "
                f"{block_b['platform']} ({block_b['score']:+.2f}) "
                f"{'disagree on direction' if opposed else 'differ in degree'}"
                f" by {gap:.2f}."
                + (" Headlines and the crowd pointing opposite ways is the "
                   "reading worth investigating, not the average of the two."
                   if opposed else ""))

    return notes


def _drivers(ticker: str, institutional: pd.DataFrame,
             stocktwits: pd.DataFrame,
             reddit: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    The most strongly worded items on each side, with their platform.

    Filtered to items that actually name the ticker. Both upstream streams
    carry adjacent content - Yahoo's per-ticker feed includes sector pieces
    about other companies, and StockTwits users routinely cash-tag five
    symbols in one post. Without the filter the top "bullish NVDA driver"
    was a headline about Progress Software, which is worse than no driver
    at all: it looks like evidence.

    When filtering leaves too little to be useful the unfiltered set is
    returned with `filtered: False`, so the caller can caption it honestly
    rather than showing a thin list as though it were the whole picture.
    """
    frames = []
    for frame, platform in ((institutional, "Yahoo Finance"),
                            (stocktwits, "StockTwits"), (reddit, "Reddit")):
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        text_column = "title" if "title" in working.columns else "body"
        if text_column not in working.columns or "sentiment" not in working.columns:
            continue
        score_column = ("driver_sentiment"
                        if "driver_sentiment" in working.columns
                        else "sentiment")
        working = working[[text_column, score_column]].rename(
            columns={text_column: "text", score_column: "sentiment"})
        working["platform"] = (
            frame["platform"] if "platform" in frame.columns else platform)
        frames.append(working)

    if not frames:
        return {"bullish": [], "bearish": [], "filtered": False, "mentions": 0}

    combined = pd.concat(frames, ignore_index=True)
    combined["sentiment"] = pd.to_numeric(combined["sentiment"],
                                          errors="coerce")
    combined = combined.dropna(subset=["sentiment"])
    if combined.empty:
        return {"bullish": [], "bearish": [], "filtered": False, "mentions": 0}

    symbol = ticker.strip().upper()
    mentions = combined[
        combined["text"].str.upper().str.contains(
            rf"(?:^|[^A-Z]){symbol}(?:[^A-Z]|$)", regex=True, na=False)]

    # Below a handful of matches the filtered view is less informative than
    # the unfiltered one, so say which is being shown rather than silently
    # switching.
    filtered = len(mentions) >= 4
    working = mentions if filtered else combined

    def top(ascending: bool) -> List[Dict[str, Any]]:
        subset = working.sort_values("sentiment", ascending=ascending).head(4)
        return [{"text": row["text"][:180], "platform": row["platform"],
                 "sentiment": round(float(row["sentiment"]), 3)}
                for _, row in subset.iterrows()]

    return {"bullish": top(ascending=False), "bearish": top(ascending=True),
            "filtered": filtered, "mentions": int(len(mentions))}


@cached(ttl=config.TTL.news, namespace="social_sentiment")
def composite_sentiment(ticker: str) -> Dict[str, Any]:
    """
    The fused multi-platform sentiment read for one ticker.

    Returns a stable structure:

        ticker, as_of
        score            -1.00..+1.00, or None when nothing resolved
        band             EXTREMELY BULLISH .. EXTREMELY BEARISH, or NO SIGNAL
        platforms        {institutional, retail, reddit} each with score,
                         samples, weight, applied_weight and a note
        confidence       {score 0..1, coverage, volume, agreement, components}
        signal_divergence  where platforms disagree, and where one went dark
        drivers          strongest bullish and bearish items, with platform
        disclaimer

    Weights renormalise over the platforms that actually answered. A silent
    platform must not drag the fused score toward zero - that would read as
    "the crowd is neutral" when the truth is "we could not hear one of them",
    and those are different claims.
    """
    ticker = ticker.strip().upper()

    try:
        institutional_frame = news.get_ticker_news(ticker, limit=30, score=True)
    except Exception as exc:
        log.warning("Yahoo headlines failed for %s: %s", ticker, exc)
        institutional_frame = pd.DataFrame()

    try:
        stocktwits_frame = get_stocktwits(ticker)
    except Exception as exc:
        log.warning("StockTwits failed for %s: %s", ticker, exc)
        stocktwits_frame = pd.DataFrame()

    reddit_frame = get_reddit(ticker)
    reddit_failures = reddit_frame.attrs.get("failed", []) if hasattr(
        reddit_frame, "attrs") else []

    platforms: Dict[str, Dict[str, Any]] = {
        "institutional": score_platform(
            institutional_frame, "Yahoo Finance headlines",
            "No headlines returned for this ticker."),
        "retail": score_stocktwits(stocktwits_frame),
        "reddit": score_platform(
            reddit_frame, "Reddit",
            "No posts found in "
            + ", ".join(f"r/{s}" for s in config.SOCIAL_SUBREDDITS)
            + (f" (failed: {', '.join(reddit_failures)})"
               if reddit_failures else " within the search window.")),
    }

    for name, block in platforms.items():
        block["weight"] = config.SENTIMENT_WEIGHTS[name]

    # Renormalise over what answered.
    resolved = {name: block for name, block in platforms.items()
                if block.get("score") is not None}
    total_weight = sum(block["weight"] for block in resolved.values())

    fused: Optional[float] = None
    if total_weight > 0:
        fused = sum(block["score"] * block["weight"]
                    for block in resolved.values()) / total_weight
        fused = round(max(-1.0, min(1.0, fused)), 4)

    for name, block in platforms.items():
        block["applied_weight"] = (
            round(block["weight"] / total_weight, 3)
            if name in resolved and total_weight else 0.0)

    if reddit_failures:
        platforms["reddit"]["note"] = (
            (platforms["reddit"].get("note", "") + " ")
            + f"Subreddits that did not answer: {', '.join(reddit_failures)}. "
              "Reddit rate-limits aggressively; this is usually transient."
        ).strip()

    return {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc),
        "score": fused,
        "band": band_for(fused),
        "platforms": platforms,
        "confidence": _confidence(platforms),
        "signal_divergence": _divergence(platforms),
        "drivers": _drivers(ticker, institutional_frame, stocktwits_frame,
                            reddit_frame),
        "frames": {
            "institutional": institutional_frame,
            "retail": stocktwits_frame,
            "reddit": reddit_frame,
        },
        "disclaimer": (
            "Sentiment measures what people are saying, not whether they are "
            "right. Crowds are loudest at turning points. Not advice."
        ),
    }


def to_schema(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    The report as a plain JSON-serialisable object.

    `composite_sentiment` carries DataFrames and a datetime for the UI; this
    strips them so the result can be dumped straight to JSON for an API
    response or a downstream model.
    """
    as_of = report.get("as_of")
    return {
        "ticker": report.get("ticker"),
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else None,
        "sentiment_score": report.get("score"),
        "sentiment_band": report.get("band"),
        "confidence_score": report.get("confidence", {}).get("score"),
        "confidence_components": report.get("confidence", {}).get("components", []),
        "platforms": {
            name: {key: value for key, value in block.items()
                   if key != "frame"}
            for name, block in report.get("platforms", {}).items()
        },
        "signal_divergence": report.get("signal_divergence", []),
        "key_drivers": report.get("drivers", {}),
        "disclaimer": report.get("disclaimer"),
    }


def _tag_direction(tag: Optional[str], lexicon_score: float) -> float:
    """
    Signed strength for one post: the user's tag if present, else the lexicon.

    Magnitude comes from the lexicon so ordering is preserved; a tagged post
    whose wording is bland floors at 0.5 so it does not sort below untagged
    noise.
    """
    if tag == "Bullish":
        return max(abs(lexicon_score), 0.5)
    if tag == "Bearish":
        return -max(abs(lexicon_score), 0.5)
    return lexicon_score


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """StockTwits stamps ISO-8601 with a trailing Z."""
    if not value:
        return None
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None


__all__ = [
    "get_stocktwits", "get_reddit", "score_stocktwits", "score_platform",
    "composite_sentiment", "band_for", "to_schema",
]

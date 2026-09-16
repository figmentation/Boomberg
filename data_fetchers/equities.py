"""
data_fetchers/equities.py :: Module A - Equity & Fundamentals Analysis.

Bloomberg equivalents: <EQUITY> GP (price graph), DES (description),
FA (financial analysis), RV (relative value).

Data sources, all free:
  * yfinance          - OHLCV, quotes, options chains, company profile.
  * SEC EDGAR         - Filings index + XBRL "company facts" for real,
                        as-reported financial statements. Official JSON API
                        at data.sec.gov: no key, no scraping, no rate cost
                        beyond a 10 req/s fair-use ceiling and a mandatory
                        descriptive User-Agent.
  * OpenBB SDK        - Used opportunistically if the user installed it.

Everything returns pandas objects and never raises into the UI layer: a
failed fetch yields an empty DataFrame/dict so the page still renders.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from utils.cache import cached, get_session
from utils.rate_limiter import (
    circuit_breaker,
    is_transient_error,
    retry_with_backoff,
    throttled,
)

log = logging.getLogger("openterm.equities")

# yfinance is imported lazily-ish but at module scope so failures surface once.
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    log.error("yfinance import failed: %s", exc)
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False

# Make yfinance raise network failures instead of logging them and handing
# back an empty frame. An empty frame is also what a mistyped symbol gets, so
# in the default mode an outage and a typo were indistinguishable and no
# circuit breaker could trip on one without tripping on the other. Measured
# with the network cut: history("AAPL") came back empty by default and
# raised curl's ConnectionError in this mode, while a bad symbol on a live
# network raises an HTTP 404 - permanent, and ignored by the breaker.
if YFINANCE_AVAILABLE:
    try:
        yf.config.debug.hide_exceptions = False
    except Exception:  # pragma: no cover - yfinance < 1.0 has no config object
        pass

# OpenBB is a heavy optional extra. Detect once, use if present.
try:
    from openbb import obb  # type: ignore

    OPENBB_AVAILABLE = True
except Exception:
    obb = None  # type: ignore[assignment]
    OPENBB_AVAILABLE = False


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Circuit breakers, one per upstream, shared by every fetcher that calls it.
#
# They sit INSIDE @retry_with_backoff, so each failed attempt counts and the
# failure that trips the circuit ends the retry loop. Outside it, a dead
# network had to exhaust three whole retry budgets before anything failed
# fast. Once open, CircuitOpen is raised without a request; the retry layer
# does not retry it, on_giveup returns the usual empty value, and @cached
# serves the last good one.
#
# Only network-shaped failures count. A 404 for a mistyped symbol proves the
# upstream is up and must not blank quotes on every other page, and neither
# must a legitimately empty answer such as a symbol with no listed options.
_yahoo_circuit = circuit_breaker("yfinance", failure_threshold=2,
                                 recovery_timeout=60.0,
                                 trip_on=is_transient_error, count_empty=False)
_sec_circuit = circuit_breaker("sec", failure_threshold=3,
                               recovery_timeout=120.0,
                               trip_on=is_transient_error, count_empty=False)


# ==========================================================================
# PRICE DATA
# ==========================================================================
@cached(ttl=config.TTL.daily_bars, namespace="equity_history")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
@_yahoo_circuit
def get_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Historical OHLCV bars.

    Args:
        ticker:      Yahoo symbol. "AAPL", "BTC-USD", "CL=F", "^GSPC", "EURUSD=X".
        period:      1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        interval:    1m 2m 5m 15m 30m 60m 1d 1wk 1mo
                     (Yahoo caps intraday history: 1m -> 7d, <1d -> 60d)
        auto_adjust: Adjust OHLC for splits/dividends. Left off by default so
                     candles match what the user sees on any other chart.

    Returns:
        DataFrame indexed by tz-aware datetime with columns
        Open/High/Low/Close/Volume, or an empty frame on failure.
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    ticker = normalize_ticker(ticker)
    df = yf.Ticker(ticker).history(
        period=period, interval=interval, auto_adjust=auto_adjust,
        actions=False, timeout=config.NET.request_timeout,
    )

    if df is None or df.empty:
        raise ValueError(f"No price history returned for {ticker}")

    # yfinance occasionally returns a MultiIndex column set for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
    df = df.dropna(subset=["Close"])
    df.index.name = "Date"
    return df


@cached(ttl=config.TTL.quote, namespace="equity_quote")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: {})
@_yahoo_circuit
def get_quote(ticker: str) -> Dict[str, Any]:
    """
    Current (15-minute delayed) snapshot quote.

    Uses `fast_info` where possible - it hits a lightweight endpoint and
    avoids the expensive `.info` blob. Falls back to a 2-day history diff if
    Yahoo's quote service is being unreliable, which it periodically is.
    """
    if not YFINANCE_AVAILABLE:
        return {}

    ticker = normalize_ticker(ticker)
    tk = yf.Ticker(ticker)

    price = prev_close = None
    volume = day_high = day_low = market_cap = None
    market_time = None

    try:
        fi = tk.fast_info
        price = _safe_float(fi.get("lastPrice"))
        prev_close = _safe_float(fi.get("previousClose"))
        volume = _safe_float(fi.get("lastVolume"))
        day_high = _safe_float(fi.get("dayHigh"))
        day_low = _safe_float(fi.get("dayLow"))
        market_cap = _safe_float(fi.get("marketCap"))
    except Exception as exc:
        # The history fallback below uses the same network. When that is
        # down, trying it doubles the cost of every attempt for nothing.
        if is_transient_error(exc):
            raise
        log.debug("fast_info failed for %s: %s", ticker, exc)

    # The exchange's own timestamp for this price, off the chart metadata the
    # fast_info call above already fetched - no extra request. Without it the
    # only timestamp on a quote was the moment we asked for it, which reads
    # as "current" on a Monday for a price that last moved on Friday.
    try:
        epoch = _safe_float((getattr(tk, "history_metadata", None) or {})
                            .get("regularMarketTime"))
        if epoch:
            market_time = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except Exception as exc:
        log.debug("No market timestamp for %s: %s", ticker, exc)

    # Fallback: derive from recent daily bars.
    if price is None or prev_close is None:
        hist = tk.history(period="5d", interval="1d", timeout=10)
        if hist is None or hist.empty:
            raise ValueError(f"No quote data for {ticker}")
        closes = hist["Close"].dropna()
        price = price if price is not None else _safe_float(closes.iloc[-1])
        if prev_close is None and len(closes) >= 2:
            prev_close = _safe_float(closes.iloc[-2])
        if volume is None and "Volume" in hist:
            volume = _safe_float(hist["Volume"].iloc[-1])

    change = None
    change_pct = None
    if price is not None and prev_close:
        change = price - prev_close
        change_pct = (change / prev_close) * 100.0

    fetched_at = datetime.now(timezone.utc).isoformat()
    return {
        "ticker": ticker,
        "price": price,
        "previous_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "day_high": day_high,
        "day_low": day_low,
        "market_cap": market_cap,
        # When the price last traded, per the exchange. None when Yahoo did
        # not say - a fallback to the fetch time would be a made-up trade time.
        "market_time": market_time,
        "fetched_at": fetched_at,
        # Kept for callers that read `timestamp`: the trade time when known,
        # and the fetch time otherwise.
        "timestamp": market_time or fetched_at,
    }


@cached(ttl=config.TTL.quote, namespace="equity_quotes_batch")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: {})
@_yahoo_circuit
def get_quotes_batch(tickers: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """
    Snapshot quotes for many symbols in ONE Yahoo round trip.

    This is what powers the ticker tape - calling get_quote() twelve times
    would burn twelve requests every rerun and get us throttled fast.

    Args:
        tickers: Tuple (must be hashable for the cache key).
    """
    if not YFINANCE_AVAILABLE or not tickers:
        return {}

    symbols = [normalize_ticker(t) for t in tickers]

    # 2 days of daily bars gives us last close + previous close in one call.
    raw = yf.download(
        tickers=" ".join(symbols),
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        timeout=config.NET.request_timeout,
    )

    if raw is None or raw.empty:
        _raise_if_download_offline(symbols[0])
        raise ValueError("Batch quote download returned nothing")

    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        try:
            # Single-symbol downloads come back without the ticker level.
            frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            closes = frame["Close"].dropna()
            if closes.empty:
                continue

            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
            change = last - prev
            out[sym] = {
                "ticker": sym,
                "price": last,
                "previous_close": prev,
                "change": change,
                "change_pct": (change / prev * 100.0) if prev else 0.0,
                "volume": _safe_float(frame["Volume"].dropna().iloc[-1])
                if "Volume" in frame else None,
            }
        except Exception as exc:
            log.debug("Batch quote parse failed for %s: %s", sym, exc)

    if not out:
        raise ValueError("Batch quote produced no usable rows")
    return out


def _raise_if_download_offline(symbol: str) -> None:
    """
    Surface a network failure that yf.download swallowed.

    yf.download never raises: it catches each symbol's error, logs it and
    returns an empty frame, so a dead network looks like a batch of symbols
    that had no bars and the breaker never trips on the one call that runs
    on every page. Only on that empty result, one symbol is re-requested
    through Ticker.history, which raises network errors in the mode set at
    import. A reachable Yahoo returns or 404s here, and the caller's
    "returned nothing" error stands.
    """
    yf.Ticker(symbol).history(period="5d", interval="1d",
                              timeout=config.NET.request_timeout)


# ==========================================================================
# CURRENCY
# ==========================================================================
# Yahoo quotes a few exchanges in the minor unit: London in pence ("GBp" -
# the lower-case p is the only thing separating it from "GBP"), Johannesburg
# in cents, Tel Aviv in agorot. A price of 250 there is 2.50 in the major
# currency, and converting it as 250 overstates the position a hundredfold.
_MINOR_UNITS: Dict[str, Tuple[str, float]] = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ZAC": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}

# Prefixes for money in tiles and prose. Anything not listed is written as
# its ISO code, which is unambiguous if less pretty than a symbol.
_CURRENCY_PREFIX: Dict[str, str] = {
    "USD": "$", "SGD": "S$", "HKD": "HK$", "AUD": "A$", "CAD": "C$",
    "NZD": "NZ$",
}


def currency_unit(code: Optional[str]) -> Tuple[Optional[str], float]:
    """
    The major currency a quote is in, and how many quote units make one.

    'GBp' -> ('GBP', 100.0), 'SGD' -> ('SGD', 1.0). The minor-unit table is
    matched before anything is upper-cased, because upper-casing 'GBp' turns
    pence into pounds. Unknown or empty codes return (None, 1.0).
    """
    text = str(code or "").strip()
    if not text:
        return None, 1.0
    if text in _MINOR_UNITS:
        return _MINOR_UNITS[text]
    return text.upper(), 1.0


def currency_prefix(code: Optional[str]) -> str:
    """'USD' -> '$', 'SGD' -> 'S$', 'EUR' -> 'EUR '."""
    major = str(code or "").strip().upper()
    return _CURRENCY_PREFIX.get(major, f"{major} " if major else "")


@cached(ttl=config.TTL.listing, namespace="equity_currency")
@throttled("yfinance")
@retry_with_backoff()
@_yahoo_circuit
def get_quote_currency(ticker: str) -> str:
    """
    The currency Yahoo quotes a listing in: 'USD' for VOO, 'SGD' for D05.SI,
    'GBp' for a London line.

    This is `currency`, not `financialCurrency`. PDD trades in USD on Nasdaq
    but reports in CNY; the price is what gets multiplied by a quantity, so
    the price's currency is the one that matters.

    `fast_info` answers from lightweight chart metadata; the company-info
    whitelist is the fallback. Raises rather than returning a default when
    neither knows - "probably USD" must not be cached for a week, and
    `@cached` serves the last known answer when there is one.
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance unavailable")

    symbol = normalize_ticker(ticker)
    code = None
    try:
        code = yf.Ticker(symbol).fast_info.currency
    except Exception as exc:
        log.debug("fast_info currency failed for %s: %s", symbol, exc)

    if not code:
        code = (get_company_info(symbol) or {}).get("currency")

    if not code:
        raise ValueError(f"No quote currency for {symbol}")
    return str(code).strip()


@cached(ttl=config.TTL.fx, namespace="fx_rates")
def get_fx_rates(currencies: Tuple[str, ...], base: str = "USD") -> Dict[str, float]:
    """
    Units of `base` per one unit of each currency: {'USD': 1.0, 'SGD': 0.788}.

    Read off Yahoo's `{CCY}{BASE}=X` crosses in one batch round trip; the base
    itself maps to 1.0 without a request. A currency whose cross returns
    nothing is left out of the result, not filled in - a guessed rate puts a
    wrong number into every total, and the caller can say what is missing.

    Args:
        currencies: Major ISO codes. Tuple, so the cache key is hashable.
        base:       The currency to convert into.

    Raises when no requested cross came back at all, so `@cached` serves the
    last good rates instead of caching an empty answer for the whole TTL.
    """
    base = str(base).strip().upper()
    rates: Dict[str, float] = {base: 1.0}
    wanted = sorted({str(code).strip().upper() for code in currencies if code})
    pairs = {code: f"{code}{base}=X" for code in wanted if code != base}
    if not pairs:
        return rates

    quotes = get_quotes_batch(tuple(sorted(pairs.values())))
    for code, pair in pairs.items():
        price = _safe_float((quotes.get(pair) or {}).get("price"))
        if price is not None and price > 0:
            rates[code] = price

    if len(rates) == 1:
        raise ValueError(f"No FX rates returned for {', '.join(pairs.values())}")
    return rates


# ==========================================================================
# TECHNICAL INDICATORS
# ==========================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. `adjust=False` matches TradingView/TA-Lib."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothed moving average, correctly seeded.

    Wilder defines the series in two stages:
        1. The first value is a SIMPLE mean of the first `period` observations.
        2. Every later value is  avg[i] = (avg[i-1] * (period-1) + x[i]) / period.

    Stage 2 is exactly `ewm(alpha=1/period, adjust=False)`. Stage 1 is not:
    pandas seeds an unadjusted EWM from the first observation, so a single
    early value dominates and the series stays biased for dozens of bars. On
    Wilder's own published example that error puts RSI at 50.7 instead of
    70.5 - enough to flip an overbought reading to neutral.

    So we blank the warm-up region, plant the simple mean at the seed index,
    and let ewm carry the recursion from there.

    `values` is expected to be a diff-derived series whose first element is
    NaN, so the first `period` real observations occupy positions 1..period.
    """
    if len(values) <= period:
        return pd.Series(np.nan, index=values.index, dtype=float)

    seeded = values.astype(float).copy()
    seed = values.iloc[1:period + 1].mean()   # the first `period` changes

    seeded.iloc[:period] = np.nan             # suppress the warm-up region
    seeded.iloc[period] = seed                # plant Wilder's seed

    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index.

    Verified against the worked example in Wilder's "New Concepts in
    Technical Trading Systems": the reference series yields 70.46 / 66.25 at
    the first two defined points, which this reproduces to within 0.01.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_average(gain, period)
    avg_loss = _wilder_average(loss, period)

    # avg_loss == 0 means the window had no down closes -> RS is infinite,
    # RSI is 100 by definition. Guard the division rather than emitting inf.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # Restore the RSI=100 case, but only where we actually have data.
    out = out.mask(avg_loss.eq(0.0) & avg_gain.notna(), 100.0)
    return out.where(avg_gain.notna(), np.nan)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence. Returns macd/signal/histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
    })


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + num_std * std,
        "bb_lower": mid - num_std * std,
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range - the volatility input for position sizing.

    Wilder-smoothed, using the same correctly-seeded average as rsi(). True
    range at the first bar is undefined (no previous close), so it is dropped
    rather than treated as high-low, which would understate the seed.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    true_range.iloc[0] = np.nan   # undefined without a prior close

    return _wilder_average(true_range, period)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Meaningful on intraday bars, not daily."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (typical * df["Volume"]).cumsum() / cum_vol


def add_indicators(
    df: pd.DataFrame,
    ema_periods: Iterable[int] = (20, 50, 200),
    rsi_period: int = 14,
    macd_params: Tuple[int, int, int] = (12, 26, 9),
    include_bollinger: bool = True,
) -> pd.DataFrame:
    """
    Attach the full indicator suite to an OHLCV frame.

    Returns a copy - the input is never mutated, so cached frames stay clean.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df if df is not None else pd.DataFrame()

    out = df.copy()
    close = out["Close"]

    for period in ema_periods:
        if len(out) >= period:
            out[f"EMA{period}"] = ema(close, period)

    out["RSI"] = rsi(close, rsi_period)

    macd_df = macd(close, *macd_params)
    out = out.join(macd_df)

    if include_bollinger:
        out = out.join(bollinger(close))

    if {"High", "Low"}.issubset(out.columns):
        out["ATR"] = atr(out)

    # Realised volatility, annualised, from log returns.
    out["LogReturn"] = np.log(close / close.shift(1))
    out["Volatility20"] = out["LogReturn"].rolling(20).std() * math.sqrt(252) * 100

    return out


def summarize_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Condense the indicator frame into the signal chips shown above the chart.

    Deliberately mechanical - these are descriptive readings of the indicators,
    not trade recommendations.
    """
    if df is None or df.empty:
        return {}

    last = df.iloc[-1]
    out: Dict[str, Any] = {}

    rsi_val = _safe_float(last.get("RSI"))
    if rsi_val is not None:
        out["rsi"] = rsi_val
        out["rsi_state"] = (
            "OVERBOUGHT" if rsi_val >= 70
            else "OVERSOLD" if rsi_val <= 30
            else "NEUTRAL"
        )

    macd_val = _safe_float(last.get("macd"))
    signal_val = _safe_float(last.get("signal"))
    if macd_val is not None and signal_val is not None:
        out["macd"] = macd_val
        out["macd_state"] = "BULLISH" if macd_val > signal_val else "BEARISH"

        # Detect a crossover in the last two bars.
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_diff = _safe_float(prev.get("macd", 0)) - _safe_float(prev.get("signal", 0))
            curr_diff = macd_val - signal_val
            if prev_diff is not None and prev_diff * curr_diff < 0:
                out["macd_cross"] = "GOLDEN" if curr_diff > 0 else "DEATH"

    close = _safe_float(last.get("Close"))
    for period in (20, 50, 200):
        col = f"EMA{period}"
        val = _safe_float(last.get(col))
        if val is not None and close is not None:
            out[col.lower()] = val
            out[f"{col.lower()}_pos"] = "ABOVE" if close > val else "BELOW"

    # Classic trend filter: 50 over 200.
    ema50, ema200 = _safe_float(last.get("EMA50")), _safe_float(last.get("EMA200"))
    if ema50 is not None and ema200 is not None:
        out["trend"] = "UPTREND" if ema50 > ema200 else "DOWNTREND"

    vol = _safe_float(last.get("Volatility20"))
    if vol is not None:
        out["volatility_annualized_pct"] = vol

    atr_val = _safe_float(last.get("ATR"))
    if atr_val is not None and close:
        out["atr"] = atr_val
        out["atr_pct"] = atr_val / close * 100.0

    return out


# ==========================================================================
# COMPANY PROFILE
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="equity_info")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: {})
@_yahoo_circuit
def get_company_info(ticker: str) -> Dict[str, Any]:
    """
    Company description, sector, and the valuation multiples Yahoo publishes.

    `.info` is a fat, slow, occasionally-flaky endpoint, hence the 24h TTL.
    """
    if not YFINANCE_AVAILABLE:
        return {}

    ticker = normalize_ticker(ticker)
    info = yf.Ticker(ticker).info or {}

    if not info or info.get("regularMarketPrice") is None and len(info) < 5:
        raise ValueError(f"Empty company info for {ticker}")

    keys = [
        "shortName", "longName", "sector", "industry", "country", "website",
        # Classification keys: the machine-readable form of sector/industry,
        # used to look up the issuer's own peer group.
        "sectorKey", "industryKey",
        # quoteType distinguishes an EQUITY from an ETF or MUTUALFUND.
        # Without it the allocation module cannot tell a fund from a company
        # with a missing sector, and silently files every ETF as
        # unclassified rather than looking through to its holdings.
        "quoteType",
        "longBusinessSummary", "fullTimeEmployees", "currency", "exchange",
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
        "priceToBook", "priceToSalesTrailing12Months", "enterpriseToEbitda",
        "enterpriseToRevenue", "pegRatio", "beta", "dividendYield",
        "payoutRatio", "profitMargins", "operatingMargins", "grossMargins",
        "returnOnEquity", "returnOnAssets", "debtToEquity", "currentRatio",
        "quickRatio", "totalRevenue", "revenueGrowth", "earningsGrowth",
        "freeCashflow", "operatingCashflow", "totalCash", "totalDebt",
        "trailingEps", "forwardEps", "bookValue", "sharesOutstanding",
        "floatShares", "shortRatio", "shortPercentOfFloat",
        "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume",
    ]
    return {k: info.get(k) for k in keys if info.get(k) is not None}


# ==========================================================================
# FUNDAMENTALS - yfinance path
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="equity_financials")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: {})
@_yahoo_circuit
def get_financial_statements(ticker: str, quarterly: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Income statement, balance sheet and cash flow from Yahoo.

    Fast and clean, but Yahoo only keeps ~4 annual / ~5 quarterly periods.
    For deeper history use `get_sec_financials()`, which reads the XBRL facts
    the company itself filed.
    """
    if not YFINANCE_AVAILABLE:
        return {}

    ticker = normalize_ticker(ticker)
    tk = yf.Ticker(ticker)

    if quarterly:
        income, balance, cash = tk.quarterly_income_stmt, tk.quarterly_balance_sheet, tk.quarterly_cashflow
    else:
        income, balance, cash = tk.income_stmt, tk.balance_sheet, tk.cashflow

    result = {
        "income_statement": _clean_statement(income),
        "balance_sheet": _clean_statement(balance),
        "cash_flow": _clean_statement(cash),
    }

    if all(df.empty for df in result.values()):
        raise ValueError(f"No financial statements available for {ticker}")
    return result


def _clean_statement(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalise a Yahoo statement frame: newest column first, no all-NaN rows."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.dropna(how="all")
    try:
        out = out[sorted(out.columns, reverse=True)]
    except Exception:
        pass
    return out


# ==========================================================================
# SEC EDGAR
# ==========================================================================
def _sec_session():
    """
    Session preconfigured for SEC fair-access rules.

    SEC requires a User-Agent identifying the requester with contact info.
    Requests without one get a 403 and repeat offenders get IP-banned.
    """
    return get_session("sec", expire_after=config.TTL.sec_filings,
                       user_agent=config.SEC_USER_AGENT)


@cached(ttl=86400 * 7, namespace="sec_ticker_map")
@throttled("sec")
@retry_with_backoff(on_giveup=lambda exc: {})
@_sec_circuit
def get_sec_ticker_map() -> Dict[str, str]:
    """
    Ticker -> zero-padded 10-digit CIK.

    The SEC publishes this as a single small JSON file; cached for a week
    since new listings are rare.
    """
    resp = _sec_session().get(SEC_TICKER_MAP_URL,
                              timeout=config.NET.request_timeout)
    resp.raise_for_status()
    data = resp.json()

    # Shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    return {
        str(row["ticker"]).upper(): str(row["cik_str"]).zfill(10)
        for row in data.values()
        if row.get("ticker")
    }


def ticker_to_cik(ticker: str) -> Optional[str]:
    """Resolve a ticker to its SEC CIK, or None if it isn't a US registrant."""
    mapping = get_sec_ticker_map()
    return mapping.get(normalize_ticker(ticker).upper())


@cached(ttl=config.TTL.sec_filings, namespace="sec_filings")
@throttled("sec")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
@_sec_circuit
def get_sec_filings(
    ticker: str,
    form_types: Tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    limit: int = 40,
) -> pd.DataFrame:
    """
    Recent EDGAR filings with direct document links.

    Args:
        form_types: Filter, e.g. ("10-K",) or ("4",) for insider transactions.
                    Pass () for everything.
        limit:      Max rows returned.

    Returns:
        DataFrame: form, filing_date, report_date, accession, description, url
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        return pd.DataFrame()

    resp = _sec_session().get(
        SEC_SUBMISSIONS_URL.format(cik=cik), timeout=config.NET.request_timeout
    )
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame({
        "form": recent.get("form", []),
        "filing_date": recent.get("filingDate", []),
        "report_date": recent.get("reportDate", []),
        "accession": recent.get("accessionNumber", []),
        "primary_doc": recent.get("primaryDocument", []),
        "description": recent.get("primaryDocDescription", []),
    })
    if df.empty:
        return df

    if form_types:
        df = df[df["form"].isin(form_types)]

    cik_int = str(int(cik))  # URL path uses the un-padded CIK
    df["url"] = df.apply(
        lambda r: (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{str(r['accession']).replace('-', '')}/{r['primary_doc']}"
        ),
        axis=1,
    )
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.sort_values("filing_date", ascending=False).head(limit)
    df["company"] = data.get("name", ticker)
    return df.reset_index(drop=True)


# XBRL us-gaap tags per statement line.
#
# Order matters: entries are tried in sequence and MERGED per period, with
# earlier tags winning for periods they cover. That ordering is why a filer
# who migrated tags mid-history still gets a continuous series - see
# _extract_xbrl_series. Put the tag carrying the most recent data first where
# they conflict; put broader/legacy tags later as gap-fillers.
#
# Variants below were discovered empirically by scanning the companyfacts of
# AAPL, MSFT, JPM, WMT and NVDA rather than guessed from the taxonomy.
_XBRL_CONCEPTS: Dict[str, Dict[str, List[str]]] = {
    "income_statement": {
        "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                    "RevenueFromContractWithCustomerIncludingAssessedTax",
                    "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
                    "SalesRevenueServicesNet"],
        # MSFT: CostOfRevenue through FY2017, CostOfGoodsAndServicesSold after.
        "Cost of Revenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue",
                            "CostOfGoodsSold", "CostOfServices", "CostOfSales"],
        "Gross Profit": ["GrossProfit"],
        "R&D Expense": ["ResearchAndDevelopmentExpense",
                        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
        "SG&A Expense": ["SellingGeneralAndAdministrativeExpense",
                         "GeneralAndAdministrativeExpense",
                         "SellingAndMarketingExpense"],
        "Operating Income": ["OperatingIncomeLoss"],
        # MSFT moved to InterestExpenseNonoperating from FY2023.
        # NOTE: Apple stops disclosing interest expense entirely after FY2023
        # (folded into "Other income/(expense), net") - no tag carries it.
        # NaN in recent Apple columns is accurate, not a coverage gap.
        "Interest Expense": ["InterestExpenseNonoperating", "InterestExpense",
                             "InterestExpenseDebt",
                             "InterestIncomeExpenseNet"],
        "Pretax Income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
        "Income Tax": ["IncomeTaxExpenseBenefit"],
        "Net Income": ["NetIncomeLoss", "ProfitLoss",
                       "NetIncomeLossAvailableToCommonStockholdersBasic"],
        "EPS Basic": ["EarningsPerShareBasic",
                      "IncomeLossFromContinuingOperationsPerBasicShare"],
        # Coca-Cola tags none of the plain EPS concepts - it reports under
        # the continuing-operations variant. Without the fallback its P/E
        # based valuations silently drop out of the comparison entirely.
        "EPS Diluted": ["EarningsPerShareDiluted",
                        "IncomeLossFromContinuingOperationsPerDilutedShare",
                        "EarningsPerShareBasicAndDiluted"],
        # Diluted share count drives the dilution metric in the fundamental
        # engine. Basic is deliberately NOT a fallback: it excludes exactly
        # the options and RSUs that dilution is measuring, so a basic count
        # under a "diluted" heading understates the very thing being tested.
        "Diluted Shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    },
    "balance_sheet": {
        "Cash & Equivalents": ["CashAndCashEquivalentsAtCarryingValue",
                               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
        "Short-Term Investments": ["ShortTermInvestments",
                                   "MarketableSecuritiesCurrent",
                                   "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
        "Accounts Receivable": ["AccountsReceivableNetCurrent",
                                "ReceivablesNetCurrent",
                                "AccountsAndOtherReceivablesNetCurrent"],
        "Inventory": ["InventoryNet"],
        "Total Current Assets": ["AssetsCurrent"],
        "PP&E Net": ["PropertyPlantAndEquipmentNet"],
        # Apple stopped disclosing goodwill separately after FY2017 (it is
        # immaterial and folded into other assets). NaN there is correct.
        "Goodwill": ["Goodwill"],
        "Total Assets": ["Assets"],
        "Accounts Payable": ["AccountsPayableCurrent",
                             "AccountsPayableAndAccruedLiabilitiesCurrent"],
        "Total Current Liabilities": ["LiabilitiesCurrent"],
        "Long-Term Debt": ["LongTermDebtNoncurrent", "LongTermDebt",
                           "LongTermDebtAndCapitalLeaseObligations"],
        "Total Liabilities": ["Liabilities"],
        "Retained Earnings": ["RetainedEarningsAccumulatedDeficit"],
        "Total Equity": ["StockholdersEquity",
                         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        # Short-term borrowings, for total debt. Filers split this several
        # ways; the current portion of long-term debt is the most consistent
        # tag, with commercial paper as the alternate for issuers that fund
        # there instead.
        "Short-Term Debt": ["LongTermDebtCurrent", "CommercialPaper",
                            "ShortTermBorrowings", "OtherShortTermBorrowings"],
    },
    "cash_flow": {
        "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities",
                                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment",
                  "PaymentsToAcquireProductiveAssets"],
        "Investing Cash Flow": ["NetCashProvidedByUsedInInvestingActivities",
                                "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
        "Financing Cash Flow": ["NetCashProvidedByUsedInFinancingActivities",
                                "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
        # AAPL/NVDA use PaymentsOfDividends recently; the CommonStock variant
        # only covers older years. Merging keeps both eras.
        "Dividends Paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
                           "PaymentsOfDistributionsToAffiliates"],
        "Buybacks": ["PaymentsForRepurchaseOfCommonStock",
                     "PaymentsForRepurchaseOfEquity"],
        # DELIBERATELY combined-only. Microsoft publishes no aggregate D&A
        # tag - it reports `Depreciation` and `AmortizationOfIntangibleAssets`
        # separately. Do NOT add those as fallbacks: a depreciation-only
        # figure rendered under a "Depreciation & Amortization" heading is an
        # understated number that looks authoritative, which is worse than a
        # blank. Leaving it NaN is the honest result.
        "Depreciation & Amortization": ["DepreciationDepletionAndAmortization",
                                        "DepreciationAmortizationAndAccretionNet",
                                        "DepreciationAndAmortization"],
        # A real cost to existing holders even though it never leaves the
        # cash flow statement, which is why the fundamental engine scores it
        # against revenue rather than ignoring it the way FCF does.
        "Stock-Based Compensation": ["ShareBasedCompensation",
                                     "AllocatedShareBasedCompensationExpense"],
        "Change in Receivables": ["IncreaseDecreaseInAccountsReceivable"],
    },
}


@cached(ttl=config.TTL.fundamentals, namespace="sec_facts")
@throttled("sec")
@retry_with_backoff(on_giveup=lambda exc: {})
@_sec_circuit
def get_sec_company_facts(ticker: str) -> Dict[str, Any]:
    """
    Raw XBRL company facts - every numeric value the company ever tagged.

    This is a large document (multi-MB for a mature filer), which is why it
    gets a 24h TTL and a dedicated HTTP cache.
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        return {}

    resp = _sec_session().get(
        SEC_COMPANYFACTS_URL.format(cik=cik), timeout=45
    )
    resp.raise_for_status()
    return resp.json()


def get_sec_financials(
    ticker: str,
    statement: str = "income_statement",
    annual: bool = True,
    periods: int = 8,
) -> pd.DataFrame:
    """
    As-reported financial statements built from SEC XBRL facts.

    This is the "scrape EDGAR instead of paying for an API" path. It reads
    the structured facts the registrant filed, so the numbers tie exactly to
    the 10-K/10-Q - no vendor normalisation in between.

    Args:
        statement: income_statement | balance_sheet | cash_flow
        annual:    True -> FY figures (10-K). False -> quarterly (10-Q).
        periods:   How many periods (columns) to return, newest first.

    Returns:
        DataFrame with line items as the index and fiscal periods as columns.
        Empty if the filer isn't a US registrant or has no matching tags.
    """
    facts = get_sec_company_facts(ticker)
    if not facts:
        return pd.DataFrame()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        # The CIK resolved but carries no us-gaap taxonomy. This is not
        # necessarily an error: after a corporate reorganisation the ticker
        # is reassigned to a newly-registered holding entity that has not
        # filed a 10-K yet, while the financial history stays under the
        # predecessor CIK. XOM currently maps to "ExxonMobil Holdings Corp"
        # (CIK 2115436), which reports only an `ffd` taxonomy.
        available = list(facts.get("facts", {}))
        log.info(
            "%s (CIK %s, %s) has no us-gaap facts; taxonomies present: %s",
            ticker, facts.get("cik"), facts.get("entityName"), available or "none",
        )
        empty = pd.DataFrame()
        empty.attrs["reason"] = (
            f"{facts.get('entityName', ticker)} has filed no XBRL financial "
            f"statements under this CIK. The ticker may have been reassigned "
            f"to a new registrant after a reorganisation, with the history "
            f"remaining under the predecessor entity. Use the Yahoo Finance "
            f"source for this name."
        )
        return empty

    concepts = _XBRL_CONCEPTS.get(statement, {})
    rows: Dict[str, Dict[str, float]] = {}

    for label, tags in concepts.items():
        series = _extract_xbrl_series(us_gaap, tags, annual=annual)
        if series:
            rows[label] = series

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T

    # Sort period columns newest-first and trim.
    try:
        ordered = sorted(df.columns, key=lambda c: str(c), reverse=True)
        df = df[ordered[:periods]]
    except Exception:
        df = df.iloc[:, :periods]

    # Preserve the canonical line-item order rather than dict/DataFrame order.
    df = df.reindex([k for k in concepts if k in df.index])
    return df


def _period_label(end: str, annual: bool) -> Optional[str]:
    """Label a fact by its own period-end date: 'FY2025' or '2025Q3'."""
    try:
        stamp = pd.Timestamp(end)
    except Exception:
        return None
    return f"FY{stamp.year}" if annual else f"{stamp.year}Q{stamp.quarter}"


def _facts_for_tag(
    node: Optional[Dict[str, Any]], annual: bool
) -> Dict[str, Tuple[str, float]]:
    """
    Extract one tag's periods. Returns {period: (filed_date, value)}.

    The `filed` date is carried out so the caller can break ties between
    tags as well as within one.
    """
    if not node:
        return {}

    units = node.get("units", {})
    if not units:
        return {}

    wanted_form = "10-K" if annual else "10-Q"

    # Pick the unit that actually carries facts for the form we want, rather
    # than the first key in dict order.
    #
    # REGRESSION: this used to be `"USD" if "USD" in units else
    # next(iter(units))`. Coca-Cola tags EarningsPerShareDiluted under BOTH
    # "pure" (four stray 10-Q facts) and "USD/shares" (fifty-one 10-K facts).
    # "USD" is not an exact key match, so the fallback took "pure", found no
    # annual facts in it, and returned nothing - KO's diluted EPS came back
    # empty from a filing that reports it perfectly well. Every P/E-based
    # valuation for that issuer then silently dropped out.
    def usable(unit_key: str) -> int:
        return sum(1 for fact in units[unit_key]
                   if str(fact.get("form", "")).startswith(wanted_form))

    preference = {"USD": 2, "USD/shares": 1}
    unit_key = max(units, key=lambda key: (usable(key), preference.get(key, 0)))
    if usable(unit_key) == 0:
        return {}
    out: Dict[str, Tuple[str, float]] = {}

    for fact in units[unit_key]:
        val = fact.get("val")
        end = fact.get("end")
        if val is None or not end:
            continue
        if not str(fact.get("form", "")).startswith(wanted_form):
            continue

        start = fact.get("start")
        if start:
            try:
                days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            except Exception:
                continue
            # ~1 year vs ~1 quarter, with slack for 52/53-week calendars.
            if annual and not (330 <= days <= 400):
                continue
            if not annual and not (60 <= days <= 115):
                continue

        period = _period_label(end, annual)
        if period is None:
            continue

        filed = str(fact.get("filed", ""))
        existing = out.get(period)
        # Within a tag, the most recently filed value is the restated one.
        if existing is None or filed > existing[0]:
            out[period] = (filed, float(val))

    return out


def _extract_xbrl_series(
    us_gaap: Dict[str, Any], tags: List[str], annual: bool
) -> Dict[str, float]:
    """
    Pull one concept's time series out of the companyfacts blob.

    CRITICAL SUBTLETY - do not "simplify" this back to using `fy`/`fp`:

    In companyfacts, a fact's `fy`/`fp` identify the *report the fact appeared
    in*, NOT the period the number describes. Apple's FY2023 revenue appears
    three times - tagged fy=2023 (its own 10-K), fy=2024 and fy=2025 (as prior
    -year comparatives). Keying on `fy` and keeping the latest filing there-
    fore labels FY2023 revenue as "FY2025", shifting the entire statement by
    two years. Verified against Apple: real FY2025 revenue is $416.161B, but
    the `fy`-keyed version reported $383.285B (the FY2023 figure) under that
    heading.

    So periods are derived from each fact's own `end` date, and facts are
    selected by duration:
      * Duration facts (income statement, cash flow) carry `start`+`end`;
        we keep spans matching the requested periodicity so a 10-K's annual
        and quarterly tags for the same concept don't collide.
      * Instant facts (balance sheet) carry only `end` and are dated by it.

    Deduplication still keeps the most recently *filed* value for a given
    period, which is what correctly surfaces restatements.

    SECOND SUBTLETY - tags are MERGED, not first-match-wins:

    Filers migrate between tags mid-history. Microsoft reported cost of
    revenue under `CostOfRevenue` through FY2017 and `CostOfGoodsAndServices
    Sold` from FY2020. Apple used `PaymentsOfDividendsCommonStock` in
    FY2016-17 and `PaymentsOfDividends` from FY2020. Returning on the first
    tag that yields *any* data therefore produced a decade-old series and
    left every recent column NaN - the line item looked simply unavailable.

    So every tag is read and results are merged per period, with earlier
    entries in `tags` taking precedence for periods they actually cover.
    """
    merged: Dict[str, Tuple[str, float]] = {}

    for tag in tags:  # priority order
        for period, (filed, value) in _facts_for_tag(us_gaap.get(tag), annual).items():
            # A higher-priority tag already covered this period - keep it.
            if period not in merged:
                merged[period] = (filed, value)

    return {period: value for period, (_, value) in merged.items()}


# ==========================================================================
# PEER COMPARISON / RELATIVE VALUE
# ==========================================================================
def get_peer_comparison(tickers: List[str]) -> pd.DataFrame:
    """
    Valuation comparables table (Bloomberg RV equivalent).

    Computes P/E, EV/EBITDA, Debt/Equity, P/B, P/S, margins and returns for
    each name. Missing metrics come back as NaN rather than dropping the row -
    a peer with no EBITDA is still worth seeing on the sheet.

    Args:
        tickers: Symbols to compare. Keep it under ~10; each costs a request.
    """
    records: List[Dict[str, Any]] = []

    for ticker in tickers:
        ticker = normalize_ticker(ticker)
        try:
            info = get_company_info(ticker)
            if not info:
                continue

            market_cap = _safe_float(info.get("marketCap"))
            ev = _safe_float(info.get("enterpriseValue"))
            total_debt = _safe_float(info.get("totalDebt"))
            equity = _safe_float(info.get("bookValue"))
            shares = _safe_float(info.get("sharesOutstanding"))

            # yfinance reports debtToEquity as a percentage (e.g. 145.0 = 1.45x).
            d_to_e = _safe_float(info.get("debtToEquity"))
            if d_to_e is not None:
                d_to_e = d_to_e / 100.0
            elif total_debt is not None and equity and shares:
                d_to_e = total_debt / (equity * shares)

            records.append({
                "Ticker": ticker,
                "Name": (info.get("shortName") or ticker)[:28],
                "Sector": info.get("sector") or "-",
                "Mkt Cap": market_cap,
                "EV": ev,
                "P/E (TTM)": _safe_float(info.get("trailingPE")),
                "P/E (Fwd)": _safe_float(info.get("forwardPE")),
                "PEG": _safe_float(info.get("pegRatio")),
                "P/B": _safe_float(info.get("priceToBook")),
                "P/S": _safe_float(info.get("priceToSalesTrailing12Months")),
                "EV/EBITDA": _safe_float(info.get("enterpriseToEbitda")),
                "EV/Rev": _safe_float(info.get("enterpriseToRevenue")),
                "Debt/Equity": d_to_e,
                "Curr Ratio": _safe_float(info.get("currentRatio")),
                "Gross Mgn %": _pct(info.get("grossMargins")),
                "Oper Mgn %": _pct(info.get("operatingMargins")),
                "Net Mgn %": _pct(info.get("profitMargins")),
                "ROE %": _pct(info.get("returnOnEquity")),
                "Rev Growth %": _pct(info.get("revenueGrowth")),
                "Div Yield %": _pct(info.get("dividendYield")),
                "Beta": _safe_float(info.get("beta")),
            })
        except Exception as exc:
            log.warning("Peer fetch failed for %s: %s", ticker, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Append a peer-median row so the user can eyeball rich vs cheap instantly.
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        median = numeric.median(numeric_only=True)
        median_row = {c: np.nan for c in df.columns}
        median_row.update(median.to_dict())
        median_row["Ticker"] = "— MEDIAN —"
        median_row["Name"] = "Peer group median"
        median_row["Sector"] = ""
        df = pd.concat([df, pd.DataFrame([median_row])], ignore_index=True)

    return df


# Above this share of its own industry's market weight, a company IS the
# industry - Apple is 99.9% of Yahoo's "consumer-electronics" - and the other
# constituents are microcaps that make a nonsense comparables table. The
# classification one level up is then the honest comparison. The test is on
# published weights, not on a list of which companies are "too big".
_INDUSTRY_DOMINANCE_CEILING = 0.5


def _constituents(kind: str, classification: Optional[str]):
    """
    Yahoo's constituent table for one industry or sector key, or None.

    `kind` is "Industry" or "Sector" - the yfinance class to instantiate.
    """
    if not classification:
        return None
    try:
        table = getattr(yf, kind)(classification).top_companies
    except Exception as exc:
        log.warning("%s constituents unavailable for %s: %s",
                    kind, classification, exc)
        return None

    if table is None or getattr(table, "empty", True):
        return None
    return table


def _peers_from(table, ticker: str, max_peers: int) -> List[str]:
    """Constituent symbols by descending market weight, focal name dropped."""
    peers: List[str] = []
    weights = (table["market weight"] if "market weight" in table.columns
               else None)

    for symbol in table.index:
        symbol = str(symbol).upper()
        if symbol == ticker or symbol in peers:
            continue
        # A zero-weight constituent is delisted or untraded, not a comparable.
        if weights is not None and not (float(weights.get(symbol) or 0.0) > 0):
            continue
        peers.append(symbol)
        if len(peers) >= max_peers:
            break
    return peers


@cached(ttl=config.TTL.fundamentals, namespace="equity_peers")
def suggest_peers(ticker: str, max_peers: int = 6) -> List[str]:
    """
    Comparables drawn from the issuer's own Yahoo classification.

    Yahoo assigns each listed company an `industryKey` and a `sectorKey` and
    publishes each one's constituents with market weights. Both describe this
    specific company; neither is a guess. The industry is preferred, except
    where the company dominates it (see `_INDUSTRY_DOMINANCE_CEILING`), in
    which case the sector is used instead.

    This used to consult seven hand-written sector lists in config and, for
    anything that matched none of them, fall through to megacap tech - so a
    regional bank or a biotech was quietly compared against AAPL and NVDA.
    The multiples rendered fine and meant nothing.

    Returns:
        [ticker, *peers]. Just [ticker] when neither classification yields a
        usable constituent list: the comparables table then reports that it
        has no peers rather than inventing some.
    """
    ticker = normalize_ticker(ticker).upper()
    if not YFINANCE_AVAILABLE:
        return [ticker]

    info = get_company_info(ticker) or {}
    wanted = max(max_peers - 1, 0)

    industry = _constituents("Industry", info.get("industryKey"))
    if industry is not None:
        own_weight = 0.0
        if "market weight" in industry.columns:
            own_weight = float(industry["market weight"].get(ticker) or 0.0)

        if own_weight < _INDUSTRY_DOMINANCE_CEILING:
            peers = _peers_from(industry, ticker, wanted)
            if peers:
                return [ticker] + peers

    sector = _constituents("Sector", info.get("sectorKey"))
    if sector is not None:
        peers = _peers_from(sector, ticker, wanted)
        if peers:
            return [ticker] + peers

    return [ticker]


# ==========================================================================
# OPTIONS
# ==========================================================================
@cached(ttl=config.TTL.intraday, namespace="equity_options")
@throttled("yfinance")
@retry_with_backoff(on_giveup=lambda exc: {})
@_yahoo_circuit
def get_options_chain(ticker: str, expiry: Optional[str] = None) -> Dict[str, Any]:
    """
    Options chain for one expiry, plus a put/call ratio.

    Args:
        expiry: "YYYY-MM-DD". Defaults to the nearest listed expiry.
    """
    if not YFINANCE_AVAILABLE:
        return {}

    ticker = normalize_ticker(ticker)
    tk = yf.Ticker(ticker)

    expiries = list(tk.options or [])
    if not expiries:
        return {}

    chosen = expiry if expiry in expiries else expiries[0]
    chain = tk.option_chain(chosen)

    calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
    puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()

    call_oi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0.0
    put_oi = float(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0.0

    return {
        "expiries": expiries,
        "selected_expiry": chosen,
        "calls": calls,
        "puts": puts,
        "call_open_interest": call_oi,
        "put_open_interest": put_oi,
        "put_call_ratio": (put_oi / call_oi) if call_oi else None,
    }


# ==========================================================================
# HELPERS
# ==========================================================================
# The only single-letter exchange suffixes Yahoo uses: TSX Venture, Frankfurt,
# Tokyo and London. Every other exchange code is two or more letters, so a
# lone letter after a dot is otherwise a share class. Matches both Yahoo's
# published suffix table and yfinance's own MIC -> suffix map.
_SINGLE_LETTER_EXCHANGES = frozenset({"V", "F", "T", "L"})

# Bloomberg yellow keys that change how the root is read. EQUITY is the
# default reading; the other three name asset classes Yahoo spells
# differently (^GSPC, EURUSD=X, CL=F).
_YELLOW_KEYS = frozenset({"EQUITY", "INDEX", "COMDTY", "CURNCY"})

# Longer than any symbol with share class, exchange code and yellow key.
# Input past it is a paste accident; normalising it would only mint a junk
# cache key and a request Yahoo refuses.
_MAX_TICKER_INPUT = 48

# The first token of a Bloomberg equity ticker: a root, possibly carrying a
# Yahoo suffix already or a slashed share class (BRK/B).
_SYMBOL_ROOT = re.compile(r"[A-Z0-9][A-Z0-9./-]{0,11}")

# Bloomberg exchange code -> Yahoo suffix. Bloomberg prints composite codes
# (US, LN, GR) and venue codes (UN, UW, GY) interchangeably, so both are
# listed. None marks China's composite, where the venue is read off the
# number. An unlisted code is dropped and the root quoted as-is - which is
# what happened to every code before this table, and why VOD LN used to
# quote Vodafone's US ADR in dollars.
_BLOOMBERG_EXCHANGES: Dict[str, Optional[str]] = {
    # United States
    "US": "", "UN": "", "UW": "", "UQ": "", "UA": "", "UP": "", "UR": "",
    "UF": "", "UV": "",
    # Canada
    "CN": ".TO", "CT": ".TO", "CV": ".V",
    # Europe
    "LN": ".L", "GR": ".DE", "GY": ".DE", "GF": ".F", "FP": ".PA",
    "NA": ".AS", "BB": ".BR", "IM": ".MI", "SM": ".MC", "SW": ".SW",
    "SE": ".SW", "VX": ".SW", "SS": ".ST", "DC": ".CO", "NO": ".OL",
    "FH": ".HE", "ID": ".IR", "PL": ".LS", "AV": ".VI",
    # Asia-Pacific
    "JP": ".T", "JT": ".T", "HK": ".HK", "CH": None, "CG": ".SS",
    "CS": ".SZ", "KS": ".KS", "KQ": ".KQ", "TT": ".TW", "SP": ".SI",
    "AU": ".AX", "AT": ".AX", "NZ": ".NZ", "IN": ".NS", "IS": ".NS",
    "IB": ".BO", "MK": ".KL", "IJ": ".JK", "TB": ".BK",
    # Latin America, Africa, Middle East
    "BZ": ".SA", "BS": ".SA", "MM": ".MX", "SJ": ".JO", "IT": ".TA",
}

# Bloomberg index tickers whose Yahoo symbol is not simply "^" + the root.
_BLOOMBERG_INDICES: Dict[str, str] = {
    "SPX": "^GSPC", "INDU": "^DJI", "CCMP": "^IXIC", "NDX": "^NDX",
    "RTY": "^RUT", "VIX": "^VIX", "MOVE": "^MOVE", "DXY": "DX-Y.NYB",
    "SPTSX": "^GSPTSE", "UKX": "^FTSE", "DAX": "^GDAXI", "CAC": "^FCHI",
    "SX5E": "^STOXX50E", "AEX": "^AEX", "IBEX": "^IBEX", "SMI": "^SSMI",
    "FTSEMIB": "FTSEMIB.MI", "NKY": "^N225", "HSI": "^HSI",
    "SHCOMP": "000001.SS", "SZCOMP": "399001.SZ", "KOSPI": "^KS11",
    "TWSE": "^TWII", "AS51": "^AXJO", "STI": "^STI", "SENSEX": "^BSESN",
    "NIFTY": "^NSEI", "IBOV": "^BVSP", "MEXBOL": "^MXX",
    # Treasury yields, which Bloomberg files under INDEX.
    "USGG3M": "^IRX", "USGG5YR": "^FVX", "USGG10YR": "^TNX",
    "USGG30YR": "^TYX",
}

# Generic futures: Bloomberg root -> Yahoo root. A commodity root missing
# here is tried as-is (KC1 -> KC=F). An index future must be listed, because
# an unknown root under INDEX is an index, not a contract.
_BLOOMBERG_COMMODITY_ROOTS: Dict[str, str] = {
    "CO": "BZ",                                 # Brent
    "XB": "RB", "LC": "LE", "LH": "HE",         # RBOB, live cattle, lean hogs
    "W": "ZW", "C": "ZC", "S": "ZS", "SM": "ZM", "BO": "ZL", "O": "ZO",
    "TY": "ZN", "US": "ZB", "FV": "ZF", "TU": "ZT",
}
_BLOOMBERG_INDEX_FUTURES: Dict[str, str] = {
    "ES": "ES", "NQ": "NQ", "DM": "YM", "RTY": "RTY",
}

# Only the first generic is mapped: Yahoo's =F symbol is the front contract,
# and serving it for CL2 would put the wrong month's price under the name.
_GENERIC_FUTURE = re.compile(r"([A-Z]{1,3})([1-9][0-9]?)")

# Crosses quoted base-USD by market convention; other ISO codes are USD-base,
# so "EUR Curncy" is EUR/USD and "JPY Curncy" is USD/JPY, as on Bloomberg.
_QUOTED_AGAINST_USD = frozenset({"EUR", "GBP", "AUD", "NZD"})
_BLOOMBERG_CRYPTO: Dict[str, str] = {
    "XBT": "BTC", "XET": "ETH", "BTC": "BTC", "ETH": "ETH",
}


def is_bloomberg_listing(tokens: Sequence[str]) -> bool:
    """
    True for a Bloomberg equity ticker typed without its yellow key:
    VOD LN, BRK B, BRK B US. Lets the command bar route those to the equity
    page instead of rejecting LN or B as an unknown function.
    """
    if not 2 <= len(tokens) <= 3:
        return False
    root, rest = str(tokens[0]).upper(), [str(t).upper() for t in tokens[1:]]
    if rest[-1] in _BLOOMBERG_EXCHANGES:
        rest.pop()
    return (bool(_SYMBOL_ROOT.fullmatch(root)) and len(rest) <= 1
            and all(len(t) == 1 and t.isalpha() for t in rest))


def _bloomberg_equity(parts: List[str]) -> str:
    """["VOD", "LN"] -> "VOD.L"; ["BRK", "B", "US"] -> "BRK-B"; ["AAPL"] -> "AAPL"."""
    root, rest = parts[0].replace("/", "-"), parts[1:]
    suffix: Optional[str] = ""
    if rest and rest[-1] in _BLOOMBERG_EXCHANGES:
        code = rest.pop()
        suffix = _BLOOMBERG_EXCHANGES[code]
        if suffix is None:
            # Shanghai numbers its issues from 6 (A shares) and 9 (B shares);
            # Shenzhen from 0, 2 and 3.
            suffix = ".SS" if root.startswith(("6", "9")) else ".SZ"
        if code == "HK" and root.isdigit():
            root = root.zfill(4)            # Yahoo pads Hong Kong: 0700.HK
        if "." in root:
            suffix = ""                     # root already has a Yahoo suffix
    if len(rest) == 1 and len(rest[0]) == 1 and rest[0].isalpha():
        root = f"{root}-{rest[0]}"
    return root + (suffix or "")


def _bloomberg_index(symbol: str) -> str:
    """SPX -> ^GSPC, ES1 -> ES=F, GDAXI -> ^GDAXI."""
    if symbol.startswith("^") or "=" in symbol or "." in symbol:
        return symbol
    if symbol in _BLOOMBERG_INDICES:
        return _BLOOMBERG_INDICES[symbol]
    generic = _GENERIC_FUTURE.fullmatch(symbol)
    if generic and generic.group(1) in _BLOOMBERG_INDEX_FUTURES:
        if generic.group(2) != "1":
            return symbol
        return f"{_BLOOMBERG_INDEX_FUTURES[generic.group(1)]}=F"
    return f"^{symbol}"


def _bloomberg_commodity(symbol: str) -> str:
    """CL1 -> CL=F, CO1 -> BZ=F, W1 -> ZW=F. Later generics pass through."""
    if "=" in symbol:
        return symbol
    generic = _GENERIC_FUTURE.fullmatch(symbol)
    if not generic or generic.group(2) != "1":
        return symbol
    root = generic.group(1)
    return f"{_BLOOMBERG_COMMODITY_ROOTS.get(root, root)}=F"


def _bloomberg_currency(pair: str) -> str:
    """EURUSD -> EURUSD=X, JPY -> USDJPY=X, XBTUSD -> BTC-USD."""
    pair = pair.replace("/", "")
    if "=" in pair or "-" in pair:
        return pair
    crypto = _BLOOMBERG_CRYPTO.get(pair[:3])
    if crypto:
        return f"{crypto}-{pair[3:] or 'USD'}"
    if len(pair) == 6 and pair.isalpha():
        return f"{pair}=X"
    if len(pair) == 3 and pair.isalpha():
        return f"{pair}USD=X" if pair in _QUOTED_AGAINST_USD else f"USD{pair}=X"
    return pair


def normalize_ticker(ticker: str) -> str:
    """
    Coerce user input into a Yahoo-compatible symbol.

    Handles whitespace, lowercase, the class-share dot/dash mismatch
    (BRK.B -> BRK-B), and Bloomberg tickers, whose exchange code and yellow
    key carry what Yahoo encodes in the symbol itself:

        VOD LN Equity  -> VOD.L        BRK/B US Equity -> BRK-B
        700 HK         -> 0700.HK      SPX Index       -> ^GSPC
        EURUSD Curncy  -> EURUSD=X     CL1 Comdty      -> CL=F

    A US share class whose letter is also an exchange code reads as the
    exchange when typed with a dot: MKC.V is kept as a TSX Venture symbol, so
    McCormick's voting stock has to be typed the way Yahoo lists it, MKC-V.
    The asymmetry decides it - a share class has a dash spelling that passes
    through untouched, and an exchange listing has no other spelling at all.

    Input longer than any real ticker returns "" rather than a truncated
    symbol that would quote something else.
    """
    if not ticker:
        return ""

    t = str(ticker).strip().upper()
    if len(t) > _MAX_TICKER_INPUT:
        return ""

    parts = t.split()
    if not parts:
        return ""

    yellow = parts.pop() if len(parts) > 1 and parts[-1] in _YELLOW_KEYS else None
    if yellow == "INDEX":
        return _bloomberg_index("".join(parts))
    if yellow == "CURNCY":
        return _bloomberg_currency("".join(parts))
    if yellow == "COMDTY":
        return _bloomberg_commodity("".join(parts))

    t = _bloomberg_equity(parts)

    # Yahoo uses '-' for share classes; users type '.'. But leave real
    # suffixes alone (.TO, .L, .HK, ...) - those are exchange codes. Length
    # alone only protects the two-letter ones: without the named set, VOD.L
    # became VOD-L, which Yahoo does not list, and every London, Frankfurt,
    # Tokyo and TSX Venture holding silently returned no quote.
    if "." in t:
        root, _, suffix = t.rpartition(".")
        if (len(suffix) == 1 and suffix.isalpha()
                and suffix not in _SINGLE_LETTER_EXCHANGES):
            t = f"{root}-{suffix}"

    return t


def _safe_float(value: Any) -> Optional[float]:
    """float() that returns None for None/NaN/inf/non-numeric instead of raising."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _pct(value: Any) -> Optional[float]:
    """Convert a 0-1 ratio to a percentage, tolerating None/NaN."""
    val = _safe_float(value)
    return None if val is None else val * 100.0


def format_large_number(value: Optional[float], currency: str = "$") -> str:
    """1.234e9 -> '$1.23B'. Used across every metric tile in the UI."""
    val = _safe_float(value)
    if val is None:
        return "—"

    sign = "-" if val < 0 else ""
    val = abs(val)

    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if val >= threshold:
            return f"{sign}{currency}{val / threshold:.2f}{suffix}"
    return f"{sign}{currency}{val:,.2f}"


__all__ = [
    "get_history", "get_quote", "get_quotes_batch",
    "get_quote_currency", "get_fx_rates", "currency_unit", "currency_prefix",
    "add_indicators", "summarize_technicals",
    "ema", "sma", "rsi", "macd", "bollinger", "atr", "vwap",
    "get_company_info", "get_financial_statements",
    "get_sec_filings", "get_sec_financials", "get_sec_company_facts",
    "ticker_to_cik", "get_peer_comparison", "suggest_peers",
    "get_options_chain", "normalize_ticker", "is_bloomberg_listing",
    "format_large_number",
    "YFINANCE_AVAILABLE", "OPENBB_AVAILABLE",
]

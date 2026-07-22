"""
Technical indicator correctness.

REGRESSION CONTEXT
------------------
RSI and ATR were both initially implemented as a bare
`ewm(alpha=1/period, adjust=False)`. That applies Wilder's recursion
correctly but seeds the average from the *first single observation* rather
than the simple mean of the first `period` observations. On Wilder's own
published example the error produced RSI = 50.66 where the correct value is
70.46 - enough to turn an overbought reading into a neutral one.

The bug is invisible in the UI: the chart still draws a plausible-looking
oscillator. Only a reference vector catches it, which is why these tests
exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_fetchers import equities


# ==========================================================================
# Ground truth
# ==========================================================================
def rsi_reference(prices, period: int = 14):
    """
    Textbook Wilder RSI, written as literally as possible.

    Deliberately a slow explicit loop - it is the oracle the vectorised
    implementation is checked against, so clarity beats speed.
    """
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    out = [np.nan] * len(prices)

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    return pd.Series(out)


def atr_reference(df: pd.DataFrame, period: int = 14):
    """Textbook Wilder ATR via explicit loop."""
    n = len(df)
    true_ranges = []
    for i in range(1, n):
        true_ranges.append(max(
            df["High"].iloc[i] - df["Low"].iloc[i],
            abs(df["High"].iloc[i] - df["Close"].iloc[i - 1]),
            abs(df["Low"].iloc[i] - df["Close"].iloc[i - 1]),
        ))

    out = [np.nan] * n
    avg = sum(true_ranges[:period]) / period
    out[period] = avg
    for i in range(period, len(true_ranges)):
        avg = (avg * (period - 1) + true_ranges[i]) / period
        out[i + 1] = avg
    return pd.Series(out, index=df.index)


# ==========================================================================
# RSI
# ==========================================================================
class TestRSI:
    def test_matches_wilder_published_values(self, wilder_prices):
        """The specific numbers that exposed the seeding bug."""
        result = equities.rsi(wilder_prices, 14)
        assert result.iloc[14] == pytest.approx(70.46, abs=0.01)
        assert result.iloc[15] == pytest.approx(66.25, abs=0.01)

    def test_matches_reference_implementation(self, wilder_prices):
        """Full-series agreement with the textbook loop."""
        mine = equities.rsi(wilder_prices, 14)
        reference = rsi_reference(list(wilder_prices), 14)

        for i in range(14, len(wilder_prices)):
            assert mine.iloc[i] == pytest.approx(reference[i], abs=1e-9), (
                f"divergence at index {i}"
            )

    def test_seed_is_simple_mean_not_first_observation(self):
        """
        Direct guard on the regression, stated as a contrast.

        A series whose first change is a large loss followed by steady gains
        maximally separates the two seeding strategies. Rather than assert a
        magic threshold, this reconstructs the *buggy* calculation and
        requires that we do not match it: correct Wilder gives 56.52 here,
        the first-observation seed gives 13.95 - a 42-point error.
        """
        prices = pd.Series([100.0, 90.0] + [90.0 + i for i in range(1, 30)])

        mine = equities.rsi(prices, 14)
        reference = rsi_reference(list(prices), 14)

        # 1. We match the textbook oracle.
        assert mine.iloc[14] == pytest.approx(reference[14], abs=1e-9)

        # 2. We do NOT match the naive seeding that caused the bug.
        delta = prices.diff()
        naive_gain = delta.clip(lower=0).ewm(
            alpha=1 / 14, adjust=False, min_periods=14).mean()
        naive_loss = (-delta).clip(lower=0).ewm(
            alpha=1 / 14, adjust=False, min_periods=14).mean()
        naive = (100 - 100 / (1 + naive_gain / naive_loss.replace(0, np.nan))).iloc[14]

        assert abs(mine.iloc[14] - naive) > 10, (
            f"RSI matches the first-observation seed ({naive:.2f}) - "
            f"the Wilder seeding fix has regressed"
        )

    def test_warmup_is_nan(self, wilder_prices):
        """No value may be emitted before `period` changes are available."""
        result = equities.rsi(wilder_prices, 14)
        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].notna().all()

    def test_bounded_zero_to_hundred(self, ohlcv):
        result = equities.rsi(ohlcv["Close"], 14).dropna()
        assert len(result) > 0
        assert result.between(0, 100).all()

    def test_monotonic_rise_gives_100(self):
        """No down closes -> avg_loss is 0 -> RSI is 100 by definition."""
        result = equities.rsi(pd.Series(np.arange(1, 40, dtype=float)), 14)
        assert result.dropna().eq(100.0).all()

    def test_monotonic_fall_gives_0(self):
        result = equities.rsi(pd.Series(np.arange(40, 1, -1, dtype=float)), 14)
        assert result.dropna().eq(0.0).all()

    def test_flat_series_does_not_raise(self):
        """Zero gain and zero loss - the 0/0 case must not produce inf."""
        result = equities.rsi(pd.Series([50.0] * 30), 14)
        assert not np.isinf(result.dropna()).any()

    def test_too_short_returns_all_nan(self):
        assert equities.rsi(pd.Series([1.0, 2.0, 3.0]), 14).isna().all()


# ==========================================================================
# ATR
# ==========================================================================
class TestATR:
    def test_matches_reference_implementation(self, ohlcv):
        """ATR shared RSI's seeding bug; same oracle, same guard."""
        mine = equities.atr(ohlcv, 14)
        reference = atr_reference(ohlcv, 14)

        for i in range(14, len(ohlcv)):
            assert mine.iloc[i] == pytest.approx(reference.iloc[i], abs=1e-9), (
                f"divergence at index {i}"
            )

    def test_is_positive(self, ohlcv):
        assert (equities.atr(ohlcv, 14).dropna() > 0).all()

    def test_warmup_is_nan(self, ohlcv):
        assert equities.atr(ohlcv, 14).iloc[:14].isna().all()


# ==========================================================================
# EMA / MACD / Bollinger
# ==========================================================================
class TestEMA:
    def test_warmup_respects_min_periods(self):
        result = equities.ema(pd.Series(np.arange(1, 101, dtype=float)), 10)
        assert result.iloc[:9].isna().all()
        assert result.notna().iloc[9:].all()

    def test_constant_series_equals_constant(self):
        result = equities.ema(pd.Series([7.0] * 40), 10).dropna()
        assert result.eq(7.0).all()

    def test_recursion_is_unadjusted(self):
        """
        adjust=False means ema[i] = a*x[i] + (1-a)*ema[i-1]. Verify one step
        explicitly so a silent switch to adjust=True is caught.
        """
        series = pd.Series(np.arange(1, 60, dtype=float))
        span = 10
        alpha = 2 / (span + 1)
        result = equities.ema(series, span)

        expected = alpha * series.iloc[30] + (1 - alpha) * result.iloc[29]
        assert result.iloc[30] == pytest.approx(expected, abs=1e-12)


class TestMACD:
    def test_columns(self, ohlcv):
        result = equities.macd(ohlcv["Close"])
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_identity(self, ohlcv):
        """histogram must always equal macd - signal."""
        result = equities.macd(ohlcv["Close"]).dropna()
        assert np.allclose(result["macd"] - result["signal"], result["histogram"])

    def test_macd_equals_ema_difference(self, ohlcv):
        close = ohlcv["Close"]
        result = equities.macd(close, 12, 26, 9)
        expected = equities.ema(close, 12) - equities.ema(close, 26)
        assert np.allclose(result["macd"].dropna(), expected.dropna())


class TestBollinger:
    def test_bands_ordered(self, ohlcv):
        bands = equities.bollinger(ohlcv["Close"], 20, 2.0).dropna()
        assert (bands["bb_upper"] >= bands["bb_mid"]).all()
        assert (bands["bb_mid"] >= bands["bb_lower"]).all()

    def test_width_scales_with_std(self, ohlcv):
        narrow = equities.bollinger(ohlcv["Close"], 20, 1.0).dropna()
        wide = equities.bollinger(ohlcv["Close"], 20, 3.0).dropna()
        assert ((wide["bb_upper"] - wide["bb_lower"])
                > (narrow["bb_upper"] - narrow["bb_lower"])).all()


# ==========================================================================
# Composition
# ==========================================================================
class TestAddIndicators:
    def test_does_not_mutate_input(self, ohlcv):
        """Cached frames are shared; mutating one would poison the cache."""
        before = list(ohlcv.columns)
        equities.add_indicators(ohlcv)
        assert list(ohlcv.columns) == before

    def test_attaches_expected_columns(self, ohlcv):
        result = equities.add_indicators(ohlcv)
        for column in ("EMA20", "EMA50", "EMA200", "RSI", "macd", "signal",
                       "histogram", "ATR", "Volatility20"):
            assert column in result.columns, f"missing {column}"

    def test_skips_emas_longer_than_history(self):
        """A 30-bar frame cannot have an EMA200; it must be omitted, not NaN-filled."""
        short = pd.DataFrame({
            "Open": np.arange(30, dtype=float), "High": np.arange(30, dtype=float) + 1,
            "Low": np.arange(30, dtype=float) - 1, "Close": np.arange(30, dtype=float),
            "Volume": np.full(30, 1000),
        })
        result = equities.add_indicators(short)
        assert "EMA20" in result.columns
        assert "EMA200" not in result.columns

    def test_empty_frame_is_safe(self):
        assert equities.add_indicators(pd.DataFrame()).empty

    def test_two_row_frame_does_not_raise(self):
        tiny = pd.DataFrame({
            "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [1.0, 1.0],
            "Close": [2.0, 3.0], "Volume": [10, 20],
        })
        assert len(equities.add_indicators(tiny)) == 2


class TestSummarizeTechnicals:
    def test_empty_returns_empty_dict(self):
        assert equities.summarize_technicals(pd.DataFrame()) == {}

    def test_reports_rsi_state(self, ohlcv):
        summary = equities.summarize_technicals(equities.add_indicators(ohlcv))
        if "rsi" in summary:
            assert summary["rsi_state"] in ("OVERBOUGHT", "OVERSOLD", "NEUTRAL")

    def test_overbought_classification(self):
        """A relentless uptrend must classify as OVERBOUGHT, not NEUTRAL."""
        rising = pd.DataFrame({
            "Open": np.arange(300, dtype=float), "High": np.arange(300, dtype=float) + 1,
            "Low": np.arange(300, dtype=float) - 1, "Close": np.arange(300, dtype=float),
            "Volume": np.full(300, 1000),
        })
        summary = equities.summarize_technicals(equities.add_indicators(rising))
        assert summary["rsi_state"] == "OVERBOUGHT"
        assert summary["trend"] == "UPTREND"

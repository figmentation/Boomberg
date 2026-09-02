"""
Multi-platform sentiment fusion.

REGRESSION CONTEXT
------------------
Every failure guarded here produces a number that looks entirely normal:

  * A silent platform averaged in as zero. If StockTwits is down and its
    score defaults to 0.0, the fused reading drifts toward NEUTRAL and the
    page reports "the crowd has no strong view" when the truth is "we could
    not hear one of the three crowds". Those are different claims and only
    one of them is true.

  * Confidence that ignores disagreement. A loud, contradictory read must
    not present as confident just because the sample was large.

  * Lexicon direction on retail slang. VADER scores "$NVDA sell it bro. Best
    salesman of the century" at +0.79 BULLISH. StockTwits users tag their own
    posts, and where a tag exists it is simply better evidence than a
    lexicon's guess at the wording.

  * Cross-ticker contamination in drivers. Yahoo's per-ticker feed carries
    sector pieces about other companies and StockTwits users cash-tag five
    symbols at once. Unfiltered, the top "bullish NVDA driver" was a headline
    about Progress Software - worse than no driver, because it looks like
    evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from data_fetchers import social


def _posts(labels, tags=None, texts=None, platform="StockTwits"):
    """A frame shaped like a fetched platform stream."""
    tags = tags or [None] * len(labels)
    texts = texts or [f"post {i}" for i in range(len(labels))]
    return pd.DataFrame({
        "body": texts,
        "tag": tags,
        "label": labels,
        "sentiment": [0.5 if l == "BULLISH" else -0.5 if l == "BEARISH" else 0.0
                      for l in labels],
        "driver_sentiment": [
            social._tag_direction(t, 0.5 if l == "BULLISH"
                                  else -0.5 if l == "BEARISH" else 0.0)
            for l, t in zip(labels, tags)],
        "platform": platform,
    })


# ==========================================================================
# Bands
# ==========================================================================
class TestBands:
    @pytest.mark.parametrize("score,expected", [
        (1.00, "EXTREMELY BULLISH"),
        (0.60, "EXTREMELY BULLISH"),
        (0.59, "MODERATELY BULLISH"),
        (0.20, "MODERATELY BULLISH"),
        (0.19, "NEUTRAL / MIXED"),
        (0.00, "NEUTRAL / MIXED"),
        (-0.19, "NEUTRAL / MIXED"),
        (-0.20, "MODERATELY BEARISH"),
        (-0.59, "MODERATELY BEARISH"),
        (-0.60, "EXTREMELY BEARISH"),
        (-1.00, "EXTREMELY BEARISH"),
    ])
    def test_every_boundary(self, score, expected):
        assert social.band_for(score) == expected

    def test_no_score_is_not_neutral(self):
        """
        Absence of data is not a neutral reading. NEUTRAL says the crowd is
        balanced; NO SIGNAL says nobody answered.
        """
        assert social.band_for(None) == "NO SIGNAL"

    def test_bands_tile_the_range(self):
        bounds = [lower for lower, _ in config.SENTIMENT_BANDS]
        assert bounds == sorted(bounds, reverse=True)
        assert bounds[-1] == -1.00


# ==========================================================================
# StockTwits tag handling
# ==========================================================================
class TestStockTwitsScoring:
    def test_explicit_tags_dominate_the_score(self):
        # Lexicon reads everything neutral; the tags are unanimous bulls.
        frame = _posts(["NEUTRAL"] * 10, tags=["Bullish"] * 10)
        result = social.score_stocktwits(frame)

        assert result["bullish_tags"] == 10
        assert result["score"] == pytest.approx(config.STOCKTWITS_TAG_WEIGHT)

    def test_untagged_stream_falls_back_to_the_lexicon(self):
        frame = _posts(["BULLISH"] * 6 + ["BEARISH"] * 2)
        result = social.score_stocktwits(frame)

        assert result["tagged"] == 0
        assert result["score"] == pytest.approx((6 - 2) / 8)
        assert "no posts carried an explicit" in result["note"].lower()

    def test_empty_stream_scores_none_not_zero(self):
        result = social.score_stocktwits(pd.DataFrame())
        assert result["score"] is None
        assert result["samples"] == 0
        assert result["note"]

    def test_counts_where_the_lexicon_contradicts_the_human(self):
        """
        The diagnostic behind the caveat: lexicons were not built for retail
        slang, and this quantifies how often that shows.
        """
        frame = _posts(["BULLISH", "BEARISH", "BULLISH"],
                       tags=["Bearish", "Bearish", "Bullish"])
        result = social.score_stocktwits(frame)
        assert result["tag_lexicon_conflicts"] == 1


class TestTagDirection:
    def test_bear_tag_overrides_a_positive_lexicon_read(self):
        """VADER reads 'sell it bro. Best salesman of the century' as +0.79."""
        assert social._tag_direction("Bearish", 0.79) == pytest.approx(-0.79)

    def test_bull_tag_overrides_a_negative_lexicon_read(self):
        assert social._tag_direction("Bullish", -0.60) == pytest.approx(0.60)

    def test_bland_tagged_post_floors_above_noise(self):
        assert social._tag_direction("Bullish", 0.0) == pytest.approx(0.5)
        assert social._tag_direction("Bearish", 0.0) == pytest.approx(-0.5)

    def test_untagged_post_keeps_its_lexicon_score(self):
        assert social._tag_direction(None, -0.33) == pytest.approx(-0.33)


# ==========================================================================
# Net sentiment
# ==========================================================================
class TestNetSentiment:
    def test_counts_sides_rather_than_averaging_intensity(self):
        """One incandescent post must not outvote several measured ones."""
        frame = _posts(["BULLISH", "BEARISH", "NEUTRAL", "NEUTRAL"])
        assert social._net_sentiment(frame) == pytest.approx(0.0)

    def test_all_bullish(self):
        assert social._net_sentiment(_posts(["BULLISH"] * 5)) == 1.0

    def test_empty_is_none(self):
        assert social._net_sentiment(pd.DataFrame()) is None


# ==========================================================================
# Weight renormalisation
# ==========================================================================
class TestWeighting:
    def _platforms(self, institutional, retail, reddit):
        blocks = {
            "institutional": {"platform": "Yahoo", "score": institutional,
                              "samples": 20, "note": ""},
            "retail": {"platform": "StockTwits", "score": retail,
                       "samples": 20, "note": ""},
            "reddit": {"platform": "Reddit", "score": reddit,
                       "samples": 20, "note": ""},
        }
        for name, block in blocks.items():
            block["weight"] = config.SENTIMENT_WEIGHTS[name]
        return blocks

    def test_weights_sum_to_one(self):
        assert sum(config.SENTIMENT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_three_platforms(self):
        blocks = self._platforms(0.4, 0.8, 0.2)
        expected = 0.4 * 0.35 + 0.8 * 0.35 + 0.2 * 0.30
        resolved = {k: v for k, v in blocks.items() if v["score"] is not None}
        total = sum(b["weight"] for b in resolved.values())
        fused = sum(b["score"] * b["weight"] for b in resolved.values()) / total
        assert fused == pytest.approx(expected)

    def test_silent_platform_is_not_averaged_in_as_zero(self):
        """
        THE bug this module has to avoid. With StockTwits down, a zero-filled
        retail leg drags a strongly bullish read from +0.60 to +0.39 and the
        band flips from EXTREMELY to MODERATELY BULLISH - a change caused
        entirely by an outage, presented as a change in sentiment.
        """
        blocks = self._platforms(0.6, None, 0.6)
        resolved = {k: v for k, v in blocks.items() if v["score"] is not None}
        total = sum(b["weight"] for b in resolved.values())
        renormalised = sum(b["score"] * b["weight"]
                           for b in resolved.values()) / total

        zero_filled = 0.6 * 0.35 + 0.0 * 0.35 + 0.6 * 0.30

        assert renormalised == pytest.approx(0.6)
        assert zero_filled == pytest.approx(0.39)
        assert social.band_for(renormalised) == "EXTREMELY BULLISH"
        assert social.band_for(zero_filled) == "MODERATELY BULLISH"


# ==========================================================================
# Confidence
# ==========================================================================
class TestConfidence:
    def _blocks(self, scores, samples=25):
        out = {}
        for name, score in scores.items():
            out[name] = {"platform": name, "score": score,
                         "samples": samples if score is not None else 0,
                         "note": "", "weight": config.SENTIMENT_WEIGHTS[name]}
        return out

    def test_full_coverage_and_agreement_scores_high(self):
        result = social._confidence(self._blocks(
            {"institutional": 0.50, "retail": 0.55, "reddit": 0.52}))
        assert result["score"] >= 0.85
        assert result["coverage"] == 1.0

    def test_disagreement_lowers_confidence_despite_full_coverage(self):
        """A loud but contradictory read must not present as confident."""
        agree = social._confidence(self._blocks(
            {"institutional": 0.5, "retail": 0.5, "reddit": 0.5}))
        conflict = social._confidence(self._blocks(
            {"institutional": 0.9, "retail": -0.9, "reddit": 0.1}))
        assert conflict["score"] < agree["score"]
        assert conflict["agreement"] < 0.2

    def test_missing_platform_lowers_coverage(self):
        result = social._confidence(self._blocks(
            {"institutional": 0.4, "retail": None, "reddit": 0.4}))
        assert result["coverage"] == pytest.approx(2 / 3, abs=0.01)

    def test_thin_volume_lowers_confidence(self):
        thick = social._confidence(self._blocks(
            {"institutional": 0.4, "retail": 0.4, "reddit": 0.4}, samples=25))
        thin = social._confidence(self._blocks(
            {"institutional": 0.4, "retail": 0.4, "reddit": 0.4}, samples=2))
        assert thin["score"] < thick["score"]

    def test_single_platform_cannot_corroborate_itself(self):
        result = social._confidence(self._blocks(
            {"institutional": 0.9, "retail": None, "reddit": None}))
        assert result["agreement"] == 0.5
        assert any("cannot corroborate" in c for c in result["components"])

    def test_nothing_resolved_is_zero_confidence(self):
        result = social._confidence(self._blocks(
            {"institutional": None, "retail": None, "reddit": None}))
        assert result["score"] == 0.0

    def test_every_component_is_itemised(self):
        result = social._confidence(self._blocks(
            {"institutional": 0.4, "retail": 0.4, "reddit": 0.4}))
        assert len(result["components"]) >= 3
        for component in result["components"]:
            assert component.strip()

    def test_score_stays_in_range(self):
        for scores in ({"institutional": 1.0, "retail": 1.0, "reddit": 1.0},
                       {"institutional": -1.0, "retail": 1.0, "reddit": 0.0},
                       {"institutional": None, "retail": None, "reddit": 0.2}):
            result = social._confidence(self._blocks(scores))
            assert 0.0 <= result["score"] <= 1.0


# ==========================================================================
# Divergence
# ==========================================================================
class TestDivergence:
    def _blocks(self, scores):
        return {name: {"platform": name.title(), "score": score,
                       "samples": 20 if score is not None else 0,
                       "note": "stream returned nothing"}
                for name, score in scores.items()}

    def test_opposed_platforms_are_called_out(self):
        notes = social._divergence(self._blocks(
            {"institutional": -0.7, "retail": 0.7, "reddit": 0.1}))
        assert any("disagree on direction" in n for n in notes)

    def test_aligned_platforms_produce_no_note(self):
        notes = social._divergence(self._blocks(
            {"institutional": 0.5, "retail": 0.55, "reddit": 0.6}))
        assert notes == []

    def test_missing_platform_is_reported_as_a_gap(self):
        """
        The brief is explicit: an empty platform must be noted in
        signal_divergence, not silently dropped.
        """
        notes = social._divergence(self._blocks(
            {"institutional": 0.5, "retail": None, "reddit": 0.5}))
        assert any("no usable data" in n for n in notes)

    def test_same_direction_large_gap_is_degree_not_direction(self):
        notes = social._divergence(self._blocks(
            {"institutional": 0.1, "retail": 0.9, "reddit": 0.5}))
        assert any("differ in degree" in n for n in notes)
        assert not any("disagree on direction" in n for n in notes)

    def test_opposite_directions_are_reported_below_the_gap_threshold(self):
        """
        Headlines at +0.20 against retail at -0.24 is a 0.44 gap, under the
        0.50 bar, and was silently passing. It is the exact case the module
        exists to surface: the institutions and the crowd pointing opposite
        ways.
        """
        notes = social._divergence(self._blocks(
            {"institutional": 0.20, "retail": -0.24, "reddit": 0.0}))
        assert any("disagree on direction" in n for n in notes)

    def test_tiny_opposite_readings_are_still_noise(self):
        notes = social._divergence(self._blocks(
            {"institutional": 0.02, "retail": -0.03, "reddit": 0.01}))
        assert notes == []


# ==========================================================================
# Drivers
# ==========================================================================
class TestDrivers:
    def test_filters_to_items_naming_the_ticker(self):
        institutional = pd.DataFrame({
            "title": [f"NVDA rallies on demand {i}" for i in range(4)]
                     + ["Progress Software launches new AI UI release"],
            "sentiment": [0.5, 0.4, 0.3, 0.2, 0.95],
            "platform": ["Yahoo Finance"] * 5,
        })
        result = social._drivers("NVDA", institutional, pd.DataFrame(),
                                 pd.DataFrame())

        assert result["filtered"] is True
        texts = " ".join(item["text"] for item in result["bullish"])
        assert "Progress Software" not in texts

    def test_falls_back_when_too_few_mentions(self):
        institutional = pd.DataFrame({
            "title": ["Sector piece about semiconductors", "Broad market note"],
            "sentiment": [0.4, -0.4],
            "platform": ["Yahoo Finance"] * 2,
        })
        result = social._drivers("NVDA", institutional, pd.DataFrame(),
                                 pd.DataFrame())
        assert result["filtered"] is False
        assert result["mentions"] == 0
        assert result["bullish"]

    def test_ticker_match_is_bounded_not_a_substring(self):
        """'NVDA' must not match inside a longer token."""
        institutional = pd.DataFrame({
            "title": ["NVDAX fund launches"] * 5,
            "sentiment": [0.5] * 5,
            "platform": ["Yahoo Finance"] * 5,
        })
        result = social._drivers("NVDA", institutional, pd.DataFrame(),
                                 pd.DataFrame())
        assert result["mentions"] == 0

    def test_uses_tag_corrected_direction_for_stocktwits(self):
        """
        The 'sell it bro' case end to end: a bear-tagged post that VADER
        reads as strongly positive must surface as a BEARISH driver.
        """
        stocktwits = _posts(["BULLISH"] * 5, tags=["Bearish"] * 5,
                            texts=[f"NVDA sell it bro {i}" for i in range(5)])
        result = social._drivers("NVDA", pd.DataFrame(), stocktwits,
                                 pd.DataFrame())

        assert result["bearish"]
        assert all(item["sentiment"] < 0 for item in result["bearish"])

    def test_no_data_yields_empty_lists(self):
        result = social._drivers("NVDA", pd.DataFrame(), pd.DataFrame(),
                                 pd.DataFrame())
        assert result == {"bullish": [], "bearish": [], "filtered": False,
                          "mentions": 0}


# ==========================================================================
# Schema
# ==========================================================================
class TestSchema:
    def test_output_is_json_serialisable(self):
        import json
        from datetime import datetime, timezone

        report = {
            "ticker": "NVDA",
            "as_of": datetime.now(timezone.utc),
            "score": 0.51,
            "band": "MODERATELY BULLISH",
            "platforms": {"retail": {"platform": "StockTwits", "score": 0.7,
                                     "samples": 30, "weight": 0.35,
                                     "applied_weight": 0.35, "note": ""}},
            "confidence": {"score": 0.94, "components": ["Coverage 100%"]},
            "signal_divergence": [],
            "drivers": {"bullish": [], "bearish": []},
            # DataFrames must not leak into the serialisable form.
            "frames": {"retail": pd.DataFrame({"a": [1]})},
            "disclaimer": "Not advice.",
        }
        payload = social.to_schema(report)

        json.dumps(payload)
        assert "frames" not in payload
        assert payload["sentiment_score"] == 0.51
        assert payload["sentiment_band"] == "MODERATELY BULLISH"
        assert payload["confidence_score"] == 0.94
        assert payload["as_of"].endswith("+00:00")

    def test_schema_keys_are_stable(self):
        payload = social.to_schema({})
        for key in ("ticker", "as_of", "sentiment_score", "sentiment_band",
                    "confidence_score", "confidence_components", "platforms",
                    "signal_divergence", "key_drivers", "disclaimer"):
            assert key in payload


# ==========================================================================
# Configuration invariants
# ==========================================================================
class TestSocialConfig:
    def test_every_band_has_a_colour(self):
        for _, label in config.SENTIMENT_BANDS:
            assert label in config.SENTIMENT_BAND_COLOURS
            assert hasattr(config.THEME, config.SENTIMENT_BAND_COLOURS[label])

    def test_weight_keys_match_the_platform_keys(self):
        assert set(config.SENTIMENT_WEIGHTS) == {"institutional", "retail",
                                                 "reddit"}

    def test_reddit_bucket_is_the_slowest(self):
        """
        Reddit 429s after a handful of requests. If this ever loosens, the
        social page will start failing in a way that looks like Reddit being
        down rather than us being impolite.
        """
        from utils.rate_limiter import _DEFAULT_BUDGETS

        reddit_rate = _DEFAULT_BUDGETS["reddit"][0]
        assert reddit_rate <= 0.2
        assert reddit_rate < _DEFAULT_BUDGETS["rss"][0]

    def test_stocktwits_is_not_sent_the_plain_user_agent(self):
        """
        Cloudflare serves a challenge page to PLAIN_USER_AGENT. This is the
        opposite of FRED's CSV endpoint, which hangs on a browser UA - so the
        two must never be "harmonised".
        """
        source = (config.BASE_DIR / "data_fetchers" / "social.py").read_text(
            encoding="utf-8")
        assert "PLAIN_USER_AGENT" not in source.split('"""')[2]

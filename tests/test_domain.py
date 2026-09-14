"""
Domain logic: command parsing, ticker normalisation, AIS decoding, curve
analysis, and sentiment scoring.

These are the pure functions between the network and the UI. They are cheap
to test and are where silent wrongness hides most easily - a mislabelled
flag state or a sentinel value treated as a real heading produces a plausible
map with wrong data on it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from data_fetchers import equities, macro
from data_fetchers import maritime as mt


# ==========================================================================
# Command parser
# ==========================================================================
class TestCommandParser:
    @pytest.fixture(autouse=True)
    def _app(self):
        import app
        self.parse = app.parse_command

    @pytest.mark.parametrize("raw,module,subject", [
        ("AAPL EQUITY", "equity", "AAPL"),
        ("NVDA GP", "equity", "NVDA"),
        ("SUEZ SHIP", "maritime", "SUEZ"),
        ("US10Y MACRO", "macro", "US10Y"),
        ("EUROPE FLY", "aviation", "EUROPE"),
        ("ENERGY TOP", "news", "ENERGY"),
    ])
    def test_subject_function_pairs(self, raw, module, subject):
        result = self.parse(raw)
        assert result["valid"]
        assert result["module"] == module
        assert result["subject"] == subject

    @pytest.mark.parametrize("raw,module", [
        ("YCRV", "macro"), ("FLY", "aviation"), ("NEWS", "news"),
        ("HOME", "home"), ("HELP", "help"), ("TOP", "news"), ("AIS", "maritime"),
    ])
    def test_bare_functions(self, raw, module):
        result = self.parse(raw)
        assert result["valid"] and result["module"] == module

    def test_bare_ticker_defaults_to_equity(self):
        result = self.parse("TSLA")
        assert result["module"] == "equity"
        assert result["subject"] == "TSLA"

    def test_nine_digit_number_is_mmsi(self):
        result = self.parse("636019825")
        assert result["module"] == "maritime"
        assert result["subject"] == "636019825"

    def test_imo_prefix_routes_to_maritime(self):
        assert self.parse("IMO9321483")["module"] == "maritime"

    def test_chokepoint_code_routes_to_maritime(self):
        result = self.parse("MALACCA")
        assert result["module"] == "maritime"
        assert result["subject"] == "MALACCA"

    def test_function_first_order(self):
        """'FLY EUROPE' must work as well as 'EUROPE FLY'."""
        result = self.parse("FLY EUROPE")
        assert result["module"] == "aviation"
        assert result["subject"] == "EUROPE"

    def test_case_insensitive(self):
        assert self.parse("aapl equity")["subject"] == "AAPL"

    def test_unknown_function_is_invalid(self):
        result = self.parse("FOO BARBAZ")
        assert not result["valid"]
        assert result["message"]

    def test_empty_input_is_invalid(self):
        assert not self.parse("")["valid"]
        assert not self.parse("   ")["valid"]

    def test_every_configured_function_resolves(self):
        """Guards against adding a COMMAND_FUNCTIONS entry with no route."""
        import app
        for token, module in config.COMMAND_FUNCTIONS.items():
            result = self.parse(token)
            assert result["valid"], f"{token} did not parse"
            assert module in app.ROUTES, f"{token} -> '{module}' has no page"


# ==========================================================================
# Ticker normalisation
# ==========================================================================
class TestNormalizeTicker:
    @pytest.mark.parametrize("raw,expected", [
        ("aapl", "AAPL"),
        ("  msft  ", "MSFT"),
        ("BRK.B", "BRK-B"),          # share class: dot -> dash
        ("BF.B", "BF-B"),
        ("AAPL US Equity", "AAPL"),  # Bloomberg-style input
        ("VOD LN Equity", "VOD"),
    ])
    def test_normalisation(self, raw, expected):
        assert equities.normalize_ticker(raw) == expected

    def test_empty_input(self):
        assert equities.normalize_ticker("") == ""

    def test_index_and_futures_preserved(self):
        assert equities.normalize_ticker("^GSPC") == "^GSPC"
        assert equities.normalize_ticker("CL=F") == "CL=F"
        assert equities.normalize_ticker("BTC-USD") == "BTC-USD"

    # Yahoo's single-letter exchange suffixes. Two-letter codes were never at
    # risk; these four were rewritten to a dash - a symbol Yahoo does not
    # list - and the holding silently returned no quote.
    @pytest.mark.parametrize("raw,expected", [
        ("VOD.L", "VOD.L"),          # London
        ("vod.l", "VOD.L"),
        ("SAP.F", "SAP.F"),          # Frankfurt
        ("7203.T", "7203.T"),        # Tokyo, numeric root
        ("FUU.V", "FUU.V"),          # TSX Venture
        ("RY.TO", "RY.TO"),          # multi-letter codes, unchanged
        ("D05.SI", "D05.SI"),
        ("0700.HK", "0700.HK"),
    ])
    def test_exchange_suffixes_preserved(self, raw, expected):
        assert equities.normalize_ticker(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("BRK.A", "BRK-A"),
        ("HEI.A", "HEI-A"),
        ("LEN.B", "LEN-B"),
        ("MOG.A", "MOG-A"),
        ("BRK-B", "BRK-B"),          # already in Yahoo form
        # A class letter that is also an exchange code: the dot form reads as
        # TSX Venture, and the dash form - how Yahoo lists it - passes through.
        ("MKC.V", "MKC.V"),
        ("MKC-V", "MKC-V"),
    ])
    def test_share_classes_still_dashed(self, raw, expected):
        assert equities.normalize_ticker(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("VOD.L", "VOD.L"),
        ("VOD.L EQUITY", "VOD.L"),
        ("brk.b equity", "BRK-B"),
    ])
    def test_command_bar_route(self, raw, expected):
        """What the command bar does: parse, then normalise the subject."""
        import app
        subject = app.parse_command(raw)["subject"]
        assert equities.normalize_ticker(subject) == expected

    def test_single_letter_set_matches_yfinance(self):
        """
        The named set is reference data. Checked against yfinance's own MIC
        map, so an exchange Yahoo adds or drops fails here rather than as a
        holding that quietly stops quoting.
        """
        from yfinance import const

        mapping = getattr(const, "_MIC_TO_YAHOO_SUFFIX", None)
        if not mapping:
            pytest.skip("yfinance no longer ships a MIC -> suffix map")
        single = {suffix.lstrip(".") for suffix in mapping.values()
                  if isinstance(suffix, str) and len(suffix.lstrip(".")) == 1}
        assert single == equities._SINGLE_LETTER_EXCHANGES

    @pytest.mark.network
    @pytest.mark.parametrize("raw", ["VOD.L", "SAP.F", "7203.T", "FUU.V", "BRK.B"])
    def test_normalised_symbol_quotes_live(self, raw):
        """The normalised form is the one Yahoo actually lists."""
        import yfinance as yf

        symbol = equities.normalize_ticker(raw)
        try:
            bars = yf.Ticker(symbol).history(period="5d")
        except Exception as exc:
            pytest.skip(f"Yahoo unreachable: {exc}")
        assert not bars.empty, f"{raw} -> {symbol} returned no data"


class TestFormatLargeNumber:
    @pytest.mark.parametrize("value,expected", [
        (1.5e12, "$1.50T"), (2.3e9, "$2.30B"), (4.5e6, "$4.50M"),
        (1234, "$1.23K"), (-2.3e9, "-$2.30B"), (None, "—"), (float("nan"), "—"),
    ])
    def test_formatting(self, value, expected):
        assert equities.format_large_number(value) == expected

    def test_custom_currency_symbol(self):
        assert equities.format_large_number(1e9, "") == "1.00B"


# ==========================================================================
# Maritime
# ==========================================================================
class TestAISSentinels:
    @pytest.mark.parametrize("sentinel", [511.0, 1023.0, 360.0])
    def test_sentinels_become_none(self, sentinel):
        """
        AIS encodes "not available" as magic numbers. Treating 511 as a real
        heading points a vessel arrow at a meaningless bearing.
        """
        assert mt._num(sentinel) is None

    @pytest.mark.parametrize("value", [0.0, 12.5, 180.0, 359.0])
    def test_real_values_kept(self, value):
        assert mt._num(value) == value

    def test_none_and_junk(self):
        assert mt._num(None) is None
        assert mt._num("abc") is None
        assert mt._num(float("nan")) is None


class TestShipTypes:
    def test_exact_code(self):
        assert mt._ship_type_label(71) == "Cargo (haz A)"
        assert mt._ship_type_label(80) == "Tanker"

    def test_rounds_down_to_decade(self):
        """Unlisted codes fall back to their ITU decade bucket."""
        assert mt._ship_type_label(79) == "Cargo"
        assert mt._ship_type_label(89) == "Tanker"

    def test_none_and_junk(self):
        assert mt._ship_type_label(None) is None
        assert mt._ship_type_label("x") is None


class TestMMSIFlags:
    @pytest.mark.parametrize("mmsi,flag", [
        ("636019825", "Liberia"),
        ("477995000", "Hong Kong"),
        ("366123456", "United States"),
        ("235103564", "United Kingdom"),
        ("563001234", "Singapore"),
    ])
    def test_known_mids(self, mmsi, flag):
        assert mt.mmsi_to_flag(mmsi) == flag

    def test_unknown_mid_is_labelled_not_guessed(self):
        assert mt.mmsi_to_flag("999123456").startswith("MID")

    def test_short_input(self):
        assert mt.mmsi_to_flag("12") == "Unknown"


class TestGeo:
    def test_haversine_known_distance(self):
        """JFK -> LHR is ~2990 nautical miles."""
        distance = mt.haversine_nm(40.6413, -73.7781, 51.4700, -0.4543)
        assert 2900 < distance < 3100

    def test_zero_distance(self):
        assert mt.haversine_nm(10, 20, 10, 20) == pytest.approx(0, abs=1e-9)

    def test_bbox_center(self):
        assert mt.bbox_center((0, 0, 10, 20)) == (5.0, 10.0)


class TestChokepointConfig:
    def test_all_bboxes_wellformed(self):
        """min < max, and coordinates inside real-world bounds."""
        for code, cp in config.CHOKEPOINTS.items():
            min_lat, min_lon, max_lat, max_lon = cp.bbox
            assert min_lat < max_lat, f"{code} latitude inverted"
            assert min_lon < max_lon, f"{code} longitude inverted"
            assert -90 <= min_lat <= 90 and -90 <= max_lat <= 90, code
            assert -180 <= min_lon <= 180 and -180 <= max_lon <= 180, code

    def test_no_hardcoded_baseline(self):
        """
        Congestion baselines are measured, never configured.

        This guards a regression that would be invisible in the UI: a
        plausible-looking SEVERE CONGESTION label computed against a number
        somebody typed in rather than one the terminal observed.
        """
        for code, cp in config.CHOKEPOINTS.items():
            assert not hasattr(cp, "baseline_vessels"), code


# ==========================================================================
# Yield curve analysis
# ==========================================================================
def _curve(pairs):
    return pd.DataFrame([
        {"tenor": t, "years": config.TENOR_YEARS[t], "yield": y,
         "date": pd.Timestamp("2026-07-22")}
        for t, y in pairs
    ])


class TestYieldCurveAnalysis:
    def test_detects_inversion(self):
        result = macro.analyze_yield_curve(
            _curve([("3M", 5.4), ("2Y", 4.8), ("10Y", 4.2), ("30Y", 4.5)]))
        assert result["is_inverted"]
        assert result["spreads"]["10Y-2Y"] < 0
        assert result["spreads"]["10Y-3M"] < 0
        assert result["shape"] == "INVERTED"
        assert result["recession_risk"] == "ELEVATED"

    def test_normal_curve(self):
        result = macro.analyze_yield_curve(
            _curve([("3M", 3.8), ("2Y", 4.2), ("10Y", 4.6), ("30Y", 5.1)]))
        assert not result["is_inverted"]
        assert result["shape"] == "NORMAL"
        assert result["recession_risk"] == "LOW"

    def test_flat_curve(self):
        result = macro.analyze_yield_curve(
            _curve([("3M", 4.30), ("2Y", 4.32), ("10Y", 4.35), ("30Y", 4.38)]))
        assert result["shape"] == "FLAT"

    def test_partial_inversion_is_moderate(self):
        """10Y-2Y inverted but 10Y-3M positive -> MODERATE, not ELEVATED."""
        result = macro.analyze_yield_curve(
            _curve([("3M", 3.5), ("2Y", 4.8), ("10Y", 4.2), ("30Y", 4.6)]))
        assert result["spreads"]["10Y-2Y"] < 0
        assert result["spreads"]["10Y-3M"] > 0
        assert result["recession_risk"] == "MODERATE"

    def test_severity_grading(self):
        result = macro.analyze_yield_curve(
            _curve([("3M", 6.0), ("2Y", 5.5), ("10Y", 4.2), ("30Y", 4.3)]))
        severities = {i["pair"]: i["severity"] for i in result["inversions"]}
        assert severities["10Y-2Y"] == "SEVERE"

    def test_spreads_in_percentage_points(self):
        """A 40bp spread must be 0.40, not 40 - the UI multiplies by 100."""
        result = macro.analyze_yield_curve(
            _curve([("2Y", 4.20), ("10Y", 4.60)]))
        assert result["spreads"]["10Y-2Y"] == pytest.approx(0.40, abs=1e-9)

    def test_empty_curve_is_safe(self):
        assert macro.analyze_yield_curve(pd.DataFrame()) == {}

    def test_insufficient_data_reports_unknown(self):
        result = macro.analyze_yield_curve(_curve([("10Y", 4.5)]))
        assert result["recession_risk"] == "UNKNOWN"


class TestMacroConfig:
    def test_every_curve_tenor_has_a_year_value(self):
        for tenor in config.YIELD_CURVE_SERIES:
            assert tenor in config.TENOR_YEARS, f"{tenor} missing from TENOR_YEARS"

    def test_tenor_years_ascending(self):
        years = [config.TENOR_YEARS[t] for t in config.YIELD_CURVE_SERIES]
        assert years == sorted(years), "YIELD_CURVE_SERIES is not maturity-ordered"


class TestPeriodInference:
    @pytest.mark.parametrize("freq,expected", [
        ("D", 252), ("W", 52), ("MS", 12), ("QS", 4),
    ])
    def test_frequency_inference(self, freq, expected):
        index = pd.date_range("2020-01-01", periods=40, freq=freq)
        series = pd.Series(range(40), index=index)
        assert macro._infer_periods_per_year(series) == expected


# ==========================================================================
# Sentiment
# ==========================================================================
class TestSentiment:
    def test_bullish_headline(self):
        from data_fetchers import news
        result = news.score_sentiment(
            "Company beats estimates, raises guidance, stock surges")
        assert result["label"] == "BULLISH"
        assert result["score"] > 0

    def test_bearish_headline(self):
        from data_fetchers import news
        result = news.score_sentiment(
            "Company misses estimates, downgrade, bankruptcy fears mount")
        assert result["label"] == "BEARISH"
        assert result["score"] < 0

    def test_empty_text_is_neutral(self):
        from data_fetchers import news
        result = news.score_sentiment("")
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.0

    def test_finance_lexicon_loaded(self):
        """
        VADER doesn't know finance vocabulary out of the box - "downgrade"
        and "beats" score 0 without the custom lexicon.
        """
        from data_fetchers import news
        analyzer = news._get_vader()
        if analyzer is None:
            pytest.skip("nltk/VADER unavailable")
        for term, score in list(config.FINANCE_LEXICON.items())[:10]:
            assert analyzer.lexicon.get(term) == score, f"{term} not merged"

    def test_score_bounded(self):
        from data_fetchers import news
        for text in ["surge rally beat upgrade soar record high buyback profit",
                     "plunge crash fraud bankruptcy default layoffs collapse"]:
            assert -1.0 <= news.score_sentiment(text)["score"] <= 1.0


class TestNewsHelpers:
    def test_clean_text_strips_html(self):
        from data_fetchers import news
        assert news._clean_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_clean_text_unescapes_entities(self):
        from data_fetchers import news
        assert "&" in news._clean_text("AT&amp;T")

    def test_article_id_stable_and_distinct(self):
        from data_fetchers import news
        a = news._article_id("Title", "http://x.com/a?utm=1")
        b = news._article_id("Title", "http://x.com/a?utm=2")
        assert a == b, "query strings should not affect identity"
        assert a != news._article_id("Other", "http://x.com/a")

    def test_time_ago_handles_none(self):
        from data_fetchers import news
        assert news.time_ago(None) == "—"


class TestFeedConfig:
    def test_all_feed_urls_wellformed(self):
        for category, feeds in config.RSS_FEEDS.items():
            for name, url in feeds:
                assert url.startswith("http"), f"{category}/{name}: {url}"
                assert name, f"unnamed feed in {category}"


# ==========================================================================
# Aviation
# ==========================================================================
class TestAviation:
    def test_operator_lookup(self):
        from data_fetchers import aviation
        assert aviation.identify_operator("FDX1234") == "FedEx Express"
        assert aviation.identify_operator("UAL99") == "United Airlines"
        assert aviation.identify_operator("ZZZ1") == "UNKNOWN"
        assert aviation.identify_operator("") == "UNKNOWN"

    def test_operator_lookup_has_one_source(self):
        """
        The designator table lives in config and nowhere else.

        A second copy inside aviation.py is how the two drifted apart before:
        the watchlist knew about operators the callsign resolver didn't.
        """
        from data_fetchers import aviation

        for fleet in config.OPERATOR_FLEETS.values():
            for designator, operator in fleet.items():
                assert aviation.identify_operator(f"{designator}123") == operator

    def test_designators_are_three_letters(self):
        """Callsign matching slices [:3]; a longer key could never match."""
        for group, fleet in config.OPERATOR_FLEETS.items():
            for designator in fleet:
                assert len(designator) == 3, f"{group}/{designator}"
                assert designator.isalnum(), f"{group}/{designator}"
                assert designator == designator.upper(), f"{group}/{designator}"

    def test_regions_wellformed(self):
        for name, (min_lat, max_lat, min_lon, max_lon) in config.AVIATION_REGIONS.items():
            assert min_lat < max_lat, name
            assert min_lon < max_lon, name

    def test_summarize_empty(self):
        from data_fetchers import aviation
        assert aviation.summarize_traffic(pd.DataFrame()) == {"total": 0}

    def test_squawk_alerts_defined(self):
        from data_fetchers import aviation
        assert aviation.SQUAWK_ALERTS["7700"][0] == "EMERGENCY"
        assert aviation.SQUAWK_ALERTS["7600"][0] == "RADIO FAIL"


# ==========================================================================
# Peer selection
# ==========================================================================
# REGRESSION CONTEXT: comparables used to come from seven hand-written sector
# lists, with megacap tech as the catch-all. Any ticker outside those lists -
# a regional bank, a biotech, a utility - was silently benchmarked against
# AAPL and NVDA, and the table of multiples looked completely normal. Peers
# now come from the issuer's own Yahoo classification or from nowhere.
class TestPeerSelection:
    def _table(self, rows):
        return pd.DataFrame(
            [{"name": s, "market weight": w} for s, w in rows],
            index=[s for s, _ in rows],
        ).rename_axis("symbol")

    def test_drops_focal_and_zero_weight_names(self):
        from data_fetchers import equities

        table = self._table([("AAPL", 0.2), ("MSFT", 0.15),
                             ("DEAD", 0.0), ("NVDA", 0.1)])
        assert equities._peers_from(table, "AAPL", 6) == ["MSFT", "NVDA"]

    def test_respects_max_peers(self):
        from data_fetchers import equities

        table = self._table([(f"P{i}", 0.1) for i in range(10)])
        assert len(equities._peers_from(table, "FOCAL", 3)) == 3

    def test_no_config_peer_groups_remain(self):
        """
        The hand-written sector lists must not come back.

        They were seductive because they always returned something. That is
        exactly the failure: a peer set is either derived from this issuer's
        classification or it does not exist.
        """
        assert not hasattr(config, "PEER_GROUPS")

    def test_dominant_issuer_falls_through_to_sector(self, monkeypatch):
        """
        Apple is 99.9% of Yahoo's consumer-electronics industry; its
        "peers" there are microcaps. Comparing against them is arithmetic,
        not analysis, so the sector is used instead.
        """
        from data_fetchers import equities

        industry = self._table([("AAPL", 0.999), ("TINY", 0.0004)])
        sector = self._table([("NVDA", 0.18), ("AAPL", 0.16), ("MSFT", 0.13)])

        monkeypatch.setattr(equities, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(equities, "get_company_info",
                            lambda t: {"industryKey": "ce", "sectorKey": "tech"})
        monkeypatch.setattr(
            equities, "_constituents",
            lambda kind, key: industry if kind == "Industry" else sector)

        assert equities.suggest_peers.__wrapped__("AAPL") == [
            "AAPL", "NVDA", "MSFT"]

    def test_returns_bare_ticker_when_unclassified(self, monkeypatch):
        from data_fetchers import equities

        monkeypatch.setattr(equities, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(equities, "get_company_info", lambda t: {})
        monkeypatch.setattr(equities, "_constituents", lambda kind, key: None)

        assert equities.suggest_peers.__wrapped__("OBSCURE") == ["OBSCURE"]


# ==========================================================================
# Chart styling
# ==========================================================================
class TestStyleFigure:
    """
    REGRESSION CONTEXT: style_figure used to pass its `title` argument
    straight to update_layout, including when it was None. That does not mean
    "no title" - Plotly serialises an empty title object, Plotly.js concludes
    a title exists, and paints the literal string "undefined" into the chart.

    Every untitled figure carried it: both gauges, the candlestick chart and
    the sentiment bar. On the recession panel it was the most prominent text
    on screen, because that gauge reads 0 and draws a zero-length arc
    whenever nothing is triggered.

    The assertion has to be on the serialised JSON. `layout.title.text` is
    None either way, so a Python-side check passes while the browser shows
    "undefined".
    """

    @pytest.fixture(autouse=True)
    def _template(self):
        """style_figure references the "openterm" template, which app.py
        registers via apply_theme(). Tests never call that."""
        from ui.terminal_theme import _register_plotly_template
        _register_plotly_template()

    @staticmethod
    def _layout(fig):
        import json
        return json.loads(fig.to_json())["layout"]

    def test_untitled_figure_serialises_no_title_key(self):
        import plotly.graph_objects as go
        from ui.terminal_theme import style_figure

        assert "title" not in self._layout(style_figure(go.Figure(), height=100))

    def test_titled_figure_keeps_its_title(self):
        import plotly.graph_objects as go
        from ui.terminal_theme import style_figure

        layout = self._layout(
            style_figure(go.Figure(), height=100, title="YIELD CURVE"))
        assert layout["title"]["text"] == "YIELD CURVE"

    def test_gauge_at_zero_has_no_stray_title(self):
        """The exact figure that shipped the bug: an all-clear recession score."""
        from ui import components as ui

        figure = ui.gauge(0.0, "COMPOSITE RISK SCORE",
                          thresholds=[(20, config.THEME.green),
                                      (45, config.THEME.amber),
                                      (100, config.THEME.red)])
        assert "title" not in self._layout(figure)
        assert figure.data[0].value == 0.0

"""
Sector drill-down: who is in a sector, how much of it they are, and filters.

REGRESSION CONTEXT
------------------
Every failure guarded here renders a table that looks entirely reasonable:

  * Yahoo's sector keys are URL slugs. yfinance's own constant table carries
    display names ("Real Estate"), and every one of those 404s - a tile wired
    to one opens an empty sector.

  * A company's `market weight` in an industry table is its share of the
    industry. Reading it as its share of the sector overstates every name in
    a small industry.

  * A partial list presented as the sector. Yahoo publishes up to fifty names
    per industry; the drill-down reports how many it lists against how many
    exist, and a failed industry is named rather than silently missing.

  * Filters that drop unrated or uncapped companies when no filter was asked
    for, or treat a typed "." as a regex wildcard.

  * A heatmap tile that cannot be clicked. Plotly cannot select heatmap
    cells, so each tile carries an invisible point Streamlit can report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import config
from data_fetchers import allocation, equities, sectors


def _members(rows):
    """A frame shaped like yfinance's top_companies: indexed by symbol."""
    frame = pd.DataFrame(rows, columns=["symbol", "name", "rating", "market weight"])
    return frame.set_index("symbol")


@pytest.fixture
def yahoo(monkeypatch):
    """
    Stub yfinance's Sector and Industry classes.

    Tests edit the state before building. An industry key missing from
    `members` raises exactly as a 404 does.
    """
    state = {
        "overview": {"companies_count": 853, "market_cap": 1_000e9},
        "industries": pd.DataFrame(
            {"name": ["Semiconductors", "Software", "Broken Things"],
             "symbol": ["^S", "^W", "^B"],
             "market weight": [0.60, 0.35, 0.05]},
            index=pd.Index(["semis", "software", "broken"], name="key")),
        "members": {
            "semis": _members([
                ("NVDA", "NVIDIA Corporation", "Strong Buy", 0.50),
                ("AVGO", "Broadcom Inc.", "Buy", 0.30),
                ("DEAD", "Delisted Co", None, 0.0),
            ]),
            "software": _members([
                ("MSFT", "Microsoft Corporation", "Buy", 0.75),
                ("ORCL", "Oracle Corporation", "Hold", 0.20),
            ]),
        },
        "leaders": _members([
            ("NVDA", "NVIDIA Corporation", "Strong Buy", 0.30),
            ("IBM", "International Business Machines", None, 0.02),
        ]),
    }

    class Sector:
        def __init__(self, key):
            self.key = key
            self.name = "Technology"
            self.overview = state["overview"]
            self.industries = state["industries"]
            self.top_companies = state["leaders"]

    class Industry:
        def __init__(self, key):
            self.key = key

        @property
        def top_companies(self):
            if self.key not in state["members"]:
                raise RuntimeError(f"HTTP Error 404 for industry {self.key}")
            return state["members"][self.key]

    monkeypatch.setattr(equities.yf, "Sector", Sector)
    monkeypatch.setattr(equities.yf, "Industry", Industry)
    return state


def _build():
    return sectors.get_sector_constituents.uncached("technology")


# ==========================================================================
# Constituents
# ==========================================================================
class TestSectorConstituents:
    def test_sector_weight_is_industry_share_times_member_share(self, yahoo):
        table = _build()["table"].set_index("ticker")
        assert table.loc["NVDA", "sector_weight_pct"] == pytest.approx(30.0)
        assert table.loc["MSFT", "sector_weight_pct"] == pytest.approx(26.25)
        assert table.loc["ORCL", "sector_weight_pct"] == pytest.approx(7.0)
        # The industry share is kept alongside, not substituted for it.
        assert table.loc["MSFT", "industry_weight_pct"] == pytest.approx(75.0)

    def test_largest_share_of_the_sector_first(self, yahoo):
        assert list(_build()["table"]["ticker"]) == [
            "NVDA", "MSFT", "AVGO", "ORCL", "IBM"]

    def test_zero_weight_member_dropped(self, yahoo):
        assert "DEAD" not in set(_build()["table"]["ticker"])

    def test_failed_industry_is_named(self, yahoo):
        data = _build()
        assert data["industries"] == 3
        assert data["failed_industries"] == ["Broken Things"]

    def test_sector_leader_missing_from_industries_is_backfilled(self, yahoo):
        ibm = _build()["table"].set_index("ticker").loc["IBM"]
        assert pd.isna(ibm["industry"])
        assert ibm["sector_weight_pct"] == pytest.approx(2.0)
        assert ibm["rating"] == sectors.UNRATED

    def test_leader_already_listed_is_not_duplicated(self, yahoo):
        assert list(_build()["table"]["ticker"]).count("NVDA") == 1

    def test_count_coverage_and_derived_cap(self, yahoo):
        data = _build()
        assert data["name"] == "Technology"
        assert data["companies_count"] == 853
        assert data["coverage_pct"] == pytest.approx(30 + 26.25 + 18 + 7 + 2)
        nvda = data["table"].set_index("ticker").loc["NVDA"]
        assert nvda["market_cap"] == pytest.approx(300e9)

    def test_missing_sector_cap_leaves_caps_blank(self, yahoo):
        yahoo["overview"] = {}
        data = _build()
        assert data["companies_count"] is None
        assert data["table"]["market_cap"].isna().all()

    def test_nothing_back_raises_so_stale_list_is_served(self, yahoo):
        yahoo["members"] = {}
        yahoo["leaders"] = None
        with pytest.raises(ValueError):
            _build()

    def test_no_industries_raises(self, yahoo):
        yahoo["industries"] = pd.DataFrame()
        with pytest.raises(ValueError):
            _build()


# ==========================================================================
# Filters
# ==========================================================================
@pytest.fixture
def table():
    """A frame shaped like get_sector_constituents()['table']."""
    return pd.DataFrame([
        ("NVDA", "NVIDIA Corporation", "Semiconductors", "Strong Buy", 50.0, 30.0, 300e9),
        ("MSFT", "Microsoft Corporation", "Software", "Buy", 75.0, 26.25, 262.5e9),
        ("AVGO", "Broadcom Inc.", "Semiconductors", "Buy", 30.0, 18.0, 180e9),
        ("ORCL", "Oracle Corporation", "Software", "Hold", 20.0, 7.0, 70e9),
        ("IBM", "International Business Machines", None, "Unrated", None, 2.0, float("nan")),
    ], columns=sectors.TABLE_COLUMNS)


def _tickers(frame):
    return list(frame["ticker"])


class TestFilterConstituents:
    def test_no_filters_returns_everything(self, table):
        assert _tickers(sectors.filter_constituents(table)) == _tickers(table)

    @pytest.mark.parametrize("query,expected", [
        ("nv", ["NVDA"]),
        ("  msft ", ["MSFT"]),
        ("corporation", ["NVDA", "MSFT", "ORCL"]),
        ("BUSINESS", ["IBM"]),
    ])
    def test_query_matches_ticker_or_name(self, table, query, expected):
        assert _tickers(sectors.filter_constituents(table, query=query)) == expected

    def test_query_is_literal_not_regex(self, table):
        # As a regex, "c.o" would match the "cro" in Microsoft.
        assert sectors.filter_constituents(table, query="c.o").empty
        assert sectors.filter_constituents(table, query="(").empty

    def test_industry_filter(self, table):
        result = sectors.filter_constituents(table, industries=["Software"])
        assert _tickers(result) == ["MSFT", "ORCL"]

    def test_rating_filter_reaches_unrated(self, table):
        result = sectors.filter_constituents(table, ratings=[sectors.UNRATED])
        assert _tickers(result) == ["IBM"]

    def test_min_cap_excludes_unknown_caps(self, table):
        result = sectors.filter_constituents(table, min_market_cap=100e9)
        assert _tickers(result) == ["NVDA", "MSFT", "AVGO"]

    def test_filters_combine(self, table):
        result = sectors.filter_constituents(
            table, industries=["Semiconductors"], ratings=["Buy"])
        assert _tickers(result) == ["AVGO"]

    def test_empty_table(self):
        assert sectors.filter_constituents(pd.DataFrame(), query="x").empty


class TestOptions:
    def test_industries_ordered_by_share_of_sector(self, table):
        assert sectors.industry_options(table) == ["Semiconductors", "Software"]

    def test_ratings_in_consensus_order_unrated_last(self, table):
        table.loc[len(table)] = ("XYZ", "Xyz", "Software", "Outperform",
                                 1.0, 0.5, 1e9)
        assert sectors.rating_options(table) == [
            "Strong Buy", "Buy", "Hold", "Outperform", sectors.UNRATED]


class TestAttachQuotes:
    def test_prices_attached_where_quoted(self, table):
        quotes = {"NVDA": {"price": 200.0, "change_pct": 1.5},
                  "IBM": {"price": 250.0, "change_pct": -0.5}}
        priced = sectors.attach_quotes(table, quotes).set_index("ticker")
        assert priced.loc["NVDA", "change_pct"] == pytest.approx(1.5)
        assert pd.isna(priced.loc["MSFT", "price"])
        assert "price" not in table.columns


# ==========================================================================
# Heatmap wiring
# ==========================================================================
class TestSectorTiles:
    def test_every_sector_tile_maps_to_a_distinct_gics_sector(self):
        keys = [key for _label, key in config.SECTOR_ETFS.values() if key]
        mapped = [allocation.normalise_sector(key) for key in keys]
        assert None not in mapped
        assert sorted(mapped) == sorted(config.GICS_SECTORS)

    def test_keys_are_slugs_not_display_names(self):
        for _label, key in config.SECTOR_ETFS.values():
            if key:
                assert key == key.lower() and " " not in key

    def test_the_market_tile_has_no_sector(self):
        assert config.SECTOR_ETFS["SPY"][1] is None


class TestClickableHeatmap:
    @pytest.fixture(autouse=True)
    def _template(self):
        """style_figure references the "openterm" template, which app.py
        registers via apply_theme(). Tests never call that."""
        from ui.terminal_theme import _register_plotly_template
        _register_plotly_template()

    @staticmethod
    def _frame():
        return pd.DataFrame({"label": ["TECH", "ENERGY", "S&P 500"],
                             "change": [1.2, -0.4, 0.5],
                             "symbol": ["XLK", "XLE", None]})

    def test_plain_heatmap_is_unchanged(self):
        from ui import components as ui

        figure = ui.heatmap(self._frame(), "change", "label", columns=2)
        assert [trace.type for trace in figure.data] == ["heatmap"]

    def test_every_cell_gets_a_clickable_point(self):
        from ui import components as ui

        figure = ui.heatmap(self._frame(), "change", "label", columns=2,
                            key_column="symbol")
        points = figure.data[1]
        assert points.type == "scatter"
        # The keyless tile and the grid padding still get a point, so a click
        # there lands on nothing rather than on the nearest sector.
        assert list(points.customdata) == ["XLK", "XLE", "", ""]
        assert list(points.x) == [0, 1, 0, 1]
        assert list(points.y) == [0, 0, 1, 1]
        # Scrolling over the grid must scroll the page, not zoom the tiles.
        assert figure.layout.xaxis.fixedrange and figure.layout.yaxis.fixedrange
        assert figure.layout.xaxis.showspikes is False

    def test_selected_tile_is_outlined(self):
        from ui import components as ui

        figure = ui.heatmap(self._frame(), "change", "label", columns=2,
                            key_column="symbol", selected="XLE")
        assert len(figure.layout.shapes) == 1
        assert figure.layout.shapes[0].x0 == pytest.approx(0.5)

    def test_no_outline_without_a_selection(self):
        from ui import components as ui

        figure = ui.heatmap(self._frame(), "change", "label", columns=2,
                            key_column="symbol", selected="NOPE")
        assert len(figure.layout.shapes) == 0

    @pytest.mark.parametrize("event,expected", [
        (None, None),
        ({"selection": {"points": []}}, None),
        ({"selection": {"points": [{"customdata": "XLK"}]}}, "XLK"),
        ({"selection": {"points": [{"customdata": ["XLF"]}]}}, "XLF"),
        ({"selection": {"points": [{"customdata": ""}]}}, None),
        (SimpleNamespace(selection=SimpleNamespace(
            points=[{"customdata": ["XLE"]}])), "XLE"),
    ])
    def test_selected_customdata(self, event, expected):
        from ui import components as ui

        assert ui.selected_customdata(event) == expected

"""
SEC XBRL extraction correctness.

REGRESSION CONTEXT
------------------
The original implementation keyed each fact by its `fy`/`fp` fields. Those
identify *the report the fact appeared in*, not the period the number
describes. A 10-K carries three years of income-statement comparatives, so
Apple's FY2023 revenue appears tagged fy=2023, fy=2024 AND fy=2025. Combined
with a "keep the most recently filed value" rule, every figure was relabelled
with the newest filing that mentioned it - shifting the entire statement two
years.

Concretely: the terminal reported $383.285B as Apple's FY2025 revenue. The
real FY2025 figure is $416.161B; $383.285B is FY2023.

Nothing about this is visible in the UI - the table renders, the numbers are
real, they are simply under the wrong headings. The fixture below reproduces
the exact fact structure from Apple's companyfacts document so the bug can
never return without a test failing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_fetchers import equities


# ==========================================================================
# Fixtures - shaped exactly like SEC's companyfacts payload
# ==========================================================================
def _duration(start, end, val, fy, filed, form="10-K", fp="FY"):
    return {"start": start, "end": end, "val": val, "fy": fy, "fp": fp,
            "form": form, "filed": filed, "accn": f"acc-{filed}"}


def _instant(end, val, fy, filed, form="10-K", fp="FY"):
    return {"end": end, "val": val, "fy": fy, "fp": fp,
            "form": form, "filed": filed, "accn": f"acc-{filed}"}


@pytest.fixture
def apple_revenue_facts():
    """
    Three fiscal years of revenue as SEC actually publishes them: each year
    restated into every subsequent 10-K with that filing's `fy` stamped on it.

    Real values, real period boundaries, real filing dates.
    """
    return {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {
                "USD": [
                    # --- FY2023: $383.285B, appears in three filings -------
                    _duration("2022-09-25", "2023-09-30", 383_285_000_000, 2023, "2023-11-03"),
                    _duration("2022-09-25", "2023-09-30", 383_285_000_000, 2024, "2024-11-01"),
                    _duration("2022-09-25", "2023-09-30", 383_285_000_000, 2025, "2025-10-31"),
                    # --- FY2024: $391.035B, appears in two ------------------
                    _duration("2023-10-01", "2024-09-28", 391_035_000_000, 2024, "2024-11-01"),
                    _duration("2023-10-01", "2024-09-28", 391_035_000_000, 2025, "2025-10-31"),
                    # --- FY2025: $416.161B, newest ---------------------------
                    _duration("2024-09-29", "2025-09-27", 416_161_000_000, 2025, "2025-10-31"),
                ]
            }
        }
    }


# ==========================================================================
# The core regression
# ==========================================================================
class TestPeriodLabelling:
    def test_periods_are_not_shifted(self, apple_revenue_facts):
        """
        THE regression test. Each fiscal year must carry its own figure.

        Under the `fy`-keyed bug this returned FY2025 -> 383.285B.
        """
        series = equities._extract_xbrl_series(
            apple_revenue_facts,
            ["RevenueFromContractWithCustomerExcludingAssessedTax"],
            annual=True,
        )

        assert series["FY2025"] == 416_161_000_000, (
            "FY2025 is not showing its own revenue - periods are shifted. "
            f"Got {series['FY2025']:,.0f}, expected 416,161,000,000."
        )
        assert series["FY2024"] == 391_035_000_000
        assert series["FY2023"] == 383_285_000_000

    def test_no_duplicate_or_missing_years(self, apple_revenue_facts):
        series = equities._extract_xbrl_series(
            apple_revenue_facts,
            ["RevenueFromContractWithCustomerExcludingAssessedTax"],
            annual=True,
        )
        assert set(series) == {"FY2023", "FY2024", "FY2025"}

    def test_label_derives_from_period_end(self):
        assert equities._period_label("2025-09-27", annual=True) == "FY2025"
        assert equities._period_label("2024-01-31", annual=True) == "FY2024"
        assert equities._period_label("2025-06-28", annual=False) == "2025Q2"
        assert equities._period_label("2025-12-31", annual=False) == "2025Q4"

    def test_malformed_date_returns_none(self):
        assert equities._period_label("not-a-date", annual=True) is None


class TestDurationFiltering:
    def test_quarterly_facts_excluded_from_annual(self):
        """
        A 10-K also tags Q4 durations for some concepts. Without duration
        filtering a ~90-day value would overwrite the annual figure for the
        same period-end year.
        """
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 416_161_000_000, 2025, "2025-10-31"),
                # Q4 stub ending in the same fiscal year, filed the same day.
                _duration("2025-06-29", "2025-09-27", 102_000_000_000, 2025, "2025-10-31"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=True)
        assert series["FY2025"] == 416_161_000_000, (
            "a quarterly duration leaked into the annual series"
        )

    def test_annual_facts_excluded_from_quarterly(self):
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 416_161_000_000, 2025,
                          "2025-10-31", form="10-Q", fp="Q4"),
                _duration("2025-06-29", "2025-09-27", 102_000_000_000, 2025,
                          "2025-10-31", form="10-Q", fp="Q4"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=False)
        assert series["2025Q3"] == 102_000_000_000

    def test_53_week_year_accepted(self):
        """Retailers run 52/53-week calendars; 371 days is still annual."""
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-01-28", "2025-02-01", 100_000_000, 2025, "2025-03-01"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=True)
        assert series == {"FY2025": 100_000_000}


class TestFormFiltering:
    def test_annual_ignores_10q(self):
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 416_000_000_000, 2025,
                          "2025-10-31", form="10-K"),
                _duration("2023-09-29", "2024-09-27", 999_000_000_000, 2024,
                          "2024-10-31", form="10-Q"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=True)
        assert "FY2024" not in series
        assert series["FY2025"] == 416_000_000_000

    def test_amended_filings_accepted(self):
        """10-K/A is still an annual report and must not be filtered out."""
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 416_000_000_000, 2025,
                          "2025-10-31", form="10-K/A"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=True)
        assert series["FY2025"] == 416_000_000_000


class TestRestatements:
    def test_latest_filing_wins_for_same_period(self):
        """
        Two values for the *same* period end: the later filing is the
        restated figure and must win. This is the one place `filed` should
        still break ties.
        """
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 400_000_000_000, 2025, "2025-10-31"),
                _duration("2024-09-29", "2025-09-27", 416_161_000_000, 2026, "2026-02-15"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Revenues"], annual=True)
        assert series["FY2025"] == 416_161_000_000


class TestInstantFacts:
    def test_balance_sheet_dated_by_end(self):
        """
        Instant facts carry no `start`. A 10-K reports the current and prior
        year-end balances; both must be dated by their own `end`.
        """
        facts = {
            "Assets": {"units": {"USD": [
                _instant("2025-09-27", 359_240_000_000, 2025, "2025-10-31"),
                _instant("2024-09-28", 364_980_000_000, 2025, "2025-10-31"),
                _instant("2024-09-28", 364_980_000_000, 2024, "2024-11-01"),
            ]}}
        }
        series = equities._extract_xbrl_series(facts, ["Assets"], annual=True)
        assert series["FY2025"] == 359_240_000_000
        assert series["FY2024"] == 364_980_000_000


class TestTagFallback:
    def test_higher_priority_tag_wins_for_shared_period(self):
        facts = {
            "Revenues": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 111, 2025, "2025-10-31"),
            ]}},
            "SalesRevenueNet": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 222, 2025, "2025-10-31"),
            ]}},
        }
        series = equities._extract_xbrl_series(
            facts, ["Revenues", "SalesRevenueNet"], annual=True)
        assert series["FY2025"] == 111

    def test_falls_through_to_second_tag(self):
        facts = {
            "SalesRevenueNet": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 222, 2025, "2025-10-31"),
            ]}},
        }
        series = equities._extract_xbrl_series(
            facts, ["Revenues", "SalesRevenueNet"], annual=True)
        assert series["FY2025"] == 222

    def test_unknown_tags_return_empty(self):
        assert equities._extract_xbrl_series({}, ["Nope"], annual=True) == {}

    def test_non_usd_unit_used_as_fallback(self):
        """EPS is denominated USD/shares, not USD."""
        facts = {
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                _duration("2024-09-29", "2025-09-27", 7.09, 2025, "2025-10-31"),
            ]}}
        }
        series = equities._extract_xbrl_series(
            facts, ["EarningsPerShareDiluted"], annual=True)
        assert series["FY2025"] == pytest.approx(7.09)


class TestTagMigration:
    """
    REGRESSION: filers switch tags mid-history.

    Microsoft reported cost of revenue under `CostOfRevenue` through FY2017
    and `CostOfGoodsAndServicesSold` from FY2020. Apple used
    `PaymentsOfDividendsCommonStock` in FY2016-17 and `PaymentsOfDividends`
    from FY2020.

    The original first-match-wins loop returned whichever tag it hit first
    and stopped - yielding a decade-old series with every recent column NaN.
    The line item simply looked unavailable in the UI.
    """

    @pytest.fixture
    def migrated_tags(self):
        """Two tags covering disjoint, non-overlapping eras."""
        return {
            "CostOfRevenue": {"units": {"USD": [          # legacy era
                _duration("2015-07-01", "2016-06-30", 32_780_000_000, 2016, "2016-07-28"),
                _duration("2016-07-01", "2017-06-30", 34_261_000_000, 2017, "2017-08-02"),
            ]}},
            "CostOfGoodsAndServicesSold": {"units": {"USD": [   # modern era
                _duration("2023-07-01", "2024-06-30", 74_114_000_000, 2024, "2024-07-30"),
                _duration("2024-07-01", "2025-06-30", 87_820_000_000, 2025, "2025-07-30"),
            ]}},
        }

    def test_merges_across_eras(self, migrated_tags):
        """Both eras must be present in one continuous series."""
        series = equities._extract_xbrl_series(
            migrated_tags,
            ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
            annual=True,
        )
        assert set(series) == {"FY2016", "FY2017", "FY2024", "FY2025"}
        assert series["FY2025"] == 87_820_000_000
        assert series["FY2016"] == 32_780_000_000

    def test_recent_periods_populated_regardless_of_tag_order(self, migrated_tags):
        """
        The exact failure mode: with the legacy tag listed first, the modern
        era used to be dropped entirely.
        """
        for order in (["CostOfRevenue", "CostOfGoodsAndServicesSold"],
                      ["CostOfGoodsAndServicesSold", "CostOfRevenue"]):
            series = equities._extract_xbrl_series(migrated_tags, order, annual=True)
            assert "FY2025" in series, f"recent period dropped with order {order}"
            assert "FY2016" in series, f"legacy period dropped with order {order}"

    def test_priority_respected_where_eras_overlap(self):
        """Merging must not override a higher-priority tag on shared periods."""
        facts = {
            "PaymentsOfDividends": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 100, 2025, "2025-10-31"),
            ]}},
            "PaymentsOfDividendsCommonStock": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 999, 2025, "2025-10-31"),
                _duration("2023-10-01", "2024-09-28", 90, 2024, "2024-11-01"),
            ]}},
        }
        series = equities._extract_xbrl_series(
            facts, ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
            annual=True)
        assert series["FY2025"] == 100, "priority tag was overridden"
        assert series["FY2024"] == 90, "gap-fill from lower-priority tag failed"


class TestNoUsGaapTaxonomy:
    """
    A ticker can resolve to a CIK with no us-gaap facts at all - e.g. XOM now
    maps to a post-reorganisation holding entity that has filed no 10-K. That
    must be reported, not rendered as a silently blank table.
    """

    def test_returns_empty_with_reason(self, monkeypatch):
        monkeypatch.setattr(equities, "get_sec_company_facts", lambda ticker: {
            "cik": 2115436,
            "entityName": "EXXON MOBIL CORP",
            "facts": {"ffd": {"SomeTag": {}}},
        })
        df = equities.get_sec_financials("XOM")
        assert df.empty
        assert "reason" in df.attrs
        assert "no XBRL financial statements" in df.attrs["reason"]

class TestMalformedInput:
    def test_missing_value_skipped(self):
        facts = {"Revenues": {"units": {"USD": [
            {"start": "2024-09-29", "end": "2025-09-27", "val": None,
             "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-10-31"},
        ]}}}
        assert equities._extract_xbrl_series(facts, ["Revenues"], annual=True) == {}

    def test_missing_end_skipped(self):
        facts = {"Revenues": {"units": {"USD": [
            {"start": "2024-09-29", "val": 1, "fy": 2025, "fp": "FY",
             "form": "10-K", "filed": "2025-10-31"},
        ]}}}
        assert equities._extract_xbrl_series(facts, ["Revenues"], annual=True) == {}

    def test_empty_units_skipped(self):
        facts = {"Revenues": {"units": {}}}
        assert equities._extract_xbrl_series(facts, ["Revenues"], annual=True) == {}


# ==========================================================================
# Statement assembly
# ==========================================================================
class TestGetSecFinancials:
    def test_columns_newest_first(self, monkeypatch, apple_revenue_facts):
        monkeypatch.setattr(
            equities, "get_sec_company_facts",
            lambda ticker: {"facts": {"us-gaap": apple_revenue_facts}},
        )
        df = equities.get_sec_financials("AAPL", "income_statement", annual=True)
        assert list(df.columns) == ["FY2025", "FY2024", "FY2023"]
        assert df.loc["Revenue", "FY2025"] == 416_161_000_000

    def test_periods_argument_truncates(self, monkeypatch, apple_revenue_facts):
        monkeypatch.setattr(
            equities, "get_sec_company_facts",
            lambda ticker: {"facts": {"us-gaap": apple_revenue_facts}},
        )
        df = equities.get_sec_financials("AAPL", "income_statement",
                                         annual=True, periods=2)
        assert list(df.columns) == ["FY2025", "FY2024"]

    def test_line_item_order_preserved(self, monkeypatch):
        """Rows must follow the canonical statement order, not dict order."""
        facts = {
            "NetIncomeLoss": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 112_010_000_000, 2025, "2025-10-31"),
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                _duration("2024-09-29", "2025-09-27", 416_161_000_000, 2025, "2025-10-31"),
            ]}},
        }
        monkeypatch.setattr(
            equities, "get_sec_company_facts",
            lambda ticker: {"facts": {"us-gaap": facts}},
        )
        df = equities.get_sec_financials("AAPL", "income_statement", annual=True)
        assert list(df.index).index("Revenue") < list(df.index).index("Net Income")

    def test_non_registrant_returns_empty(self, monkeypatch):
        monkeypatch.setattr(equities, "get_sec_company_facts", lambda ticker: {})
        assert equities.get_sec_financials("NOTREAL").empty

    def test_missing_us_gaap_returns_empty(self, monkeypatch):
        monkeypatch.setattr(equities, "get_sec_company_facts",
                            lambda ticker: {"facts": {}})
        assert equities.get_sec_financials("AAPL").empty


# ==========================================================================
# Live validation - accounting identities against real filings
# ==========================================================================
@pytest.mark.network
@pytest.mark.slow
class TestLiveFilings:
    def test_gross_profit_identity(self):
        """Revenue - COGS must equal reported Gross Profit."""
        df = equities.get_sec_financials("AAPL", "income_statement",
                                         annual=True, periods=3)
        if df.empty:
            pytest.skip("SEC EDGAR unreachable")

        column = df.columns[0]
        computed = df.loc["Revenue", column] - df.loc["Cost of Revenue", column]
        assert computed == pytest.approx(df.loc["Gross Profit", column], rel=1e-6)

    def test_balance_sheet_balances(self):
        """Assets must equal Liabilities + Equity."""
        df = equities.get_sec_financials("AAPL", "balance_sheet",
                                         annual=True, periods=3)
        if df.empty:
            pytest.skip("SEC EDGAR unreachable")

        column = df.columns[0]
        assets = df.loc["Total Assets", column]
        total = df.loc["Total Liabilities", column] + df.loc["Total Equity", column]
        assert assets == pytest.approx(total, rel=0.01)

    def test_tag_migration_lines_are_populated(self):
        """
        The concrete lines the merge fix repaired.

        Deliberately requests 14 periods so the window SPANS the tag
        migration. A short window is not a valid test here: the tag lists put
        the modern tag first, so even a broken first-match-wins loop fills
        recent columns. Only a cross-era window distinguishes the two.

        MSFT cost of revenue: CostOfRevenue (FY2012-17) ->
        CostOfGoodsAndServicesSold (FY2020-25). Merged, all 14 populate;
        first-match-wins yields roughly half.
        """
        msft = equities.get_sec_financials("MSFT", "income_statement",
                                           annual=True, periods=14)
        if msft.empty:
            pytest.skip("SEC EDGAR unreachable")

        cost = msft.loc["Cost of Revenue"]
        populated = int(cost.notna().sum())
        assert populated >= 12, (
            f"MSFT Cost of Revenue populated in only {populated}/{len(cost)} "
            f"periods - the CostOfRevenue -> CostOfGoodsAndServicesSold "
            f"migration is not being merged across eras"
        )
        # Both eras specifically.
        assert cost.get("FY2025") is not None and not pd.isna(cost["FY2025"])
        assert cost.get("FY2016") is not None and not pd.isna(cost["FY2016"])

    def test_dividends_span_tag_migration(self):
        """
        AAPL: PaymentsOfDividendsCommonStock (FY2016-17) ->
        PaymentsOfDividends (FY2020-25). FY2012 is genuinely absent - Apple
        reinstated its dividend partway through that year.
        """
        aapl = equities.get_sec_financials("AAPL", "cash_flow",
                                           annual=True, periods=14)
        if aapl.empty:
            pytest.skip("SEC EDGAR unreachable")

        dividends = aapl.loc["Dividends Paid"]
        populated = int(dividends.notna().sum())
        assert populated >= 12, (
            f"AAPL Dividends Paid populated in only {populated}/"
            f"{len(dividends)} periods - tag migration not merged"
        )
        assert not pd.isna(dividends["FY2025"])
        assert not pd.isna(dividends["FY2016"])

    def test_merged_cost_of_revenue_satisfies_gross_profit_identity(self):
        """
        Strongest available check that the merged tag is the *right* one:
        Revenue - COGS must reproduce reported Gross Profit for MSFT, whose
        cost line comes from the migrated tag.
        """
        df = equities.get_sec_financials("MSFT", "income_statement",
                                         annual=True, periods=3)
        if df.empty:
            pytest.skip("SEC EDGAR unreachable")

        column = df.columns[0]
        computed = df.loc["Revenue", column] - df.loc["Cost of Revenue", column]
        assert computed == pytest.approx(df.loc["Gross Profit", column], rel=1e-6)

    def test_revenue_is_plausible_magnitude(self):
        """
        Guards the shift bug end-to-end: a two-year slip would surface as a
        materially different newest-year revenue.
        """
        df = equities.get_sec_financials("AAPL", "income_statement",
                                         annual=True, periods=1)
        if df.empty:
            pytest.skip("SEC EDGAR unreachable")

        revenue = df.loc["Revenue", df.columns[0]]
        assert 3.5e11 < revenue < 6e11, f"implausible revenue {revenue:,.0f}"

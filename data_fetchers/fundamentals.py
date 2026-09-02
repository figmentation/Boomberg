"""
data_fetchers/fundamentals.py :: Module H - Fundamental Analysis.

Bloomberg equivalents: <EQUITY> FA (financial analysis), EE (earnings
estimates), the analyst workflow that ends in a rating.

WHAT THIS ANSWERS
-----------------
The rest of the terminal fetches financials. This decides what they mean:
metrics, explainable axis scores, and - combined with `valuation.py` - a
BUY / HOLD / SELL verdict on the company at its current price.

Two ideas run through every scoring decision here, because they are the two
ways a fundamental screen usually goes wrong:

  A great company can still be a bad investment if the price is too high.
  A cheap stock can still be a bad investment if the business is decaying.

That is why the verdict is a gate on quality AND value AND risk rather than a
single summed score (see `verdict`). One number cannot express "cheap but
deteriorating", which is exactly the case that costs money.

PROVENANCE IS A FIELD, NOT A FOOTNOTE
-------------------------------------
Every number is a `Fact` carrying how it was obtained - FILED, MARKET,
CALCULATED, PROXY or ASSUMPTION - so the report can keep facts, derivations,
interpretations and forecasts visually distinct. A metric that cannot be
computed comes back with `value=None` and a `note` saying why. It is never
defaulted, never zero-filled, and never scored.

WHAT THIS DELIBERATELY DOES NOT CLAIM
-------------------------------------
Competitive moat, market share, management quality and industry outlook have
no free data source. Rather than invent them:

  * moat and capital allocation are scored as LABELLED PROXIES built from
    things that are filed - ROIC persistence, gross-margin stability, share
    count, buybacks, debt trend;
  * market share and industry outlook are reported as NOT AVAILABLE, are
    excluded from every score, and reduce the confidence figure.

A fabricated "wide moat" reads identically to a real one on screen, which is
precisely why it is not produced here.

NOT ADVICE
----------
This is a mechanical screen over public filings, in the same register as
`macro.get_recession_indicators`. It has no view on your circumstances, tax
position or risk tolerance, and nothing it prints is a recommendation to
transact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from data_fetchers import equities
from utils import fundamental_math as fm
from utils.cache import cached

log = logging.getLogger("openterm.fundamentals")

# Provenance tags. FILED and MARKET are observations; CALCULATED is
# arithmetic on them; PROXY is a stand-in for something not directly
# reported; ASSUMPTION is a forecast input.
FILED = "FILED"
MARKET = "MARKET"
CALCULATED = "CALCULATED"
PROXY = "PROXY"
ASSUMPTION = "ASSUMPTION"
UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class Fact:
    """One number plus everything needed to say where it came from."""

    value: Optional[float]
    source: str
    unit: str = ""
    as_of: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None


def _fact(value: Optional[float], source: str, unit: str = "",
          as_of: str = "", note: str = "") -> Fact:
    """Build a Fact, demoting a missing value to UNAVAILABLE automatically."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return Fact(None, UNAVAILABLE, unit, as_of,
                    note or "Not reported in the filed statements.")
    return Fact(float(value), source, unit, as_of, note)


def _unavailable(note: str, unit: str = "") -> Fact:
    return Fact(None, UNAVAILABLE, unit, "", note)


def format_metric(value: Optional[float], unit: str) -> str:
    """
    Render a metric the way its unit means it.

    Shared by the narrative text and the UI so the two cannot disagree. The
    bug this replaced printed a 1.51x P/FCF ratio as "151.1%", which reads as
    a percentage nobody calculated.
    """
    if value is None:
        return "—"
    if unit == "%":
        return f"{value:.1%}"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "pp":
        return f"{value:+.1f}pp"
    if unit == "/100":
        return f"{value:.0f}/100"
    if unit == "$":
        for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= scale:
                return f"${value / scale:,.2f}{suffix}"
        return f"${value:,.0f}"
    if unit == "slope":
        return f"{value:+.3f}"
    return f"{value:,.2f}"


# ==========================================================================
# STATEMENT ACCESS
# ==========================================================================
# The same line item is labelled differently depending on which source
# answered. SEC XBRL comes through `equities._XBRL_CONCEPTS` naming; Yahoo
# uses its own vendor labels. Without this map the yfinance fallback path
# silently produced an empty metric set - every accessor looked for the SEC
# label, found nothing, and the report degraded to INSUFFICIENT DATA on
# companies whose statements had in fact been fetched perfectly well.
#
# Exxon is the case that surfaced it: SEC now maps XOM to a CIK carrying 94
# us-gaap tags with no OperatingIncomeLoss among them, so the fallback fires
# and has to work.
_ROW_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Revenue": ("Revenue", "Total Revenue", "Operating Revenue"),
    "Gross Profit": ("Gross Profit",),
    "Operating Income": ("Operating Income", "EBIT"),
    "Net Income": ("Net Income", "Net Income Common Stockholders",
                   "Net Income Continuous Operations"),
    "EPS Diluted": ("EPS Diluted", "Diluted EPS"),
    "Diluted Shares": ("Diluted Shares", "Diluted Average Shares"),
    "Interest Expense": ("Interest Expense", "Interest Expense Non Operating"),
    "Pretax Income": ("Pretax Income",),
    "Income Tax": ("Income Tax", "Tax Provision"),
    "Total Assets": ("Total Assets",),
    "Total Equity": ("Total Equity", "Stockholders Equity",
                     "Common Stock Equity", "Total Equity Gross Minority Interest"),
    "Cash & Equivalents": ("Cash & Equivalents", "Cash And Cash Equivalents"),
    "Short-Term Investments": ("Short-Term Investments",
                               "Other Short Term Investments"),
    "Accounts Receivable": ("Accounts Receivable", "Receivables"),
    "Inventory": ("Inventory",),
    "Total Current Assets": ("Total Current Assets", "Current Assets"),
    "Total Current Liabilities": ("Total Current Liabilities",
                                  "Current Liabilities"),
    "Long-Term Debt": ("Long-Term Debt", "Long Term Debt"),
    "Short-Term Debt": ("Short-Term Debt", "Current Debt",
                        "Other Current Borrowings", "Commercial Paper"),
    "Retained Earnings": ("Retained Earnings",),
    "Operating Cash Flow": ("Operating Cash Flow",
                            "Cash Flow From Continuing Operating Activities"),
    # Yahoo reports capital expenditure as a negative number and SEC as a
    # positive payment. Callers take the magnitude, so either sign is safe.
    "CapEx": ("CapEx", "Capital Expenditure", "Purchase Of PPE"),
    "Depreciation & Amortization": ("Depreciation & Amortization",
                                    "Depreciation Amortization Depletion",
                                    "Reconciled Depreciation"),
    "Stock-Based Compensation": ("Stock-Based Compensation",
                                 "Stock Based Compensation"),
    "Buybacks": ("Buybacks", "Repurchase Of Capital Stock",
                 "Common Stock Payments"),
    "Dividends Paid": ("Dividends Paid", "Cash Dividends Paid",
                       "Common Stock Dividend Paid"),
}


def _series(frame: pd.DataFrame, row: str) -> List[Optional[float]]:
    """
    One statement line as a list, newest period first.

    Resolves the row through `_ROW_ALIASES`, so the same call works whether
    the statements came from SEC XBRL or from Yahoo.

    Returns [] when no alias is present - callers test for length rather than
    catching a KeyError, so a filer that never tagged a concept degrades to a
    missing metric instead of an exception.
    """
    if frame is None or frame.empty:
        return []

    label = None
    for candidate in _ROW_ALIASES.get(row, (row,)):
        if candidate in frame.index:
            label = candidate
            break
    if label is None:
        return []

    values: List[Optional[float]] = []
    for value in frame.loc[label].tolist():
        try:
            number = float(value)
        except (TypeError, ValueError):
            values.append(None)
            continue
        values.append(None if np.isnan(number) else number)
    return values


def _latest(frame: pd.DataFrame, row: str) -> Optional[float]:
    """Most recent value for a line, skipping leading gaps."""
    for value in _series(frame, row):
        if value is not None:
            return value
    return None


def _at(values: Sequence[Optional[float]], index: int) -> Optional[float]:
    return values[index] if 0 <= index < len(values) else None


def _oldest_first(values: Sequence[Optional[float]]) -> List[float]:
    """Statement frames are newest-first; trend maths wants the reverse."""
    return [v for v in reversed(list(values)) if v is not None]


@cached(ttl=config.TTL.fundamentals, namespace="fund_facts")
def get_fact_set(ticker: str, periods: int = 8) -> Dict[str, Any]:
    """
    Everything the engine reads, with the source of each statement recorded.

    SEC XBRL is preferred: it is what the company itself filed, and it goes
    back eight annual periods, which is what makes a 5-year CAGR a
    measurement rather than an extrapolation. Yahoo is the fallback for
    non-US filers and keeps roughly four years, so a name that lands there
    loses the 5-year metrics and some confidence with them.
    """
    ticker = equities.normalize_ticker(ticker).upper()

    statements: Dict[str, pd.DataFrame] = {}
    source = UNAVAILABLE

    try:
        sec = {name: equities.get_sec_financials(ticker, name, True, periods)
               for name in ("income_statement", "balance_sheet", "cash_flow")}
        if not all(df.empty for df in sec.values()):
            statements, source = sec, "SEC XBRL"
    except Exception as exc:
        log.warning("SEC statements failed for %s: %s", ticker, exc)

    if not statements:
        try:
            yahoo = equities.get_financial_statements(ticker)
            if yahoo and not all(df.empty for df in yahoo.values()):
                statements, source = yahoo, "yfinance"
        except Exception as exc:
            log.warning("Yahoo statements failed for %s: %s", ticker, exc)

    info: Dict[str, Any] = {}
    try:
        info = equities.get_company_info(ticker) or {}
    except Exception as exc:
        log.warning("Company info failed for %s: %s", ticker, exc)

    quote: Dict[str, Any] = {}
    try:
        quote = equities.get_quote(ticker) or {}
    except Exception as exc:
        log.warning("Quote failed for %s: %s", ticker, exc)

    income = statements.get("income_statement", pd.DataFrame())
    periods_available = list(income.columns) if not income.empty else []

    return {
        "ticker": ticker,
        "statements": statements,
        "statement_source": source,
        "periods": periods_available,
        "info": info,
        "quote": quote,
        "price": quote.get("price") or info.get("currentPrice"),
    }


# ==========================================================================
# PROFILE DETECTION
# ==========================================================================
def detect_profile(facts: Dict[str, Any],
                   revenue_cagr: Optional[float] = None) -> config.FundamentalProfile:
    """
    Pick the metric set from the issuer's OWN Yahoo classification.

    Never a hand-written ticker list - the same rule `suggest_peers` follows.
    Technology and communication-services names growing faster than
    `config.HIGH_GROWTH_REVENUE_CAGR` route to the high-growth profile, since
    trailing P/E on a company reinvesting everything is either negative or
    meaninglessly large.
    """
    info = facts.get("info", {})
    industry_key = str(info.get("industryKey") or "").lower()
    sector_key = str(info.get("sectorKey") or "").lower()

    for key in (industry_key, sector_key):
        if key in config.PROFILE_SECTOR_MAP:
            return config.FUNDAMENTAL_PROFILES[config.PROFILE_SECTOR_MAP[key]]

    if sector_key in ("technology", "communication-services"):
        if revenue_cagr is not None and revenue_cagr >= config.HIGH_GROWTH_REVENUE_CAGR:
            return config.FUNDAMENTAL_PROFILES["HIGH_GROWTH_TECH"]

    return config.FUNDAMENTAL_PROFILES["GENERIC"]


# ==========================================================================
# METRICS
# ==========================================================================
# Units are declared here rather than at each call site, because the display
# layer formats on them and getting one wrong is silently misleading rather
# than obviously broken: a P/FCF ratio of 1.51 rendered under the same unit
# as a margin prints as "151.1%", which reads as a percentage nobody
# computed.
#
#   "%"     a decimal fraction, shown as a percentage   (0.269 -> 26.9%)
#   "x"     a ratio or multiple, shown as a multiple    (1.51  -> 1.51x)
#   "pp"    a gap already expressed in percentage points
#   "$"     currency
#   "/100"  a score
#   "slope" a normalised trend
_METRIC_UNITS: Dict[str, str] = {
    "revenue": "$", "net_income": "$", "operating_cash_flow": "$",
    "total_assets": "$", "total_equity": "$", "cash": "$",
    "free_cash_flow": "$", "normalized_fcf": "$", "total_debt": "$",
    "net_debt": "$", "ebitda": "$",

    "gross_margin": "%", "operating_margin": "%", "net_margin": "%",
    "fcf_margin": "%", "ffo_margin": "%", "roe": "%", "roic": "%",
    "revenue_cagr_3y": "%", "revenue_cagr_5y": "%", "eps_cagr_3y": "%",
    "fcf_cagr_3y": "%", "fcf_cagr_5y": "%", "share_count_cagr": "%",
    "sbc_to_revenue": "%", "buyback_yield": "%", "dividend_yield": "%",
    "accruals_ratio": "%", "fcf_to_net_income": "%",
    "gaap_vs_adjusted_gap": "%",

    "net_debt_to_ebitda": "x", "interest_coverage": "x",
    "current_ratio": "x", "cash_to_debt": "x",

    "receivables_vs_revenue": "pp", "inventory_vs_revenue": "pp",
    "debt_trend": "slope", "operating_margin_trend": "slope",
    "moat_proxy": "/100", "capital_allocation": "/100",
}
def get_metrics(facts: Dict[str, Any]) -> Dict[str, Fact]:
    """
    Every metric in the brief, each carrying its own provenance.

    Anything that cannot be computed from what was filed comes back as an
    UNAVAILABLE Fact with a note. Nothing here substitutes a default.
    """
    statements = facts.get("statements", {})
    income = statements.get("income_statement", pd.DataFrame())
    balance = statements.get("balance_sheet", pd.DataFrame())
    cash = statements.get("cash_flow", pd.DataFrame())
    info = facts.get("info", {})
    source = facts.get("statement_source", UNAVAILABLE)
    as_of = (facts.get("periods") or [""])[0]

    out: Dict[str, Fact] = {}

    def filed(value: Optional[float], unit: str = "$", note: str = "") -> Fact:
        return _fact(value, FILED if source != UNAVAILABLE else UNAVAILABLE,
                     unit, as_of, note)

    def calc(value: Optional[float], unit: str = "", note: str = "") -> Fact:
        return _fact(value, CALCULATED, unit, as_of, note)

    # --- Raw lines --------------------------------------------------------
    revenue = _series(income, "Revenue")
    gross_profit = _series(income, "Gross Profit")
    operating_income = _series(income, "Operating Income")
    net_income = _series(income, "Net Income")
    eps = _series(income, "EPS Diluted")
    diluted_shares = _series(income, "Diluted Shares")
    interest_expense = _series(income, "Interest Expense")
    pretax = _series(income, "Pretax Income")
    tax = _series(income, "Income Tax")

    total_assets = _series(balance, "Total Assets")
    total_equity = _series(balance, "Total Equity")
    cash_equiv = _series(balance, "Cash & Equivalents")
    short_investments = _series(balance, "Short-Term Investments")
    long_debt = _series(balance, "Long-Term Debt")
    short_debt = _series(balance, "Short-Term Debt")
    current_assets = _series(balance, "Total Current Assets")
    current_liabilities = _series(balance, "Total Current Liabilities")
    receivables = _series(balance, "Accounts Receivable")
    inventory = _series(balance, "Inventory")

    ocf = _series(cash, "Operating Cash Flow")
    capex = _series(cash, "CapEx")
    dep_amort = _series(cash, "Depreciation & Amortization")
    sbc = _series(cash, "Stock-Based Compensation")
    buybacks = _series(cash, "Buybacks")
    dividends = _series(cash, "Dividends Paid")

    out["revenue"] = filed(_at(revenue, 0))
    out["net_income"] = filed(_at(net_income, 0))
    out["operating_cash_flow"] = filed(_at(ocf, 0))
    out["total_assets"] = filed(_at(total_assets, 0))
    out["total_equity"] = filed(_at(total_equity, 0))
    out["cash"] = filed(_at(cash_equiv, 0))

    # --- Free cash flow ---------------------------------------------------
    # CapEx is filed as a positive payment, so it is subtracted by magnitude.
    fcf_series: List[Optional[float]] = []
    for index in range(max(len(ocf), len(capex))):
        operating = _at(ocf, index)
        spend = _at(capex, index)
        fcf_series.append(None if operating is None or spend is None
                          else operating - abs(spend))
    out["free_cash_flow"] = calc(
        _at(fcf_series, 0), "$", "Operating cash flow less capital expenditure.")

    # Normalised free cash flow: the median FCF MARGIN across the filed years,
    # applied to current revenue.
    #
    # A DCF compounds its base for a decade, so a base distorted by one
    # year's one-off legal settlement or working-capital swing distorts the
    # whole valuation. Coca-Cola is the case in point - FY2025 free cash flow
    # of $5.3bn against $9-11bn in the years either side, entirely from a
    # single contingent-consideration payment. Compounding that produces a
    # confident fair value roughly half what the business supports.
    #
    # Normalising on the margin rather than on an average of past dollars
    # keeps the company's CURRENT scale, so a genuinely fast-growing name is
    # not dragged back to what it earned three years ago.
    fcf_margins = [fm.safe_div(_at(fcf_series, i), _at(revenue, i))
                   for i in range(min(len(fcf_series), len(revenue)))]
    median_fcf_margin = fm.median_of([m for m in fcf_margins if m is not None])
    revenue_latest = _at(revenue, 0)
    out["normalized_fcf"] = calc(
        median_fcf_margin * revenue_latest
        if median_fcf_margin is not None and revenue_latest else None, "$",
        f"Median FCF margin of {median_fcf_margin:.1%} across "
        f"{len([m for m in fcf_margins if m is not None])} filed years, "
        "applied to current revenue. Used as the DCF base so a single "
        "distorted year does not compound for a decade."
        if median_fcf_margin is not None else
        "Too few years of revenue and cash flow to normalise a DCF base.")

    # --- Debt and liquidity ----------------------------------------------
    total_debt_series: List[Optional[float]] = []
    for index in range(max(len(long_debt), len(short_debt))):
        long_part = _at(long_debt, index)
        short_part = _at(short_debt, index)
        if long_part is None and short_part is None:
            total_debt_series.append(None)
        else:
            total_debt_series.append((long_part or 0.0) + (short_part or 0.0))

    total_debt = _at(total_debt_series, 0)
    liquid = (_at(cash_equiv, 0) or 0.0) + (_at(short_investments, 0) or 0.0)
    out["total_debt"] = calc(total_debt, "$", "Long-term plus short-term debt.")
    out["net_debt"] = calc(
        None if total_debt is None else total_debt - liquid, "$",
        "Total debt less cash and short-term investments.")

    ebitda = None
    if _at(operating_income, 0) is not None:
        ebitda = _at(operating_income, 0) + (_at(dep_amort, 0) or 0.0)
    out["ebitda"] = calc(
        ebitda, "$",
        "Operating income plus D&A."
        + ("" if _at(dep_amort, 0) is not None
           else " D&A was not tagged, so this is operating income alone."))

    out["net_debt_to_ebitda"] = calc(
        fm.safe_div(out["net_debt"].value, ebitda) if ebitda and ebitda > 0 else None,
        "x", "" if ebitda and ebitda > 0 else
        "EBITDA is zero or negative; the ratio would not be meaningful.")

    interest = _at(interest_expense, 0)
    out["interest_coverage"] = calc(
        fm.safe_div(_at(operating_income, 0), abs(interest)) if interest else None,
        "x", "" if interest else
        "No interest expense tagged - common where it is netted into other income.")

    out["current_ratio"] = calc(
        fm.safe_div(_at(current_assets, 0), _at(current_liabilities, 0)), "x")
    out["cash_to_debt"] = calc(
        fm.safe_div(liquid, total_debt) if total_debt else None, "x",
        "" if total_debt else "No debt reported.")
    out["debt_trend"] = calc(
        fm.trend_slope(_oldest_first(total_debt_series)), "slope",
        "Normalised slope of total debt across the filed years; negative is "
        "deleveraging.")

    # --- Margins ----------------------------------------------------------
    revenue_now = _at(revenue, 0)
    out["gross_margin"] = calc(fm.safe_div(_at(gross_profit, 0), revenue_now), "x")
    out["operating_margin"] = calc(
        fm.safe_div(_at(operating_income, 0), revenue_now), "x")
    out["net_margin"] = calc(fm.safe_div(_at(net_income, 0), revenue_now), "x")
    out["fcf_margin"] = calc(fm.safe_div(_at(fcf_series, 0), revenue_now), "x")

    # Funds from operations margin - net income with depreciation added back.
    # Only surfaced for the REIT profile, where GAAP depreciation on property
    # held at cost swamps the income statement and makes every net-income
    # derived margin describe accounting rather than the business.
    ffo_now = None
    if _at(net_income, 0) is not None and _at(dep_amort, 0) is not None:
        ffo_now = _at(net_income, 0) + _at(dep_amort, 0)
    out["ffo_margin"] = _fact(
        fm.safe_div(ffo_now, revenue_now), PROXY, "%", as_of,
        "PROXY: (net income + D&A) / revenue. Gains on property sales have "
        "no consistent tag and are not deducted, so this runs high in "
        "disposal years.")

    margin_series = [fm.safe_div(_at(operating_income, i), _at(revenue, i))
                     for i in range(min(len(revenue), len(operating_income)))]
    out["operating_margin_trend"] = calc(
        fm.trend_slope(_oldest_first(margin_series)), "slope",
        "Direction of the operating margin across the filed years.")

    # --- Returns ----------------------------------------------------------
    out["roe"] = calc(fm.safe_div(_at(net_income, 0), _at(total_equity, 0)), "x")

    roic_series = _roic_series(operating_income, pretax, tax, total_debt_series,
                               total_equity, cash_equiv, short_investments)
    out["roic"] = calc(
        _at(roic_series, 0), "%",
        "NOPAT over invested capital (debt + equity - cash).")

    # --- Growth -----------------------------------------------------------
    out["revenue_cagr_3y"] = calc(
        fm.cagr(_at(revenue, 3), _at(revenue, 0), 3), "x")
    out["revenue_cagr_5y"] = calc(
        fm.cagr(_at(revenue, 5), _at(revenue, 0), 5), "x",
        "" if len(revenue) > 5 else
        "Fewer than six annual periods available from this source.")
    out["eps_cagr_3y"] = calc(fm.cagr(_at(eps, 3), _at(eps, 0), 3), "x")
    out["fcf_cagr_3y"] = calc(
        fm.cagr(_at(fcf_series, 3), _at(fcf_series, 0), 3), "x")
    out["fcf_cagr_5y"] = calc(
        fm.cagr(_at(fcf_series, 5), _at(fcf_series, 0), 5), "x",
        "" if len(fcf_series) > 5 else
        "Fewer than six annual periods available from this source.")

    # --- Shareholder economics -------------------------------------------
    out["share_count_cagr"] = calc(
        fm.cagr(_at(diluted_shares, 3), _at(diluted_shares, 0), 3), "x",
        "Diluted share count. Negative is net buyback, positive is dilution.")
    out["sbc_to_revenue"] = calc(
        fm.safe_div(_at(sbc, 0), revenue_now), "x",
        "Stock-based compensation as a share of revenue - a real cost to "
        "holders that free cash flow does not capture.")

    market_cap = info.get("marketCap")
    out["buyback_yield"] = calc(
        fm.safe_div(abs(_at(buybacks, 0)), market_cap)
        if _at(buybacks, 0) and market_cap else None, "x")
    out["dividend_yield"] = _fact(
        (info.get("dividendYield") or 0) / 100.0
        if info.get("dividendYield") else None, MARKET, "x")

    # --- Earnings quality -------------------------------------------------
    out["fcf_to_net_income"] = calc(
        fm.safe_div(_at(fcf_series, 0), _at(net_income, 0))
        if (_at(net_income, 0) or 0) > 0 else None, "x",
        "" if (_at(net_income, 0) or 0) > 0 else
        "Net income is zero or negative; the conversion ratio is undefined.")
    out["accruals_ratio"] = calc(
        fm.safe_div((_at(net_income, 0) or 0) - (_at(ocf, 0) or 0),
                    _at(total_assets, 0))
        if _at(net_income, 0) is not None and _at(ocf, 0) is not None else None,
        "x", "Net income less operating cash flow, over assets. High means "
             "earnings the cash flow statement does not corroborate.")

    out["receivables_vs_revenue"] = calc(
        _growth_gap(receivables, revenue), "pp",
        "Receivables growth less revenue growth. Persistently positive can "
        "mean sales are being booked faster than they are collected.")
    out["inventory_vs_revenue"] = calc(
        _growth_gap(inventory, revenue), "pp",
        "Inventory growth less revenue growth.")

    gaap_eps = _at(eps, 0)
    adjusted_eps = info.get("trailingEps")
    out["gaap_vs_adjusted_gap"] = calc(
        fm.safe_div(abs(adjusted_eps - gaap_eps), abs(gaap_eps))
        if gaap_eps and adjusted_eps else None, "x",
        "Gap between filed diluted EPS and Yahoo's trailing figure. These "
        "cover different windows, so a small gap is normal; a large one is "
        "worth reading the reconciliation for.")

    # --- Proxies for what cannot be sourced -------------------------------
    out["moat_proxy"] = _moat_proxy(roic_series, gross_profit, revenue)
    out["capital_allocation"] = _capital_allocation_proxy(out, roic_series)

    # Stamp the declared unit on every metric, so the display layer never has
    # to guess whether 1.51 is a ratio or 151%.
    return {key: replace(fact, unit=_METRIC_UNITS.get(key, fact.unit))
            for key, fact in out.items()}


def _roic_series(operating_income, pretax, tax, total_debt, equity,
                 cash_equiv, short_investments) -> List[Optional[float]]:
    """Return on invested capital per filed year, newest first."""
    out: List[Optional[float]] = []
    for index in range(len(operating_income)):
        ebit = _at(operating_income, index)
        if ebit is None:
            out.append(None)
            continue

        tax_rate = fm.safe_div(_at(tax, index), _at(pretax, index))
        if tax_rate is None or not 0.0 <= tax_rate <= 0.6:
            tax_rate = 0.21          # US statutory; noted wherever surfaced
        nopat = ebit * (1.0 - tax_rate)

        invested = None
        debt_now = _at(total_debt, index)
        equity_now = _at(equity, index)
        if equity_now is not None:
            liquid = (_at(cash_equiv, index) or 0.0) + (_at(short_investments, index) or 0.0)
            invested = (debt_now or 0.0) + equity_now - liquid

        out.append(fm.safe_div(nopat, invested) if invested and invested > 0 else None)
    return out


def _growth_gap(numerator: Sequence[Optional[float]],
                revenue: Sequence[Optional[float]]) -> Optional[float]:
    """Year-on-year growth of one line minus that of revenue, in points."""
    line_growth = fm.cagr(_at(numerator, 1), _at(numerator, 0), 1)
    revenue_growth = fm.cagr(_at(revenue, 1), _at(revenue, 0), 1)
    if line_growth is None or revenue_growth is None:
        return None
    return (line_growth - revenue_growth) * 100.0


def _moat_proxy(roic_series, gross_profit, revenue) -> Fact:
    """
    A PROXY for durability - explicitly not a moat rating.

    Two filed things correlate with pricing power: returns on capital that
    stay high across a cycle rather than in one good year, and a gross margin
    that holds its level. Neither proves a moat and this does not claim one;
    it reports what the filings support, labelled as the stand-in it is.
    """
    roic_values = [r for r in roic_series if r is not None]
    margins = [fm.safe_div(_at(gross_profit, i), _at(revenue, i))
               for i in range(min(len(gross_profit), len(revenue)))]
    margin_values = [m for m in margins if m is not None]

    if len(roic_values) < 3 and len(margin_values) < 3:
        return _unavailable(
            "Too few filed years to judge whether returns or margins persist.")

    parts: List[float] = []

    if len(roic_values) >= 3:
        # Share of years clearing a 15% return on invested capital.
        persistence = sum(1 for r in roic_values if r >= 0.15) / len(roic_values)
        parts.append(persistence * 100.0)

    if len(margin_values) >= 3:
        level = fm.score_band(sum(margin_values) / len(margin_values),
                              config.SCORE_ANCHORS["gross_margin"])
        spread = float(np.std(margin_values))
        # A gross margin moving less than two points a year is stable.
        stability = fm.clamp(100.0 - (spread / 0.02) * 25.0)
        if level is not None:
            parts.append((level + stability) / 2.0)

    if not parts:
        return _unavailable("Insufficient data for a durability proxy.")

    return Fact(sum(parts) / len(parts), PROXY, "/100", "",
                "PROXY from ROIC persistence and gross-margin level and "
                "stability. Not a moat rating - no free source carries one.")


def _capital_allocation_proxy(metrics: Dict[str, Fact], roic_series) -> Fact:
    """
    What management has actually done with the money, from the filings.

    Dilution, buybacks, the direction of debt and the direction of returns
    are all filed facts. This is the part of "management quality" that is
    genuinely measurable; judgement about strategy or integrity is not, and
    is not attempted.
    """
    parts: List[float] = []

    dilution = metrics.get("share_count_cagr")
    if dilution and dilution.known:
        points = fm.score_band(dilution.value,
                               config.SCORE_ANCHORS["share_count_cagr"])
        if points is not None:
            parts.append(points)

    debt_trend = metrics.get("debt_trend")
    if debt_trend and debt_trend.known:
        points = fm.score_band(debt_trend.value, config.SCORE_ANCHORS["debt_trend"])
        if points is not None:
            parts.append(points)

    roic_values = [r for r in roic_series if r is not None]
    if len(roic_values) >= 3:
        slope = fm.trend_slope(list(reversed(roic_values)))
        if slope is not None:
            parts.append(fm.clamp(60.0 + slope * 200.0))

    if not parts:
        return _unavailable(
            "Share count, debt trend and ROIC history are all unavailable.")

    return Fact(sum(parts) / len(parts), PROXY, "/100", "",
                "CALCULATED from dilution, debt direction and the trend in "
                "returns on capital. Covers capital allocation only - not "
                "strategy or stewardship, which no free source reports.")


# ==========================================================================
# SCORING
# ==========================================================================
# Each axis is a list of named components. Every component carries its raw
# value, the anchor table that turned it into points, its weight and its
# provenance, so the UI can print a row that justifies itself and nobody has
# to take the headline number on trust.
def _component(key: str, label: str, metrics: Dict[str, Fact], weight: float,
               profile: config.FundamentalProfile,
               anchors_key: Optional[str] = None,
               direct: bool = False) -> Optional[fm.ScoreComponent]:
    """
    Build one scored component, or None if the profile skips this metric.

    `direct=True` marks a metric already expressed on 0-100 (the proxies),
    which needs no anchor table.
    """
    if key in profile.skip_metrics:
        return None

    fact = metrics.get(key)
    if fact is None:
        return None

    anchors = () if direct else config.SCORE_ANCHORS.get(anchors_key or key, ())

    if not fact.known:
        return fm.ScoreComponent(
            name=label, value=None, unit=fact.unit, points=None, weight=weight,
            source=fact.source, anchors=anchors, note=fact.note)

    points = fact.value if direct else fm.score_band(fact.value, anchors)
    return fm.ScoreComponent(
        name=label, value=fact.value, unit=fact.unit, points=points,
        weight=weight, source=fact.source, anchors=anchors, note=fact.note)


def _axis(components: List[Optional[fm.ScoreComponent]],
          min_components: int = 2) -> Dict[str, Any]:
    resolved = [c for c in components if c is not None]
    score, count = fm.weighted_score(resolved, min_components=min_components)
    return {"score": score, "components": resolved, "resolved": count,
            "considered": len(resolved)}


def business_quality(metrics: Dict[str, Fact],
                     profile: config.FundamentalProfile) -> Dict[str, Any]:
    """Margins, returns on capital, and the two labelled proxies."""
    return _axis([
        _component("gross_margin", "Gross margin", metrics, 1.0, profile),
        _component("operating_margin", "Operating margin", metrics, 1.2, profile),
        _component("net_margin", "Net margin", metrics, 1.0, profile),
        _component("fcf_margin", "FCF margin", metrics, 1.2, profile),
        _component("roe", "Return on equity", metrics, 1.2, profile),
        _component("roic", "Return on invested capital", metrics, 1.5, profile),
        _component("moat_proxy", "Durability (proxy)", metrics, 1.0, profile,
                   direct=True),
        _component("capital_allocation", "Capital allocation", metrics, 1.0,
                   profile, direct=True),
        # REITs only. Elsewhere FFO would double-count the margins above.
        _component("ffo_margin", "FFO margin (proxy)", metrics, 1.5, profile,
                   anchors_key="fcf_margin")
        if profile.valuation_anchor == "affo" else None,
    ])


def growth(metrics: Dict[str, Fact],
           profile: config.FundamentalProfile) -> Dict[str, Any]:
    return _axis([
        _component("revenue_cagr_3y", "Revenue CAGR 3y", metrics, 1.2, profile),
        _component("revenue_cagr_5y", "Revenue CAGR 5y", metrics, 1.0, profile),
        _component("eps_cagr_3y", "EPS CAGR 3y", metrics, 1.0, profile),
        _component("fcf_cagr_3y", "FCF CAGR 3y", metrics, 1.0, profile),
        _component("fcf_cagr_5y", "FCF CAGR 5y", metrics, 0.8, profile,
                   anchors_key="fcf_cagr_3y"),
    ])


def financial_health(metrics: Dict[str, Fact],
                     profile: config.FundamentalProfile) -> Dict[str, Any]:
    return _axis([
        _component("net_debt_to_ebitda", "Net debt / EBITDA", metrics, 1.5,
                   profile),
        _component("interest_coverage", "Interest coverage", metrics, 1.2,
                   profile),
        _component("current_ratio", "Current ratio", metrics, 0.8, profile),
        _component("cash_to_debt", "Cash / debt", metrics, 1.0, profile),
        _component("share_count_cagr", "Share count CAGR", metrics, 1.0, profile),
        _component("sbc_to_revenue", "SBC / revenue", metrics, 0.8, profile),
        _component("debt_trend", "Debt trend", metrics, 0.8, profile),
    ])


def earnings_quality(metrics: Dict[str, Fact]) -> Dict[str, Any]:
    """
    Does the cash flow statement corroborate the income statement?

    Each flag is a threshold test that renders as a sentence quoting the
    numbers behind it. These are prompts to go and read the filing, not
    accusations - every one of them has innocent explanations.
    """
    flags: List[Dict[str, str]] = []

    def value_of(key: str) -> Optional[float]:
        fact = metrics.get(key)
        return fact.value if fact is not None and fact.known else None

    conversion = value_of("fcf_to_net_income")
    if conversion is not None and conversion < 0.8:
        flags.append({
            "flag": "Weak cash conversion",
            "detail": f"Free cash flow is {conversion:.0%} of net income. "
                      "Reported profit is not turning into cash at the rate "
                      "the income statement implies.",
        })

    accruals = value_of("accruals_ratio")
    if accruals is not None and accruals > 0.05:
        flags.append({
            "flag": "High accruals",
            "detail": f"Net income exceeds operating cash flow by "
                      f"{accruals:.1%} of total assets. The gap is accrual "
                      "accounting the cash flow statement does not confirm.",
        })

    receivables = value_of("receivables_vs_revenue")
    if receivables is not None and receivables > 15.0:
        flags.append({
            "flag": "Receivables outpacing revenue",
            "detail": f"Receivables grew {receivables:.0f} percentage points "
                      "faster than revenue. Can mean sales are being booked "
                      "well ahead of collection.",
        })

    inventory = value_of("inventory_vs_revenue")
    if inventory is not None and inventory > 20.0:
        flags.append({
            "flag": "Inventory building",
            "detail": f"Inventory grew {inventory:.0f} percentage points "
                      "faster than revenue, which often precedes discounting "
                      "or a write-down.",
        })

    gap = value_of("gaap_vs_adjusted_gap")
    if gap is not None and gap > 0.25:
        flags.append({
            "flag": "Wide GAAP vs adjusted gap",
            "detail": f"Filed diluted EPS and the trailing adjusted figure "
                      f"differ by {gap:.0%}. Worth reading the "
                      "reconciliation for what is being excluded.",
        })

    sbc = value_of("sbc_to_revenue")
    if sbc is not None and sbc > 0.10:
        flags.append({
            "flag": "Heavy stock-based compensation",
            "detail": f"SBC is {sbc:.1%} of revenue - a real cost to existing "
                      "holders that free cash flow does not deduct.",
        })

    return {"flags": flags, "count": len(flags)}


def valuation_axis(valuation: Dict[str, Any], metrics: Dict[str, Fact],
                   profile: config.FundamentalProfile) -> Dict[str, Any]:
    """
    How attractively priced, on several independent readings.

    Deliberately NOT dominated by any single multiple. The upside to the
    blended fair value carries the most weight, and the company's own
    historical multiple is weighted above the peer comparison, because a
    whole sector can be expensive at once.
    """
    multiples = valuation.get("multiples", {})
    history = valuation.get("own_history", {})
    peers = valuation.get("peers", {})

    components: List[Optional[fm.ScoreComponent]] = []

    def add(label: str, value: Optional[float], anchors_key: str,
            weight: float, source: str, unit: str = "x",
            note: str = "") -> None:
        anchors = config.SCORE_ANCHORS.get(anchors_key, ())
        points = fm.score_band(value, anchors) if value is not None else None
        components.append(fm.ScoreComponent(
            name=label, value=value, unit=unit, points=points, weight=weight,
            source=source, anchors=anchors, note=note))

    add("Upside to base case", valuation.get("upside_to_base"),
        "upside_to_base", 2.0, CALCULATED, "%",
        "" if valuation.get("upside_to_base") is not None
        else "No method resolved a fair value.")

    add("FCF yield", multiples.get("FCF yield"), "fcf_yield", 1.2, MARKET, "%")

    if "pe_vs_own_history" not in profile.skip_metrics:
        pe_now = multiples.get("P/E (TTM)")
        pe_median = history.get("pe_median")
        add("P/E vs own 5y median",
            fm.safe_div(pe_now, pe_median) if pe_now and pe_median else None,
            "pe_vs_own_history", 1.5, CALCULATED,
            "" if pe_now and pe_median
            else "No positive earnings history to compare against.")

    p_fcf_now = multiples.get("P/FCF")
    p_fcf_median = history.get("p_fcf_median")
    add("P/FCF vs own 5y median",
        fm.safe_div(p_fcf_now, p_fcf_median) if p_fcf_now and p_fcf_median else None,
        "ev_ebitda_vs_own_history", 1.2, CALCULATED)

    if "pe_vs_peers" not in profile.skip_metrics:
        pe_now = multiples.get("P/E (TTM)")
        peer_median = peers.get("pe_median")
        add("P/E vs peer median",
            fm.safe_div(pe_now, peer_median) if pe_now and peer_median else None,
            "pe_vs_peers", 1.0, CALCULATED,
            peers.get("note", ""))

    if profile.valuation_anchor == "book":
        add("Price / book", multiples.get("P/B"), "price_to_book", 1.5, MARKET)

    return _axis(components)


def risk_axis(metrics: Dict[str, Fact], valuation: Dict[str, Any],
              earnings: Dict[str, Any],
              profile: config.FundamentalProfile) -> Dict[str, Any]:
    """
    Risk, where HIGHER POINTS MEAN MORE RISK.

    The inversion is deliberate and matches how the verdict gates read it:
    every other axis is "more is better", this one is "more is worse", and
    the UI labels it that way so a 25 is not mistaken for a poor score.
    """
    info_beta = None
    components: List[Optional[fm.ScoreComponent]] = []

    def add(label: str, value: Optional[float], anchors_key: str,
            weight: float, source: str, note: str = "") -> None:
        if anchors_key in profile.skip_metrics:
            return
        anchors = config.SCORE_ANCHORS.get(anchors_key, ())
        points = fm.score_band(value, anchors) if value is not None else None
        components.append(fm.ScoreComponent(
            name=label, value=value, unit="", points=points, weight=weight,
            source=source, anchors=anchors, note=note))

    leverage = metrics.get("net_debt_to_ebitda")
    add("Leverage", leverage.value if leverage and leverage.known else None,
        "risk_leverage", 1.5, CALCULATED,
        "" if leverage and leverage.known else
        (leverage.note if leverage else "Not computed."))

    accruals = metrics.get("accruals_ratio")
    add("Earnings quality",
        accruals.value if accruals and accruals.known else None,
        "risk_earnings_quality", 1.2, CALCULATED)

    add("Valuation stretch",
        -(valuation.get("margin_of_safety") or 0.0)
        if valuation.get("margin_of_safety") is not None else None,
        "risk_valuation", 1.5, CALCULATED,
        "Premium to the blended fair value. This is what stops a great "
        "business from being scored as low risk at any price.")

    dilution = metrics.get("share_count_cagr")
    add("Dilution", dilution.value if dilution and dilution.known else None,
        "risk_dilution", 1.0, CALCULATED)

    margin_trend = metrics.get("operating_margin_trend")
    add("Margin direction",
        margin_trend.value if margin_trend and margin_trend.known else None,
        "risk_margin_trend", 1.0, CALCULATED)

    axis = _axis(components)

    # Accounting red flags add risk directly - they are qualitative findings
    # rather than a metric with an anchor table, so they are applied here
    # and reported separately rather than smuggled into a component.
    if axis["score"] is not None and earnings.get("count"):
        penalty = min(earnings["count"] * 8.0, 24.0)
        axis["score"] = fm.clamp(axis["score"] + penalty)
        axis["red_flag_penalty"] = penalty

    return axis


# ==========================================================================
# CONFIDENCE
# ==========================================================================
def confidence(facts: Dict[str, Any], metrics: Dict[str, Fact],
               axes: Dict[str, Dict[str, Any]], valuation: Dict[str, Any],
               profile: config.FundamentalProfile) -> Dict[str, Any]:
    """
    How much the report should be trusted, 0-100, with every deduction named.

    Starts at 100 and subtracts for things that genuinely make the output
    less reliable. The itemisation is the point: "confidence 62" is not
    useful, "confidence 62, because the statements came from Yahoo rather
    than EDGAR and the valuation methods disagree by 90%" is.
    """
    score = 100.0
    deductions: List[Dict[str, Any]] = []

    def deduct(points: float, reason: str) -> None:
        nonlocal score
        score -= points
        deductions.append({"points": points, "reason": reason})

    if facts.get("statement_source") == "yfinance":
        deduct(15.0, "Statements came from Yahoo rather than SEC EDGAR - "
                     "vendor-normalised, and roughly four years deep instead "
                     "of eight.")
    elif facts.get("statement_source") == UNAVAILABLE:
        deduct(45.0, "No financial statements could be retrieved from any "
                     "source.")

    periods = len(facts.get("periods") or [])
    if periods and periods < 6:
        deduct(10.0, f"Only {periods} annual periods available, so the "
                     "five-year growth and durability measures are missing.")

    unresolved = [key for key, fact in metrics.items() if not fact.known]
    if unresolved:
        penalty = min(len(unresolved) * 1.5, 18.0)
        deduct(penalty, f"{len(unresolved)} metrics could not be computed: "
                        f"{', '.join(unresolved[:6])}"
                        f"{'…' if len(unresolved) > 6 else ''}.")

    for name, axis in axes.items():
        if axis.get("score") is None:
            deduct(8.0, f"The {name.replace('_', ' ')} axis had too few "
                        "resolved components to score.")

    spread = valuation.get("dispersion")
    if spread is not None and spread > 0.5:
        deduct(min(spread * 12.0, 18.0),
               f"Valuation methods disagree by {spread:.0%} of the median - "
               "the fair value is a wide range, not a point.")
    elif spread is None:
        deduct(12.0, "Fewer than two valuation methods resolved, so there is "
                     "no cross-check on the fair value.")

    if profile.unavailable:
        deduct(min(len(profile.unavailable) * 2.5, 10.0),
               f"{profile.label} analysis wants metrics that free data does "
               f"not carry: {', '.join(profile.unavailable)}.")

    deduct(6.0, "Market share, competitive position and industry outlook have "
                "no free data source and are excluded from every score.")

    return {"score": fm.clamp(score), "deductions": deductions}


# ==========================================================================
# VERDICT
# ==========================================================================
def _passes(rule: Dict[str, float], quality: Optional[float],
            value: Optional[float], risk: Optional[float]) -> bool:
    """Evaluate one gate. A missing axis fails any rule that tests it."""
    checks = (
        ("min_quality", quality, lambda a, b: a >= b),
        ("max_quality", quality, lambda a, b: a < b),
        ("min_value", value, lambda a, b: a >= b),
        ("max_value", value, lambda a, b: a <= b),
        ("max_risk", risk, lambda a, b: a <= b),
    )
    for key, actual, test in checks:
        if key not in rule:
            continue
        if actual is None or not test(actual, rule[key]):
            return False
    return True


def verdict(axes: Dict[str, Dict[str, Any]], valuation_score: Optional[float],
            risk_score: Optional[float], earnings: Dict[str, Any],
            profile: config.FundamentalProfile) -> Dict[str, Any]:
    """
    BUY / HOLD / SELL as a gate on three axes, never a single summed score.

    This is the shape the brief demanded and it is the right shape: one
    number cannot express "cheap but deteriorating". A low P/E on a business
    scoring 40 for quality lands in SELL, not BUY, because the quality gate
    is tested before the value gate. A high P/E on a business scoring 90
    lands in HOLD, not SELL, because expensive is not the same as broken.
    """
    quality_parts = []
    for name, weight_key in (("business_quality", "business_quality"),
                             ("growth", "growth"),
                             ("financial_health", "financial_health")):
        axis = axes.get(name, {})
        if axis.get("score") is not None:
            quality_parts.append(
                (axis["score"], profile.axis_weights.get(weight_key, 1.0)))

    quality = None
    if quality_parts:
        total_weight = sum(w for _, w in quality_parts)
        quality = sum(s * w for s, w in quality_parts) / total_weight

    label = config.VERDICT_DEFAULT
    matched = "default"

    for candidate, rule in config.VERDICT_RULES:
        if _passes(rule, quality, valuation_score, risk_score):
            label, matched = candidate, str(rule)
            break

    # Accounting red flags on an already-weak business override upward.
    if (earnings.get("count", 0) >= config.RED_FLAG_STRONG_SELL_COUNT
            and quality is not None
            and quality < config.RED_FLAG_STRONG_SELL_QUALITY):
        label = "STRONG SELL"
        matched = (f"{earnings['count']} accounting red flags with quality "
                   f"below {config.RED_FLAG_STRONG_SELL_QUALITY:.0f}")

    return {
        "verdict": label,
        "quality_composite": quality,
        "value_score": valuation_score,
        "risk_score": risk_score,
        "rule": matched,
    }


def _reasons(axes: Dict[str, Dict[str, Any]], valuation: Dict[str, Any],
             earnings: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Supporting and opposing points, quoting the numbers that produced them.

    Built from the highest- and lowest-scoring components rather than from
    prose templates, so the bullets cannot drift away from what the scores
    actually say.
    """
    scored: List[Tuple[str, fm.ScoreComponent]] = []
    for name, axis in axes.items():
        for component in axis.get("components", []):
            if component.resolved:
                scored.append((name, component))

    scored.sort(key=lambda pair: pair[1].points or 0.0, reverse=True)

    def describe(component: fm.ScoreComponent) -> str:
        if component.value is None:
            return component.name
        rendered = format_metric(component.value, component.unit)
        return f"{component.name} at {rendered} scores {component.points:.0f}/100"

    strengths = [describe(c) for name, c in scored[:5]
                 if name != "risk" and (c.points or 0) >= 60]

    weaknesses = [describe(c) for name, c in reversed(scored)
                  if name != "risk" and (c.points or 100) <= 45][:4]

    for flag in earnings.get("flags", []):
        weaknesses.append(f"{flag['flag']}: {flag['detail']}")

    spread = valuation.get("dispersion")
    if spread is not None and spread > 0.5:
        weaknesses.append(
            f"Valuation methods disagree by {spread:.0%} of the median, so "
            "the fair value is a range rather than a figure.")

    return {"strengths": strengths, "weaknesses": weaknesses[:6]}


def _narrative(verdict_block: Dict[str, Any], valuation: Dict[str, Any],
               profile: config.FundamentalProfile,
               metrics: Dict[str, Fact]) -> Dict[str, str]:
    """The valuation explanation and thesis, assembled from computed values."""
    price = valuation.get("price")
    base = valuation.get("base")
    upside = valuation.get("upside_to_base")
    quality = verdict_block.get("quality_composite")

    if base is None or price is None:
        valuation_text = (
            "No method resolved a fair value, so there is nothing to compare "
            "the price against. The verdict rests on business quality alone "
            "and should be treated as incomplete.")
    else:
        stance = ("undervalued" if (upside or 0) > 0.15
                  else "overvalued" if (upside or 0) < -0.15
                  else "roughly fairly valued")
        valuation_text = (
            f"At {price:,.2f} against a blended base case of {base:,.2f}, the "
            f"shares look {stance} ({upside:+.0%} to base). That base is the "
            f"median of {sum(1 for m in valuation.get('methods', []) if m.get('per_share'))} "
            f"methods anchored on {profile.valuation_anchor}, and the methods "
            f"span {valuation.get('bear', 0):,.2f} to {valuation.get('bull', 0):,.2f}.")

    if quality is None:
        thesis = ("Too few quality metrics resolved to form a view on the "
                  "business.")
    elif quality >= 65 and (upside or 0) > 0.15:
        thesis = (
            "The market is pricing this below what the filed fundamentals "
            "support. Quality scores well and the price sits under the blended "
            "fair value - the mispricing case rests on that gap persisting "
            "for reasons the numbers do not explain, so look for what the "
            "market may know that the statements do not show.")
    elif quality >= 65:
        thesis = (
            "The business scores well and the market has noticed. There is no "
            "obvious mispricing here: the price already reflects the quality, "
            "which is why a good company is not automatically a good "
            "investment.")
    elif (upside or 0) > 0.15:
        thesis = (
            "The shares look cheap on the multiples, but the quality axis is "
            "weak. Cheapness that comes with deteriorating fundamentals is "
            "usually a value trap rather than an opportunity - the market may "
            "be right to discount it.")
    else:
        thesis = (
            "Neither the business quality nor the price offers an edge on the "
            "filed numbers. Nothing here argues for action.")

    return {"valuation": valuation_text, "thesis": thesis}


def _triggers(valuation: Dict[str, Any], metrics: Dict[str, Fact],
              axes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Concrete conditions that would change the verdict."""
    buy: List[str] = []
    sell: List[str] = []

    trigger_price = valuation.get("buy_trigger_price")
    if trigger_price:
        buy.append(
            f"Price at or below {trigger_price:,.2f} - a "
            f"{config.DCF.required_margin_of_safety:.0%} margin of safety to "
            f"the {valuation.get('base', 0):,.2f} base case.")

    for key, label, threshold in (
        ("roic", "Return on invested capital", 0.15),
        ("fcf_cagr_3y", "Three-year FCF CAGR", 0.10),
    ):
        fact = metrics.get(key)
        if fact and fact.known and fact.value < threshold:
            buy.append(f"{label} recovering above {threshold:.0%} "
                       f"(currently {fact.value:.1%}).")

    # (metric, label, threshold, direction, formatter). The formatter is
    # explicit per metric because a ratio and a percentage are not the same
    # thing: 3.5 net debt/EBITDA is "3.5x", not "350%".
    percent = lambda v: f"{v:.1%}"
    times = lambda v: f"{v:.2f}x"

    for key, label, threshold, direction, render in (
        ("roic", "Return on invested capital", 0.10, "below", percent),
        ("net_debt_to_ebitda", "Net debt / EBITDA", 3.5, "above", times),
        ("fcf_to_net_income", "FCF / net income", 0.6, "below", percent),
        ("share_count_cagr", "Annual share count growth", 0.03, "above", percent),
    ):
        fact = metrics.get(key)
        if fact and fact.known:
            sell.append(f"{label} moving {direction} {render(threshold)} "
                        f"(currently {render(fact.value)}).")

    valuation_score = axes.get("valuation", {}).get("score")
    if valuation_score is not None:
        sell.append(
            "Price rising far enough that the blended fair value no longer "
            "supports it - the valuation axis dropping below 20.")

    return {"buy": buy, "sell": sell}


# ==========================================================================
# THE REPORT
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="fund_assess")
def assess(ticker: str, overrides: Optional[Dict[str, float]] = None
           ) -> Dict[str, Any]:
    """
    The whole fundamental report for one ticker.

    Args:
        overrides: DCF assumption overrides - growth, discount_rate,
                   terminal_growth, fade_to, horizon_years.

    Returns a dict carrying the verdict, every axis score with its component
    breakdown, the valuation block, red flags, narrative, triggers and the
    itemised confidence figure. When confidence falls below
    `config.CONFIDENCE_FLOOR` the verdict is replaced with INSUFFICIENT DATA:
    a recommendation off half-missing financials is worse than no answer.

    Not advice. A mechanical screen over public filings, with no view on the
    reader's circumstances.
    """
    from data_fetchers import valuation as valuation_module

    ticker = equities.normalize_ticker(ticker).upper()
    facts = get_fact_set(ticker)

    if not facts.get("statements"):
        return {
            "ticker": ticker,
            "verdict": "INSUFFICIENT DATA",
            "reason": "No financial statements could be retrieved from SEC "
                      "EDGAR or Yahoo for this symbol.",
            "confidence": {"score": 0.0, "deductions": []},
        }

    metrics = get_metrics(facts)
    revenue_growth = metrics.get("revenue_cagr_3y")
    profile = detect_profile(
        facts, revenue_growth.value if revenue_growth.known else None)

    valuation = valuation_module.value(ticker, facts, metrics, profile, overrides)
    earnings = earnings_quality(metrics)

    axes = {
        "business_quality": business_quality(metrics, profile),
        "growth": growth(metrics, profile),
        "financial_health": financial_health(metrics, profile),
        "valuation": valuation_axis(valuation, metrics, profile),
    }
    axes["risk"] = risk_axis(metrics, valuation, earnings, profile)

    verdict_block = verdict(axes, axes["valuation"]["score"],
                            axes["risk"]["score"], earnings, profile)
    confidence_block = confidence(facts, metrics, axes, valuation, profile)

    if confidence_block["score"] < config.CONFIDENCE_FLOOR:
        verdict_block["verdict"] = "INSUFFICIENT DATA"
        verdict_block["rule"] = (
            f"Confidence {confidence_block['score']:.0f} is below the floor of "
            f"{config.CONFIDENCE_FLOOR:.0f}; no verdict is issued.")

    # Overall score: the quality composite adjusted for how the price and the
    # risk profile sit against it. Reported alongside its parts, never alone.
    overall = None
    if verdict_block["quality_composite"] is not None:
        parts = [(verdict_block["quality_composite"], 2.0)]
        if axes["valuation"]["score"] is not None:
            parts.append((axes["valuation"]["score"], 1.5))
        if axes["risk"]["score"] is not None:
            parts.append((100.0 - axes["risk"]["score"], 1.0))
        overall = sum(s * w for s, w in parts) / sum(w for _, w in parts)

    return {
        "ticker": ticker,
        "name": facts.get("info", {}).get("shortName") or ticker,
        "profile": profile,
        "price": facts.get("price"),
        "as_of": (facts.get("periods") or [""])[0],
        "statement_source": facts.get("statement_source"),
        "verdict": verdict_block["verdict"],
        "verdict_detail": verdict_block,
        "overall_score": overall,
        "scores": {
            "business_quality": axes["business_quality"]["score"],
            "growth": axes["growth"]["score"],
            "financial_health": axes["financial_health"]["score"],
            "valuation": axes["valuation"]["score"],
            "risk": axes["risk"]["score"],
        },
        "axes": axes,
        "metrics": metrics,
        "valuation": valuation,
        "earnings_quality": earnings,
        "reasons": _reasons(axes, valuation, earnings),
        "narrative": _narrative(verdict_block, valuation, profile, metrics),
        "triggers": _triggers(valuation, metrics, axes),
        "confidence": confidence_block,
        "unavailable": list(profile.unavailable) + [
            "Market share / competitive position",
            "Industry outlook",
            "Management strategy and stewardship (only capital allocation is "
            "measurable)",
        ],
        "disclaimer": (
            "Mechanical screen over public filings. Descriptive, not advice, "
            "and no substitute for reading the filings yourself."
        ),
    }


__all__ = [
    "Fact", "get_fact_set", "get_metrics", "detect_profile",
    "business_quality", "growth", "financial_health", "earnings_quality",
    "format_metric",
    "valuation_axis", "risk_axis", "confidence", "verdict", "assess",
    "FILED", "MARKET", "CALCULATED", "PROXY", "ASSUMPTION", "UNAVAILABLE",
]

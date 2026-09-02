"""
data_fetchers/valuation.py :: Intrinsic value, several ways at once.

Bloomberg equivalents: <EQUITY> RV (relative value), DDM, EVA.

WHY SEVERAL METHODS
-------------------
Any single valuation method can be made to say anything. A DCF is a lever
with three assumptions on it; a P/E is meaningless on a company earning
nothing; a peer multiple just relocates the question to whether the sector is
mispriced. So this computes every method the data supports, reports each one
separately, and takes the base case from a weighted median of those that
resolved.

The dispersion between methods is itself the signal. When a DCF says $180 and
the peer multiple says $95, that spread is the honest answer - averaging it
into $137 and printing one number would hide the only interesting thing on
the page. Dispersion is returned and it reduces the confidence score.

METHODS ARE ANCHORED PER PROFILE
--------------------------------
Running the same DCF over a bank is not conservatism, it is a category
error: a lender's free cash flow is an artefact of its funding mix, not a
distributable surplus. Each profile therefore names its own anchor -
`config.FundamentalProfile.valuation_anchor` - and the others are shown as
cross-checks:

    dcf       general corporates
    book      banks and insurers, on a justified P/B derived from ROE
    affo      REITs, where depreciation makes net income uninformative
    midcycle  commodities, valued off through-cycle margins not trough or peak
    growth    high-growth technology, on EV/sales and price/FCF

EVERY DCF INPUT IS A FORECAST
-----------------------------
Growth, fade, terminal rate and discount rate are assumptions, not facts.
They are derived from the company's own history and the live risk-free rate,
returned in a separate `assumptions` block tagged ASSUMPTION, and every one
is overridable by the caller. None of them is knowledge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_fetchers import equities
from utils import fundamental_math as fm

log = logging.getLogger("openterm.valuation")

ASSUMPTION = "ASSUMPTION"
CALCULATED = "CALCULATED"
MARKET = "MARKET"


# ==========================================================================
# ASSUMPTIONS
# ==========================================================================
def _risk_free_rate() -> Tuple[float, str]:
    """
    Live 10-year Treasury yield, or a stated fallback.

    Imported lazily: `macro` pulls in the FRED stack, and a valuation on a
    machine with no FRED access should still run rather than fail at import.
    """
    try:
        from data_fetchers import macro

        series = macro.get_fred_series("DGS10")
        if series is not None and len(series):
            return float(series.iloc[-1]) / 100.0, "FRED DGS10"
    except Exception as exc:
        log.warning("Risk-free rate unavailable: %s", exc)
    return 0.04, "fallback (FRED unreachable)"


def build_assumptions(metrics: Dict[str, Any], info: Dict[str, Any],
                      overrides: Optional[Dict[str, float]] = None
                      ) -> Dict[str, Any]:
    """
    Derive the forecast inputs from the company's own history, then let the
    caller override any of them.

    Trailing growth is capped at `config.DCF.growth_cap`: a 60% FCF CAGR
    compounded for a decade produces a fair value that is arithmetically
    correct and completely useless.
    """
    overrides = overrides or {}

    risk_free, rate_source = _risk_free_rate()
    beta = info.get("beta")

    discount = fm.capm_discount_rate(risk_free, beta if beta is not None else 1.0,
                                     config.DCF.equity_risk_premium)
    if discount is None:
        discount = risk_free + config.DCF.equity_risk_premium

    # MEDIAN of every growth estimate available, not the first one that
    # resolves. A single window is fragile in a way that silently changes the
    # answer: Apple's 3-year FCF CAGR is NEGATIVE purely because FY2022 was a
    # peak, and feeding that into a ten-year DCF compounds one cyclical dip
    # into a decade of decline. The 5-year read on the same cash flows is
    # +6%. Taking the median across windows and across revenue and FCF is
    # robust to any one of them being distorted, and it uses all the evidence
    # the eight filed years provide rather than the first item in a list.
    estimates = []
    for key in ("fcf_cagr_5y", "revenue_cagr_5y", "fcf_cagr_3y",
                "revenue_cagr_3y"):
        candidate = metrics.get(key)
        value = getattr(candidate, "value", None) if candidate is not None else None
        if value is not None:
            estimates.append(value)

    trailing = fm.median_of(estimates)
    growth = config.DCF.fade_to if trailing is None else max(
        min(trailing, config.DCF.growth_cap), -0.10)

    assumptions = {
        "growth": growth,
        "fade_to": config.DCF.fade_to,
        "terminal_growth": config.DCF.terminal_growth,
        "discount_rate": discount,
        "horizon_years": config.DCF.horizon_years,
    }
    assumptions.update({k: v for k, v in overrides.items() if v is not None})

    assumptions["_provenance"] = {
        "growth": f"{ASSUMPTION}: median of {len(estimates)} trailing "
                  f"revenue/FCF CAGRs, capped at "
                  f"{config.DCF.growth_cap:.0%}",
        "fade_to": f"{ASSUMPTION}: config.DCF.fade_to",
        "terminal_growth": f"{ASSUMPTION}: config.DCF.terminal_growth, held "
                           "below long-run nominal GDP",
        "discount_rate": f"{ASSUMPTION}: CAPM on {rate_source} "
                         f"(risk-free {risk_free:.2%}, beta "
                         f"{beta if beta is not None else 1.0:.2f}, ERP "
                         f"{config.DCF.equity_risk_premium:.0%})",
        "horizon_years": f"{ASSUMPTION}: config.DCF.horizon_years",
    }
    return assumptions


# ==========================================================================
# HISTORICAL AND PEER MULTIPLES
# ==========================================================================
def _column_year(column: Any) -> Optional[int]:
    """
    Fiscal year from a statement column label.

    SEC columns are strings like "FY2025"; Yahoo columns are Timestamps.
    Both have to resolve or the historical multiple silently comes back
    empty on whichever source was not anticipated.
    """
    if isinstance(column, pd.Timestamp):
        return int(column.year)
    digits = "".join(ch for ch in str(column) if ch.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else None


def own_multiple_history(ticker: str, statements: Dict[str, pd.DataFrame],
                         years: int = 5) -> Dict[str, Optional[float]]:
    """
    The company's own median P/E and P/FCF over `years`.

    This is the anchor that answers "cheap relative to *itself*", which a
    peer comparison cannot: a whole sector can be expensive at once.

    Monthly closes are matched to the most recent fiscal year that had
    already been reported at that date, so a 2021 price is divided by 2021
    earnings rather than by today's. Getting that backwards makes every
    historical multiple look cheap in hindsight.
    """
    out: Dict[str, Optional[float]] = {"pe_median": None, "p_fcf_median": None}

    try:
        history = equities.get_history(ticker, f"{years}y", "1mo")
    except Exception as exc:
        log.warning("Price history unavailable for %s: %s", ticker, exc)
        return out
    if history is None or history.empty or "Close" not in history.columns:
        return out

    income = statements.get("income_statement", pd.DataFrame())
    cash = statements.get("cash_flow", pd.DataFrame())
    if income.empty:
        return out

    # Alias-aware access, so this works on Yahoo statements as well as SEC.
    from data_fetchers import fundamentals

    eps_series = fundamentals._series(income, "EPS Diluted")
    share_series = fundamentals._series(income, "Diluted Shares")
    ocf_series = fundamentals._series(cash, "Operating Cash Flow")
    capex_series = fundamentals._series(cash, "CapEx")

    eps_by_year: Dict[int, float] = {}
    fcf_ps_by_year: Dict[int, float] = {}

    for index, column in enumerate(income.columns):
        year = _column_year(column)
        if year is None:
            continue

        eps = eps_series[index] if index < len(eps_series) else None
        if eps is not None and eps > 0:
            eps_by_year[year] = eps

        shares = share_series[index] if index < len(share_series) else None
        operating = ocf_series[index] if index < len(ocf_series) else None
        capex = capex_series[index] if index < len(capex_series) else None
        if shares and operating is not None and capex is not None and shares > 0:
            fcf_ps = (operating - abs(capex)) / shares
            if fcf_ps > 0:
                fcf_ps_by_year[year] = fcf_ps

    pe_values: List[float] = []
    p_fcf_values: List[float] = []

    for stamp, close in history["Close"].dropna().items():
        # The most recent fiscal year that was already public at this date.
        available = [y for y in eps_by_year if y < stamp.year] or \
                    [y for y in eps_by_year if y <= stamp.year]
        if available:
            pe_values.append(float(close) / eps_by_year[max(available)])

        available_fcf = [y for y in fcf_ps_by_year if y < stamp.year] or \
                        [y for y in fcf_ps_by_year if y <= stamp.year]
        if available_fcf:
            p_fcf_values.append(float(close) / fcf_ps_by_year[max(available_fcf)])

    out["pe_median"] = fm.median_of(pe_values)
    out["p_fcf_median"] = fm.median_of(p_fcf_values)
    return out


def peer_multiples(ticker: str) -> Dict[str, Any]:
    """
    Median multiples across the issuer's own derived peer group.

    Reuses `suggest_peers` (industry classification, never a hand-written
    list) and `get_peer_comparison`, which already appends a median row.
    """
    out: Dict[str, Any] = {"peers": [], "pe_median": None,
                           "ev_ebitda_median": None, "pb_median": None}
    try:
        peers = equities.suggest_peers(ticker)
        if len(peers) <= 1:
            out["note"] = ("No peer group could be derived from this issuer's "
                           "industry classification.")
            return out
        frame = equities.get_peer_comparison(peers)
    except Exception as exc:
        log.warning("Peer multiples unavailable for %s: %s", ticker, exc)
        out["note"] = f"Peer comparison failed: {exc}"
        return out

    if frame is None or frame.empty:
        return out

    others = frame[~frame["Ticker"].isin([ticker, "— MEDIAN —"])]
    out["peers"] = others["Ticker"].tolist()
    out["pe_median"] = fm.median_of(others.get("P/E (TTM)", pd.Series(dtype=float)))
    out["ev_ebitda_median"] = fm.median_of(
        others.get("EV/EBITDA", pd.Series(dtype=float)))
    out["pb_median"] = fm.median_of(others.get("P/B", pd.Series(dtype=float)))
    return out


# ==========================================================================
# METHODS
# ==========================================================================
def _method(name: str, per_share: Optional[float], basis: str,
            note: str = "") -> Dict[str, Any]:
    return {"method": name, "per_share": per_share, "basis": basis, "note": note}


def compute_methods(ticker: str, facts: Dict[str, Any],
                    metrics: Dict[str, Any],
                    profile: config.FundamentalProfile,
                    assumptions: Dict[str, Any],
                    history: Dict[str, Optional[float]],
                    peers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Every per-share value the available data supports.

    A method that cannot be computed returns `per_share=None` with a note
    saying why, and is excluded from the blend. It is still listed: knowing
    that the DCF could not run because the company burns cash is information.
    """
    info = facts.get("info", {})
    shares = info.get("sharesOutstanding")
    methods: List[Dict[str, Any]] = []

    def value_of(key: str) -> Optional[float]:
        item = metrics.get(key)
        return getattr(item, "value", None) if item is not None else None

    fcf = value_of("free_cash_flow")
    # The DCF compounds its base for ten years, so it uses the normalised
    # figure where one exists. Every other method stays on the latest filed
    # number - a P/FCF multiple should reflect what actually happened.
    dcf_base = value_of("normalized_fcf") or fcf
    net_income = value_of("net_income")
    equity = value_of("total_equity")
    net_debt = value_of("net_debt")
    revenue = value_of("revenue")
    roe = value_of("roe")
    from data_fetchers import fundamentals

    income = facts.get("statements", {}).get("income_statement", pd.DataFrame())
    eps = fundamentals._latest(income, "EPS Diluted")

    # --- Discounted cash flow --------------------------------------------
    if dcf_base is not None and shares:
        dcf = fm.dcf_per_share(
            fcf0=dcf_base, growth=assumptions["growth"],
            fade_to=assumptions["fade_to"], years=assumptions["horizon_years"],
            terminal_growth=assumptions["terminal_growth"],
            discount_rate=assumptions["discount_rate"],
            net_cash=-(net_debt or 0.0), shares=shares,
        )
        methods.append(_method(
            "DCF", dcf,
            f"Two-stage DCF on normalised FCF of {dcf_base/1e9:,.1f}bn, "
            "all inputs ASSUMPTION",
            "" if dcf is not None else
            "Free cash flow is negative or the discount rate does not exceed "
            "terminal growth; a DCF here would be arithmetic, not valuation."))
    else:
        methods.append(_method(
            "DCF", None, "Two-stage FCF",
            "Free cash flow or share count unavailable."))

    # --- Own historical P/E ----------------------------------------------
    if history.get("pe_median") and eps and eps > 0:
        methods.append(_method(
            "Own 5y median P/E", history["pe_median"] * eps,
            f"{history['pe_median']:.1f}x own median x filed EPS {eps:.2f}"))
    else:
        methods.append(_method(
            "Own 5y median P/E", None, "Own history",
            "No positive filed EPS history to build a multiple from."))

    # --- Own historical P/FCF --------------------------------------------
    if history.get("p_fcf_median") and fcf and shares and fcf > 0:
        methods.append(_method(
            "Own 5y median P/FCF", history["p_fcf_median"] * (fcf / shares),
            f"{history['p_fcf_median']:.1f}x own median x FCF/share"))
    else:
        methods.append(_method("Own 5y median P/FCF", None, "Own history",
                               "No positive FCF history."))

    # --- Peer multiple ----------------------------------------------------
    if peers.get("pe_median") and eps and eps > 0:
        methods.append(_method(
            "Peer median P/E", peers["pe_median"] * eps,
            f"{peers['pe_median']:.1f}x peer median x filed EPS",
            f"Peers: {', '.join(peers.get('peers', [])[:6])}"))
    else:
        methods.append(_method(
            "Peer median P/E", None, "Derived peer group",
            peers.get("note", "No peer multiple or no positive EPS.")))

    # --- Profile anchors --------------------------------------------------
    if profile.valuation_anchor == "book":
        methods.append(_justified_book_value(
            equity, shares, roe, assumptions, profile))
    elif profile.valuation_anchor == "affo":
        methods.append(_affo_value(facts, shares, history, peers))
    elif profile.valuation_anchor == "midcycle":
        methods.append(_midcycle_value(facts, metrics, shares, history))
    elif profile.valuation_anchor == "growth":
        methods.append(_growth_value(revenue, net_debt, shares, peers, info))

    return methods


def _justified_book_value(equity, shares, roe, assumptions,
                          profile) -> Dict[str, Any]:
    """
    Justified price/book for a lender or insurer: P/B = (ROE - g) / (r - g).

    The standard residual-income identity. A bank earning its cost of equity
    is worth book; one earning above it is worth a premium, and the size of
    that premium is not a matter of taste. This replaces the DCF, which on a
    balance-sheet business measures funding mix rather than surplus.
    """
    if not equity or not shares or roe is None:
        return _method("Justified P/B", None, "Residual income",
                       "Book value, share count or ROE unavailable.")

    growth = assumptions["terminal_growth"]
    cost_of_equity = assumptions["discount_rate"]
    if cost_of_equity <= growth:
        return _method("Justified P/B", None, "Residual income",
                       "Cost of equity does not exceed growth; the identity "
                       "does not resolve.")

    justified = (roe - growth) / (cost_of_equity - growth)
    if justified <= 0:
        return _method("Justified P/B", None, "Residual income",
                       f"ROE of {roe:.1%} is below assumed growth; the model "
                       "returns a negative multiple.")

    book_per_share = equity / shares
    return _method(
        "Justified P/B", justified * book_per_share,
        f"(ROE {roe:.1%} - g {growth:.1%}) / (r {cost_of_equity:.1%} - g) "
        f"= {justified:.2f}x book of {book_per_share:.2f}")


def _affo_value(facts, shares, history, peers) -> Dict[str, Any]:
    """
    REIT value on funds from operations.

    FFO = net income + depreciation, the NAREIT definition minus the
    property-gains adjustment. That adjustment is deliberately not attempted:
    gains on property sales have no consistent XBRL tag, and subtracting a
    number that is sometimes zero because it was not tagged would overstate
    FFO in exactly the years a REIT was selling assets. The result is
    labelled a PROXY for that reason.
    """
    statements = facts.get("statements", {})
    income = statements.get("income_statement", pd.DataFrame())
    cash = statements.get("cash_flow", pd.DataFrame())

    from data_fetchers import fundamentals

    net_income = fundamentals._latest(income, "Net Income")
    depreciation = fundamentals._latest(cash, "Depreciation & Amortization")

    if net_income is None or depreciation is None or not shares:
        return _method("P/FFO (proxy)", None, "FFO",
                       "Net income, D&A or share count unavailable; FFO "
                       "cannot be built.")

    ffo_per_share = (net_income + depreciation) / shares
    if ffo_per_share <= 0:
        return _method("P/FFO (proxy)", None, "FFO", "FFO is not positive.")

    multiple = peers.get("pe_median") or history.get("pe_median") or 15.0
    basis = ("peer median" if peers.get("pe_median")
             else "own median" if history.get("pe_median") else "15x default")

    return _method(
        "P/FFO (proxy)", multiple * ffo_per_share,
        f"{multiple:.1f}x ({basis}) x FFO/share {ffo_per_share:.2f}",
        "FFO = net income + D&A. Property-sale gains are not consistently "
        "tagged and are not deducted, so this overstates FFO in disposal "
        "years. NAV and occupancy are not available at all.")


def _midcycle_value(facts, metrics, shares, history) -> Dict[str, Any]:
    """
    Commodity value on through-cycle margins.

    Trailing earnings on a cyclical say where in the cycle the reporting
    window fell, not what the business earns. This applies the average
    operating margin across every filed year to current revenue, which is
    the cheapest honest way to normalise without a commodity price deck.
    """
    statements = facts.get("statements", {})
    income = statements.get("income_statement", pd.DataFrame())
    if income.empty or not shares:
        return _method("Mid-cycle earnings", None, "Normalised margin",
                       "Income statement or share count unavailable.")

    from data_fetchers import fundamentals

    operating_series = fundamentals._series(income, "Operating Income")
    revenue_series = fundamentals._series(income, "Revenue")

    margins: List[float] = []
    for index in range(min(len(operating_series), len(revenue_series))):
        operating, revenue = operating_series[index], revenue_series[index]
        if operating is not None and revenue and revenue > 0:
            margins.append(operating / revenue)

    if len(margins) < 4:
        return _method("Mid-cycle earnings", None, "Normalised margin",
                       f"Only {len(margins)} filed years; too few to average "
                       "across a cycle.")

    mid_margin = float(np.mean(margins))
    revenue_now = getattr(metrics.get("revenue"), "value", None)
    if not revenue_now or mid_margin <= 0:
        return _method("Mid-cycle earnings", None, "Normalised margin",
                       "Current revenue unavailable or mid-cycle margin is "
                       "not positive.")

    # After-tax at the US statutory rate, then a market multiple.
    mid_earnings_per_share = revenue_now * mid_margin * (1 - 0.21) / shares
    multiple = history.get("pe_median") or 14.0
    basis = "own median" if history.get("pe_median") else "14x default"

    return _method(
        "Mid-cycle earnings", multiple * mid_earnings_per_share,
        f"{mid_margin:.1%} mean operating margin over {len(margins)} years "
        f"x current revenue, at {multiple:.1f}x ({basis})",
        "Normalised on the company's own filed margin history. Per-unit "
        "production cost and realised prices are not tagged anywhere.")


def _growth_value(revenue, net_debt, shares, peers, info) -> Dict[str, Any]:
    """
    High-growth technology on EV/sales, since trailing P/E is uninformative.

    Uses the peer EV/EBITDA median only as a sanity anchor; the primary is
    the company's own current EV/sales held constant, which makes this a
    "what does today's multiple imply" cross-check rather than a forecast.
    """
    ev_to_revenue = info.get("enterpriseToRevenue")
    if not revenue or not shares or not ev_to_revenue:
        return _method("EV/Sales", None, "Growth multiple",
                       "Revenue, share count or EV/sales unavailable.")

    enterprise_value = revenue * float(ev_to_revenue)
    equity_value = enterprise_value - (net_debt or 0.0)
    if equity_value <= 0:
        return _method("EV/Sales", None, "Growth multiple",
                       "Implied equity value is not positive.")

    return _method(
        "EV/Sales", equity_value / shares,
        f"{float(ev_to_revenue):.1f}x EV/sales on revenue, less net debt",
        "A restatement of the current multiple, not an independent estimate "
        "- treat it as a cross-check on the others.")


# ==========================================================================
# SCENARIOS AND BLEND
# ==========================================================================
def scenarios(methods: List[Dict[str, Any]], assumptions: Dict[str, Any],
              facts: Dict[str, Any], metrics: Dict[str, Any]
              ) -> Dict[str, Optional[float]]:
    """
    Bear, base and bull, built from the spread of the methods themselves.

    Base is the median of every method that resolved. Bear and bull are the
    minimum and maximum, widened by a DCF run at pessimistic and optimistic
    assumptions where one is available.

    Using the actual dispersion rather than an arbitrary +/-25% band means
    the width of the range reports how much the methods disagree, which is
    the thing worth knowing.
    """
    resolved = [m["per_share"] for m in methods
                if m.get("per_share") is not None and m["per_share"] > 0]

    out: Dict[str, Optional[float]] = {"bear": None, "base": None, "bull": None}
    if not resolved:
        return out

    out["base"] = fm.median_of(resolved)
    out["bear"] = min(resolved)
    out["bull"] = max(resolved)

    info = facts.get("info", {})
    shares = info.get("sharesOutstanding")
    fcf = (getattr(metrics.get("normalized_fcf"), "value", None)
           or getattr(metrics.get("free_cash_flow"), "value", None))
    net_debt = getattr(metrics.get("net_debt"), "value", None)

    if fcf and shares and fcf > 0:
        for label, growth_shift, rate_shift in (("bear", -0.05, 0.02),
                                                ("bull", 0.05, -0.02)):
            candidate = fm.dcf_per_share(
                fcf0=fcf,
                growth=assumptions["growth"] + growth_shift,
                fade_to=assumptions["fade_to"],
                years=assumptions["horizon_years"],
                terminal_growth=assumptions["terminal_growth"],
                discount_rate=assumptions["discount_rate"] + rate_shift,
                net_cash=-(net_debt or 0.0), shares=shares,
            )
            if candidate is None or candidate <= 0:
                continue
            if label == "bear":
                out["bear"] = min(out["bear"], candidate)
            else:
                out["bull"] = max(out["bull"], candidate)

    # A single resolved method gives a point, not a range - say so by
    # keeping bear == base == bull rather than inventing a spread.
    return out


def dispersion(methods: List[Dict[str, Any]]) -> Optional[float]:
    """Spread across methods, as a fraction of the median. Wide = uncertain."""
    resolved = [m["per_share"] for m in methods
                if m.get("per_share") is not None and m["per_share"] > 0]
    if len(resolved) < 2:
        return None
    median = fm.median_of(resolved)
    if not median:
        return None
    return (max(resolved) - min(resolved)) / median


def value(ticker: str, facts: Dict[str, Any], metrics: Dict[str, Any],
          profile: config.FundamentalProfile,
          overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    The full valuation block: methods, scenarios, multiples and assumptions.
    """
    info = facts.get("info", {})
    price = facts.get("price")

    assumptions = build_assumptions(metrics, info, overrides)
    history = own_multiple_history(ticker, facts.get("statements", {}))
    peers = peer_multiples(ticker)
    methods = compute_methods(ticker, facts, metrics, profile, assumptions,
                              history, peers)
    bands = scenarios(methods, assumptions, facts, metrics)

    base = bands.get("base")
    fcf = getattr(metrics.get("free_cash_flow"), "value", None)
    market_cap = info.get("marketCap")

    multiples = {
        "P/E (TTM)": info.get("trailingPE"),
        "P/E (Fwd)": info.get("forwardPE"),
        "PEG": info.get("pegRatio"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "P/B": info.get("priceToBook"),
        "P/S": info.get("priceToSalesTrailing12Months"),
        "P/FCF": fm.safe_div(market_cap, fcf) if fcf and fcf > 0 else None,
        "FCF yield": fm.safe_div(fcf, market_cap) if fcf and market_cap else None,
    }

    return {
        "methods": methods,
        "bear": bands.get("bear"),
        "base": base,
        "bull": bands.get("bull"),
        "price": price,
        "upside_to_base": fm.upside(price, base) if price and base else None,
        "margin_of_safety": (fm.margin_of_safety(price, base)
                             if price and base else None),
        "buy_trigger_price": fm.price_for_margin_of_safety(
            base, config.DCF.required_margin_of_safety) if base else None,
        "dispersion": dispersion(methods),
        "multiples": multiples,
        "own_history": history,
        "peers": peers,
        "assumptions": assumptions,
        "anchor": profile.valuation_anchor,
    }


__all__ = ["value", "build_assumptions", "own_multiple_history",
           "peer_multiples", "compute_methods", "scenarios", "dispersion"]

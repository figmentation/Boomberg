"""
Open-Terminal :: A free, open-source Bloomberg Terminal alternative.

Run with:
    streamlit run app.py

Layout mirrors a real terminal:
    * Scrolling market tape across the top
    * Command bar accepting "<SUBJECT> <FUNCTION>" syntax (AAPL EQUITY,
      SUEZ SHIP, US10Y MACRO, ...)
    * Module tabs for direct navigation
    * Status bar with data-source health

Every module degrades gracefully. A dead upstream shows a stale-data notice
or an empty state, never a traceback in the user's face.
"""

from __future__ import annotations

import html
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import config

# --------------------------------------------------------------------------
# Page config must be the first Streamlit call in the script.
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="OPEN-TERMINAL",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": "Open-Terminal — free financial & OSINT intelligence terminal."},
)

from data_fetchers import aviation, equities, macro, maritime, news  # noqa: E402
from data_fetchers import company_intel, portfolio, supply_chain  # noqa: E402
from data_fetchers import macro_regime  # noqa: E402
from data_fetchers import fundamentals, social  # noqa: E402
from data_fetchers import allocation  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import maps  # noqa: E402
from ui.terminal_theme import THEME, apply_theme  # noqa: E402
from utils.cache import clear_all_caches, get_cache  # noqa: E402
from utils.rate_limiter import (  # noqa: E402
    bucket_status,
    circuit_status,
    reset_all_circuits,
)

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("openterm.app")

apply_theme()


# ==========================================================================
# COMMAND PARSER
# ==========================================================================
def parse_command(raw: str) -> Dict[str, Any]:
    """
    Parse a Bloomberg-style command into a route.

    Grammar (deliberately forgiving - terminals reward muscle memory, not
    strict syntax):

        <SUBJECT> <FUNCTION>    AAPL EQUITY / SUEZ SHIP / US10Y MACRO
        <FUNCTION>              NEWS / YCRV / FLY / HELP
        <SUBJECT>               AAPL          -> defaults to EQUITY
        <9-digit number>        636019825     -> defaults to AIS vessel lookup

    Returns:
        {module, subject, raw, valid, message}
    """
    result: Dict[str, Any] = {
        "module": None, "subject": None, "raw": raw,
        "valid": False, "message": "",
    }

    if not raw or not raw.strip():
        return result

    tokens = raw.strip().upper().split()
    result["raw"] = " ".join(tokens)

    # HELP is special-cased so it works from anywhere.
    if tokens[0] in ("HELP", "?", "H"):
        result.update({"module": "help", "valid": True})
        return result

    # Single token.
    if len(tokens) == 1:
        token = tokens[0]

        if token in config.COMMAND_FUNCTIONS:
            result.update({"module": config.COMMAND_FUNCTIONS[token], "valid": True})
            return result

        if token in config.CHOKEPOINTS:
            result.update({"module": "maritime", "subject": token, "valid": True})
            return result

        # A bare 9-digit number is an MMSI.
        if token.isdigit() and len(token) == 9:
            result.update({"module": "maritime", "subject": token, "valid": True,
                           "message": f"Interpreting {token} as an MMSI."})
            return result

        if token.startswith("IMO") and token[3:].strip().isdigit():
            result.update({"module": "maritime", "subject": token, "valid": True})
            return result

        # Anything else: treat as a ticker.
        result.update({"module": "equity", "subject": token, "valid": True,
                       "message": f"Defaulting to EQUITY for {token}."})
        return result

    # Multi-token: last token is the function, the rest is the subject.
    function = tokens[-1]
    subject = " ".join(tokens[:-1])

    if function in config.COMMAND_FUNCTIONS:
        result.update({
            "module": config.COMMAND_FUNCTIONS[function],
            "subject": subject,
            "valid": True,
        })
        return result

    # Maybe the *first* token is the function ("FLY EUROPE").
    if tokens[0] in config.COMMAND_FUNCTIONS:
        result.update({
            "module": config.COMMAND_FUNCTIONS[tokens[0]],
            "subject": " ".join(tokens[1:]),
            "valid": True,
        })
        return result

    result["message"] = (
        f"Unrecognised function '{function}'. Type HELP for the command list."
    )
    return result


# ==========================================================================
# SESSION STATE
# ==========================================================================
def init_state() -> None:
    """Seed session state on first run."""
    defaults = {
        "module": "home",
        "ticker": config.DEFAULT_TICKER,
        "chokepoint": "SUEZ",
        "vessel_query": "",
        "aviation_region": "EUROPE",
        "macro_view": "YIELD CURVE",
        "news_category": "MARKETS",
        "command_history": [],
        "last_command": "",
        "chart_period": "1y",
        "chart_interval": "1d",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def execute_command(raw: str) -> None:
    """Route a command string, updating session state."""
    parsed = parse_command(raw)

    if not parsed["valid"]:
        if parsed["message"]:
            st.session_state["command_error"] = parsed["message"]
        return

    st.session_state.pop("command_error", None)
    st.session_state["module"] = parsed["module"]
    st.session_state["last_command"] = parsed["raw"]

    history = st.session_state["command_history"]
    if parsed["raw"] and (not history or history[-1] != parsed["raw"]):
        history.append(parsed["raw"])
        st.session_state["command_history"] = history[-25:]

    subject = parsed["subject"]
    if not subject:
        return

    module = parsed["module"]
    if module in ("equity", "supply_chain", "fundamentals"):
        # These modules key off the same ticker, so "NVDA SPLC" and "NVDA FA"
        # both set it and switching pages keeps the name you were looking at.
        st.session_state["ticker"] = equities.normalize_ticker(subject)
    elif module == "maritime":
        if subject in config.CHOKEPOINTS:
            st.session_state["chokepoint"] = subject
            st.session_state["vessel_query"] = ""
        else:
            st.session_state["vessel_query"] = subject
    elif module == "aviation":
        if subject in config.AVIATION_REGIONS:
            st.session_state["aviation_region"] = subject
    elif module == "news":
        # "NVDA SOCIAL" should land on the social tab already looking at NVDA.
        # Anything that normalises to a plausible symbol sets the shared
        # ticker; a category word like ENERGY does not.
        symbol = equities.normalize_ticker(subject).upper()
        if symbol and symbol.isalpha() and len(symbol) <= 5:
            st.session_state["ticker"] = symbol
    elif module == "portfolio":
        # "NVDA WATCH" adds a symbol without a trip to the editor. The add is
        # idempotent and the row is deletable, so a typo costs one click.
        symbol = equities.normalize_ticker(subject)
        if portfolio.add_to_watchlist(symbol):
            st.session_state["portfolio_toast"] = f"{symbol} added to watchlist."
    elif module == "macro":
        upper = subject.upper()
        if any(token in upper for token in ("CPI", "INFLATION", "PCE")):
            st.session_state["macro_view"] = "INFLATION"
        elif any(token in upper for token in ("YCRV", "CURVE", "10Y", "2Y", "30Y")):
            st.session_state["macro_view"] = "YIELD CURVE"
        elif any(token in upper for token in ("JOBS", "UNEMPLOY", "PAYROLL")):
            st.session_state["macro_view"] = "LABOR"
    elif module == "news":
        upper = subject.upper()
        for category in config.RSS_FEEDS:
            if upper in category or category.split(" / ")[0] in upper:
                st.session_state["news_category"] = category
                break


# ==========================================================================
# CHROME: TAPE, COMMAND BAR, SIDEBAR
# ==========================================================================
@st.cache_data(ttl=config.TTL.quote, show_spinner=False)
def _tape_quotes() -> Dict[str, Dict[str, Any]]:
    """Batched tape quotes. Streamlit-cached on top of the SQLite layer so
    reruns within the TTL don't even touch the DB."""
    symbols = tuple(sym for sym, _ in config.TAPE_SYMBOLS)
    try:
        return equities.get_quotes_batch(symbols)
    except Exception as exc:
        log.warning("Tape quotes failed: %s", exc)
        return {}


def render_tape() -> None:
    labels = dict(config.TAPE_SYMBOLS)
    ui.ticker_tape(_tape_quotes(), labels)


def render_command_bar() -> None:
    """Top command input plus quick-jump function buttons."""
    left, right = st.columns([5, 1])

    with left:
        command = st.text_input(
            "COMMAND",
            key="command_input",
            placeholder="AAPL EQUITY   |   SUEZ SHIP   |   US10Y MACRO   |   FLY   |   NEWS   |   HELP",
            label_visibility="collapsed",
        )

    with right:
        submitted = st.button("▶ EXECUTE", use_container_width=True)

    # Streamlit's text_input fires on Enter, so either path runs the command.
    if command and (submitted or command != st.session_state.get("_prev_command")):
        st.session_state["_prev_command"] = command
        execute_command(command)
        # Rerun explicitly. Without it the new module renders in this pass but
        # the *previous* module's widgets survive below it, because Streamlit
        # only reconciles elements it re-encounters at the same position. The
        # nav buttons already rerun; the command bar must do the same.
        st.rerun()

    if st.session_state.get("command_error"):
        ui.alert(st.session_state["command_error"], "warn")

    # Function-key style shortcuts.
    shortcuts = [
        ("HOME", "home"), ("EQUITY", "equity"), ("SPLC", "supply_chain"),
        ("SHIP", "maritime"), ("FLY", "aviation"), ("MACRO", "macro"),
        ("FA", "fundamentals"), ("NEWS", "news"), ("PF", "portfolio"),
        ("HELP", "help"),
    ]
    cols = st.columns(len(shortcuts))
    for col, (label, module) in zip(cols, shortcuts):
        with col:
            active = st.session_state["module"] == module
            if st.button(f"{'▸ ' if active else ''}{label}", key=f"nav_{module}",
                         use_container_width=True):
                st.session_state["module"] = module
                st.rerun()


def render_sidebar() -> None:
    """Data-source health, cache controls, diagnostics."""
    with st.sidebar:
        st.markdown(
            f'<div style="color:{THEME.amber};font-size:17px;font-weight:700;'
            f'letter-spacing:0.16em;border-bottom:2px solid {THEME.amber};'
            f'padding-bottom:5px;margin-bottom:9px;">◆ OPEN-TERMINAL</div>',
            unsafe_allow_html=True,
        )
        st.caption("Free financial & OSINT intelligence")

        # --- Credentials --------------------------------------------------
        st.markdown("### DATA SOURCES")
        credentials = config.credential_status()

        source_rows = [
            ("yfinance", equities.YFINANCE_AVAILABLE, "Equities, FX, futures"),
            ("SEC EDGAR", True, "Filings & XBRL (keyless)"),
            ("FRED", True, "Macro — keyless CSV fallback active"
                          if not credentials["FRED"] else "Macro (API key set)"),
            ("OpenSky", True, "Aviation — 4000 credits/day"
                              if credentials["OpenSky"] else "Aviation — anonymous, 400/day"),
            ("AISStream", credentials["AISStream"], "Maritime AIS"
                          if credentials["AISStream"] else "No key — maritime limited"),
            ("GDELT", True, "Global OSINT (keyless)"),
        ]

        for name, healthy, note in source_rows:
            state = "live" if healthy else "off"
            st.markdown(
                f'<div style="font-size:11px;margin:3px 0;">'
                f'{ui.status_dot(state, name)}'
                f'<div style="color:{THEME.muted};font-size:9px;margin-left:12px;">{note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not credentials["FRED"] or not credentials["AISStream"]:
            with st.expander("⚙ ADD FREE API KEYS"):
                st.markdown(
                    "All optional, all free, no card required. Create a `.env` "
                    "file next to `app.py`:\n\n"
                    "```\nFRED_API_KEY=...\nOPENSKY_CLIENT_ID=...\n"
                    "OPENSKY_CLIENT_SECRET=...\nAISSTREAM_API_KEY=...\n```\n\n"
                    "- **FRED** — fredaccount.stlouisfed.org/apikeys\n"
                    "- **OpenSky** — opensky-network.org (Account → API Client)\n"
                    "- **AISStream** — aisstream.io\n\n"
                    "Without them the terminal still runs: FRED falls back to "
                    "its keyless CSV endpoint and OpenSky to anonymous access. "
                    "Maritime is the one module that genuinely needs a key."
                )

        st.divider()

        # --- Cache --------------------------------------------------------
        st.markdown("### CACHE")
        stats = get_cache().stats()
        if stats:
            total_entries = sum(s["entries"] for s in stats)
            total_bytes = sum(s["bytes"] for s in stats)
            st.markdown(
                f'<div style="font-size:11px;color:{THEME.cyan};">'
                f'{total_entries} entries · {total_bytes / 1024:.0f} KB</div>',
                unsafe_allow_html=True,
            )
            with st.expander("BY NAMESPACE"):
                st.dataframe(
                    pd.DataFrame(stats)[["namespace", "entries", "hits"]],
                    use_container_width=True, hide_index=True,
                )
        else:
            st.caption("Cache empty")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("PURGE", use_container_width=True):
                clear_all_caches()
                st.cache_data.clear()
                reset_all_circuits()
                st.success("Cache cleared")
                st.rerun()
        with col_b:
            if st.button("REFRESH", use_container_width=True):
                st.cache_data.clear()
                # Give tripped sources another chance on an explicit refresh.
                reset_all_circuits()
                st.rerun()

        # --- Circuit breakers ---------------------------------------------
        circuits = circuit_status()
        tripped = {name: state for name, state in circuits.items()
                   if state != "CLOSED"}
        if tripped:
            st.markdown("### UPSTREAM STATUS")
            for name, state in tripped.items():
                st.markdown(
                    f'<div style="font-size:11px;color:'
                    f'{THEME.red if state == "OPEN" else THEME.amber};">'
                    f'⚠ {name.upper()}: {state}</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "Failing fast to avoid stalling the UI. Cached values are "
                "served where available; press REFRESH to retry now."
            )

        st.divider()

        # --- History ------------------------------------------------------
        history = st.session_state.get("command_history", [])
        if history:
            st.markdown("### RECENT")
            for cmd in reversed(history[-8:]):
                if st.button(cmd, key=f"hist_{cmd}", use_container_width=True):
                    execute_command(cmd)
                    st.rerun()

        if config.DEBUG:
            st.divider()
            with st.expander("RATE LIMIT BUCKETS"):
                st.json(bucket_status())
            with st.expander("CIRCUIT BREAKERS"):
                st.json(circuit_status() or {"(none exercised yet)": ""})


# ==========================================================================
# PAGE: HOME
# ==========================================================================
@st.cache_data(ttl=config.TTL.quote, show_spinner=False)
def _home_quotes(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """
    Quotes for the home dashboard.

    Separate from _tape_quotes because the dashboard shows names the tape
    doesn't carry (^RUT). Reusing the tape's dict left those tiles blank.
    """
    try:
        return equities.get_quotes_batch(symbols)
    except Exception as exc:
        log.warning("Home quotes failed: %s", exc)
        return {}


def page_home() -> None:
    ui.module_header("OPEN-TERMINAL", "MARKET OVERVIEW")

    # Macro regime banner. Wrapped in _safe and tolerant of an empty verdict:
    # a FRED outage must not take the landing page with it.
    verdict = _safe(macro_regime.classify, default={})
    if verdict and verdict.get("regime"):
        accent = _regime_accent(verdict)
        stamp = verdict.get("as_of")
        st.markdown(
            f'<div style="border-left:3px solid {accent};'
            f'background:{THEME.bg_panel};padding:6px 13px;margin-bottom:10px;'
            f'font-size:11px;">'
            f'<span style="color:{THEME.muted};">MACRO REGIME</span>&nbsp;&nbsp;'
            f'<span style="color:{accent};letter-spacing:0.08em;">'
            f'{verdict["regime"]}</span>'
            f'<span style="color:{THEME.muted};"> — {verdict["posture"]} · '
            f'growth {verdict["growth_z"]:+.2f} · '
            f'inflation {verdict["inflation_z"]:+.2f} · '
            f'{verdict["conviction"]} conviction · '
            f'{pd.Timestamp(stamp).strftime("%b %Y") if stamp is not None else ""}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # --- Index tiles -------------------------------------------------------
    st.markdown("### MAJOR INDICES")
    index_symbols = [("^GSPC", "S&P 500"), ("^IXIC", "NASDAQ"),
                     ("^DJI", "DOW JONES"), ("^RUT", "RUSSELL 2000"),
                     ("^VIX", "VIX"), ("^TNX", "US 10Y")]
    commodity_symbols = [("CL=F", "WTI CRUDE"), ("BZ=F", "BRENT"),
                         ("GC=F", "GOLD"), ("SI=F", "SILVER"),
                         ("BTC-USD", "BITCOIN"), ("ETH-USD", "ETHER")]

    # One batched request covering every tile on this page.
    quotes = _home_quotes(tuple(
        sym for sym, _ in index_symbols + commodity_symbols
    ))
    tiles = []
    for symbol, label in index_symbols:
        quote = quotes.get(symbol, {})
        tiles.append(ui.metric_tile(
            label, quote.get("price"), quote.get("change_pct"), "%",
            accent=THEME.green if (quote.get("change") or 0) >= 0 else THEME.red,
        ))
    ui.metric_row(tiles, columns=6)

    # --- Commodities & crypto ---------------------------------------------
    st.markdown("### COMMODITIES & DIGITAL ASSETS")
    tiles = []
    for symbol, label in commodity_symbols:
        quote = quotes.get(symbol, {})
        tiles.append(ui.metric_tile(
            label, quote.get("price"), quote.get("change_pct"), "%",
            accent=THEME.green if (quote.get("change") or 0) >= 0 else THEME.red,
        ))
    ui.metric_row(tiles, columns=6)

    left, right = st.columns([3, 2])

    # --- Sector heatmap ----------------------------------------------------
    with left:
        st.markdown("### SECTOR PERFORMANCE")
        sector_etfs = {
            "XLK": "TECH", "XLF": "FINANCIALS", "XLE": "ENERGY",
            "XLV": "HEALTH", "XLI": "INDUSTRIAL", "XLY": "CONS DISC",
            "XLP": "CONS STAPLE", "XLU": "UTILITIES", "XLB": "MATERIALS",
            "XLRE": "REAL ESTATE", "XLC": "COMM SVCS", "SPY": "S&P 500",
        }
        try:
            sector_quotes = equities.get_quotes_batch(tuple(sector_etfs))
            rows = [
                {"label": name, "change": sector_quotes[symbol]["change_pct"]}
                for symbol, name in sector_etfs.items()
                if symbol in sector_quotes
            ]
            if rows:
                ui.render_chart(
                    ui.heatmap(pd.DataFrame(rows), "change", "label",
                               "DAILY % CHANGE", height=300, columns=4)
                )
            else:
                ui.alert("Sector data unavailable.", "warn")
        except Exception as exc:
            ui.alert(f"Sector heatmap unavailable: {exc}", "warn")

    # --- Macro snapshot ----------------------------------------------------
    with right:
        st.markdown("### YIELD CURVE")
        try:
            curve = macro.get_yield_curve()
            if not curve.empty:
                analysis = macro.analyze_yield_curve(curve)
                ui.render_chart(_yield_curve_figure(curve, analysis, height=290))

                if analysis.get("is_inverted"):
                    pairs = ", ".join(i["pair"] for i in analysis["inversions"])
                    ui.alert(f"CURVE INVERTED — {pairs}", "warn")
                else:
                    ui.alert(
                        f"Curve {analysis.get('shape', 'NORMAL').lower()} — "
                        f"recession risk {analysis.get('recession_risk', 'UNKNOWN').lower()}",
                        "ok",
                    )
            else:
                ui.alert("Yield curve data unavailable.", "warn")
        except Exception as exc:
            ui.alert(f"Curve unavailable: {exc}", "warn")

    # --- News + chokepoints ------------------------------------------------
    news_col, ship_col = st.columns([3, 2])

    with news_col:
        st.markdown("### TOP HEADLINES")
        try:
            headlines = _cached_news(("MARKETS",), 8, 25)
            if not headlines.empty:
                ui.news_feed(headlines, max_rows=14)
            else:
                ui.alert("No headlines retrieved.", "warn")
        except Exception as exc:
            ui.alert(f"News unavailable: {exc}", "warn")

    with ship_col:
        st.markdown("### CHOKEPOINT STATUS")
        if not config.AISSTREAM_API_KEY:
            ui.alert(
                "Maritime tracking needs a free AISStream key. "
                "See the sidebar.", "warn",
            )
            st.markdown(
                "".join(
                    f'<div style="padding:5px 8px;border-bottom:1px solid {THEME.grid};'
                    f'font-size:11px;">'
                    f'<span style="color:{THEME.amber};">{cp.name}</span><br>'
                    f'<span style="color:{THEME.muted};font-size:9px;">'
                    f'{cp.description}</span></div>'
                    for cp in config.CHOKEPOINTS.values()
                ),
                unsafe_allow_html=True,
            )
        else:
            if st.button("SCAN ALL CHOKEPOINTS", use_container_width=True):
                with st.spinner("Listening to AIS streams…"):
                    summary = maritime.get_all_chokepoints(stream_seconds=10)
                ui.styled_table(summary, height=280)


# ==========================================================================
# PAGE: EQUITY
# ==========================================================================
def page_equity() -> None:
    ticker = st.session_state["ticker"]

    controls = st.columns([2, 1, 1, 1, 1])
    with controls[0]:
        entered = st.text_input("TICKER", value=ticker, key="eq_ticker").upper()
        if entered and entered != ticker:
            st.session_state["ticker"] = equities.normalize_ticker(entered)
            st.rerun()
    with controls[1]:
        period = st.selectbox("PERIOD",
                              ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                              index=3, key="eq_period")
    with controls[2]:
        interval = st.selectbox("INTERVAL",
                                ["1d", "1wk", "1mo", "1h", "30m", "15m", "5m"],
                                index=0, key="eq_interval")
    with controls[3]:
        show_bollinger = st.checkbox("BOLLINGER", value=False)
    with controls[4]:
        show_volume = st.checkbox("VOLUME", value=True)

    quote = _safe(equities.get_quote, ticker, default={})
    info = _safe(equities.get_company_info, ticker, default={})

    name = info.get("longName") or info.get("shortName") or ticker
    exchange = info.get("exchange", "")
    ui.module_header(f"{ticker} — {name}",
                     f"{info.get('sector', '')} {('· ' + exchange) if exchange else ''}")

    if not quote:
        ui.alert(
            f"No market data for '{ticker}'. Check the symbol — Yahoo uses "
            f"'-' for share classes (BRK-B) and suffixes for foreign listings "
            f"(VOD.L, 7203.T).", "error",
        )
        return

    ui.quote_tiles(quote, info)

    tabs = st.tabs(["CHART", "FUNDAMENTALS", "SEC FILINGS",
                    "PEER COMPARISON", "OPTIONS", "NEWS", "PEOPLE",
                    "SUPPLY CHAIN", "PROFILE"])

    # ---- CHART -----------------------------------------------------------
    with tabs[0]:
        history = _safe(equities.get_history, ticker, period, interval,
                        default=pd.DataFrame())
        if history.empty:
            ui.alert(
                f"No price history. Yahoo caps intraday history: 1m→7d, "
                f"other intraday→60d. Try a shorter period or a daily interval.",
                "warn",
            )
        else:
            enriched = equities.add_indicators(history)
            signals = equities.summarize_technicals(enriched)

            if signals:
                badges = []
                if signals.get("trend"):
                    badges.append(ui.badge(
                        signals["trend"],
                        "green" if signals["trend"] == "UPTREND" else "red"))
                if signals.get("rsi_state"):
                    color = {"OVERBOUGHT": "red", "OVERSOLD": "green"}.get(
                        signals["rsi_state"], "cyan")
                    badges.append(ui.badge(
                        f"RSI {signals.get('rsi', 0):.1f} {signals['rsi_state']}", color))
                if signals.get("macd_state"):
                    badges.append(ui.badge(
                        f"MACD {signals['macd_state']}",
                        "green" if signals["macd_state"] == "BULLISH" else "red"))
                if signals.get("macd_cross"):
                    badges.append(ui.badge(
                        f"{signals['macd_cross']} CROSS",
                        "green" if signals["macd_cross"] == "GOLDEN" else "red"))
                if signals.get("volatility_annualized_pct"):
                    badges.append(ui.badge(
                        f"VOL {signals['volatility_annualized_pct']:.1f}%", "amber"))
                st.markdown("".join(badges), unsafe_allow_html=True)

            ui.render_chart(ui.candlestick_chart(
                enriched, ticker,
                show_volume=show_volume, show_bollinger=show_bollinger,
            ))

            with st.expander("INDICATOR DETAIL"):
                st.json({k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in signals.items()})

    # ---- FUNDAMENTALS ----------------------------------------------------
    with tabs[1]:
        source = st.radio(
            "SOURCE", ["SEC EDGAR (as reported)", "Yahoo Finance"],
            horizontal=True, key="fund_source",
        )
        frequency = st.radio("PERIOD", ["Annual", "Quarterly"],
                             horizontal=True, key="fund_freq")
        annual = frequency == "Annual"

        if source.startswith("SEC"):
            st.caption(
                "Pulled from the registrant's own XBRL filings via SEC EDGAR's "
                "public API. Figures tie to the 10-K/10-Q exactly. US filers only."
            )
            for label, key in (("INCOME STATEMENT", "income_statement"),
                               ("BALANCE SHEET", "balance_sheet"),
                               ("CASH FLOW", "cash_flow")):
                st.markdown(f"#### {label}")
                statement = _safe(equities.get_sec_financials, ticker, key, annual,
                                  default=pd.DataFrame())
                ui.statement_table(statement)
        else:
            statements = _safe(equities.get_financial_statements, ticker,
                               not annual, default={})
            for label, key in (("INCOME STATEMENT", "income_statement"),
                               ("BALANCE SHEET", "balance_sheet"),
                               ("CASH FLOW", "cash_flow")):
                st.markdown(f"#### {label}")
                ui.statement_table(statements.get(key, pd.DataFrame()))

    # ---- SEC FILINGS -----------------------------------------------------
    with tabs[2]:
        forms = st.multiselect(
            "FORM TYPES", ["10-K", "10-Q", "8-K", "4", "DEF 14A", "S-1", "13F-HR"],
            default=["10-K", "10-Q", "8-K"], key="filing_forms",
        )
        filings = _safe(equities.get_sec_filings, ticker, tuple(forms), 50,
                        default=pd.DataFrame())

        if filings.empty:
            ui.alert(
                f"No EDGAR filings for '{ticker}'. Only US registrants file "
                f"with the SEC — foreign issuers without an ADR won't appear.",
                "warn",
            )
        else:
            st.caption(f"{len(filings)} filings · {filings.iloc[0]['company']}")
            display = filings[["form", "filing_date", "report_date",
                               "description", "url"]].copy()
            display["filing_date"] = display["filing_date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                display, use_container_width=True, hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("DOCUMENT", display_text="OPEN →"),
                    "form": st.column_config.TextColumn("FORM", width="small"),
                },
            )

    # ---- PEERS -----------------------------------------------------------
    with tabs[3]:
        default_peers = _safe(equities.suggest_peers, ticker, default=[ticker])
        if len(default_peers) <= 1:
            ui.alert(
                f"No comparables derived for {ticker}. Yahoo publishes no "
                "industry or sector constituent list for it, and the terminal "
                "does not assert peers it cannot source — type your own set "
                "below.", "warn",
            )
        peer_input = st.text_input(
            "PEER SET (comma-separated)",
            value=", ".join(default_peers), key="peer_input",
        )
        peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]

        if st.button("RUN COMPARISON", key="run_peers") or peers:
            with st.spinner(f"Fetching {len(peers)} names…"):
                comparison = _safe(equities.get_peer_comparison, peers,
                                   default=pd.DataFrame())
            if comparison.empty:
                ui.alert("Peer comparison returned no data.", "warn")
            else:
                ui.styled_table(
                    comparison,
                    highlight_columns=["Rev Growth %", "ROE %", "Net Mgn %",
                                       "Oper Mgn %", "Gross Mgn %"],
                )
                st.caption(
                    "Default peers are the constituents of this issuer's own "
                    "industry classification, ranked by market weight — not a "
                    "hand-written sector list. Multiples from Yahoo Finance. A "
                    "negative or blank P/E means trailing losses. EV/EBITDA is "
                    "generally the more comparable multiple across capital "
                    "structures."
                )

    # ---- OPTIONS ---------------------------------------------------------
    with tabs[4]:
        chain = _safe(equities.get_options_chain, ticker, default={})
        if not chain or not chain.get("expiries"):
            ui.alert("No listed options for this symbol.", "warn")
        else:
            expiry = st.selectbox("EXPIRY", chain["expiries"], key="opt_expiry")
            if expiry != chain["selected_expiry"]:
                chain = _safe(equities.get_options_chain, ticker, expiry, default=chain)

            ratio = chain.get("put_call_ratio")
            ui.metric_row([
                ui.metric_tile("PUT/CALL OI", ratio,
                               subtitle="above 1.0 = put-heavy",
                               accent=THEME.red if (ratio or 0) > 1 else THEME.green),
                ui.metric_tile("CALL OI", chain.get("call_open_interest"),
                               value_format="{:,.0f}"),
                ui.metric_tile("PUT OI", chain.get("put_open_interest"),
                               value_format="{:,.0f}"),
            ], columns=3)

            call_col, put_col = st.columns(2)
            columns = ["strike", "lastPrice", "bid", "ask", "volume",
                       "openInterest", "impliedVolatility"]
            with call_col:
                st.markdown("#### CALLS")
                calls = chain["calls"]
                ui.styled_table(calls[[c for c in columns if c in calls.columns]],
                                height=420)
            with put_col:
                st.markdown("#### PUTS")
                puts = chain["puts"]
                ui.styled_table(puts[[c for c in columns if c in puts.columns]],
                                height=420)

    # ---- NEWS ------------------------------------------------------------
    with tabs[5]:
        ticker_news = _safe(news.get_ticker_news, ticker, 40, default=pd.DataFrame())
        if ticker_news.empty:
            ui.alert(f"No recent news for {ticker}.", "warn")
        else:
            summary = news.sentiment_summary(ticker_news)
            ui.metric_row([
                ui.metric_tile("HEADLINES", summary.get("total"), value_format="{:,.0f}"),
                ui.metric_tile("NET SENTIMENT", summary.get("net_sentiment"),
                               subtitle=summary.get("mood", ""),
                               accent=THEME.green if (summary.get("net_sentiment") or 0) > 0 else THEME.red),
                ui.metric_tile("BULLISH", summary.get("bullish"),
                               accent=THEME.green, value_format="{:,.0f}"),
                ui.metric_tile("BEARISH", summary.get("bearish"),
                               accent=THEME.red, value_format="{:,.0f}"),
            ], columns=4)
            ui.news_feed(ticker_news, max_rows=40)

    # ---- PEOPLE ----------------------------------------------------------
    # Both of the tabs below are gated behind a button. Streamlit executes
    # every tab body on every rerun, not just the visible one, so an eager
    # fetch here would download a 10-K and issue a dozen Wikidata lookups
    # each time anyone opened the equity page for a new ticker.
    with tabs[6]:
        if _load_on_demand("people", ticker,
                           "Fetch the executive roster and verified accounts"):
            render_people(ticker, name, info)

    # ---- SUPPLY CHAIN ----------------------------------------------------
    with tabs[7]:
        st.caption(
            "A compact view. The full SPLC module — network map, geographic "
            "exposure, commodity and credit risk — is one command away: "
            "type SPLC, or the ticker followed by SPLC."
        )
        if _load_on_demand("splc", ticker,
                           "Mine the latest 10-K for counterparty disclosures"):
            render_supply_chain_summary(ticker)

    # ---- PROFILE ---------------------------------------------------------
    with tabs[8]:
        if not info:
            ui.alert("Company profile unavailable.", "warn")
        else:
            summary_text = info.get("longBusinessSummary")
            if summary_text:
                st.markdown(f"**BUSINESS DESCRIPTION**\n\n{summary_text}")
            st.divider()

            left_col, right_col = st.columns(2)
            fields = [
                ("Sector", "sector"), ("Industry", "industry"),
                ("Country", "country"), ("Employees", "fullTimeEmployees"),
                ("Website", "website"), ("Currency", "currency"),
                ("Beta", "beta"), ("52W High", "fiftyTwoWeekHigh"),
                ("52W Low", "fiftyTwoWeekLow"), ("Avg Volume", "averageVolume"),
                ("Analyst Target", "targetMeanPrice"),
                ("Recommendation", "recommendationKey"),
                ("Analysts Covering", "numberOfAnalystOpinions"),
                ("Short % of Float", "shortPercentOfFloat"),
            ]
            for index, (label, key) in enumerate(fields):
                value = info.get(key)
                if value is None:
                    continue
                target = left_col if index % 2 == 0 else right_col
                with target:
                    if isinstance(value, (int, float)):
                        display = f"{value:,.2f}" if abs(value) < 1e6 else \
                                  equities.format_large_number(value, "")
                    else:
                        display = str(value)
                    st.markdown(
                        f'<div style="border-bottom:1px solid {THEME.grid};padding:3px 0;">'
                        f'<span style="color:{THEME.muted};font-size:10px;">{label}</span>'
                        f'<span style="float:right;color:{THEME.cyan};">{display}</span></div>',
                        unsafe_allow_html=True,
                    )


# ==========================================================================
# PEOPLE  (management roster + verified social presence)
# ==========================================================================
def _load_on_demand(name: str, ticker: str, prompt: str) -> bool:
    """
    Gate an expensive tab behind one click, then remember the choice.

    Streamlit has no server-side notion of which tab is visible - every tab
    body runs on every rerun. Without this, opening the equity page would pay
    for a 10-K download and a round of Wikidata lookups whether or not the
    user ever looked at these tabs. Once loaded for a ticker it stays loaded,
    so navigating away and back doesn't re-prompt.
    """
    key = f"_loaded_{name}_{ticker.upper()}"
    if st.session_state.get(key):
        return True

    st.caption(f"{prompt}. Not loaded automatically — it costs a filing "
               f"download and a few seconds.")
    if st.button("LOAD", key=f"load_{name}_{ticker}", use_container_width=False):
        st.session_state[key] = True
        return True
    return False


def render_people(ticker: str, company_name: str, info: Dict[str, Any]) -> None:
    """Executive roster with the compensation the proxy discloses."""
    with st.spinner("Resolving officers and verified accounts…"):
        officers = _safe(company_intel.get_executives, ticker,
                         default=pd.DataFrame())
        corporate = _safe(company_intel.get_company_social, ticker, company_name,
                          default={})

    # --- The company's own accounts ---------------------------------------
    st.markdown("### CORPORATE ACCOUNTS")
    if corporate.get("social"):
        st.markdown(ui.social_links(corporate["social"], size=12),
                    unsafe_allow_html=True)
        if corporate.get("unverified"):
            ui.alert(
                "The matched Wikidata entity doesn't describe a company. These "
                "handles may belong to a different subject of the same name.",
                "warn",
            )
        if corporate.get("description"):
            st.caption(corporate["description"])
    else:
        ui.alert(
            "No verified social accounts found. Handles come from Wikidata's "
            "curated properties rather than guesswork — an unmapped company "
            "shows nothing here rather than a plausible-looking wrong link.",
            "warn",
        )

    st.divider()

    # --- Officers ----------------------------------------------------------
    if officers.empty:
        ui.alert(
            f"No officer data for {ticker}. yfinance sources this from the "
            f"DEF 14A proxy statement, which foreign private issuers "
            f"(filing 20-F) and non-US listings do not file.", "warn",
        )
        return

    summary = company_intel.compensation_summary(officers)
    currency = info.get("currency", "USD")
    symbol = "$" if currency in ("USD", "") else ""

    ui.metric_row([
        ui.metric_tile("OFFICERS", summary.get("officers"), value_format="{:,.0f}"),
        ui.metric_tile("CEO PAY", (summary.get("ceo_pay") or 0) / 1e6,
                       subtitle=f"{currency} millions", value_format="{:,.2f}"),
        ui.metric_tile("MEDIAN PAY", (summary.get("median_comp") or 0) / 1e6,
                       subtitle=f"{currency} millions", value_format="{:,.2f}"),
        ui.metric_tile("CEO vs PEERS", summary.get("ceo_vs_peers"),
                       subtitle="× median officer", value_format="{:,.2f}×",
                       accent=THEME.red if (summary.get("ceo_vs_peers") or 0) > 4
                       else THEME.amber),
        ui.metric_tile("MEAN AGE", summary.get("mean_age"), value_format="{:,.0f}"),
    ], columns=5)

    st.caption(
        "Compensation is the proxy's summary-table total for the last "
        "reported fiscal year. 'CEO vs peers' compares the CEO to the median "
        "of the other named officers — it is NOT the SEC's CEO-to-median-"
        "employee pay ratio, which needs a figure yfinance doesn't carry."
    )

    left, right = st.columns(2)
    for index, (_, row) in enumerate(officers.iterrows()):
        with (left if index % 2 == 0 else right):
            st.markdown(ui.executive_card(row, symbol), unsafe_allow_html=True)

    with st.expander("ROSTER AS TABLE"):
        columns = [c for c in ("name", "title", "age", "total_pay",
                               "twitter_handle", "bio") if c in officers.columns]
        st.dataframe(officers[columns], use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: SUPPLY CHAIN  (SPLC)
# ==========================================================================
def render_supply_chain_summary(ticker: str) -> None:
    """The condensed panel embedded in the equity page's tab."""
    counterparties = _safe(supply_chain.get_counterparties, ticker,
                           default=pd.DataFrame())
    if counterparties.empty:
        ui.alert(
            "No concentration disclosures found in the latest 10-K. US GAAP "
            "only forces disclosure above 10% of revenue, so a diversified "
            "issuer legitimately reports none.", "warn",
        )
        return

    named = counterparties[counterparties["named"]]
    single = counterparties[~counterparties["is_aggregate"]]
    largest_single = single["pct_of_revenue"].max() if not single.empty else None

    ui.metric_row([
        ui.metric_tile("DISCLOSURES", len(counterparties), value_format="{:,.0f}"),
        ui.metric_tile("NAMED", len(named), value_format="{:,.0f}",
                       accent=THEME.green if len(named) else THEME.muted),
        ui.metric_tile("LARGEST SINGLE", largest_single,
                       subtitle="% of revenue", value_format="{:,.0f}%",
                       accent=THEME.red if (largest_single or 0) >= 25
                       else THEME.amber),
    ], columns=3)

    display = counterparties[["counterparty", "relationship",
                              "pct_of_revenue", "named", "is_aggregate"]].copy()
    ui.styled_table(display, height=260)


def page_supply_chain() -> None:
    ticker = st.session_state["ticker"]
    ui.module_header("SUPPLY CHAIN ANALYSIS",
                     "COUNTERPARTY · GEOGRAPHIC · COMMODITY · CREDIT EXPOSURE")

    controls = st.columns([3, 1])
    with controls[0]:
        entered = st.text_input("TICKER", value=ticker, key="splc_ticker").upper()
        if entered and entered != ticker:
            st.session_state["ticker"] = equities.normalize_ticker(entered)
            st.rerun()
    with controls[1]:
        st.write("")
        if st.button("REBUILD", use_container_width=True):
            # Every figure on this page is memoised in the project's SQLite
            # cache, which st.cache_data.clear() does not reach - clearing it
            # dropped the tape quotes and left the map untouched. Flag the
            # rebuild and let the fetch below refetch this issuer.
            st.session_state["splc_refresh"] = True
            st.rerun()

    st.markdown(
        f'<div style="border-left:3px solid {THEME.amber};padding:6px 10px;'
        f'background:{THEME.bg_panel};font-size:11px;color:{THEME.muted};'
        f'margin-bottom:8px;">'
        f'Every relationship below is mined from this issuer\'s own SEC '
        f'filings and links back to the source document. Bloomberg\'s SPLC '
        f'adds analyst-curated links and proprietary estimates for pairs '
        f'nobody discloses — that layer is not reproducible from free data, '
        f'so it is absent here rather than guessed at.</div>',
        unsafe_allow_html=True,
    )

    rebuilding = st.session_state.pop("splc_refresh", False)
    spinner = ("Refetching filings and rebuilding the network…" if rebuilding
               else "Mining filings and assembling the network…")

    with st.spinner(spinner):
        if rebuilding:
            _safe(supply_chain.refresh, ticker)
        network = _safe(supply_chain.build_network, ticker, default={})
        counterparties = _safe(supply_chain.get_counterparties, ticker,
                               default=pd.DataFrame())

    stats = network.get("stats", {}) if network else {}
    verdict, explanation = supply_chain.concentration_verdict(stats)

    top = stats.get("max_customer_pct") or 0
    ui.alert(f"{verdict} — {explanation}",
             "error" if top >= 50 else "warn" if top >= 25 else "ok")

    ui.metric_row([
        ui.metric_tile("COUNTERPARTIES", stats.get("counterparties"),
                       value_format="{:,.0f}"),
        ui.metric_tile("NAMED", stats.get("named"), value_format="{:,.0f}",
                       subtitle="issuer identified them",
                       accent=THEME.green if stats.get("named") else THEME.muted),
        ui.metric_tile("CUSTOMERS", stats.get("customers"), value_format="{:,.0f}"),
        ui.metric_tile("SUPPLIERS", stats.get("suppliers"), value_format="{:,.0f}"),
        ui.metric_tile("COMPARABLES", stats.get("peers"), value_format="{:,.0f}",
                       subtitle=stats.get("industry") or "industry peers"),
        ui.metric_tile("TOP EXPOSURE", stats.get("max_customer_pct"),
                       subtitle="% of revenue", value_format="{:,.0f}%",
                       accent=THEME.red if top >= 25 else THEME.amber),
    ], columns=6)

    tabs = st.tabs(["NETWORK MAP", "COUNTERPARTY DETAIL", "GEOGRAPHIC EXPOSURE",
                    "COMMODITY DEPENDENCY", "CREDIT RISK"])

    # ---- NETWORK ---------------------------------------------------------
    with tabs[0]:
        if not network or not network.get("nodes"):
            ui.alert("Nothing to map for this issuer.", "warn")
        else:
            ui.render_chart(ui.supply_chain_graph(network),
                            key=f"splc_net_{ticker}")
            st.caption(
                "Suppliers left, customers right, each hub carrying the total "
                "count for that side. A thick solid edge is a named "
                "counterparty; a dotted edge is a disclosure where the issuer "
                "withheld the name. Comparables below are the issuer's own "
                "industry classification ranked by market weight — a peer "
                "group derived per company, not a fixed list."
            )

    # ---- COUNTERPARTIES --------------------------------------------------
    with tabs[1]:
        if counterparties.empty:
            ui.alert(
                "No concentration disclosure in the latest 10-K. Above 10% of "
                "revenue a US issuer must disclose the exposure — but it is "
                "not obliged to name the partner, and most decline.", "warn",
            )
        else:
            for _, row in counterparties.iterrows():
                accent = THEME.green if row["named"] else THEME.muted
                tag = ui.badge(row["relationship"].upper(),
                               "green" if row["relationship"] == "customer" else "cyan")
                if row.get("is_aggregate"):
                    tag += ui.badge("COMBINED GROUP", "cyan")
                elif not row["named"]:
                    tag += ui.badge("NAME WITHHELD", "muted")
                # pd.notna, not truthiness: an unnamed row carries NaN here,
                # and NaN is truthy - it rendered a badge reading "NAN".
                if pd.notna(row["counterparty_ticker"]):
                    tag += ui.badge(row["counterparty_ticker"], "amber")

                st.markdown(
                    f'<div style="border:1px solid {THEME.border};border-left:3px '
                    f'solid {accent};padding:7px 10px;margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:{THEME.white};font-size:13px;font-weight:700;">'
                    f'{row["counterparty"]}</span>'
                    f'<span style="color:{THEME.amber};font-size:14px;">'
                    f'{row["pct_of_revenue"]:.0f}%</span></div>'
                    f'<div style="margin:3px 0;">{tag}</div>'
                    f'<div style="color:{THEME.muted};font-size:10px;'
                    f'font-style:italic;">“{row["quote"]}”</div>'
                    f'<a href="{row["source_url"]}" target="_blank" '
                    f'style="font-size:9px;color:{THEME.cyan};">SOURCE FILING →</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---- GEOGRAPHY -------------------------------------------------------
    with tabs[2]:
        geography = _safe(supply_chain.get_geographic_revenue, ticker,
                          default=pd.DataFrame())
        if geography.empty:
            ui.alert(
                "This issuer doesn't break revenue out by geography in its "
                "10-K. Some report only property and equipment by region, or "
                "disaggregate by end market instead.", "warn",
            )
        else:
            coverage = geography["coverage_pct"].iloc[0] \
                if "coverage_pct" in geography.columns else None
            if pd.notna(coverage) and coverage < 95:
                ui.alert(
                    f"These regions account for {coverage:.0f}% of "
                    f"consolidated revenue — the issuer breaks out only its "
                    f"largest markets, so shares below are of the disclosed "
                    f"portion, not of total revenue.", "warn",
                )

            map_col, bar_col = st.columns([3, 2])
            with map_col:
                ui.render_chart(maps.exposure_map(geography),
                                key=f"splc_geo_{ticker}")
            with bar_col:
                ui.render_chart(ui.exposure_bars(
                    geography, "region", "pct", "SHARE OF DISCLOSED REVENUE",
                    height=340))

            ui.styled_table(
                geography[["region", "revenue", "pct"]],
                highlight_columns=["pct"], height=220,
            )
            st.caption(f"Source: {geography.iloc[0]['source']} — parsed from "
                       f"the filing's XBRL.")

    # ---- COMMODITY -------------------------------------------------------
    with tabs[3]:
        commodities = _safe(supply_chain.get_commodity_exposure, ticker,
                            default=pd.DataFrame())
        ui.alert(
            "This is a statistical association, not a disclosed input cost. A "
            "high correlation can mean the company sells the commodity, buys "
            "it, or merely shares a demand cycle with it. Direction is not "
            "identified here.", "warn",
        )
        if commodities.empty:
            st.info("No commodity correlation above the 0.15 threshold.")
        else:
            ui.render_chart(ui.exposure_bars(
                commodities, "commodity", "correlation",
                "RETURN CORRELATION — 2Y DAILY", height=330, suffix=""))
            ui.styled_table(
                commodities[["commodity", "correlation", "beta",
                             "r_squared", "observations"]],
                highlight_columns=["correlation"], height=240,
            )

    # ---- CREDIT ----------------------------------------------------------
    with tabs[4]:
        credit = _safe(supply_chain.get_credit_risk, ticker, default={})
        if not credit or credit.get("z_score") is None:
            ui.alert(
                credit.get("note", "Z-score unavailable.")
                if credit else "Z-score unavailable.", "warn",
            )
        else:
            z = credit["z_score"]
            color = {"SAFE": THEME.green, "GREY": THEME.amber,
                     "DISTRESS": THEME.red}.get(credit["band"], THEME.muted)
            ui.metric_row([
                ui.metric_tile("ALTMAN Z", z, value_format="{:,.2f}",
                               subtitle=credit["band"], accent=color),
                ui.metric_tile("DEFAULT RISK", credit["risk"], accent=color),
                ui.metric_tile("MODEL FIT", credit["model_fit"],
                               subtitle=credit.get("sector", ""),
                               accent=THEME.amber
                               if credit["model_fit"] == "POOR" else THEME.green),
            ], columns=3)

            if credit["model_fit"] == "POOR":
                ui.alert(credit["note"], "warn")

            components = pd.DataFrame(
                [{"component": key.replace("_", " ").title(), "value": value}
                 for key, value in credit["components"].items()
                 if value is not None])
            if not components.empty:
                ui.render_chart(ui.exposure_bars(
                    components, "component", "value",
                    "Z-SCORE COMPONENTS", height=280, suffix=""))
            st.caption(
                "Z = 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + "
                "1.0·Sales/TA. Above 2.99 safe, 1.81–2.99 grey, below 1.81 "
                "distress. Calibrated on manufacturers — treat banks, "
                "insurers and REITs as out of scope."
            )


# ==========================================================================
# PAGE: MARITIME
# ==========================================================================
def page_maritime() -> None:
    ui.module_header("MARITIME OSINT", "AIS VESSEL TRACKING & CHOKEPOINT MONITOR")

    availability = maritime.source_availability()
    live_sources = [name for name, meta in availability.items() if meta["available"]]

    if not live_sources:
        ui.alert(
            "No AIS source is configured. The fastest fix is a free API key "
            "from aisstream.io — add AISSTREAM_API_KEY to your .env and "
            "restart.", "error",
        )
        with st.expander("SOURCE OPTIONS", expanded=True):
            for name, meta in availability.items():
                st.markdown(
                    f"**{name}** — {meta['quality']}  \n"
                    f"<span style='color:{THEME.muted};font-size:11px;'>"
                    f"{meta['note']}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "---\n"
                "**On the scraper options:** MarineTraffic and VesselFinder "
                "both prohibit automated collection in their terms of service "
                "and sit behind Cloudflare. Open-Terminal implements those "
                "paths but ships them disabled. Enabling them "
                "(`OPENTERM_ALLOW_SCRAPERS=1`) is your call, and they will "
                "break whenever those sites change. A $25 RTL-SDR dongle gives "
                "you better local data with none of that fragility."
            )
    else:
        st.markdown(
            " ".join(ui.badge(f"{name.upper()} ACTIVE", "green") for name in live_sources),
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["CHOKEPOINT MONITOR", "VESSEL LOOKUP", "AREA SCAN"])

    # ---- CHOKEPOINTS -----------------------------------------------------
    with tabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            selected = st.selectbox(
                "CORRIDOR",
                list(config.CHOKEPOINTS),
                index=list(config.CHOKEPOINTS).index(st.session_state["chokepoint"])
                if st.session_state["chokepoint"] in config.CHOKEPOINTS else 0,
                format_func=lambda code: config.CHOKEPOINTS[code].name,
                key="cp_select",
            )
            st.session_state["chokepoint"] = selected
        with col_b:
            listen = st.slider("LISTEN (sec)", 5, 60, 20, key="cp_listen")
        with col_c:
            st.write("")
            scan = st.button("SCAN CORRIDOR", use_container_width=True)

        chokepoint = config.CHOKEPOINTS[selected]
        st.caption(chokepoint.description)

        if scan or st.session_state.get("cp_last") == selected:
            st.session_state["cp_last"] = selected
            with st.spinner(f"Collecting AIS over {chokepoint.name} ({listen}s)…"):
                status = _safe(maritime.get_chokepoint_status, selected, float(listen),
                               default={})

            if not status or status.get("vessel_count", 0) == 0:
                ui.alert(status.get("coverage_note",
                                    "No vessels observed in this corridor."), "warn")
            else:
                congestion = status.get("congestion_ratio")
                status_color = {
                    "SEVERE CONGESTION": THEME.red, "ELEVATED": THEME.amber,
                    "NORMAL": THEME.green, "LIGHT TRAFFIC": THEME.cyan,
                }.get(status.get("status"), THEME.muted)

                ui.metric_row([
                    ui.metric_tile("VESSELS", status.get("vessel_count"),
                                   value_format="{:,.0f}", accent=status_color),
                    ui.metric_tile("ANCHORED", status.get("anchored_count"),
                                   subtitle="queue", value_format="{:,.0f}"),
                    ui.metric_tile("UNDER WAY", status.get("underway_count"),
                                   subtitle="flow", value_format="{:,.0f}"),
                    ui.metric_tile("TANKERS", status.get("tanker_count"),
                                   value_format="{:,.0f}", accent=THEME.amber),
                    ui.metric_tile(
                        "CONGESTION", congestion,
                        subtitle=("× measured baseline" if congestion is not None
                                  else "baseline building"),
                        value_format="{:.2f}", accent=status_color),
                    ui.metric_tile("MEAN DRAUGHT", status.get("mean_draught_m"),
                                   subtitle="m — laden proxy"),
                ], columns=6)

                if status.get("baseline_note"):
                    st.caption(status["baseline_note"])

                vessels = status.get("vessels", pd.DataFrame())

                map_col, detail_col = st.columns([3, 2])
                with map_col:
                    if maps.FOLIUM_AVAILABLE:
                        fmap = maps.vessel_map(vessels, bbox=chokepoint.bbox, zoom=8)
                        maps.render_folium(fmap, height=520, key=f"cpmap_{selected}")
                    else:
                        ui.render_chart(maps.vessel_map_plotly(
                            vessels, chokepoint.bbox, chokepoint.name.upper()))

                with detail_col:
                    composition = status.get("composition")
                    if composition:
                        st.markdown("#### FLEET COMPOSITION")
                        ui.render_chart(ui.line_chart(
                            {"Vessels": pd.Series(composition)},
                            "BY SHIP TYPE", height=240,
                        ) if False else _composition_figure(composition))

                    st.markdown("#### VESSEL LIST")
                    if not vessels.empty:
                        columns = [c for c in ("name", "mmsi", "ship_type",
                                               "sog_kts", "nav_status", "destination")
                                   if c in vessels.columns]
                        st.dataframe(vessels[columns].head(60),
                                     use_container_width=True, hide_index=True,
                                     height=270)

                st.caption(
                    f"Source: {status.get('source', 'unknown')} · "
                    f"Counts reflect AIS coverage in this box, not a census. "
                    f"Low numbers can mean light traffic or thin receiver coverage."
                )
        else:
            st.info("Press SCAN CORRIDOR to collect live AIS for this area.")

        st.divider()
        st.markdown("### ALL CORRIDORS")
        if st.button("SCAN ALL (slow)", key="scan_all"):
            with st.spinner("Sweeping every corridor…"):
                summary = maritime.get_all_chokepoints(stream_seconds=10)
            ui.styled_table(summary, highlight_columns=["Congestion"])

            if maps.FOLIUM_AVAILABLE:
                statuses = [
                    maritime.get_chokepoint_status(code, stream_seconds=1)
                    for code in config.CHOKEPOINTS
                ]
                maps.render_folium(maps.chokepoint_overview_map(statuses),
                                   height=440, key="cp_overview")

    # ---- VESSEL LOOKUP ---------------------------------------------------
    with tabs[1]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            identifier = st.text_input(
                "MMSI OR IMO",
                value=st.session_state.get("vessel_query", ""),
                placeholder="636019825  or  IMO9321483",
                key="vessel_input",
            )
        with col_b:
            listen = st.slider("LISTEN (sec)", 10, 90, 30, key="vessel_listen")
        with col_c:
            st.write("")
            lookup = st.button("TRACK VESSEL", use_container_width=True)

        if lookup and identifier:
            with st.spinner(f"Listening for {identifier} ({listen}s)…"):
                result = _safe(maritime.track_vessel, identifier, float(listen),
                               default={})

            if not result.get("found"):
                ui.alert(result.get("note", "Vessel not found."), "warn")
            else:
                vessel = result["vessel"]
                ui.metric_row([
                    ui.metric_tile("VESSEL", vessel.get("name") or vessel.get("mmsi")),
                    ui.metric_tile("FLAG", result.get("flag", "—")),
                    ui.metric_tile("SPEED", vessel.get("sog_kts"), subtitle="knots"),
                    ui.metric_tile("COURSE", vessel.get("cog_deg"), subtitle="degrees"),
                    ui.metric_tile("DRAUGHT", vessel.get("draught_m"), subtitle="metres"),
                    ui.metric_tile("STATUS", vessel.get("nav_status") or "—"),
                ], columns=6)

                frame = pd.DataFrame([vessel])
                if maps.FOLIUM_AVAILABLE:
                    maps.render_folium(
                        maps.vessel_map(frame, zoom=9), height=460, key="vessel_map")
                else:
                    ui.render_chart(maps.vessel_map_plotly(frame, title="VESSEL POSITION"))

                with st.expander("FULL AIS RECORD"):
                    st.json(vessel)

                st.caption(
                    "Flag state comes from the MMSI's country prefix. Note that "
                    "flag of registry says nothing about beneficial ownership — "
                    "Panama, Liberia and the Marshall Islands are flags of "
                    "convenience."
                )

    # ---- AREA SCAN -------------------------------------------------------
    with tabs[2]:
        st.markdown("Define a custom bounding box to sweep.")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            min_lat = st.number_input("MIN LAT", -90.0, 90.0, 29.0, 0.5)
        with col_b:
            max_lat = st.number_input("MAX LAT", -90.0, 90.0, 32.0, 0.5)
        with col_c:
            min_lon = st.number_input("MIN LON", -180.0, 180.0, 32.0, 0.5)
        with col_d:
            max_lon = st.number_input("MAX LON", -180.0, 180.0, 34.0, 0.5)

        listen = st.slider("LISTEN (sec)", 5, 90, 25, key="area_listen")

        if st.button("SCAN AREA", use_container_width=True):
            bbox = (min_lat, min_lon, max_lat, max_lon)
            with st.spinner(f"Collecting AIS ({listen}s)…"):
                vessels = _safe(maritime.get_vessels_in_area, bbox, 800, float(listen),
                                default=pd.DataFrame())

            if vessels.empty:
                ui.alert("No vessels observed in this box.", "warn")
            else:
                st.success(f"{len(vessels)} vessels")
                view = st.radio("VIEW", ["MARKERS", "DENSITY"], horizontal=True)
                if view == "DENSITY":
                    ui.render_chart(maps.density_map(vessels, "VESSEL DENSITY"))
                elif maps.FOLIUM_AVAILABLE:
                    maps.render_folium(maps.vessel_map(vessels, bbox=bbox),
                                       height=520, key="area_map")
                else:
                    ui.render_chart(maps.vessel_map_plotly(vessels, bbox))

                ui.styled_table(vessels, height=340)


def _composition_figure(composition: Dict[str, int]):
    """Horizontal bar of ship types in a corridor."""
    import plotly.graph_objects as go
    from ui.terminal_theme import style_figure

    items = sorted(composition.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items],
        orientation="h", marker=dict(color=THEME.amber),
        text=[str(v) for _, v in items], textposition="auto",
    ))
    return style_figure(fig, height=250, title="FLEET COMPOSITION", showlegend=False)


# ==========================================================================
# PAGE: AVIATION
# ==========================================================================
def page_aviation() -> None:
    ui.module_header("AVIATION OSINT", "LIVE ADS-B VIA OPENSKY NETWORK")

    if aviation.is_authenticated():
        st.markdown(ui.badge("OAUTH2 — 4000 CREDITS/DAY", "green"),
                    unsafe_allow_html=True)
    else:
        st.markdown(
            ui.badge("ANONYMOUS — 400 CREDITS/DAY", "amber") +
            ui.badge("REGISTER FREE AT OPENSKY-NETWORK.ORG FOR 10×", "muted"),
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["LIVE TRAFFIC", "FLEET WATCHLIST",
                    "AIRCRAFT TRACK", "AIRPORT FLOW"])

    # ---- LIVE TRAFFIC ----------------------------------------------------
    with tabs[0]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            region = st.selectbox(
                "REGION", list(config.AVIATION_REGIONS),
                index=list(config.AVIATION_REGIONS).index(
                    st.session_state["aviation_region"]),
                key="av_region",
            )
            st.session_state["aviation_region"] = region
        with col_b:
            color_by = st.selectbox("COLOUR BY",
                                    ["altitude_ft", "speed_kts", "none"],
                                    key="av_color")
        with col_c:
            st.write("")
            refresh = st.button("REFRESH", use_container_width=True)

        if region == "GLOBAL":
            st.caption(
                "A global query costs roughly 4× the API credits of a regional "
                "one. Prefer a region unless you need worldwide coverage."
            )

        with st.spinner(f"Querying OpenSky for {region}…"):
            states = _safe(aviation.get_states_by_region, region,
                           default=pd.DataFrame())

        if states.empty:
            ui.alert(
                "No aircraft returned. Either the credit budget is exhausted "
                "(anonymous access is capped at 400/day) or OpenSky is "
                "temporarily unavailable. Cached data will be served if any "
                "exists.", "warn",
            )
        else:
            states = aviation.add_operator_column(states)
            summary = aviation.summarize_traffic(states)

            ui.metric_row([
                ui.metric_tile("TRACKED", summary.get("total"), value_format="{:,.0f}"),
                ui.metric_tile("AIRBORNE", summary.get("airborne"), value_format="{:,.0f}"),
                ui.metric_tile("ON GROUND", summary.get("on_ground"), value_format="{:,.0f}"),
                ui.metric_tile("MEAN ALT", summary.get("mean_altitude_ft"),
                               subtitle="ft", value_format="{:,.0f}"),
                ui.metric_tile("MEAN SPEED", summary.get("mean_speed_kts"),
                               subtitle="kts", value_format="{:,.0f}"),
                ui.metric_tile("EMERGENCIES", summary.get("emergency_count", 0),
                               value_format="{:,.0f}",
                               accent=THEME.red if summary.get("emergency_count") else THEME.amber),
            ], columns=6)

            if summary.get("emergencies"):
                for emergency in summary["emergencies"]:
                    ui.alert(
                        f"SQUAWK {emergency.get('squawk')} — {emergency.get('alert')} — "
                        f"{emergency.get('callsign') or emergency.get('icao24')} "
                        f"({emergency.get('origin_country')})",
                        "error",
                    )

            ui.render_chart(maps.flight_map(
                states, f"LIVE TRAFFIC — {region}",
                color_by=None if color_by == "none" else color_by,
            ))

            left, right = st.columns(2)
            with left:
                st.markdown("#### TOP ORIGIN COUNTRIES")
                countries = summary.get("top_countries", {})
                if countries:
                    ui.styled_table(
                        pd.DataFrame(list(countries.items()),
                                     columns=["Country", "Aircraft"]),
                        numeric_format="{:,.0f}", height=280,
                    )
            with right:
                st.markdown("#### IDENTIFIED OPERATORS")
                known = states[states["operator"] != "UNKNOWN"]
                if known.empty:
                    st.caption("No callsigns matched the operator table.")
                else:
                    ui.styled_table(
                        known["operator"].value_counts().reset_index()
                        .rename(columns={"operator": "Operator", "count": "Aircraft"}),
                        numeric_format="{:,.0f}", height=280,
                    )

            with st.expander("RAW STATE VECTORS"):
                columns = [c for c in ("icao24", "callsign", "operator",
                                       "origin_country", "altitude_ft",
                                       "speed_kts", "true_track", "phase",
                                       "squawk", "source")
                           if c in states.columns]
                st.dataframe(states[columns].head(400),
                             use_container_width=True, hide_index=True)

    # ---- WATCHLIST -------------------------------------------------------
    with tabs[1]:
        st.caption(
            "Fleets are matched on the ICAO operator designator each aircraft "
            "is broadcasting in its callsign, so every row is something "
            "OpenSky can see right now — no stored list of airframe "
            "identities. Institutional operators only: Open-Terminal ships no "
            "private-individual watchlist; sustained tracking of a named "
            "person's aircraft is treated very differently in law from "
            "fleet-level analysis, and OpenSky's terms restrict it."
        )

        wl_col, region_col = st.columns(2)
        with wl_col:
            watchlist = st.selectbox("FLEET",
                                     ["ALL"] + list(config.OPERATOR_FLEETS),
                                     key="av_watchlist")
        with region_col:
            wl_region = st.selectbox("SEARCH REGION",
                                     list(config.AVIATION_REGIONS),
                                     key="av_wl_region")

        if st.button("QUERY WATCHLIST", use_container_width=True):
            with st.spinner(f"Matching live callsigns over {wl_region}…"):
                tracked = _safe(aviation.track_watchlist,
                                None if watchlist == "ALL" else watchlist,
                                wl_region,
                                default=pd.DataFrame())

            if tracked.empty:
                ui.alert(
                    f"No aircraft from this fleet are transmitting over "
                    f"{wl_region} right now. Try a wider region — absence of "
                    "a signal is not evidence of anything.", "warn",
                )
            else:
                st.success(f"{len(tracked)} aircraft transmitting")
                ui.render_chart(maps.flight_map(tracked, "WATCHLIST POSITIONS",
                                                show_labels=True))
                columns = [c for c in ("operator", "callsign", "icao24",
                                       "watchlist", "altitude_ft", "speed_kts",
                                       "phase", "origin_country")
                           if c in tracked.columns]
                ui.styled_table(tracked[columns])

        with st.expander("FLEET DESIGNATORS"):
            st.caption(
                "ICAO Doc 8585 three-letter operator designators. These are a "
                "published standard, not an inventory — the aircraft behind "
                "each one are whatever is airborne at query time."
            )
            for group, entries in config.OPERATOR_FLEETS.items():
                st.markdown(f"**{group}**")
                st.dataframe(
                    pd.DataFrame(list(entries.items()),
                                 columns=["Designator", "Operator"]),
                    use_container_width=True, hide_index=True,
                )

    # ---- TRACK -----------------------------------------------------------
    with tabs[2]:
        icao = st.text_input("ICAO24 HEX ADDRESS", placeholder="a1b2c3",
                             key="track_icao").strip().lower()

        if icao and st.button("FETCH TRACK", use_container_width=True):
            track = _safe(aviation.get_aircraft_track, icao, default=pd.DataFrame())
            if track.empty:
                ui.alert(
                    "No track available. OpenSky's /tracks endpoint is "
                    "officially experimental and is often closed to anonymous "
                    "callers — an empty result here is common and not "
                    "necessarily an error.", "warn",
                )
            else:
                ui.render_chart(maps.flight_track_map(
                    track, f"TRACK — {track.iloc[0].get('callsign') or icao}"))
                ui.styled_table(track[["timestamp", "latitude", "longitude",
                                       "altitude_ft", "true_track"]], height=300)

            history = _safe(aviation.get_flights_by_aircraft, icao, 14,
                            default=pd.DataFrame())
            if not history.empty:
                st.markdown("#### RECENT FLIGHTS (14 DAYS)")
                ui.styled_table(history)

    # ---- AIRPORT ---------------------------------------------------------
    with tabs[3]:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            airport = st.text_input("AIRPORT ICAO", value="KJFK",
                                    key="airport_icao").strip().upper()
        with col_b:
            direction = st.selectbox("DIRECTION", ["arrival", "departure"],
                                     key="airport_dir")
        with col_c:
            hours = st.slider("HOURS BACK", 1, 48, 12, key="airport_hours")

        if st.button("QUERY AIRPORT", use_container_width=True):
            traffic = _safe(aviation.get_airport_traffic, airport, hours, direction,
                            default=pd.DataFrame())
            if traffic.empty:
                ui.alert(
                    f"No {direction} data for {airport}. Check the ICAO code "
                    f"(4 letters — KJFK not JFK) and note that OpenSky's "
                    f"coverage of some regions is thin.", "warn",
                )
            else:
                traffic["operator"] = traffic["callsign"].map(aviation.identify_operator)
                ui.metric_row([
                    ui.metric_tile("FLIGHTS", len(traffic), value_format="{:,.0f}"),
                    ui.metric_tile("PER HOUR", len(traffic) / max(hours, 1),
                                   value_format="{:,.1f}"),
                    ui.metric_tile("OPERATORS",
                                   traffic["operator"].nunique(), value_format="{:,.0f}"),
                ], columns=3)
                ui.styled_table(traffic, height=460)


# ==========================================================================
# PAGE: MACRO
# ==========================================================================
def page_macro() -> None:
    ui.module_header("MACROECONOMICS", "YIELD CURVES · INFLATION · POLICY")

    if not config.FRED_API_KEY:
        st.caption(
            "No FRED API key configured — using FRED's keyless CSV endpoint. "
            "Everything works; you just lose series metadata. Free key at "
            "fredaccount.stlouisfed.org/apikeys."
        )

    tabs = st.tabs(["REGIME", "YIELD CURVE", "RECESSION SIGNALS", "INFLATION",
                    "LABOR & GROWTH", "LIQUIDITY", "GLOBAL"])

    # ---- REGIME ----------------------------------------------------------
    with tabs[0]:
        _render_regime_tab()

    # ---- YIELD CURVE -----------------------------------------------------
    with tabs[1]:
        curve = _safe(macro.get_yield_curve, default=pd.DataFrame())

        if curve.empty:
            ui.alert("Yield curve data unavailable from every source.", "error")
        else:
            analysis = macro.analyze_yield_curve(curve)

            spreads = analysis.get("spreads", {})
            ui.metric_row([
                ui.metric_tile("SHAPE", analysis.get("shape", "—"),
                               accent=THEME.red if analysis.get("is_inverted") else THEME.green),
                ui.metric_tile("10Y-2Y", (spreads.get("10Y-2Y") or 0) * 100,
                               subtitle="basis points", value_format="{:+,.0f}",
                               accent=THEME.red if (spreads.get("10Y-2Y") or 0) < 0 else THEME.green),
                ui.metric_tile("10Y-3M", (spreads.get("10Y-3M") or 0) * 100,
                               subtitle="basis points", value_format="{:+,.0f}",
                               accent=THEME.red if (spreads.get("10Y-3M") or 0) < 0 else THEME.green),
                ui.metric_tile("30D CHANGE", analysis.get("spread_change_30d_bps"),
                               subtitle=analysis.get("curve_direction", ""),
                               value_format="{:+,.0f}"),
                ui.metric_tile("RECESSION RISK", analysis.get("recession_risk", "—"),
                               accent={"ELEVATED": THEME.red, "MODERATE": THEME.amber,
                                       "WATCH": THEME.amber}.get(
                                   analysis.get("recession_risk"), THEME.green)),
            ], columns=5)

            if analysis.get("is_inverted"):
                for inversion in analysis["inversions"]:
                    ui.alert(
                        f"{inversion['pair']} INVERTED — {inversion['spread_bps']:+.0f}bp "
                        f"({inversion['severity']})", "warn",
                    )
            st.caption(analysis.get("recession_note", ""))

            left, right = st.columns([3, 2])
            with left:
                ui.render_chart(_yield_curve_figure(curve, analysis, height=420))
            with right:
                history = _safe(macro.get_yield_curve_history, 730,
                                default=pd.DataFrame())
                if not history.empty and {"10Y", "2Y", "3M"}.issubset(history.columns):
                    series_map = {
                        "10Y-2Y": history["10Y"] - history["2Y"],
                        "10Y-3M": history["10Y"] - history["3M"],
                    }
                    fig = ui.line_chart(series_map, "SPREAD HISTORY", "%", height=420)
                    fig.add_hline(y=0, line=dict(color=THEME.red, width=1.4,
                                                 dash="dash"))
                    ui.render_chart(fig)

            st.markdown("#### CURVE DETAIL")
            display = curve.copy()
            display["yield"] = display["yield"].round(3)
            ui.styled_table(display[["tenor", "yield", "date"]],
                            numeric_format="{:.3f}")

            # Policy expectations
            st.markdown("#### POLICY EXPECTATIONS")
            expectations = _safe(macro.get_rate_expectations, default={})
            if expectations:
                ui.metric_row([
                    ui.metric_tile("FED FUNDS (EFFECTIVE)",
                                   expectations.get("effective_rate"), subtitle="%"),
                    ui.metric_tile("TARGET UPPER",
                                   expectations.get("current_target_upper"), subtitle="%"),
                    ui.metric_tile("FUTURES IMPLIED",
                                   expectations.get("implied_rate_front_contract"),
                                   subtitle="front ZQ contract"),
                    ui.metric_tile("IMPLIED MOVES",
                                   expectations.get("implied_moves"),
                                   subtitle=expectations.get("policy_bias", "")),
                    ui.metric_tile("2Y SIGNAL", expectations.get("ust2y_signal", "—"),
                                   subtitle=f"{expectations.get('ust2y_3m_change_bps', 0):+.0f}bp / 3m"),
                ], columns=5)
                st.caption(
                    "Derived from public Fed Funds futures prices (implied rate "
                    "= 100 − price). This is a directional read, not CME "
                    "FedWatch's probability distribution."
                )

    # ---- RECESSION -------------------------------------------------------
    with tabs[2]:
        indicators = _safe(macro.get_recession_indicators, default={})

        if not indicators or not indicators.get("signals"):
            ui.alert("Recession indicators unavailable.", "warn")
        else:
            gauge_col, table_col = st.columns([1, 2])
            with gauge_col:
                ui.render_chart(ui.gauge(
                    indicators["score"], "COMPOSITE RISK SCORE",
                    thresholds=[(20, THEME.green), (45, THEME.amber), (100, THEME.red)],
                    suffix="", height=260,
                ))
                st.markdown(
                    f'<div style="text-align:center;color:{THEME.amber};'
                    f'font-size:15px;font-weight:700;letter-spacing:0.12em;">'
                    f'{indicators["level"]}</div>'
                    f'<div style="text-align:center;color:{THEME.muted};font-size:10px;">'
                    f'{indicators["triggered_count"]} of {indicators["total_count"]} '
                    f'signals triggered</div>',
                    unsafe_allow_html=True,
                )

            with table_col:
                st.markdown("#### SIGNAL DETAIL")
                for signal in indicators["signals"]:
                    color = THEME.red if signal["triggered"] else THEME.green
                    marker = "⚠ TRIGGERED" if signal["triggered"] else "✔ CLEAR"
                    st.markdown(
                        f'<div style="border-left:3px solid {color};'
                        f'background:{THEME.bg_panel};padding:7px 12px;margin:5px 0;">'
                        f'<span style="color:{THEME.amber};font-size:12px;">'
                        f'{signal["name"]}</span>'
                        f'<span style="float:right;color:{color};font-size:10px;">'
                        f'{marker}</span><br>'
                        f'<span style="color:{THEME.muted};font-size:10px;">'
                        f'{signal["detail"]} · weight {signal["weight"]:.0f}</span></div>',
                        unsafe_allow_html=True,
                    )

            st.caption(
                "This aggregates public FRED series into a weighted score. It "
                "is a summary of what the data currently says, not a forecast, "
                "and it carries no confidence interval. Every one of these "
                "signals has produced false positives historically."
            )

    # ---- SERIES BROWSERS -------------------------------------------------
    for tab, group in ((tabs[3], "INFLATION"), (tabs[4], "LABOR"),
                       (tabs[5], "LIQUIDITY")):
        with tab:
            _render_series_group(group)
            if group == "LABOR":
                st.divider()
                _render_series_group("GROWTH")

    # ---- GLOBAL ----------------------------------------------------------
    with tabs[6]:
        indicator = st.selectbox(
            "WORLD BANK INDICATOR",
            list(config.WORLD_BANK_INDICATORS),
            format_func=lambda code: config.WORLD_BANK_INDICATORS[code],
            key="wb_indicator",
        )
        countries = st.text_input(
            "COUNTRIES (ISO2, semicolon-separated)",
            value="US;CN;JP;DE;GB;IN", key="wb_countries",
        )

        data = _safe(macro.get_world_bank_indicator, indicator, countries, 20,
                     default=pd.DataFrame())
        if data.empty:
            ui.alert("World Bank returned no data for this selection.", "warn")
        else:
            series_map = {
                country: group.set_index("year")["value"]
                for country, group in data.groupby("country")
            }
            ui.render_chart(ui.line_chart(
                series_map, config.WORLD_BANK_INDICATORS[indicator],
                height=440,
            ))
            ui.styled_table(
                data.pivot(index="year", columns="country", values="value")
                .sort_index(ascending=False).head(20)
            )


def _regime_accent(verdict: Dict[str, Any]) -> str:
    """Theme colour for a regime verdict, or muted when there is no call."""
    return getattr(THEME, verdict.get("colour") or "", THEME.muted)


def _regime_quadrant_figure(verdict: Dict[str, Any], height: int = 360):
    """
    Where the two axis scores put us on the growth/inflation matrix.

    `ui.gauge` is semicircular and reads one number; this reads two, so it is
    a small scatter rather than a new reusable component. Styling goes through
    the shared `style_figure`, so it inherits the terminal palette and needs
    no CSS of its own.
    """
    import plotly.graph_objects as go
    from ui.terminal_theme import style_figure

    bound = 2.5
    fig = go.Figure()

    # Quadrant backgrounds, in the same order as config.REGIME_QUADRANTS.
    corners = [
        ((0, bound), (-bound, 0), "GOLDILOCKS", THEME.green),
        ((0, bound), (0, bound), "OVERHEATING", THEME.amber),
        ((-bound, 0), (0, bound), "STAGFLATION", THEME.red),
        ((-bound, 0), (-bound, 0), "DEFLATIONARY\nBUST", THEME.cyan),
    ]
    for (x0, x1), (y0, y1), label, colour in corners:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(width=0),
                      fillcolor=f"rgba({ui._hex_to_rgb(colour)},0.07)",
                      layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=9, color=colour, family=THEME.font_mono),
            opacity=0.75,
        )

    for axis in ("x", "y"):
        fig.add_shape(
            type="line", layer="below",
            x0=-bound if axis == "x" else 0, x1=bound if axis == "x" else 0,
            y0=0 if axis == "x" else -bound, y1=0 if axis == "x" else bound,
            line=dict(color=THEME.border, width=1),
        )

    growth = verdict.get("growth_z")
    inflation = verdict.get("inflation_z")
    if growth is not None and inflation is not None:
        accent = _regime_accent(verdict)
        fig.add_trace(go.Scatter(
            x=[growth], y=[inflation], mode="markers+text",
            marker=dict(size=17, color=accent, symbol="circle",
                        line=dict(color=THEME.bg, width=2)),
            text=[verdict.get("regime", "")], textposition="top center",
            textfont=dict(size=10, color=accent, family=THEME.font_mono),
            hovertemplate=(f"Growth {growth:+.2f}<br>"
                           f"Inflation {inflation:+.2f}<extra></extra>"),
            showlegend=False,
        ))

    fig.update_xaxes(title_text="GROWTH MOMENTUM (z)", range=[-bound, bound],
                     zeroline=False)
    fig.update_yaxes(title_text="INFLATION MOMENTUM (z)", range=[-bound, bound],
                     zeroline=False)
    return style_figure(fig, height=height, title="REGIME MATRIX",
                        showlegend=False)


def _regime_overlay_figure(indicator: pd.Series, indicator_label: str,
                           benchmark: pd.Series, history: pd.Series,
                           height: int = 420):
    """
    One indicator against the S&P 500, both rebased to 100, regimes shaded.

    `ui.line_chart` has no secondary y-axis, and rather than add one both
    series are standardised to mean 0 / sd 1. Rebasing to 100 was the first
    attempt and is wrong for half these series: CFNAI and NFCI oscillate
    about zero, so dividing by a first value of -0.08 inverts the line and
    multiplies it by a thousand - the chart came out spanning -5,000 to
    20,000. Standardising needs no meaningful zero, and unlike a secondary
    axis it cannot be slid until two lines appear to agree.

    The regime bands are added afterwards with `add_vrect`, the same
    post-processing the yield-curve tab already does with `add_hline`.
    """
    from utils import macro_analytics as ma

    series_map = {}
    scaled_indicator = ma.standardize(indicator)
    if not scaled_indicator.empty:
        series_map[indicator_label] = scaled_indicator
    scaled_benchmark = ma.standardize(benchmark)
    if not scaled_benchmark.empty:
        series_map["S&P 500"] = scaled_benchmark

    fig = ui.line_chart(series_map, f"{indicator_label} vs S&P 500",
                        "STANDARDISED (z)", height=height)

    if history is not None and not history.empty:
        for start, end, label in ma.runs(history):
            colour = getattr(
                THEME,
                dict((v[0], v[2]) for v in config.REGIME_QUADRANTS.values())
                .get(label, ""),
                THEME.muted,
            )
            fig.add_vrect(
                x0=start, x1=end, layer="below", line_width=0,
                fillcolor=f"rgba({ui._hex_to_rgb(colour)},0.10)",
            )

    return fig


def _render_regime_tab() -> None:
    """
    The macro regime card, metric table and overlay.

    Every element here is an existing component from `ui/components.py`; the
    only bespoke figure is the two-axis quadrant scatter, which has no
    equivalent among them.
    """
    verdict = _safe(macro_regime.classify, default={})

    if not verdict or not verdict.get("regime"):
        ui.alert(
            verdict.get("reason", "Regime classification unavailable."), "warn")
        st.caption(
            "The quadrant is only reported when both axes clear "
            f"{config.REGIME_MIN_CONTRIBUTORS} resolved contributors. Fewer "
            "than that and the call would describe which FRED fetches "
            "succeeded, not the economy."
        )
        return

    accent = _regime_accent(verdict)
    as_of = verdict.get("as_of")
    as_of_label = (pd.Timestamp(as_of).strftime("%b %Y")
                   if as_of is not None else "—")

    # --- Headline card ----------------------------------------------------
    curve = _safe(macro.get_yield_curve, default=pd.DataFrame())
    spreads = (macro.analyze_yield_curve(curve).get("spreads", {})
               if not curve.empty else {})
    liquidity = _safe(macro.get_net_liquidity, 2, default=pd.DataFrame())

    ui.metric_row([
        ui.metric_tile("REGIME", verdict["regime"], subtitle=as_of_label,
                       accent=accent),
        ui.metric_tile("GROWTH", verdict["growth_z"],
                       subtitle=f"z · {verdict['growth_n']} inputs",
                       value_format="{:+.2f}",
                       accent=THEME.green if verdict["growth_z"] > 0 else THEME.red),
        ui.metric_tile("INFLATION", verdict["inflation_z"],
                       subtitle=f"z · {verdict['inflation_n']} inputs",
                       value_format="{:+.2f}",
                       accent=THEME.red if verdict["inflation_z"] > 0 else THEME.green),
        ui.metric_tile("CONVICTION", verdict["conviction"],
                       subtitle=f"{verdict['run_months']}m in regime · "
                                f"{verdict.get('flips_12m', 0)} flips/12m",
                       accent={"HIGH": THEME.green, "MODERATE": THEME.amber}
                       .get(verdict["conviction"], THEME.muted)),
        ui.metric_tile("10Y-2Y", (spreads.get("10Y-2Y") or 0) * 100,
                       subtitle="basis points", value_format="{:+,.0f}",
                       accent=THEME.red if (spreads.get("10Y-2Y") or 0) < 0
                       else THEME.green),
        ui.metric_tile(
            "NET LIQUIDITY",
            float(liquidity["net_liquidity_bn"].iloc[-1])
            if not liquidity.empty else None,
            subtitle="$bn · WALCL − TGA − RRP", value_format="{:,.0f}"),
    ], columns=6)

    st.markdown(
        f'<div style="border-left:3px solid {accent};background:{THEME.bg_panel};'
        f'padding:8px 14px;margin:6px 0 12px 0;">'
        f'<span style="color:{accent};font-size:13px;letter-spacing:0.1em;">'
        f'{verdict["regime"]}</span>'
        f'<span style="color:{THEME.muted};font-size:11px;"> — '
        f'{verdict["posture"]}</span></div>',
        unsafe_allow_html=True,
    )

    if liquidity.empty and liquidity.attrs.get("reason"):
        ui.alert(liquidity.attrs["reason"], "warn")

    # --- Matrix + contributors -------------------------------------------
    matrix_col, detail_col = st.columns([2, 3])

    with matrix_col:
        ui.render_chart(_regime_quadrant_figure(verdict))

    with detail_col:
        st.markdown("#### AXIS CONTRIBUTORS")
        contributors = pd.DataFrame(verdict.get("contributors", []))
        if not contributors.empty:
            display = contributors.rename(columns={
                "label": "Contributor", "axis": "Axis", "z": "Momentum z",
                "weight": "Weight", "note": "Note",
            })
            ui.styled_table(
                display[["Contributor", "Axis", "Momentum z", "Weight", "Note"]],
                highlight_columns=["Momentum z"], height=250,
            )
        st.caption(
            "Each contributor is its 3-month annualised momentum, z-scored "
            "against its own 3-year history and sign-corrected so a higher "
            "score always means a stronger axis. The axis score is their "
            "weighted mean; weights live in config.REGIME_INPUTS."
        )
        if verdict.get("flips_12m", 0) >= 4:
            st.caption(
                f"This call has changed {verdict['flips_12m']} times in the "
                "last 12 months. Momentum classification is genuinely "
                "unstable when both scores sit near zero — treat a LOW "
                "conviction reading as 'no clear regime', not as a forecast."
            )

    # --- Metric table -----------------------------------------------------
    st.markdown("#### MACRO METRICS")
    metrics = _safe(macro_regime.metric_table, default=pd.DataFrame())
    if metrics.empty:
        ui.alert("No macro metrics resolved.", "warn")
    else:
        ui.styled_table(
            metrics[["Metric", "Series", "Level", "1M Chg", "YoY", "3M Mom",
                     "Z (1Y)", "Z (3Y)", "Signal", "As Of"]],
            highlight_columns=["1M Chg", "YoY", "3M Mom", "Z (1Y)", "Z (3Y)"],
        )
        st.caption(
            "Change columns follow each series' own units: percentage series "
            "(unemployment, spreads, NFCI) move in percentage POINTS, index "
            "series (CPI, payrolls, PPI) in percent. As Of is that series' "
            "last real print — they differ, because macro data has a ragged "
            "edge and nothing here is forward-filled into a change column."
        )

    # --- Overlay ----------------------------------------------------------
    st.markdown("#### INDICATOR VS S&P 500")
    panel = _safe(macro_regime.build_panel, default=pd.DataFrame())
    if panel.empty:
        return

    choices = [s.series_id for s in macro_regime._all_specs()
               if s.series_id in panel.columns]
    labels = {s.series_id: s.label for s in macro_regime._all_specs()}

    chosen = st.selectbox(
        "INDICATOR", choices, key="regime_overlay",
        format_func=lambda sid: f"{sid} — {labels.get(sid, sid)}",
    )

    benchmark = _safe(equities.get_history, "^GSPC", "10y", "1mo",
                      default=pd.DataFrame())
    benchmark_close = (benchmark["Close"] if not benchmark.empty
                       and "Close" in benchmark.columns else pd.Series(dtype=float))
    if not benchmark_close.empty:
        benchmark_close.index = pd.to_datetime(
            benchmark_close.index).tz_localize(None)

    indicator = panel[chosen].dropna()
    if not indicator.empty and not benchmark_close.empty:
        start = max(indicator.index.min(), benchmark_close.index.min())
        indicator = indicator[indicator.index >= start]
        benchmark_close = benchmark_close[benchmark_close.index >= start]

    history = _safe(macro_regime.regime_history, panel, default=pd.Series(dtype=object))
    ui.render_chart(_regime_overlay_figure(
        indicator, labels.get(chosen, chosen), benchmark_close, history))

    st.caption(
        "Shaded bands are the classified regime for each month. They are "
        "built from today's revised data placed at the date it describes — "
        "FRED stamps an observation at the start of its period, but a July "
        "CPI print is not public until mid-August and is revised for months "
        "after. This strip therefore looks more prescient than any real-time "
        "reading was, and is a description of history, not a backtest. "
        "Neither it nor the regime call is investment advice."
    )


def _render_series_group(group: str) -> None:
    """Chart + table for one MACRO_SERIES group."""
    st.markdown(f"### {group}")
    with st.spinner(f"Loading {group.lower()} series…"):
        series_map = _safe(macro.get_macro_series_group, group, 12, default={})

    if not series_map:
        ui.alert(f"No {group.lower()} data available.", "warn")
        return

    tiles = []
    for series_id, series in series_map.items():
        if series is None or series.empty:
            continue
        latest = float(series.iloc[-1])
        previous = float(series.iloc[-2]) if len(series) > 1 else None
        change = latest - previous if previous is not None else None
        tiles.append(ui.metric_tile(
            series_id, latest, change, "",
            subtitle=str(series.name)[:34],
            value_format="{:,.2f}",
        ))
    ui.metric_row(tiles, columns=min(len(tiles), 5) or 1)

    chosen = st.multiselect(
        "PLOT SERIES", list(series_map),
        default=list(series_map)[:3],
        format_func=lambda sid: f"{sid} — {series_map[sid].name}",
        key=f"plot_{group}",
    )
    if chosen:
        ui.render_chart(ui.line_chart(
            {f"{sid}": series_map[sid] for sid in chosen},
            group, height=400,
        ))


def _yield_curve_figure(curve: pd.DataFrame, analysis: Dict[str, Any],
                        height: int = 420):
    """The curve itself, with inverted segments highlighted in red."""
    import plotly.graph_objects as go
    from ui.terminal_theme import style_figure

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve["years"], y=curve["yield"],
        mode="lines+markers+text",
        text=[f"{y:.2f}" for y in curve["yield"]],
        textposition="top center",
        textfont=dict(size=9, color=THEME.muted),
        line=dict(color=THEME.amber, width=2.4),
        marker=dict(size=8, color=THEME.amber,
                    line=dict(color=THEME.bg, width=1.4)),
        name="US Treasury",
        hovertemplate="%{customdata}<br>%{y:.3f}%<extra></extra>",
        customdata=curve["tenor"],
    ))

    # Mark segments where yields fall as maturity rises.
    for index in range(len(curve) - 1):
        if curve["yield"].iloc[index + 1] < curve["yield"].iloc[index]:
            fig.add_trace(go.Scatter(
                x=curve["years"].iloc[index:index + 2],
                y=curve["yield"].iloc[index:index + 2],
                mode="lines", line=dict(color=THEME.red, width=4),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_xaxes(type="log", title_text="MATURITY (YEARS)",
                     tickvals=list(curve["years"]),
                     ticktext=list(curve["tenor"]))
    fig.update_yaxes(title_text="YIELD (%)")

    as_of = analysis.get("as_of")
    title = "US TREASURY YIELD CURVE"
    if as_of is not None:
        try:
            title += f" — {pd.Timestamp(as_of).strftime('%Y-%m-%d')}"
        except Exception:
            pass

    return style_figure(fig, height=height, title=title, showlegend=False)



# ==========================================================================
# PAGE: FUNDAMENTAL ANALYSIS
# ==========================================================================
def _fa_accent(verdict: str) -> str:
    return getattr(THEME, config.VERDICT_COLOURS.get(verdict, "muted"), THEME.muted)


def _fa_component_rows(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Every scored component as a row, including the ones that did not resolve.

    This table is the answer to "why 86?". Each row carries the raw value,
    the anchor table that turned it into points, the weight it was given and
    where the number came from - so the headline score can be recomputed by
    hand from what is on screen.
    """
    rows: List[Dict[str, Any]] = []
    for axis_name, axis in report.get("axes", {}).items():
        for component in axis.get("components", []):
            rows.append({
                "Axis": axis_name.replace("_", " ").title(),
                "Component": component.name,
                "Value": fundamentals.format_metric(component.value,
                                                    component.unit),
                "Points": component.points,
                "Weight": component.weight,
                "Source": component.source,
                "Anchors": component.anchor_label(),
                "Note": (component.note or "")[:90],
            })
    return pd.DataFrame(rows)


def page_fundamentals() -> None:
    ui.module_header("FUNDAMENTAL ANALYSIS",
                     "BUSINESS QUALITY · FINANCIAL HEALTH · VALUATION · VERDICT")

    ticker = st.session_state.get("ticker", config.DEFAULT_TICKER)

    control, override_col = st.columns([1, 3])
    with control:
        entered = st.text_input("TICKER", value=ticker, key="fa_ticker")
        entered = equities.normalize_ticker(entered).upper()
        if entered and entered != ticker:
            st.session_state["ticker"] = entered
        ticker = entered or ticker

    with override_col:
        with st.expander("DCF ASSUMPTIONS — override the derived defaults"):
            st.caption(
                "These are FORECAST ASSUMPTIONS, not facts. Defaults come "
                "from the company's own history and the live 10-year "
                "Treasury yield. Change them and the fair value changes — "
                "that sensitivity is the point."
            )
            a, b, c = st.columns(3)
            with a:
                growth_override = st.number_input(
                    "GROWTH (%)", value=0.0, step=0.5, format="%.1f",
                    key="fa_growth",
                    help="0 keeps the derived default.")
            with b:
                discount_override = st.number_input(
                    "DISCOUNT RATE (%)", value=0.0, step=0.5, format="%.1f",
                    key="fa_discount", help="0 keeps the CAPM-derived rate.")
            with c:
                terminal_override = st.number_input(
                    "TERMINAL GROWTH (%)", value=0.0, step=0.1, format="%.1f",
                    key="fa_terminal", help="0 keeps the configured default.")

    overrides: Dict[str, float] = {}
    if growth_override:
        overrides["growth"] = growth_override / 100.0
    if discount_override:
        overrides["discount_rate"] = discount_override / 100.0
    if terminal_override:
        overrides["terminal_growth"] = terminal_override / 100.0

    if not ticker:
        ui.alert("Enter a ticker to run the analysis.", "warn")
        return

    with st.spinner(f"Reading {ticker} filings and building the valuation…"):
        report = _safe(fundamentals.assess, ticker,
                       overrides or None, default={})

    if not report:
        ui.alert(f"Fundamental analysis unavailable for {ticker}.", "error")
        return

    if report.get("verdict") == "INSUFFICIENT DATA" and "reason" in report:
        ui.alert(report["reason"], "warn")
        return

    verdict = report["verdict"]
    accent = _fa_accent(verdict)
    profile = report["profile"]
    scores = report["scores"]
    valuation = report["valuation"]

    # --- Verdict banner ---------------------------------------------------
    st.markdown(
        f'<div style="border-left:4px solid {accent};background:{THEME.bg_panel};'
        f'padding:12px 16px;margin:4px 0 12px 0;">'
        f'<div style="color:{THEME.muted};font-size:10px;letter-spacing:0.14em;">'
        f'FUNDAMENTAL VERDICT · {report["name"]} · {profile.label} · '
        f'{report["statement_source"]} {report["as_of"]}</div>'
        f'<div style="color:{accent};font-size:26px;font-weight:700;'
        f'letter-spacing:0.08em;">{verdict}</div>'
        f'<div style="color:{THEME.muted};font-size:11px;">'
        f'{report["verdict_detail"]["rule"]}</div></div>',
        unsafe_allow_html=True,
    )

    # --- Scores -----------------------------------------------------------
    ui.metric_row([
        ui.metric_tile("OVERALL", report.get("overall_score"), subtitle="/100",
                       value_format="{:.0f}", accent=accent),
        ui.metric_tile("BUSINESS QUALITY", scores.get("business_quality"),
                       subtitle="/100", value_format="{:.0f}"),
        ui.metric_tile("GROWTH", scores.get("growth"), subtitle="/100",
                       value_format="{:.0f}"),
        ui.metric_tile("FINANCIAL HEALTH", scores.get("financial_health"),
                       subtitle="/100", value_format="{:.0f}"),
        ui.metric_tile("VALUATION", scores.get("valuation"),
                       subtitle="/100 · higher = cheaper", value_format="{:.0f}"),
        ui.metric_tile("RISK", scores.get("risk"),
                       subtitle="/100 · higher = worse", value_format="{:.0f}",
                       accent=THEME.red if (scores.get("risk") or 0) > 55
                       else THEME.green),
    ], columns=6)

    # --- Intrinsic value --------------------------------------------------
    st.markdown("### INTRINSIC VALUE")
    ui.metric_row([
        ui.metric_tile("BEAR", valuation.get("bear"), value_format="{:,.2f}",
                       accent=THEME.red),
        ui.metric_tile("BASE", valuation.get("base"), value_format="{:,.2f}",
                       accent=THEME.amber),
        ui.metric_tile("BULL", valuation.get("bull"), value_format="{:,.2f}",
                       accent=THEME.green),
        ui.metric_tile("CURRENT PRICE", report.get("price"),
                       value_format="{:,.2f}", accent=THEME.cyan),
        ui.metric_tile("UPSIDE TO BASE",
                       (valuation.get("upside_to_base") or 0) * 100
                       if valuation.get("upside_to_base") is not None else None,
                       subtitle="%", value_format="{:+.1f}",
                       accent=THEME.green if (valuation.get("upside_to_base") or 0) > 0
                       else THEME.red),
        ui.metric_tile("MARGIN OF SAFETY",
                       (valuation.get("margin_of_safety") or 0) * 100
                       if valuation.get("margin_of_safety") is not None else None,
                       subtitle="%", value_format="{:+.1f}",
                       accent=THEME.green if (valuation.get("margin_of_safety") or 0) > 0
                       else THEME.red),
    ], columns=6)

    dispersion = valuation.get("dispersion")
    if dispersion is not None and dispersion > 0.5:
        ui.alert(
            f"Valuation methods disagree by {dispersion:.0%} of the median. "
            "The fair value is a range, not a figure — read the method table "
            "below rather than the base case alone.", "warn")

    # --- Narrative --------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("#### WHY")
        for reason in report["reasons"]["strengths"] or ["No component scored above 60."]:
            st.markdown(
                f'<div style="border-left:2px solid {THEME.green};'
                f'padding:4px 10px;margin:4px 0;font-size:11px;'
                f'color:{THEME.cyan};">{reason}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### KEY RISKS")
        for risk in report["reasons"]["weaknesses"] or ["No component scored below 45."]:
            st.markdown(
                f'<div style="border-left:2px solid {THEME.red};'
                f'padding:4px 10px;margin:4px 0;font-size:11px;'
                f'color:{THEME.cyan};">{risk}</div>', unsafe_allow_html=True)

    st.markdown("#### VALUATION")
    st.caption(report["narrative"]["valuation"])
    st.markdown("#### THESIS")
    st.caption(report["narrative"]["thesis"])

    trigger_buy, trigger_sell = st.columns(2)
    with trigger_buy:
        st.markdown("#### BUY TRIGGERS")
        for item in report["triggers"]["buy"] or ["None identified."]:
            st.caption(f"• {item}")
    with trigger_sell:
        st.markdown("#### SELL TRIGGERS")
        for item in report["triggers"]["sell"] or ["None identified."]:
            st.caption(f"• {item}")

    # --- Detail tabs ------------------------------------------------------
    detail = st.tabs(["SCORE BREAKDOWN", "VALUATION METHODS", "METRICS",
                      "EARNINGS QUALITY", "ASSUMPTIONS", "CONFIDENCE"])

    with detail[0]:
        st.caption(
            "Every component that fed a score. Points come from interpolating "
            "the raw value across the anchor table shown — no fitted curves "
            "and no hidden constants, so the headline score can be "
            "recomputed by hand from this table."
        )
        breakdown = _fa_component_rows(report)
        if breakdown.empty:
            ui.alert("No components scored.", "warn")
        else:
            ui.styled_table(breakdown, numeric_format="{:,.3f}")

    with detail[1]:
        methods = pd.DataFrame(valuation.get("methods", []))
        if not methods.empty:
            methods = methods.rename(columns={
                "method": "Method", "per_share": "Value / share",
                "basis": "Basis", "note": "Note"})
            ui.styled_table(methods, numeric_format="{:,.2f}")
        st.caption(
            f"Anchored on {valuation.get('anchor')} for the "
            f"{profile.label.lower()} profile. {profile.note}")
        multiples = pd.DataFrame(
            [{"Multiple": k, "Value": v}
             for k, v in (valuation.get("multiples") or {}).items()])
        if not multiples.empty:
            ui.styled_table(multiples, numeric_format="{:,.2f}")

    with detail[2]:
        metric_rows = [{
            "Metric": key.replace("_", " ").title(),
            # Rendered through the engine's own formatter so a ratio never
            # prints as a percentage and a dollar figure never prints raw.
            "Value": fundamentals.format_metric(fact.value, fact.unit),
            "Source": fact.source,
            "Note": (fact.note or "")[:110],
        } for key, fact in report["metrics"].items()]
        ui.styled_table(pd.DataFrame(metric_rows))
        st.caption(
            "FILED and MARKET are observations. CALCULATED is arithmetic on "
            "them. PROXY is a stand-in for something with no direct source. "
            "UNAVAILABLE means the data was not there — never a substituted "
            "default."
        )

    with detail[3]:
        flags = report["earnings_quality"]["flags"]
        if not flags:
            ui.alert("No earnings-quality flags raised on the filed numbers.",
                     "ok")
        for flag in flags:
            st.markdown(
                f'<div style="border-left:3px solid {THEME.amber};'
                f'background:{THEME.bg_panel};padding:7px 12px;margin:5px 0;">'
                f'<span style="color:{THEME.amber};font-size:12px;">'
                f'{flag["flag"]}</span><br>'
                f'<span style="color:{THEME.muted};font-size:10px;">'
                f'{flag["detail"]}</span></div>', unsafe_allow_html=True)
        st.caption(
            "These are prompts to go and read the filing, not accusations. "
            "Every one of them has innocent explanations."
        )

    with detail[4]:
        assumptions = valuation.get("assumptions", {})
        provenance = assumptions.get("_provenance", {})
        rows = [{
            "Assumption": key.replace("_", " ").title(),
            "Value": value,
            "Derivation": provenance.get(key, ""),
        } for key, value in assumptions.items() if not key.startswith("_")]
        ui.styled_table(pd.DataFrame(rows), numeric_format="{:,.4f}")
        ui.alert(
            "Every row here is a forecast, not a fact. The fair value is only "
            "as good as these, which is why they are editable above.", "warn")

    with detail[5]:
        confidence = report["confidence"]
        ui.metric_row([
            ui.metric_tile("CONFIDENCE", confidence["score"], subtitle="/100",
                           value_format="{:.0f}",
                           accent=THEME.green if confidence["score"] >= 70
                           else THEME.amber if confidence["score"] >= 50
                           else THEME.red),
        ], columns=4)
        for deduction in confidence["deductions"]:
            st.markdown(
                f'<div style="font-size:11px;color:{THEME.muted};'
                f'padding:3px 0;">−{deduction["points"]:.0f} · '
                f'{deduction["reason"]}</div>', unsafe_allow_html=True)

        st.markdown("#### NOT AVAILABLE AT ANY PRICE")
        st.caption(
            "These are part of a real fundamental assessment and no free "
            "source carries them. They are excluded from every score rather "
            "than estimated:"
        )
        for item in report.get("unavailable", []):
            st.caption(f"• {item}")

    st.caption(report["disclaimer"])


# ==========================================================================
# PAGE: NEWS
# ==========================================================================
@st.cache_data(ttl=config.TTL.news, show_spinner=False)
def _cached_news(categories: Tuple[str, ...], limit_per_feed: int,
                 max_articles: int) -> pd.DataFrame:
    return news.get_news(list(categories), limit_per_feed, max_articles)


def _social_accent(band: str) -> str:
    return getattr(THEME, config.SENTIMENT_BAND_COLOURS.get(band, "muted"),
                   THEME.muted)


def _render_social_sentiment() -> None:
    """
    Fused multi-platform sentiment for one ticker.

    Built from the existing components. The one thing this panel must not do
    is present a fused number without the platform breakdown beside it: the
    interesting case is when headlines and the crowd point opposite ways, and
    a single score is exactly where that information goes to die.
    """
    ticker = st.session_state.get("ticker", config.DEFAULT_TICKER)

    control, _ = st.columns([1, 3])
    with control:
        entered = st.text_input("TICKER", value=ticker, key="social_ticker")
        entered = equities.normalize_ticker(entered).upper()
        if entered and entered != ticker:
            st.session_state["ticker"] = entered
        ticker = entered or ticker

    if not ticker:
        ui.alert("Enter a ticker to fuse sentiment across platforms.", "warn")
        return

    with st.spinner(f"Reading headlines, StockTwits and Reddit for {ticker}…"):
        report = _safe(social.composite_sentiment, ticker, default={})

    if not report:
        ui.alert(f"Sentiment fusion unavailable for {ticker}.", "error")
        return

    band = report["band"]
    accent = _social_accent(band)
    score = report["score"]
    confidence = report["confidence"]

    if score is None:
        ui.alert(
            "No platform returned usable data for this ticker, so there is no "
            "score. That is a gap in coverage, not a neutral reading — the "
            "two are different claims.", "warn")

    st.markdown(
        f'<div style="border-left:4px solid {accent};background:{THEME.bg_panel};'
        f'padding:12px 16px;margin:4px 0 12px 0;">'
        f'<div style="color:{THEME.muted};font-size:10px;letter-spacing:0.14em;">'
        f'SOCIAL SENTIMENT · {ticker}</div>'
        f'<div style="color:{accent};font-size:26px;font-weight:700;'
        f'letter-spacing:0.08em;">{band}'
        f'<span style="font-size:16px;color:{THEME.muted};"> &nbsp;'
        f'{"" if score is None else f"{score:+.2f}"}</span></div></div>',
        unsafe_allow_html=True,
    )

    platforms = report["platforms"]
    ui.metric_row([
        ui.metric_tile("FUSED SCORE", score, subtitle="-1.00 to +1.00",
                       value_format="{:+.2f}", accent=accent),
        ui.metric_tile("CONFIDENCE", confidence["score"], subtitle="0.00 to 1.00",
                       value_format="{:.2f}",
                       accent=THEME.green if confidence["score"] >= 0.7
                       else THEME.amber if confidence["score"] >= 0.45
                       else THEME.red),
        ui.metric_tile("INSTITUTIONAL", platforms["institutional"]["score"],
                       subtitle=f"{platforms['institutional']['samples']} headlines · "
                                f"{platforms['institutional']['weight']:.0%}",
                       value_format="{:+.2f}"),
        ui.metric_tile("RETAIL", platforms["retail"]["score"],
                       subtitle=f"{platforms['retail']['samples']} posts · "
                                f"{platforms['retail']['weight']:.0%}",
                       value_format="{:+.2f}"),
        ui.metric_tile("REDDIT", platforms["reddit"]["score"],
                       subtitle=f"{platforms['reddit']['samples']} posts · "
                                f"{platforms['reddit']['weight']:.0%}",
                       value_format="{:+.2f}"),
        ui.metric_tile("BULL / BEAR TAGS",
                       platforms["retail"].get("tagged"),
                       subtitle=f"{platforms['retail'].get('bullish_tags', 0)} bull / "
                                f"{platforms['retail'].get('bearish_tags', 0)} bear",
                       value_format="{:,.0f}", accent=THEME.cyan),
    ], columns=6)

    # --- Divergence -------------------------------------------------------
    divergence = report["signal_divergence"]
    if divergence:
        st.markdown("#### SIGNAL DIVERGENCE")
        for note in divergence:
            ui.alert(note, "warn")
    else:
        ui.alert("Platforms agree within tolerance and all returned data.", "ok")

    left, right = st.columns(2)
    with left:
        st.markdown("#### BULLISH DRIVERS")
        for item in report["drivers"].get("bullish", []) or [None]:
            if item is None:
                st.caption("None found.")
                break
            st.markdown(
                f'<div style="border-left:2px solid {THEME.green};'
                f'padding:4px 10px;margin:4px 0;font-size:11px;color:{THEME.cyan};">'
                f'<span style="color:{THEME.muted};">[{item["platform"]}] '
                f'{item["sentiment"]:+.2f}</span> {html.escape(item["text"])}</div>',
                unsafe_allow_html=True)
    with right:
        st.markdown("#### BEARISH DRIVERS")
        for item in report["drivers"].get("bearish", []) or [None]:
            if item is None:
                st.caption("None found.")
                break
            st.markdown(
                f'<div style="border-left:2px solid {THEME.red};'
                f'padding:4px 10px;margin:4px 0;font-size:11px;color:{THEME.cyan};">'
                f'<span style="color:{THEME.muted};">[{item["platform"]}] '
                f'{item["sentiment"]:+.2f}</span> {html.escape(item["text"])}</div>',
                unsafe_allow_html=True)

    drivers = report["drivers"]
    if not drivers.get("filtered", False) and drivers.get("mentions") is not None:
        st.caption(
            f"Only {drivers['mentions']} items named {ticker} directly, too few "
            "to filter on, so these are drawn from the whole stream and may "
            "reference other companies."
        )

    detail = st.tabs(["PLATFORM DETAIL", "CONFIDENCE", "RAW POSTS", "JSON"])

    with detail[0]:
        rows = [{
            "Platform": block["platform"],
            "Score": block["score"],
            "Samples": block["samples"],
            "Weight": block["weight"],
            "Applied": block["applied_weight"],
            "Note": (block.get("note") or "")[:150],
        } for block in platforms.values()]
        ui.styled_table(pd.DataFrame(rows), numeric_format="{:,.3f}",
                        highlight_columns=["Score"])
        st.caption(
            "Applied weight renormalises over the platforms that answered. A "
            "silent platform is not counted as neutral — that would read as "
            "'the crowd has no view' when the truth is 'we could not hear "
            "one of them'."
        )
        conflicts = platforms["retail"].get("tag_lexicon_conflicts")
        if conflicts:
            st.caption(
                f"On {conflicts} StockTwits posts the user's own Bull/Bear tag "
                "contradicted the lexicon's reading of the same text. Lexicons "
                "were not built for retail slang or sarcasm — where a tag "
                "exists it is treated as the better evidence."
            )

    with detail[1]:
        ui.metric_row([
            ui.metric_tile("CONFIDENCE", confidence["score"],
                           value_format="{:.2f}"),
            ui.metric_tile("COVERAGE", confidence["coverage"],
                           value_format="{:.2f}"),
            ui.metric_tile("VOLUME", confidence["volume"],
                           value_format="{:.2f}"),
            ui.metric_tile("AGREEMENT", confidence["agreement"],
                           value_format="{:.2f}"),
        ], columns=4)
        for component in confidence["components"]:
            st.caption(f"• {component}")

    with detail[2]:
        frames = report.get("frames", {})
        for label, frame in (("YAHOO HEADLINES", frames.get("institutional")),
                             ("STOCKTWITS", frames.get("retail")),
                             ("REDDIT", frames.get("reddit"))):
            st.markdown(f"**{label}**")
            if frame is None or frame.empty:
                st.caption("No data returned.")
                continue
            columns = [c for c in ("title", "body", "tag", "label", "sentiment",
                                   "platform", "subreddit", "source",
                                   "created_at", "published")
                       if c in frame.columns]
            st.dataframe(frame[columns].head(40), use_container_width=True,
                         hide_index=True, height=220)

    with detail[3]:
        st.caption(
            "The report as a JSON-serialisable object — score, band, "
            "confidence, per-platform breakdown, divergence and drivers."
        )
        st.json(social.to_schema(report))

    st.caption(report["disclaimer"])


def page_news() -> None:
    ui.module_header("OSINT NEWS TERMINAL", "RSS · GDELT · SENTIMENT")

    engine = news.sentiment_engine_status()
    st.markdown(
        ui.badge(f"ENGINE: {engine['active'].upper()}",
                 "green" if engine["active"] != "none" else "red") +
        (ui.badge("INSTALL transformers+torch FOR FINBERT", "muted")
         if engine["active"] == "vader" else ""),
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["NEWS FEED", "SOCIAL SENTIMENT", "GDELT OSINT",
                    "TRENDING TERMS"])

    # ---- FEED ------------------------------------------------------------
    with tabs[0]:
        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
        with col_a:
            categories = st.multiselect(
                "DESKS", list(config.RSS_FEEDS),
                default=[st.session_state["news_category"]]
                if st.session_state["news_category"] in config.RSS_FEEDS
                else ["MARKETS"],
                key="news_cats",
            )
        with col_b:
            sentiment_filter = st.selectbox(
                "SENTIMENT", ["ALL", "BULLISH", "BEARISH", "NEUTRAL"],
                key="news_sent")
        with col_c:
            hours = st.slider("HOURS BACK", 1, 72, 24, key="news_hours")
        with col_d:
            query = st.text_input("SEARCH", key="news_query")

        if not categories:
            ui.alert("Select at least one desk.", "warn")
            return

        with st.spinner("Aggregating feeds…"):
            articles = _cached_news(tuple(categories), 25, 250)

        if articles.empty:
            ui.alert(
                "No headlines retrieved. Publishers occasionally rotate feed "
                "URLs — check your connection first.", "error",
            )
            return

        filtered = news.filter_news(articles, query, sentiment_filter,
                                    hours_back=hours)
        summary = news.sentiment_summary(filtered)

        ui.metric_row([
            ui.metric_tile("HEADLINES", summary.get("total"), value_format="{:,.0f}"),
            ui.metric_tile("MOOD", summary.get("mood", "—"),
                           accent=THEME.green if summary.get("mood") == "RISK-ON"
                           else THEME.red if summary.get("mood") == "RISK-OFF"
                           else THEME.amber),
            ui.metric_tile("NET SENTIMENT", summary.get("net_sentiment"),
                           value_format="{:+.3f}"),
            ui.metric_tile("BULLISH", summary.get("bullish"),
                           accent=THEME.green, value_format="{:,.0f}"),
            ui.metric_tile("BEARISH", summary.get("bearish"),
                           accent=THEME.red, value_format="{:,.0f}"),
            ui.metric_tile("NEUTRAL", summary.get("neutral"),
                           accent=THEME.muted, value_format="{:,.0f}"),
        ], columns=6)

        ui.news_feed(filtered, max_rows=70)

        st.caption(
            "Sentiment scores the tone of the writing, not the market impact. "
            "\"Beats estimates, shares fall\" scores positive. Use it to triage "
            "what to read."
        )

    # ---- SOCIAL SENTIMENT ------------------------------------------------
    with tabs[1]:
        _render_social_sentiment()

    # ---- GDELT -----------------------------------------------------------
    with tabs[2]:
        st.caption(
            "GDELT monitors global news in 65+ languages with a genuinely open "
            "API — no key, no quota page."
        )

        mode = st.radio("MODE", ["PRESET SWEEPS", "CUSTOM QUERY"],
                        horizontal=True, key="gdelt_mode")
        timespan = st.selectbox("TIMESPAN", ["1h", "24h", "3d", "7d"],
                                index=1, key="gdelt_span")

        if mode == "PRESET SWEEPS":
            if st.button("RUN OSINT SWEEP", use_container_width=True):
                with st.spinner("Querying GDELT…"):
                    results = _safe(news.get_geopolitical_osint, timespan, default={})

                if not results:
                    ui.alert("GDELT returned nothing.", "warn")
                else:
                    for theme, frame in results.items():
                        summary = news.sentiment_summary(frame)
                        st.markdown(
                            f"### {theme} "
                            f"<span style='color:{THEME.muted};font-size:11px;'>"
                            f"{summary.get('total', 0)} articles · "
                            f"{summary.get('mood', '')}</span>",
                            unsafe_allow_html=True,
                        )
                        ui.news_feed(frame, max_rows=12)
        else:
            query = st.text_input(
                "GDELT QUERY",
                value='"supply chain" (disruption OR shortage)',
                key="gdelt_query",
            )
            st.caption(
                "Supports quoted phrases, OR, and operators like "
                "`domain:reuters.com`, `sourcecountry:china`, "
                "`theme:ECON_STOCKMARKET`."
            )
            if st.button("QUERY GDELT", use_container_width=True):
                with st.spinner("Querying…"):
                    results = _safe(news.query_gdelt, query, timespan, 100,
                                    default=pd.DataFrame())
                if results.empty:
                    ui.alert("No results. Check the query syntax.", "warn")
                else:
                    summary = news.sentiment_summary(results)
                    ui.metric_row([
                        ui.metric_tile("ARTICLES", summary.get("total"),
                                       value_format="{:,.0f}"),
                        ui.metric_tile("MOOD", summary.get("mood", "—")),
                        ui.metric_tile("NET", summary.get("net_sentiment"),
                                       value_format="{:+.3f}"),
                    ], columns=3)
                    ui.news_feed(results, max_rows=60)

    # ---- TRENDING --------------------------------------------------------
    with tabs[3]:
        with st.spinner("Analysing corpus…"):
            corpus = _cached_news(tuple(config.RSS_FEEDS), 25, 400)

        if corpus.empty:
            ui.alert("No corpus available.", "warn")
        else:
            terms = news.extract_trending_terms(corpus, top_n=30)
            if terms.empty:
                ui.alert("Not enough text to extract terms.", "warn")
            else:
                left, right = st.columns([2, 1])
                with left:
                    ui.render_chart(ui.heatmap(
                        terms.rename(columns={"avg_sentiment": "value"}),
                        "value", "term",
                        "TRENDING TERMS BY SENTIMENT", height=380, columns=5,
                    ))
                with right:
                    ui.styled_table(terms, height=380)

                st.caption(
                    f"Across {len(corpus)} headlines. Frequency-ranked with "
                    f"finance stopwords removed; sentiment is the mean across "
                    f"articles containing each term."
                )


# ==========================================================================
# PAGE: HELP
# ==========================================================================
def page_help() -> None:
    ui.module_header("HELP", "COMMAND REFERENCE")

    st.code(config.HELP_TEXT, language=None)

    left, right = st.columns(2)

    with left:
        st.markdown("### MODULES")
        st.markdown("""
| Module | Command | What it does |
|---|---|---|
| **Equity** | `AAPL EQUITY` | Candles + EMA/RSI/MACD, XBRL statements from SEC EDGAR, peer comps, options chain, ticker news |
| **Maritime** | `SUEZ SHIP` | AIS positions, chokepoint congestion vs locally measured baselines, vessel lookup by MMSI/IMO |
| **Aviation** | `EUROPE FLY` | Live ADS-B state vectors, fleet tracking by ICAO designator, aircraft tracks, airport flow |
| **Macro** | `YCRV` | Treasury curve + inversion detection, recession composite, FRED series, World Bank |
| **News** | `NEWS` | RSS aggregation with sentiment, GDELT OSINT sweeps, trending terms |
        """)

    with right:
        st.markdown("### DATA SOURCES & LIMITS")
        st.markdown("""
| Source | Cost | Limit |
|---|---|---|
| yfinance | Free | Unofficial; 15-min delayed; throttles on abuse |
| SEC EDGAR | Free | 10 req/s, descriptive User-Agent required |
| FRED | Free | Key optional — keyless CSV fallback built in |
| OpenSky | Free | 400 credits/day anon, 4000 with OAuth2 |
| AISStream | Free key | Websocket firehose, global |
| GDELT | Free | No key, no published quota |
| World Bank | Free | No key |
        """)

    st.divider()
    st.markdown("### DESIGN NOTES")
    st.markdown(f"""
**Caching.** Everything network-bound goes through a SQLite TTL cache
(`{config.DATA_DIR}`). Daily bars cache for 1 hour, scrapes for 15 minutes,
fundamentals for 24 hours. If an upstream fails and a stale entry exists,
the stale value is served rather than an error — check the timestamp in the
module header.

**Rate limiting.** Every request passes a per-provider token bucket, and
failures retry with exponential backoff plus jitter, honouring `Retry-After`
when the server sends one.

**What this is not.** Free data has real limits and you should know them:
prices are delayed, AIS coverage is patchy outside major shipping lanes,
OpenSky's receiver network thins out over oceans and much of the global
south, and headline sentiment is a triage tool rather than a signal. None of
the analytics here are investment advice.

**On scraping.** Open-Terminal prefers official free APIs over scraping
wherever one exists — SEC's XBRL API instead of parsing 10-K HTML, FRED's CSV
endpoint instead of the website, aisstream instead of MarineTraffic. The
Playwright scrapers for MarineTraffic and VesselFinder are implemented but
ship disabled: both sites prohibit automated collection and both are behind
Cloudflare. Enabling them is a decision you make deliberately with
`OPENTERM_ALLOW_SCRAPERS=1`.
    """)


# ==========================================================================
# ERROR HANDLING HELPERS
# ==========================================================================
def _safe(func, *args, default: Any = None, **kwargs) -> Any:
    """
    Call a fetcher, converting any exception into a UI warning + default.

    The fetchers already swallow most failures via @retry_with_backoff's
    on_giveup, so reaching this handler means something genuinely unexpected
    happened - worth surfacing, but not worth crashing the page over.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        log.error("%s failed: %s", getattr(func, "__name__", func), exc,
                  exc_info=config.DEBUG)
        ui.alert(f"{getattr(func, '__name__', 'fetch')}: {exc}", "warn")
        if config.DEBUG:
            st.code(traceback.format_exc())
        return default


# ==========================================================================
# MAIN
# ==========================================================================
# ==========================================================================
# PAGE: PORTFOLIO  (PF / WATCH / BRIEF)
# ==========================================================================
def _editor(frame: pd.DataFrame, columns: List[str], key: str,
            config_map: Dict[str, Any]) -> pd.DataFrame:
    """Editable grid seeded with the stored rows and one blank row to type in."""
    seed = frame.copy()
    if seed.empty:
        seed = pd.DataFrame([{column: None for column in columns}])
    return st.data_editor(
        seed, key=key, num_rows="dynamic", use_container_width=True,
        hide_index=True, column_config=config_map,
    )


def _brief_stamp(brief: Dict[str, Any]) -> str:
    built = str(brief.get("built_at", ""))[:16].replace("T", " ")
    return f"{brief.get('edition', '—')} · built {built} SGT" if built else str(
        brief.get("edition", "—"))


def _render_allocation(valued: pd.DataFrame) -> None:
    """
    Sector exposure against a benchmark, with the dollar moves to close it.

    Every number here is arithmetic on live data: sector labels come from the
    issuer's own classification, funds are looked through to their published
    weights, and the benchmark targets are read off a real fund rather than
    typed into a table that would rot.
    """
    if valued is None or valued.empty:
        ui.alert("Add holdings before running an allocation review.", "warn")
        return

    choice_col, _ = st.columns([1, 2])
    with choice_col:
        benchmark_key = st.selectbox(
            "BENCHMARK", list(config.ALLOCATION_BENCHMARKS),
            format_func=lambda k: (f"{config.ALLOCATION_BENCHMARKS[k].label} "
                                   f"({config.ALLOCATION_BENCHMARKS[k].proxy})"),
            key="alloc_benchmark")

    with st.spinner("Classifying holdings and reading benchmark weights…"):
        report = _safe(allocation.rebalance, valued, benchmark_key, default={})

    if not report:
        ui.alert("Allocation review unavailable.", "error")
        return

    benchmark = report["benchmark"]
    money = equities.currency_prefix(config.BASE_CURRENCY)
    st.caption(
        f"{benchmark.get('label', benchmark_key)} — {benchmark.get('description', '')} "
        f"Target weights are {benchmark.get('proxy')}'s current published "
        "sector weightings, read live rather than hardcoded."
    )

    if report.get("note"):
        ui.alert(report["note"], "warn")

    exposure = report.get("exposure")
    if exposure is None or exposure.empty:
        return

    # --- Headline ---------------------------------------------------------
    largest = exposure.iloc[0]
    rows = report.get("rows")
    biggest_drift = (rows.iloc[0] if isinstance(rows, pd.DataFrame)
                     and not rows.empty else None)

    ui.metric_row([
        ui.metric_tile("BOOK VALUE", report.get("total_value"),
                       subtitle=f"{config.BASE_CURRENCY} priced", value_format="{:,.0f}"),
        ui.metric_tile("SECTORS HELD", int(
            (exposure["sector"] != allocation.UNCLASSIFIED).sum()),
                       subtitle=f"of {len(config.GICS_SECTORS)} GICS",
                       value_format="{:,.0f}"),
        ui.metric_tile("LARGEST SECTOR", largest["weight_pct"],
                       subtitle=largest["sector"], value_format="{:.1f}"),
        ui.metric_tile("BIGGEST DRIFT",
                       biggest_drift["Drift %"] if biggest_drift is not None else None,
                       subtitle=(biggest_drift["Sector"]
                                 if biggest_drift is not None else ""),
                       value_format="{:+.1f}",
                       accent=THEME.red if biggest_drift is not None
                       and abs(biggest_drift["Drift %"]) >= config.REBALANCE_MIN_DRIFT_PCT
                       else THEME.green),
        ui.metric_tile("UNCLASSIFIED", report.get("unclassified_pct"),
                       subtitle="% of book", value_format="{:.1f}",
                       accent=THEME.amber if (report.get("unclassified_pct") or 0) > 5
                       else THEME.green),
    ], columns=5)

    if report.get("lookthrough"):
        names = ", ".join(
            f"{item['ticker']} ({item['sectors']} sectors, mostly "
            f"{item['largest']})" for item in report["lookthrough"])
        st.caption(
            f"Fund holdings looked through to their constituent sectors: "
            f"{names}. A fund is not a single-sector position, and filing one "
            "under its largest sector would overstate that sector by the whole "
            "position."
        )

    if report.get("unclassified"):
        for item in report["unclassified"]:
            ui.alert(
                f"{item['ticker']} ({money}{item['market_value']:,.0f}) could not be "
                f"placed in a sector: {item['reason']} It is excluded from the "
                "drift maths rather than spread across sectors.", "warn")

    # --- Allocation vs benchmark -----------------------------------------
    chart = exposure.copy()
    chart["bar_label"] = [
        f"{w:.1f}%   {money}{v:,.0f}"
        for w, v in zip(chart["weight_pct"], chart["market_value"])]

    left, right = st.columns([1, 1])
    with left:
        ui.render_chart(
            ui.donut(chart, "sector", "market_value",
                     title="SECTOR SPREAD",
                     height=int(min(max(150 + 32 * len(chart), 300), 700)),
                     center_value=f"{money}{report.get('total_value', 0):,.0f}",
                     center_label="book value"),
            key="alloc_donut")

    with right:
        if isinstance(rows, pd.DataFrame) and not rows.empty:
            drift = rows[rows["Drift %"].abs() > 0.01].copy()
            drift["bar_label"] = [f"{d:+.1f}pp" for d in drift["Drift %"]]
            ui.render_chart(
                ui.exposure_bars(drift, "Sector", "Drift %",
                                 title=f"DRIFT VS {benchmark.get('label', '')}".upper(),
                                 suffix="pp",
                                 height=int(min(max(150 + 32 * len(drift), 260), 700)),
                                 text_col="bar_label"),
                key="alloc_drift")

    # --- Actions ----------------------------------------------------------
    st.markdown("#### REBALANCING ACTIONS")
    actions = report.get("actions", [])
    if not actions:
        ui.alert(
            f"No sector drifts by more than {config.REBALANCE_MIN_DRIFT_PCT:.0f} "
            "percentage points. Nothing here is worth the spread and tax to "
            "fix.", "ok")
    else:
        colours = {"TRIM": THEME.red, "ADD": THEME.green, "OPEN": THEME.amber}
        for action in actions:
            colour = colours.get(action["action"], THEME.muted)
            st.markdown(
                f'<div style="border-left:3px solid {colour};'
                f'background:{THEME.bg_panel};padding:7px 12px;margin:5px 0;">'
                f'<span style="color:{colour};font-size:12px;letter-spacing:0.08em;">'
                f'{action["action"]} · {action["sector"]}</span>'
                f'<span style="float:right;color:{THEME.amber};font-size:12px;">'
                f'{money}{action["amount"]:,.0f}</span><br>'
                f'<span style="color:{THEME.muted};font-size:10px;">'
                f'{html.escape(action["detail"])}</span></div>',
                unsafe_allow_html=True)

    detail = st.tabs(["SECTOR TABLE", "CONCENTRATION", "JSON"])

    with detail[0]:
        ui.render_chart(
            ui.exposure_bars(chart, "sector", "weight_pct",
                             title="CURRENT SECTOR EXPOSURE",
                             height=int(min(max(150 + 32 * len(chart), 260), 700)),
                             color=THEME.cyan, text_col="bar_label"),
            key="alloc_current")
        if isinstance(rows, pd.DataFrame) and not rows.empty:
            ui.styled_table(rows, numeric_format="{:,.2f}",
                            highlight_columns=["Drift %", "Adjust $"])
        st.caption(
            "Current % is measured over the classified book only. Including "
            "an unclassified slug in the denominator would understate every "
            "sector by the same amount and make the whole book look "
            "underweight."
        )

    with detail[1]:
        stats = report.get("concentration", {})
        ui.metric_row([
            ui.metric_tile("POSITIONS", stats.get("positions"),
                           value_format="{:,.0f}"),
            ui.metric_tile("LARGEST", stats.get("largest"),
                           subtitle=stats.get("largest_ticker") or "",
                           value_format="{:.1f}"),
            ui.metric_tile("TOP 3", stats.get("top3_pct"), subtitle="%",
                           value_format="{:.1f}"),
            ui.metric_tile("EFFECTIVE", stats.get("effective_positions"),
                           subtitle="1/HHI", value_format="{:.1f}"),
        ], columns=4)
        st.caption(
            "Effective positions is 1/HHI — how many equally-weighted names "
            "the book behaves like. A twenty-name portfolio where one holding "
            "is 60% has an effective count near three, and that is the number "
            "that describes the risk."
        )
        for flag in stats.get("flags", []):
            ui.alert(flag, "warn")

    with detail[2]:
        st.json(allocation.to_schema(report))

    st.caption(report.get("disclaimer", ""))


def _render_position_weights(valued: pd.DataFrame) -> None:
    """
    Position weights, with the context that makes them mean something.

    The previous version was a bare bar chart at a fixed 280px. Three things
    were wrong with it and all three were about usability rather than
    correctness:

      * Fixed height. Six positions got fat bars with dead space; twenty got
        squashed into unreadable slivers. The height now scales with the row
        count.
      * Percent only. "VOO 54.9%" is the less useful half of the answer when
        the reader wants to know what that is in money. Bars now carry both.
      * No concentration read. A weight list does not tell you that six
        positions behave like fewer than three, which is the number that
        actually describes the risk. The effective count (1/HHI) does.

    It also says out loud when unpriced positions are missing from the
    picture. They were being dropped silently, so a book with a dead symbol
    showed weights that summed to 100% of something smaller than the book.
    """
    weights = valued.dropna(subset=["weight"])
    if weights.empty:
        return
    money = equities.currency_prefix(config.BASE_CURRENCY)

    stats = allocation.concentration(valued)

    ui.metric_row([
        ui.metric_tile("LARGEST", stats.get("largest"),
                       subtitle=stats.get("largest_ticker") or "",
                       value_format="{:.1f}",
                       accent=THEME.red
                       if (stats.get("largest") or 0) >= config.POSITION_CONCENTRATION_PCT
                       else THEME.cyan),
        ui.metric_tile("TOP 3", stats.get("top3_pct"), subtitle="% of book",
                       value_format="{:.1f}"),
        ui.metric_tile("EFFECTIVE POSITIONS", stats.get("effective_positions"),
                       subtitle=f"of {stats.get('positions', 0)} held · 1/HHI",
                       value_format="{:.1f}",
                       accent=THEME.amber
                       if (stats.get("effective_positions") or 99) < 5
                       else THEME.green),
        ui.metric_tile("PRICED", stats.get("positions"),
                       subtitle=f"of {len(valued)} positions",
                       value_format="{:,.0f}"),
    ], columns=4)

    chart = weights[["ticker", "weight", "market_value"]].copy()
    chart["bar_label"] = [
        f"{w:.1f}%   {money}{v:,.0f}" if pd.notna(v) else f"{w:.1f}%"
        for w, v in zip(chart["weight"], chart["market_value"])]

    # Bars and donut answer different questions and are both worth having:
    # bars rank and compare precisely, the donut shows share of the whole.
    bars_col, donut_col = st.columns([3, 2])
    with bars_col:
        ui.render_chart(
            ui.exposure_bars(chart, "ticker", "weight", title="POSITION WEIGHT",
                             # Roughly one row of breathing room per position,
                             # floored so a two-name book is not a stub and
                             # capped so a fifty-name book still fits a screen.
                             height=int(min(max(150 + 34 * len(chart), 260), 900)),
                             color=THEME.cyan, text_col="bar_label"),
            key="pf_weights")
    with donut_col:
        total = float(chart["market_value"].sum(skipna=True))
        ui.render_chart(
            ui.donut(chart, "ticker", "market_value", title="POSITION SPREAD",
                     height=int(min(max(150 + 34 * len(chart), 260), 900)),
                     center_value=f"{money}{total:,.0f}",
                     center_label="priced book",
                     # Past a dozen names the slivers stop being readable and
                     # start being decoration.
                     max_slices=12),
            key="pf_weight_donut")

    for flag in stats.get("flags", []):
        ui.alert(flag, "warn")


def page_portfolio() -> None:
    ui.module_header(
        "PORTFOLIO & WATCHLIST",
        "HOLDINGS · MARK-TO-MARKET · 08:00 SGT BRIEF",
    )

    toast = st.session_state.pop("portfolio_toast", None)
    if toast:
        ui.alert(toast, "ok")

    stored_holdings = portfolio.holdings()
    stored_watchlist = portfolio.watchlist()

    with st.spinner("Marking positions to market…"):
        valued = portfolio.value_positions(stored_holdings)
    summary = portfolio.portfolio_summary(valued)

    now_sgt = portfolio.sgt_now()
    edition = portfolio.edition_date(now_sgt)
    next_build = portfolio.next_edition_at(now_sgt)

    day_pnl = summary.get("day_pnl")
    total_pnl = summary.get("pnl")
    base = summary.get("currency") or config.BASE_CURRENCY
    money = equities.currency_prefix(base)

    ui.metric_row([
        ui.metric_tile("MARKET VALUE", summary.get("market_value"),
                       value_format=f"{money}{{:,.0f}}",
                       subtitle=f"{summary.get('positions', 0)} positions · {base}"),
        ui.metric_tile("DAY P&L", day_pnl, value_format=f"{money}{{:+,.0f}}",
                       subtitle="since previous close",
                       accent=THEME.green if (day_pnl or 0) >= 0 else THEME.red),
        ui.metric_tile("TOTAL P&L", total_pnl, value_format=f"{money}{{:+,.0f}}",
                       subtitle=(f"{summary['pnl_pct']:+,.1f}% on cost"
                                 if summary.get("pnl_pct") is not None
                                 else "no cost basis"),
                       accent=THEME.green if (total_pnl or 0) >= 0 else THEME.red),
        ui.metric_tile("WATCHLIST", len(stored_watchlist), value_format="{:,.0f}",
                       subtitle="symbols tracked"),
        ui.metric_tile("BRIEF EDITION", edition.strftime("%d %b"),
                       subtitle=f"next {next_build:%d %b} 08:00 SGT"),
        ui.metric_tile("SGT NOW", now_sgt.strftime("%H:%M"),
                       subtitle=now_sgt.strftime("%a %d %b")),
    ], columns=6)

    tabs = st.tabs(["HOLDINGS", "ALLOCATION", "WATCHLIST",
                    "MORNING BRIEF"])

    # ---- HOLDINGS --------------------------------------------------------
    with tabs[0]:
        st.caption(
            "Edit any cell, use the blank row to add a position, select a row "
            "and press delete to remove it. Nothing is written until you save. "
            "Cost basis is optional — leave it empty and the position still "
            "marks to market, it just cannot show a return."
        )
        edited = _editor(
            stored_holdings, portfolio.HOLDING_COLUMNS, "pf_holdings_editor",
            {
                "ticker": st.column_config.TextColumn("TICKER", width="small"),
                "quantity": st.column_config.NumberColumn("QTY", format="%.4f"),
                "cost_basis": st.column_config.NumberColumn(
                    "COST BASIS", format="%.4f", help="Average price paid"),
                "note": st.column_config.TextColumn("NOTE"),
            },
        )

        left, right = st.columns([1, 5])
        with left:
            if st.button("SAVE", key="pf_save_holdings",
                         use_container_width=True):
                portfolio.save(edited, stored_watchlist)
                st.success("Holdings saved.")
                st.rerun()
        with right:
            st.caption(f"Stored at `{config.PORTFOLIO_FILE}`")

        if valued.empty:
            ui.alert("No positions yet. Add a row above and save.", "warn")
        else:
            # BASIS and LAST stay in the listing's own currency so they match
            # the broker statement; every column that gets summed is in base.
            display = valued[[
                "ticker", "quantity", "currency", "cost_basis", "price",
                "change_pct", "day_pnl", "market_value", "cost", "pnl",
                "pnl_pct", "weight",
            ]].rename(columns={
                "ticker": "TICKER", "quantity": "QTY", "currency": "CCY",
                "cost_basis": "BASIS", "price": "LAST", "change_pct": "CHG %",
                "day_pnl": f"DAY P&L {base}", "market_value": f"MKT VALUE {base}",
                "cost": f"COST {base}", "pnl": f"P&L {base}",
                "pnl_pct": "P&L %", "weight": "WEIGHT %",
            })
            ui.styled_table(
                display,
                highlight_columns=["CHG %", f"DAY P&L {base}", f"P&L {base}",
                                   "P&L %"],
            )

            crosses = portfolio.fx_applied(valued)
            if crosses:
                st.caption(
                    "BASIS and LAST are in each listing's own currency (CCY). "
                    f"Day P&L, value, cost and P&L are in {base}, converting "
                    + ", ".join(f"{code} at {rate:,.4f}"
                                for code, rate in sorted(crosses.items()))
                    + ". Both legs of P&L use today's rate, so it is the "
                    "local-market return and leaves out currency moves since "
                    "purchase."
                )

            unpriced = valued[valued["unpriced_reason"].notna()]
            for reason, group in unpriced.groupby("unpriced_reason", sort=False):
                names = ", ".join(group["ticker"])
                if reason == "no quote":
                    ui.alert(
                        f"{names} returned no quote — check the symbol is the "
                        "one Yahoo lists.", "warn")
                else:
                    ui.alert(
                        f"{names}: {reason}, so left out of every total, weight "
                        "and sector rather than converted at a guessed rate.",
                        "warn")

            _render_position_weights(valued)

    # ---- ALLOCATION ------------------------------------------------------
    with tabs[1]:
        _render_allocation(valued)

    # ---- WATCHLIST -------------------------------------------------------
    with tabs[2]:
        st.caption(
            "Symbols you are following but do not hold. `NVDA WATCH` from the "
            "command bar adds one without coming here."
        )
        edited_watch = _editor(
            stored_watchlist, portfolio.WATCHLIST_COLUMNS, "pf_watch_editor",
            {
                "ticker": st.column_config.TextColumn("TICKER", width="small"),
                "note": st.column_config.TextColumn("NOTE"),
            },
        )
        if st.button("SAVE", key="pf_save_watch"):
            portfolio.save(stored_holdings, edited_watch)
            st.success("Watchlist saved.")
            st.rerun()

        quotes = portfolio.watchlist_quotes(stored_watchlist)
        if quotes.empty:
            ui.alert("Watchlist is empty.", "warn")
        else:
            ui.styled_table(
                quotes.rename(columns={
                    "ticker": "TICKER", "price": "LAST", "change": "CHG",
                    "change_pct": "CHG %", "volume": "VOLUME", "note": "NOTE",
                }),
                highlight_columns=["CHG", "CHG %"],
            )

    # ---- MORNING BRIEF ---------------------------------------------------
    with tabs[3]:
        if stored_holdings.empty:
            ui.alert(
                "The brief is built from your holdings — add positions first.",
                "warn")
            return

        archive = portfolio.stored_editions()
        controls = st.columns([2, 1, 3])
        with controls[0]:
            options = [edition] + [d for d in archive if d != edition]
            chosen = st.selectbox(
                "EDITION", options, index=0,
                format_func=lambda d: (f"{d:%a %d %b %Y}"
                                       + (" · today" if d == edition else "")),
            )
        with controls[1]:
            st.write("")
            rebuild = st.button("REBUILD", key="pf_brief_rebuild",
                                use_container_width=True)

        if chosen == edition:
            with st.spinner("Reading the tape for your positions…"):
                brief = portfolio.get_brief(force=rebuild)
        else:
            brief = portfolio.load_edition(chosen) or {}

        if not brief or brief.get("empty"):
            ui.alert("Nothing recorded for this edition.", "warn")
            return

        summary_block = brief.get("summary", {})
        net = summary_block.get("net_sentiment")
        mood = portfolio.mood_label(net)

        st.markdown("###### YOUR PORTFOLIO THIS MORNING")
        narrative = brief.get("narrative")
        if narrative:
            as_of = str(brief.get("book", {}).get("as_of", ""))[:16].replace("T", " ")
            st.markdown(
                f'<div style="border-left:3px solid {THEME.cyan};'
                f'background:{THEME.bg_panel};padding:12px 16px;margin:4px 0 10px;'
                f'font-size:13px;line-height:1.65;color:{THEME.white};">'
                f'{html.escape(narrative)}'
                + (f'<div style="margin-top:8px;font-size:10px;color:{THEME.muted};">'
                   f'Book figures as of {as_of} SGT</div>' if as_of else "")
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "This edition was built before the portfolio summary existed, "
                "so it carries headlines only.")

        st.markdown(
            f'<div style="border-left:3px solid {THEME.amber};padding:6px 10px;'
            f'background:{THEME.bg_panel};font-size:11px;color:{THEME.muted};'
            f'margin-bottom:8px;">'
            f'EDITION {_brief_stamp(brief)} — {summary_block.get("stories", 0)} '
            f'stories across {summary_block.get("symbols", 0)} positions. '
            f'Stories are ranked by sentiment strength weighted by position '
            f'size, so a soft story about a large holding outranks a loud one '
            f'about a small holding.</div>',
            unsafe_allow_html=True,
        )

        movers = summary_block.get("movers") or []
        ui.metric_row([
            ui.metric_tile("PORTFOLIO MOOD", mood,
                           subtitle=(f"net {net:+.2f}" if net is not None
                                     else "no scored stories"),
                           accent=THEME.green if mood == "RISK-ON"
                           else THEME.red if mood == "RISK-OFF" else THEME.amber),
            ui.metric_tile("STORIES", summary_block.get("stories"),
                           value_format="{:,.0f}", subtitle="in this edition"),
        ] + [
            ui.metric_tile(f"MOVER · {m['ticker']}", m["change_pct"],
                           value_format="{:+,.2f}%", subtitle="last session",
                           accent=THEME.green if m["change_pct"] >= 0
                           else THEME.red)
            for m in movers
        ], columns=2 + len(movers))

        st.markdown("###### TOP STORIES FOR THIS BOOK")
        top = pd.DataFrame(brief.get("top_stories", []))
        if top.empty:
            ui.alert(
                "No scored stories this edition — headlines were retrieved but "
                "all scored neutral.", "warn")
        else:
            top["published"] = pd.to_datetime(top["published"], errors="coerce",
                                              utc=True)
            top["source"] = top["ticker"] + " · " + top["source"]
            ui.news_feed(top, max_rows=12)

        st.markdown("###### BY POSITION")
        for position in brief.get("positions", []):
            weight = position.get("weight")
            heading = (f"{position['ticker']}  ·  "
                       f"{weight:,.1f}% of book" if weight is not None
                       else position["ticker"])
            counts = (f"{position['bullish']}▲ / {position['bearish']}▼ / "
                      f"{position['neutral']}■")
            with st.expander(f"{heading}   —   {counts}"):
                stories = pd.DataFrame(position.get("stories", []))
                if stories.empty:
                    st.caption("No headlines retrieved for this symbol.")
                    continue
                stories["published"] = pd.to_datetime(
                    stories["published"], errors="coerce", utc=True)
                ui.news_feed(stories, max_rows=8)


ROUTES = {
    "home": page_home,
    "equity": page_equity,
    "supply_chain": page_supply_chain,
    "maritime": page_maritime,
    "aviation": page_aviation,
    "macro": page_macro,
    "fundamentals": page_fundamentals,
    "news": page_news,
    "portfolio": page_portfolio,
    "help": page_help,
}


def main() -> None:
    init_state()
    render_sidebar()
    render_tape()
    render_command_bar()
    st.markdown("<hr/>", unsafe_allow_html=True)

    module = st.session_state.get("module", "home")
    page = ROUTES.get(module, page_home)

    try:
        page()
    except Exception as exc:
        log.exception("Page '%s' crashed", module)
        ui.alert(f"Module error: {exc}", "error")
        if config.DEBUG:
            st.code(traceback.format_exc())
        else:
            st.caption("Set OPENTERM_DEBUG=1 for a full traceback.")

    # --- Status bar --------------------------------------------------------
    credentials = config.credential_status()
    ui.status_bar({
        "MODULE": module.upper(),
        "LAST CMD": st.session_state.get("last_command") or "—",
        "TIME": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "FRED": "KEY" if credentials["FRED"] else "KEYLESS",
        "OPENSKY": "OAUTH2" if credentials["OpenSky"] else "ANON",
        "AIS": "LIVE" if credentials["AISStream"] else "OFF",
        "PYTHON": f"{sys.version_info.major}.{sys.version_info.minor}",
    })


if __name__ == "__main__":
    main()

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
    if module == "equity":
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
        ("HOME", "home"), ("EQUITY", "equity"), ("SHIP", "maritime"),
        ("FLY", "aviation"), ("MACRO", "macro"), ("NEWS", "news"),
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
                    "PEER COMPARISON", "OPTIONS", "NEWS", "PROFILE"])

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
        default_peers = equities.suggest_peers(ticker)
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
                    "Multiples from Yahoo Finance. A negative or blank P/E "
                    "means trailing losses. EV/EBITDA is generally the more "
                    "comparable multiple across capital structures."
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

    # ---- PROFILE ---------------------------------------------------------
    with tabs[6]:
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
                congestion = status.get("congestion_ratio", 0)
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
                    ui.metric_tile("CONGESTION", congestion, subtitle="× baseline",
                                   value_format="{:.2f}", accent=status_color),
                    ui.metric_tile("MEAN DRAUGHT", status.get("mean_draught_m"),
                                   subtitle="m — laden proxy"),
                ], columns=6)

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
            "Institutional airframes only — national carriers, cargo fleets "
            "and state transports drawn from public registries. Open-Terminal "
            "ships no private-individual watchlist; sustained tracking of a "
            "named person's aircraft is treated very differently in law from "
            "fleet-level analysis, and OpenSky's terms restrict it."
        )

        watchlist = st.selectbox("WATCHLIST",
                                 ["ALL"] + list(config.AIRCRAFT_WATCHLISTS),
                                 key="av_watchlist")

        if st.button("QUERY WATCHLIST", use_container_width=True):
            with st.spinner("Querying transponder addresses…"):
                tracked = _safe(aviation.track_watchlist,
                                None if watchlist == "ALL" else watchlist,
                                default=pd.DataFrame())

            if tracked.empty:
                ui.alert(
                    "No watchlist aircraft are currently transmitting. This is "
                    "normal — most of these airframes are parked with "
                    "transponders off most of the time. Absence of a signal "
                    "is not evidence of anything.", "warn",
                )
            else:
                st.success(f"{len(tracked)} aircraft transmitting")
                ui.render_chart(maps.flight_map(tracked, "WATCHLIST POSITIONS",
                                                show_labels=True))
                columns = [c for c in ("label", "callsign", "icao24", "watchlist",
                                       "altitude_ft", "speed_kts", "phase",
                                       "origin_country")
                           if c in tracked.columns]
                ui.styled_table(tracked[columns])

        with st.expander("CONFIGURED AIRFRAMES"):
            for group, entries in config.AIRCRAFT_WATCHLISTS.items():
                st.markdown(f"**{group}**")
                st.dataframe(
                    pd.DataFrame(list(entries.items()),
                                 columns=["ICAO24", "Description"]),
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

    tabs = st.tabs(["YIELD CURVE", "RECESSION SIGNALS", "INFLATION",
                    "LABOR & GROWTH", "LIQUIDITY", "GLOBAL"])

    # ---- YIELD CURVE -----------------------------------------------------
    with tabs[0]:
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
    with tabs[1]:
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
    for tab, group in ((tabs[2], "INFLATION"), (tabs[3], "LABOR"),
                       (tabs[4], "LIQUIDITY")):
        with tab:
            _render_series_group(group)
            if group == "LABOR":
                st.divider()
                _render_series_group("GROWTH")

    # ---- GLOBAL ----------------------------------------------------------
    with tabs[5]:
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
# PAGE: NEWS
# ==========================================================================
@st.cache_data(ttl=config.TTL.news, show_spinner=False)
def _cached_news(categories: Tuple[str, ...], limit_per_feed: int,
                 max_articles: int) -> pd.DataFrame:
    return news.get_news(list(categories), limit_per_feed, max_articles)


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

    tabs = st.tabs(["NEWS FEED", "GDELT OSINT", "TRENDING TERMS"])

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

    # ---- GDELT -----------------------------------------------------------
    with tabs[1]:
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
    with tabs[2]:
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
| **Maritime** | `SUEZ SHIP` | AIS positions, chokepoint congestion, vessel lookup by MMSI/IMO |
| **Aviation** | `EUROPE FLY` | Live ADS-B state vectors, fleet watchlists, aircraft tracks, airport flow |
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
ROUTES = {
    "home": page_home,
    "equity": page_equity,
    "maritime": page_maritime,
    "aviation": page_aviation,
    "macro": page_macro,
    "news": page_news,
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

"""
ui/components.py :: Reusable terminal widgets.

Ticker tape, metric tiles, headers, news rows, heatmaps, candlestick charts,
and the sparkline/gauge primitives the pages compose from. Everything here is
presentation only - no network calls, no business logic.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import THEME
from ui.terminal_theme import (
    AMBER_SCALE,
    HEATMAP_SCALE,
    PLOTLY_CONFIG,
    color_for_change,
    css_class_for_change,
    style_figure,
)


# ==========================================================================
# TICKER TAPE
# ==========================================================================
def ticker_tape(quotes: Dict[str, Dict[str, Any]],
                labels: Optional[Dict[str, str]] = None) -> None:
    """
    Scrolling market banner across the top of the app.

    Args:
        quotes: {symbol: {price, change, change_pct}} from
                equities.get_quotes_batch().
        labels: Optional {symbol: display_name} overrides.
    """
    if not quotes:
        st.markdown(
            '<div class="ot-tape"><div class="ot-tape-inner">'
            '<span class="ot-tape-item ot-flat">MARKET DATA UNAVAILABLE — '
            'CHECK CONNECTION</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    labels = labels or {}
    items: List[str] = []

    for symbol, quote in quotes.items():
        price = quote.get("price")
        change = quote.get("change")
        change_pct = quote.get("change_pct")
        if price is None:
            continue

        css = css_class_for_change(change)
        arrow = "▲" if (change or 0) > 0 else "▼" if (change or 0) < 0 else "■"
        name = html.escape(labels.get(symbol, symbol))

        # Indices and crypto need different precision than a $4 stock.
        decimals = 2 if price >= 1 else 4
        pct_text = f"{change_pct:+.2f}%" if change_pct is not None else "—"

        items.append(
            f'<span class="ot-tape-item">'
            f'<span class="ot-tape-sym">{name}</span>'
            f'<span class="ot-tape-px">{price:,.{decimals}f}</span> '
            f'<span class="{css}">{arrow} {pct_text}</span>'
            f'</span>'
        )

    # Duplicate the strip so the CSS marquee loops without a visible gap.
    strip = "".join(items)
    st.markdown(
        f'<div class="ot-tape"><div class="ot-tape-inner">{strip}{strip}</div></div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# HEADERS & PANELS
# ==========================================================================
def module_header(title: str, subtitle: str = "",
                  meta: Optional[str] = None) -> None:
    """Amber module bar with a right-aligned timestamp/metadata slot."""
    meta = meta if meta is not None else datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    sub = f' <span style="color:{THEME.muted};font-size:11px;">// {html.escape(subtitle)}</span>' if subtitle else ""

    st.markdown(
        f'<div class="ot-header">'
        f'<div class="ot-header-title">{html.escape(title)}{sub}</div>'
        f'<div class="ot-header-meta">{html.escape(meta)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    """Bordered panel with an uppercase amber caption."""
    st.markdown(
        f'<div class="ot-panel">'
        f'<div class="ot-panel-title">{html.escape(title)}</div>'
        f'{body_html}</div>',
        unsafe_allow_html=True,
    )


def alert(message: str, level: str = "error") -> None:
    """Terminal-styled alert. level: error | warn | ok"""
    css = {"error": "ot-alert", "warn": "ot-alert ot-alert-warn",
           "ok": "ot-alert ot-alert-ok"}.get(level, "ot-alert")
    icon = {"error": "✖", "warn": "⚠", "ok": "✔"}.get(level, "●")
    st.markdown(
        f'<div class="{css}">{icon}&nbsp;&nbsp;{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = "cyan") -> str:
    """Inline pill. Returns HTML - compose several before writing them out."""
    return f'<span class="ot-badge ot-badge-{color}">{html.escape(str(text))}</span>'


def status_dot(state: str = "live", label: str = "") -> str:
    """Pulsing status indicator. state: live | warn | off"""
    css = {"live": "ot-dot-live", "warn": "ot-dot-warn"}.get(state, "ot-dot-off")
    return f'<span class="ot-dot {css}"></span>{html.escape(label)}'


# ==========================================================================
# METRIC TILES
# ==========================================================================
def metric_tile(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    delta_suffix: str = "%",
    subtitle: str = "",
    accent: Optional[str] = None,
    value_format: str = "{:,.2f}",
) -> str:
    """
    One metric tile as HTML.

    Returns the markup instead of rendering, so callers can lay several out
    in a single st.markdown and avoid Streamlit's inter-element padding.
    """
    accent = accent or THEME.amber

    if value is None or (isinstance(value, float) and np.isnan(value)):
        display = "—"
    elif isinstance(value, (int, float, np.number)):
        display = value_format.format(value)
    else:
        display = str(value)

    delta_html = ""
    if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
        color = color_for_change(delta)
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "■"
        delta_html = (
            f'<div class="ot-tile-delta" style="color:{color};">'
            f'{arrow} {delta:+,.2f}{delta_suffix}</div>'
        )

    sub_html = f'<div class="ot-tile-sub">{html.escape(subtitle)}</div>' if subtitle else ""

    return (
        f'<div class="ot-tile" style="border-left-color:{accent};">'
        f'<div class="ot-tile-label">{html.escape(label)}</div>'
        f'<div class="ot-tile-value">{html.escape(display)}</div>'
        f'{delta_html}{sub_html}</div>'
    )


def metric_row(tiles: List[str], columns: Optional[int] = None) -> None:
    """Render pre-built tile HTML across a responsive grid."""
    if not tiles:
        return
    columns = columns or min(len(tiles), 6)
    cols = st.columns(columns)
    for index, tile in enumerate(tiles):
        with cols[index % columns]:
            st.markdown(tile, unsafe_allow_html=True)


def quote_tiles(quote: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> None:
    """The standard price header block on the equity page."""
    from data_fetchers.equities import format_large_number

    info = info or {}
    price = quote.get("price")
    change = quote.get("change")
    change_pct = quote.get("change_pct")

    tiles = [
        metric_tile("LAST PRICE", price, change_pct, "%",
                    accent=color_for_change(change)),
        metric_tile("CHANGE", change, subtitle="vs previous close",
                    accent=color_for_change(change), value_format="{:+,.2f}"),
        metric_tile("DAY RANGE",
                    f"{quote.get('day_low') or 0:,.2f} – {quote.get('day_high') or 0:,.2f}"
                    if quote.get("day_low") else "—",
                    subtitle="low – high"),
        metric_tile("VOLUME", format_large_number(quote.get("volume"), ""),
                    subtitle="shares traded"),
        metric_tile("MKT CAP",
                    format_large_number(quote.get("market_cap") or info.get("marketCap"))),
        metric_tile("P/E (TTM)", info.get("trailingPE"),
                    subtitle=f"Fwd {info.get('forwardPE'):.1f}" if info.get("forwardPE") else ""),
    ]
    metric_row(tiles, columns=6)


# ==========================================================================
# CHARTS
# ==========================================================================
def candlestick_chart(
    df: pd.DataFrame,
    ticker: str = "",
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    show_bollinger: bool = False,
    ema_periods: Sequence[int] = (20, 50, 200),
    height: int = 720,
) -> go.Figure:
    """
    Multi-pane price chart: candles + volume + RSI + MACD.

    Expects a frame from equities.add_indicators(). Panes are added only if
    the corresponding columns exist, so a short history that couldn't compute
    EMA200 simply omits that line rather than erroring.
    """
    if df is None or df.empty:
        return style_figure(go.Figure(), height=height,
                            title="NO DATA AVAILABLE")

    panes = 1 + int(show_volume) + int(show_rsi and "RSI" in df) \
              + int(show_macd and "macd" in df)

    # Price gets the lion's share; indicator panes are thin strips.
    heights = [0.56] + [(1 - 0.56) / max(panes - 1, 1)] * (panes - 1) if panes > 1 else [1.0]

    fig = make_subplots(
        rows=panes, cols=1, shared_xaxes=True,
        vertical_spacing=0.025, row_heights=heights,
    )

    row = 1

    # --- Price ------------------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker or "PRICE",
            increasing=dict(line=dict(color=THEME.green, width=1),
                            fillcolor=THEME.green),
            decreasing=dict(line=dict(color=THEME.red, width=1),
                            fillcolor=THEME.red),
        ),
        row=row, col=1,
    )

    ema_colors = {20: THEME.amber, 50: THEME.cyan, 200: THEME.magenta}
    for period in ema_periods:
        col = f"EMA{period}"
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=f"EMA{period}",
                           line=dict(color=ema_colors.get(period, THEME.white),
                                     width=1.2)),
                row=row, col=1,
            )

    if show_bollinger and {"bb_upper", "bb_lower"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                       line=dict(color=THEME.muted, width=0.8, dash="dot")),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                       line=dict(color=THEME.muted, width=0.8, dash="dot"),
                       fill="tonexty", fillcolor="rgba(122,127,134,0.07)"),
            row=row, col=1,
        )

    fig.update_yaxes(title_text="PRICE", row=row, col=1, side="right")

    # --- Volume -----------------------------------------------------------
    if show_volume and "Volume" in df.columns:
        row += 1
        colors = np.where(df["Close"] >= df["Open"],
                          "rgba(0,255,102,0.5)", "rgba(255,59,59,0.5)")
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="VOL",
                   marker=dict(color=colors), showlegend=False),
            row=row, col=1,
        )
        fig.update_yaxes(title_text="VOL", row=row, col=1, side="right")

    # --- RSI --------------------------------------------------------------
    if show_rsi and "RSI" in df.columns:
        row += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)",
                       line=dict(color=THEME.amber, width=1.3)),
            row=row, col=1,
        )
        for level, color in ((70, THEME.red), (30, THEME.green), (50, THEME.grid)):
            fig.add_hline(y=level, line=dict(color=color, width=0.7, dash="dash"),
                          row=row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1,
                         side="right")

    # --- MACD -------------------------------------------------------------
    if show_macd and "macd" in df.columns:
        row += 1
        hist_colors = np.where(df["histogram"] >= 0,
                               "rgba(0,255,102,0.55)", "rgba(255,59,59,0.55)")
        fig.add_trace(
            go.Bar(x=df.index, y=df["histogram"], name="HIST",
                   marker=dict(color=hist_colors), showlegend=False),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(color=THEME.cyan, width=1.2)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["signal"], name="SIGNAL",
                       line=dict(color=THEME.amber, width=1.2)),
            row=row, col=1,
        )
        fig.update_yaxes(title_text="MACD", row=row, col=1, side="right")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.005,
                    xanchor="right", x=1),
        bargap=0.05,
    )

    # Hide weekend/holiday gaps on daily bars so candles sit flush.
    if len(df) > 1:
        try:
            median_gap = pd.Series(df.index).diff().median()
            if median_gap and median_gap >= pd.Timedelta(days=1):
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        except Exception:
            pass

    return style_figure(fig, height=height, showlegend=True)


def line_chart(
    series_map: Dict[str, pd.Series],
    title: str = "",
    y_title: str = "",
    height: int = 380,
    fill: bool = False,
) -> go.Figure:
    """Multi-series line chart. Keys become legend entries."""
    fig = go.Figure()

    for index, (name, series) in enumerate(series_map.items()):
        if series is None or len(series) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=str(name),
            mode="lines", line=dict(width=1.6),
            fill="tozeroy" if fill and index == 0 else None,
        ))

    fig.update_yaxes(title_text=y_title)
    return style_figure(fig, height=height, title=title)


def sparkline(series: pd.Series, height: int = 52,
              color: Optional[str] = None) -> go.Figure:
    """Tiny inline trend chart for metric tiles. No axes, no legend."""
    fig = go.Figure()

    if series is not None and len(series) > 1:
        color = color or (
            THEME.green if series.iloc[-1] >= series.iloc[0] else THEME.red
        )
        fig.add_trace(go.Scatter(
            x=list(range(len(series))), y=series.values, mode="lines",
            line=dict(color=color, width=1.4),
            fill="tozeroy",
            fillcolor=f"rgba({_hex_to_rgb(color)},0.15)",
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def heatmap(
    df: pd.DataFrame,
    value_column: str,
    label_column: str,
    title: str = "",
    height: int = 380,
    columns: int = 6,
) -> go.Figure:
    """
    Grid heatmap of a single metric (typically % change) across instruments.

    Lays values out in a `columns`-wide grid rather than a treemap so cells
    stay equally weighted and comparable at a glance.
    """
    if df is None or df.empty or value_column not in df.columns:
        return style_figure(go.Figure(), height=height, title="NO DATA")

    data = df.dropna(subset=[value_column]).copy()
    if data.empty:
        return style_figure(go.Figure(), height=height, title="NO DATA")

    values = data[value_column].tolist()
    labels = data[label_column].astype(str).tolist()

    rows = int(np.ceil(len(values) / columns))
    padded = values + [np.nan] * (rows * columns - len(values))
    padded_labels = labels + [""] * (rows * columns - len(labels))

    grid = np.array(padded, dtype=float).reshape(rows, columns)
    label_grid = np.array(padded_labels).reshape(rows, columns)

    # Symmetric colour scale so +2% and -2% read equally intense.
    bound = float(np.nanmax(np.abs(grid))) or 1.0

    text = np.array([
        [f"{label_grid[r][c]}<br><b>{grid[r][c]:+.2f}%</b>"
         if not np.isnan(grid[r][c]) else ""
         for c in range(columns)]
        for r in range(rows)
    ])

    fig = go.Figure(go.Heatmap(
        z=grid, text=text, texttemplate="%{text}",
        textfont=dict(size=11, family=THEME.font_mono),
        colorscale=HEATMAP_SCALE, zmid=0, zmin=-bound, zmax=bound,
        showscale=True,
        colorbar=dict(title="%", tickfont=dict(color=THEME.muted, size=9),
                      thickness=11, len=0.75),
        hoverinfo="text", xgap=2, ygap=2,
    ))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    return style_figure(fig, height=height, title=title, showlegend=False)


def gauge(
    value: float,
    title: str = "",
    min_value: float = 0,
    max_value: float = 100,
    thresholds: Optional[List[Tuple[float, str]]] = None,
    height: int = 220,
    suffix: str = "",
) -> go.Figure:
    """
    Semicircular gauge for composite scores (recession risk, congestion).

    Args:
        thresholds: [(upper_bound, color)] band definitions, ascending.
    """
    thresholds = thresholds or [
        (max_value * 0.33, THEME.green),
        (max_value * 0.66, THEME.amber),
        (max_value, THEME.red),
    ]

    steps = []
    lower = min_value
    for upper, color in thresholds:
        steps.append({"range": [lower, upper],
                      "color": f"rgba({_hex_to_rgb(color)},0.22)"})
        lower = upper

    needle_color = THEME.green
    for upper, color in thresholds:
        if value <= upper:
            needle_color = color
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=suffix, font=dict(size=30, color=THEME.white,
                                             family=THEME.font_mono)),
        title=dict(text=title, font=dict(size=11, color=THEME.muted)),
        gauge=dict(
            axis=dict(range=[min_value, max_value],
                      tickcolor=THEME.muted,
                      tickfont=dict(size=9, color=THEME.muted)),
            bar=dict(color=needle_color, thickness=0.62),
            bgcolor=THEME.bg_panel,
            borderwidth=1, bordercolor=THEME.border,
            steps=steps,
        ),
    ))

    return style_figure(fig, height=height, showlegend=False)


# ==========================================================================
# NEWS FEED
# ==========================================================================
def news_feed(df: pd.DataFrame, max_rows: int = 60,
              show_source: bool = True) -> None:
    """
    Terminal-style scrolling news list with colour-coded sentiment.

    Rendered as one HTML block rather than per-row Streamlit elements -
    60 individual st.markdown calls is visibly slow.
    """
    from data_fetchers.news import time_ago

    if df is None or df.empty:
        alert("No headlines retrieved. Feeds may be unreachable.", "warn")
        return

    rows: List[str] = []
    for _, article in df.head(max_rows).iterrows():
        score = article.get("sentiment")
        label = article.get("label", "NEUTRAL")

        if label == "BULLISH":
            color, marker = THEME.green, "▲"
        elif label == "BEARISH":
            color, marker = THEME.red, "▼"
        else:
            color, marker = THEME.muted, "■"

        score_text = f"{marker} {score:+.2f}" if score is not None and not pd.isna(score) else marker

        title = html.escape(str(article.get("title", "")))[:200]
        link = str(article.get("link", "") or "")
        title_html = (
            f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer">{title}</a>'
            if link else title
        )

        source_html = (
            f'<span class="ot-news-src">{html.escape(str(article.get("source", ""))[:20])}</span>'
            if show_source else ""
        )

        rows.append(
            f'<div class="ot-news-row">'
            f'<span class="ot-news-time">{html.escape(time_ago(article.get("published")))}</span>'
            f'{source_html}'
            f'<span class="ot-news-title">{title_html}</span>'
            f'<span class="ot-news-score" style="color:{color};">{score_text}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="max-height:620px;overflow-y:auto;border:1px solid {THEME.border};'
        f'background:{THEME.bg_panel};">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def sentiment_bar(summary: Dict[str, Any], height: int = 150) -> go.Figure:
    """Horizontal stacked bar of bullish / neutral / bearish counts."""
    fig = go.Figure()

    for label, color in (("bullish", THEME.green), ("neutral", THEME.muted),
                         ("bearish", THEME.red)):
        count = summary.get(label, 0)
        fig.add_trace(go.Bar(
            y=["SENTIMENT"], x=[count], name=label.upper(),
            orientation="h", marker=dict(color=color),
            text=[str(count)], textposition="inside",
            textfont=dict(color="#000", size=12, family=THEME.font_mono),
        ))

    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=-0.3))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, height=height)


# ==========================================================================
# TABLES
# ==========================================================================
def styled_table(
    df: pd.DataFrame,
    numeric_format: str = "{:,.2f}",
    highlight_columns: Optional[List[str]] = None,
    height: Optional[int] = None,
    use_container_width: bool = True,
) -> None:
    """
    DataFrame with terminal formatting and red/green on signed columns.

    Args:
        highlight_columns: Columns where sign should drive the text colour
                           (change %, spreads, sentiment).
    """
    if df is None or df.empty:
        alert("No data to display.", "warn")
        return

    highlight_columns = highlight_columns or []
    styler = df.style

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        styler = styler.format(
            {col: numeric_format for col in numeric_cols}, na_rep="—"
        )

    def sign_color(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"color: {THEME.muted}"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if numeric > 0:
            return f"color: {THEME.green}"
        if numeric < 0:
            return f"color: {THEME.red}"
        return f"color: {THEME.muted}"

    valid_highlights = [c for c in highlight_columns if c in df.columns]
    if valid_highlights:
        styler = styler.map(sign_color, subset=valid_highlights)

    styler = styler.set_properties(**{
        "background-color": THEME.bg_panel,
        "color": THEME.cyan,
        "font-family": THEME.font_mono,
        "font-size": "12px",
        "border": f"1px solid {THEME.grid}",
    })

    # `height=None` is rejected outright by current Streamlit
    # (StreamlitInvalidHeightError) rather than treated as "auto", so the
    # kwarg has to be omitted entirely when the caller didn't specify one.
    kwargs: Dict[str, Any] = {"use_container_width": use_container_width}
    if height is not None:
        kwargs["height"] = height

    st.dataframe(styler, **kwargs)


def statement_table(df: pd.DataFrame, scale: float = 1e6,
                    scale_label: str = "$ millions") -> None:
    """
    Financial statement display, scaled to readable units.

    Filings report raw dollars; nobody wants to read 394,328,000,000.
    """
    if df is None or df.empty:
        # Fetchers attach a `reason` when they know *why* there is no data;
        # that is far more useful than a generic "unavailable".
        reason = (df.attrs.get("reason") if df is not None else None)
        alert(reason or "Statement unavailable for this filer.", "warn")
        return

    scaled = df.copy()
    for col in scaled.columns:
        scaled[col] = pd.to_numeric(scaled[col], errors="coerce") / scale

    # Per-share figures must not be scaled.
    for row_label in scaled.index:
        if "EPS" in str(row_label).upper() or "PER SHARE" in str(row_label).upper():
            scaled.loc[row_label] = pd.to_numeric(df.loc[row_label], errors="coerce")

    scaled.columns = [_format_period(c) for c in scaled.columns]

    st.caption(f"Figures in {scale_label} except per-share amounts")
    styled_table(scaled, numeric_format="{:,.0f}")


def _format_period(col: Any) -> str:
    """Turn a period column label into something compact."""
    if isinstance(col, (pd.Timestamp, datetime)):
        return col.strftime("%Y-%m")
    return str(col)


# ==========================================================================
# STATUS BAR
# ==========================================================================
def status_bar(items: Dict[str, str]) -> None:
    """Bottom strip of key/value diagnostics."""
    parts = [
        f'<span><span style="color:{THEME.muted};">{html.escape(k)}:</span> '
        f'<span style="color:{THEME.cyan};">{html.escape(str(v))}</span></span>'
        for k, v in items.items()
    ]
    st.markdown(
        f'<div class="ot-statusbar">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_chart(fig: go.Figure, key: Optional[str] = None) -> None:
    """st.plotly_chart with the terminal's standard config applied."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


# ==========================================================================
# PEOPLE
# ==========================================================================
# Platform -> (glyph, accent). Text glyphs rather than icon fonts: the theme
# overrides font-family globally and a webfont icon would render as its own
# ligature name.
_SOCIAL_STYLE: Dict[str, Tuple[str, str]] = {
    "twitter": ("X", THEME.white),
    "linkedin": ("in", THEME.cyan),
    "facebook": ("f", THEME.cyan),
    "instagram": ("IG", THEME.magenta),
    "youtube": ("YT", THEME.red),
    "tiktok": ("TT", THEME.magenta),
    "mastodon": ("M", THEME.cyan),
    "website": ("WWW", THEME.amber),
    "wikidata": ("WD", THEME.muted),
}


def social_links(social: Dict[str, Any], size: int = 11) -> str:
    """
    Row of social handles as linked pills. Returns HTML.

    `social` is {platform: {"handle": ..., "url": ...}} as produced by
    company_intel. Unknown platforms are rendered with their own name.
    """
    if not social:
        return (f'<span style="color:{THEME.muted};font-size:{size}px;">'
                f'no verified accounts</span>')

    parts = []
    for platform, entry in social.items():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        # str(), not truthiness: a missing value arriving from a DataFrame is
        # NaN, and NaN is truthy - it reached .startswith() and crashed the
        # whole page with "'float' object has no attribute 'startswith'".
        if url is None or not isinstance(url, str) or not url.strip():
            continue

        glyph, color = _SOCIAL_STYLE.get(platform, (platform.upper()[:3], THEME.cyan))
        raw_handle = entry.get("handle")
        handle = raw_handle if isinstance(raw_handle, str) else ""
        if platform == "website":
            label = "site"
        elif handle.startswith("http") or not handle:
            label = platform
        else:
            label = handle[:26]
        parts.append(
            f'<a href="{html.escape(entry["url"])}" target="_blank" '
            f'style="display:inline-block;border:1px solid {THEME.border};'
            f'padding:1px 6px;margin:2px 4px 2px 0;font-size:{size}px;'
            f'color:{color} !important;text-decoration:none;">'
            f'<b>{html.escape(glyph)}</b>&nbsp;'
            f'<span style="color:{THEME.muted};">{html.escape(label)}</span></a>'
        )

    return "".join(parts) or (
        f'<span style="color:{THEME.muted};font-size:{size}px;">'
        f'no verified accounts</span>')


def executive_card(row: pd.Series, currency: str = "$") -> str:
    """One executive as a bordered card. Returns HTML."""
    name = html.escape(str(row.get("name") or "—"))
    title = html.escape(str(row.get("title") or ""))

    age = row.get("age")
    age_text = f"age {int(age)}" if pd.notna(age) else ""

    pay = row.get("total_pay")
    if pd.notna(pay) and pay:
        pay_text = f"{currency}{pay / 1e6:,.2f}M"
        pay_color = THEME.amber
    else:
        pay_text = "not disclosed"
        pay_color = THEME.muted

    # Every field here can arrive as NaN from the officers DataFrame, so each
    # is tested with pd.notna rather than for truthiness.
    def _value(column: str) -> Optional[str]:
        value = row.get(column)
        return str(value) if value is not None and pd.notna(value) else None

    social: Dict[str, Any] = {}
    if _value("twitter"):
        social["twitter"] = {"handle": _value("twitter_handle") or "profile",
                             "url": _value("twitter")}
    if _value("linkedin"):
        social["linkedin"] = {"handle": "profile", "url": _value("linkedin")}
    if _value("wikidata_url"):
        social["wikidata"] = {"handle": "entity", "url": _value("wikidata_url")}

    bio = _value("bio")
    bio_html = (f'<div style="color:{THEME.muted};font-size:9px;margin-top:2px;">'
                f'{html.escape(bio[:90])}</div>') if bio else ""

    return (
        f'<div style="border:1px solid {THEME.border};border-left:3px solid '
        f'{THEME.amber};padding:7px 9px;margin-bottom:6px;background:{THEME.bg_panel};">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<span style="color:{THEME.white};font-size:13px;font-weight:700;">{name}</span>'
        f'<span style="color:{pay_color};font-size:11px;">{pay_text}</span>'
        f'</div>'
        f'<div style="color:{THEME.cyan};font-size:10px;">{title}'
        f'<span style="color:{THEME.muted};"> {("· " + age_text) if age_text else ""}</span>'
        f'</div>'
        f'{bio_html}'
        f'<div style="margin-top:4px;">{social_links(social, size=10)}</div>'
        f'</div>'
    )


# ==========================================================================
# SUPPLY CHAIN NETWORK
# ==========================================================================
# Geometry in data units. The layout is fixed rather than force-directed: a
# supply chain has an inherent direction, and a spring layout hides it.
# Suppliers feed the focal company from the left, customers draw from it on
# the right, and industry comparables hang below - the shape a terminal user
# reads without needing a legend.
# Cards are sized generously against the axis range: the boxes scale with the
# container but the type does not, so a tight box overflows on narrow screens.
_CARD_W, _CARD_H, _CARD_GAP = 2.62, 0.62, 0.24
_COL_X, _HUB_X = 3.38, 1.30
_FOCAL_W, _FOCAL_H = 3.05, 1.50
_COMP_W, _COMP_H = 1.52, 0.54
_MAX_CARDS = 6          # per column; the remainder is reported in the hub


def _rect(fig, x, y, w, h, border, fill, width=1):
    """Card outline centred on (x, y)."""
    fig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2,
                  y0=y - h / 2, y1=y + h / 2, layer="below",
                  line=dict(color=border, width=width), fillcolor=fill)


def _text(fig, x, y, text, color, size=9, anchor="center"):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, align="left",
                       xanchor=anchor, yanchor="middle",
                       font=dict(size=size, color=color, family=THEME.font_mono))


def _wire(fig, start, end, color, width=1.0, dash="solid"):
    fig.add_shape(type="line", x0=start[0], y0=start[1], x1=end[0], y1=end[1],
                  layer="below", line=dict(color=color, width=width, dash=dash))


def _stack(count, size, gap):
    """Centres for `count` cards laid out symmetrically about zero."""
    pitch = size + gap
    return [(index - (count - 1) / 2) * pitch for index in range(count)]


def _card_lines(node, quantified):
    """Title and metric line for one counterparty card."""
    label = str(node.get("label") or node["id"])
    if len(label) > 20:
        label = label[:19] + "…"

    pct = node.get("pct")
    if pct:
        metric = f"REV {pct:,.1f}%"
    elif quantified:
        metric = "SHARE NOT QUANTIFIED"
    else:
        metric = "DISCLOSED"

    if not node.get("named", True):
        metric += " · WITHHELD"
    return label, metric


def supply_chain_graph(network: Dict[str, Any], height: int = 640) -> go.Figure:
    """
    Counterparty map in the shape a supply-chain terminal draws it.

    Suppliers column -> aggregation hub -> focal company -> aggregation hub ->
    customers column, with industry comparables on a spur below. Card size is
    constant and the exposure rides in the metric line, which reads more
    precisely than node area for the handful of relationships a 10-K names.

    Each hub reports the *total* count on that side; the column shows the
    largest few. A dotted edge is a disclosure where the issuer withheld the
    counterparty's name.
    """
    nodes = network.get("nodes", [])
    stats = network.get("stats", {}) or {}

    if not nodes:
        return style_figure(go.Figure(), height=height, title="NO NETWORK DATA")

    focal = next((n for n in nodes if n.get("tier") == "focal"), None)
    suppliers = [n for n in nodes if n.get("tier") == "upstream"]
    customers = [n for n in nodes if n.get("tier") == "downstream"]
    comps = [n for n in nodes if n.get("tier") == "peer"]

    def by_share(group):
        return sorted(group,
                      key=lambda n: (n.get("pct") is None, -(n.get("pct") or 0)))

    shown_suppliers = by_share(suppliers)[:_MAX_CARDS]
    shown_customers = by_share(customers)[:_MAX_CARDS]

    fig = go.Figure()
    hover_x, hover_y, hover_text = [], [], []
    top = stats.get("max_customer_pct")

    # ---- focal company ---------------------------------------------------
    _rect(fig, 0, 0, _FOCAL_W, _FOCAL_H, THEME.amber, THEME.bg_raised, width=2)
    focal_label = str((focal or {}).get("label") or "—")
    if len(focal_label) > 24:
        focal_label = focal_label[:23] + "…"
    _text(fig, 0, 0, "<br>".join([
        f"<b>{focal_label}</b>",
        f"<span style='color:{THEME.muted}'>{(focal or {}).get('id', '')}</span>",
        f"DISCLOSED {stats.get('counterparties', 0)}"
        f"   NAMED {stats.get('named', 0)}",
        f"TOP CUSTOMER {top:,.0f}%" if top else "TOP CUSTOMER —",
    ]), THEME.white, size=10)

    # ---- one side of the chain -------------------------------------------
    def draw_side(group, total, sign, colour, heading):
        hub = (sign * _HUB_X, 0.0)
        if not group:
            _text(fig, sign * _COL_X, 0.0, f"NO {heading} DISCLOSED",
                  THEME.muted, size=9)
            return

        _rect(fig, hub[0], hub[1], 0.30, 0.30, colour, THEME.bg_panel)
        _wire(fig, hub, (sign * _FOCAL_W / 2, 0), colour, 1.4)
        _text(fig, hub[0], 0.44, f"<b>{total:,}</b> {heading}", colour, size=9)

        for node, y in zip(group, _stack(len(group), _CARD_H, _CARD_GAP)):
            x = sign * _COL_X
            named = node.get("named", True)
            edge_colour = colour if named else THEME.grid
            _rect(fig, x, y, _CARD_W, _CARD_H, edge_colour, THEME.bg_panel)

            title, metric = _card_lines(node, bool(top))
            left = x - _CARD_W / 2 + 0.12
            _text(fig, left, y + 0.11, f"<b>{title}</b>", THEME.white,
                  size=9, anchor="left")
            _text(fig, left, y - 0.13,
                  f"<span style='color:{THEME.muted}'>{metric}</span>",
                  THEME.muted, size=8, anchor="left")

            _wire(fig, (x - sign * _CARD_W / 2, y), hub, edge_colour,
                  width=max(1.0, min(5.0, (node.get("pct") or 0) / 12 + 1)),
                  dash="solid" if named else "dot")

            detail = f"<b>{node.get('label')}</b><br>{heading.title()[:-1]}"
            if node.get("pct"):
                detail += f"<br>{node['pct']:,.1f}% of revenue"
            if not named:
                detail += "<br><i>name withheld in filing</i>"
            hover_x.append(x)
            hover_y.append(y)
            hover_text.append(detail)

    draw_side(shown_suppliers, stats.get("suppliers", 0), -1,
              THEME.cyan, "SUPPLIERS")
    draw_side(shown_customers, stats.get("customers", 0), 1,
              THEME.green, "CUSTOMERS")

    # ---- industry comparables --------------------------------------------
    rows = max(len(shown_suppliers), len(shown_customers), 1)
    column_half = rows * (_CARD_H + _CARD_GAP) / 2
    comp_y = -max(column_half, _FOCAL_H / 2) - 1.35

    if comps:
        hub = (0.0, comp_y + 0.62)
        _rect(fig, hub[0], hub[1], 0.30, 0.30, THEME.muted, THEME.bg_panel)
        _wire(fig, (0, -_FOCAL_H / 2), hub, THEME.muted, 1.2, dash="dot")
        industry = str(stats.get("industry") or "industry").upper()
        _text(fig, 0, hub[1] + 0.36,
              f"<b>{len(comps)}</b> COMPARABLES · {industry}",
              THEME.muted, size=9)

        for node, x in zip(comps, _stack(len(comps), _COMP_W, 0.14)):
            _rect(fig, x, comp_y, _COMP_W, _COMP_H, THEME.border,
                  THEME.bg_panel)
            _wire(fig, hub, (x, comp_y + _COMP_H / 2), THEME.border, 1.0,
                  dash="dot")
            weight = node.get("weight")
            _text(fig, x, comp_y + 0.10, f"<b>{node['id']}</b>", THEME.white,
                  size=9)
            if weight:
                _text(fig, x, comp_y - 0.12,
                      f"<span style='color:{THEME.muted}'>"
                      f"{weight * 100:,.1f}% WT</span>", THEME.muted, size=8)

            hover_x.append(x)
            hover_y.append(comp_y)
            hover_text.append(
                f"<b>{node.get('label')}</b><br>Industry comparable"
                + (f"<br>{weight * 100:,.1f}% of industry market weight"
                   if weight else ""))

    # ---- hover layer -----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        hovertext=hover_text, hoverinfo="text", showlegend=False,
    ))

    # ---- frame -----------------------------------------------------------
    # Ranges are pinned so the map always opens whole. Shapes and annotations
    # carry no autoscale extent, so without this the axes fit only the
    # invisible hover layer and clip every card.
    top_y = max(column_half + 0.85, _FOCAL_H / 2 + 0.85)
    bottom_y = (comp_y - _COMP_H) if comps else -top_y
    half_width = _COL_X + _CARD_W / 2 + 0.22

    fig.update_xaxes(range=[-half_width, half_width], visible=False)
    fig.update_yaxes(range=[bottom_y - 0.45, top_y], visible=False)

    return style_figure(fig, height=height, title="", showlegend=False)

def donut(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str = "",
    height: int = 340,
    hole: float = 0.58,
    center_label: str = "",
    center_value: str = "",
    value_prefix: str = "$",
    max_slices: Optional[int] = None,
) -> go.Figure:
    """
    Share-of-whole donut for sector or position mix.

    Complements `exposure_bars` rather than replacing it: bars rank and
    compare precisely, a donut answers "how much of the book is this" at a
    glance. The allocation page shows both because those are different
    questions.

    Args:
        max_slices:   Beyond this many, the smallest are pooled into OTHER.
                      The pooled row keeps its real total and the caption
                      says how many names are in it - the alternative is
                      twenty unreadable slivers.
        center_label: Small caption in the hole.
        center_value: Large figure in the hole - usually the total, which is
                      the one number a donut otherwise throws away.

    Slice order is preserved as passed (`sort=False`), so the donut reads in
    the same order as the table beside it. Plotly's default re-sorts, which
    silently breaks that correspondence.
    """
    if df is None or df.empty or value_col not in df.columns:
        return style_figure(go.Figure(), height=height, title=title or "NO DATA")

    frame = df[[label_col, value_col]].copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[value_col])
    frame = frame[frame[value_col] > 0]
    if frame.empty:
        return style_figure(go.Figure(), height=height, title=title or "NO DATA")

    frame = frame.sort_values(value_col, ascending=False)

    pooled = 0
    if max_slices and len(frame) > max_slices:
        head = frame.head(max_slices - 1)
        tail = frame.iloc[max_slices - 1:]
        pooled = len(tail)
        frame = pd.concat([head, pd.DataFrame([{
            label_col: f"OTHER ({pooled})",
            value_col: float(tail[value_col].sum()),
        }])], ignore_index=True)

    total = float(frame[value_col].sum())

    fig = go.Figure(go.Pie(
        labels=frame[label_col].astype(str),
        values=frame[value_col],
        hole=hole,
        sort=False,
        direction="clockwise",
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=10, family=THEME.font_mono, color=THEME.bg),
        # A hairline in the page background separates adjacent slices whose
        # colours are close, which matters on a dark theme where several of
        # the palette entries are saturated.
        marker=dict(line=dict(color=THEME.bg, width=1.5)),
        hovertemplate=("%{label}<br>" + value_prefix +
                       "%{value:,.0f}  ·  %{percent}<extra></extra>"),
    ))

    if center_value or center_label:
        fig.add_annotation(
            text=(f'<span style="font-size:17px;color:{THEME.white};">'
                  f'{center_value}</span>'
                  + (f'<br><span style="font-size:9px;color:{THEME.muted};">'
                     f'{center_label}</span>' if center_label else "")),
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(family=THEME.font_mono),
        )

    fig = style_figure(fig, height=height, title=title, showlegend=True)
    fig.update_layout(
        legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle",
                    font=dict(size=9)),
        margin=dict(l=8, r=8, t=42, b=8),
    )
    return fig


def exposure_bars(df: pd.DataFrame, label_col: str, value_col: str,
                  title: str = "", height: int = 300,
                  suffix: str = "%", color: Optional[str] = None,
                  text_col: Optional[str] = None) -> go.Figure:
    """
    Horizontal bars for revenue-by-region / commodity-correlation panels.

    Args:
        color:    One colour for every bar. Pass this for magnitudes that are
                  always positive - portfolio weights, for instance - where
                  the default red/green would paint the whole chart green and
                  imply a gain that isn't being measured.
        text_col: Column holding pre-formatted bar labels. Without it a bar
                  can only say "54.9%", which is the least useful half of the
                  answer when the reader wants to know what that is in money.
    """
    if df.empty:
        return style_figure(go.Figure(), height=height, title=title)

    frame = df.sort_values(value_col)
    colors = ([color] * len(frame) if color
              else [THEME.green if v >= 0 else THEME.red for v in frame[value_col]])

    labels = (frame[text_col].astype(str).tolist()
              if text_col and text_col in frame.columns
              else [f"{v:,.1f}{suffix}" for v in frame[value_col]])

    fig = go.Figure(go.Bar(
        x=frame[value_col], y=frame[label_col], orientation="h",
        marker=dict(color=colors),
        text=labels,
        textposition="auto",
        hovertemplate="%{y}: %{x:,.2f}" + suffix + "<extra></extra>",
    ))
    return style_figure(fig, height=height, title=title, showlegend=False)


# ==========================================================================
# HELPERS
# ==========================================================================
def _hex_to_rgb(hex_color: str) -> str:
    """'#00FF66' -> '0,255,102' for rgba() strings."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "255,176,0"
    return ",".join(str(int(hex_color[i:i + 2], 16)) for i in (0, 2, 4))


__all__ = [
    "ticker_tape", "module_header", "panel", "alert", "badge", "status_dot",
    "metric_tile", "metric_row", "quote_tiles",
    "candlestick_chart", "line_chart", "sparkline", "heatmap", "gauge",
    "news_feed", "sentiment_bar",
    "styled_table", "statement_table", "status_bar", "render_chart",
    "social_links", "executive_card", "supply_chain_graph", "exposure_bars",
    "donut",
]

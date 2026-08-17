"""
ui/terminal_theme.py :: Bloomberg-style dark theme for Streamlit and Plotly.

Injects a CSS layer over Streamlit's defaults and provides matching Plotly
templates so charts don't look pasted in from a different application.

Palette lives in config.THEME so the fetchers and the UI agree on what
"bullish green" means.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from config import THEME

# ==========================================================================
# CSS
# ==========================================================================
TERMINAL_CSS = f"""
<style>
/* ---------- Font ---------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {{
    --ot-bg:        {THEME.bg};
    --ot-panel:     {THEME.bg_panel};
    --ot-raised:    {THEME.bg_raised};
    --ot-border:    {THEME.border};
    --ot-grid:      {THEME.grid};
    --ot-amber:     {THEME.amber};
    --ot-green:     {THEME.green};
    --ot-red:       {THEME.red};
    --ot-cyan:      {THEME.cyan};
    --ot-white:     {THEME.white};
    --ot-muted:     {THEME.muted};
    --ot-magenta:   {THEME.magenta};
}}

/* ---------- Global shell -------------------------------------------- */
html, body, [class*="css"], .stApp {{
    background-color: var(--ot-bg) !important;
    color: var(--ot-cyan);
    font-family: {THEME.font_mono} !important;
    font-size: 13px;
}}

.main .block-container {{
    padding-top: 0.6rem;
    padding-bottom: 2rem;
    max-width: 100%;
}}

/* Hide Streamlit chrome - this is a terminal, not a web app. */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton, [data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stToolbarActions"], [data-testid="stMainMenu"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}

/* One thing in that header has to survive: the chevron that reopens a
   collapsed sidebar. Streamlit renders it inside the toolbar, so the old
   `[data-testid="stToolbar"] {{ display: none }}` removed the only way back
   in - and the collapsed flag is persisted in localStorage, so the sidebar
   stayed gone across reloads. The header itself remains visibility:hidden,
   which keeps it click-through; only this subtree is painted back. */
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * {{ visibility: visible !important; }}
[data-testid="stExpandSidebarButton"] button {{
    color: var(--ot-amber) !important;
    border-radius: 0 !important;
}}

/* ---------- Typography ----------------------------------------------- */
h1, h2, h3, h4, h5, h6 {{
    color: var(--ot-amber) !important;
    font-family: {THEME.font_mono} !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem !important;
}}
h1 {{ font-size: 1.35rem !important; border-bottom: 2px solid var(--ot-amber);
      padding-bottom: 0.35rem; }}
h2 {{ font-size: 1.05rem !important; }}
h3 {{ font-size: 0.92rem !important; color: var(--ot-cyan) !important; }}

p, span, div, label, li {{ font-family: {THEME.font_mono} !important; }}

/* Icons are the exception. Streamlit draws Material Symbols as ligatures in a
   <span>, so the blanket mono override above printed the literal ligature name
   ("keyboard_arrow_right") instead of a chevron - expander arrows and the
   sidebar collapse control both render through this. */
[data-testid="stIconMaterial"],
.material-icons, .material-symbols-rounded,
span[class*="material-symbols"], span[class*="material-icons"] {{
    font-family: "Material Symbols Rounded", "Material Icons" !important;
    font-weight: 400 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}}

a {{ color: var(--ot-cyan) !important; text-decoration: none; }}
a:hover {{ color: var(--ot-amber) !important; text-decoration: underline; }}
code {{ background: var(--ot-raised) !important; color: var(--ot-green) !important;
        border: 1px solid var(--ot-border); padding: 1px 5px; }}

/* ---------- Sidebar --------------------------------------------------- */
[data-testid="stSidebar"] {{
    background-color: var(--ot-panel) !important;
    border-right: 1px solid var(--ot-border);
    width: 300px !important;
}}
[data-testid="stSidebar"] > div {{ background-color: var(--ot-panel) !important; }}

/* Text colour, but only on the elements that carry text. The old blanket
   `[data-testid="stSidebar"] *` rule repainted every child - captions,
   icons, dataframe cells - a flat cyan. */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {{ color: var(--ot-cyan); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: var(--ot-amber) !important; }}
[data-testid="stSidebar"] h3 {{
    font-size: 0.78rem !important;
    margin: 0.5rem 0 0.25rem !important;
    border-bottom: 1px solid var(--ot-grid);
    padding-bottom: 3px;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: var(--ot-muted) !important;
    font-size: 10px !important;
    line-height: 1.45;
}}

/* Reclaim the empty 48px band Streamlit reserves above the sidebar body,
   without clipping the collapse chevron that lives in it. */
[data-testid="stSidebarHeader"] {{
    padding: 0 !important;
    height: 1.6rem !important;
    min-height: 1.6rem !important;
}}
[data-testid="stSidebarUserContent"] {{ padding: 0 0 1.5rem !important; }}
[data-testid="stSidebarContent"] {{ padding-left: 12px; padding-right: 12px; }}
[data-testid="stSidebarCollapseButton"] button {{
    color: var(--ot-muted) !important;
    border-radius: 0 !important;
}}
[data-testid="stSidebarCollapseButton"] button:hover {{
    color: var(--ot-amber) !important;
    background: var(--ot-raised) !important;
}}

/* Long unbroken strings (API-key URLs, cache namespaces) were forcing the
   sidebar wider than its own column and adding a horizontal scrollbar. */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] .stMarkdown {{
    overflow-wrap: anywhere;
    word-break: break-word;
}}
[data-testid="stSidebar"] ul {{ padding-left: 1.1rem !important; margin-bottom: 0.4rem; }}
[data-testid="stSidebar"] li {{ font-size: 11px; margin-bottom: 2px; }}

/* Sidebar buttons: full width, wrapping labels. Recent-command buttons carry
   arbitrary text ("AAPL EQUITY") and were overflowing their box. */
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    min-height: 26px;
    padding: 3px 6px !important;
    font-size: 10px !important;
    letter-spacing: 0.06em;
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.25;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{ gap: 6px; }}

/* Expanders sat on the same colour as the sidebar itself, so they read as
   loose text rather than a control. Lift them onto the raised tone. */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: var(--ot-raised) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    background-color: var(--ot-raised) !important;
    padding: 4px 8px !important;
    font-size: 10px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
    background-color: var(--ot-border) !important;
}}

/* Code samples inherited a blue-grey default that clashed with the palette. */
[data-testid="stSidebar"] pre, [data-testid="stSidebar"] [data-testid="stCode"] {{
    background: #000 !important;
    border: 1px solid var(--ot-border);
    border-radius: 0 !important;
}}
[data-testid="stSidebar"] pre code {{
    background: transparent !important;
    border: none !important;
    font-size: 10px !important;
    color: var(--ot-green) !important;
}}
[data-testid="stSidebar"] hr {{ margin: 0.5rem 0 !important; }}

/* ---------- Inputs ---------------------------------------------------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
    background-color: #000 !important;
    color: var(--ot-amber) !important;
    border: 1px solid var(--ot-border) !important;
    border-radius: 0 !important;
    font-family: {THEME.font_mono} !important;
    font-weight: 500;
    letter-spacing: 0.08em;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: var(--ot-amber) !important;
    box-shadow: 0 0 0 1px var(--ot-amber) !important;
}}
.stTextInput input::placeholder {{ color: #4A4E54 !important; }}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
    background-color: #000 !important;
    border: 1px solid var(--ot-border) !important;
    border-radius: 0 !important;
    color: var(--ot-cyan) !important;
}}
div[data-baseweb="popover"] li {{
    background-color: var(--ot-panel) !important;
    color: var(--ot-cyan) !important;
}}
div[data-baseweb="popover"] li:hover {{
    background-color: var(--ot-raised) !important;
    color: var(--ot-amber) !important;
}}

/* ---------- Buttons --------------------------------------------------- */
.stButton > button, .stDownloadButton > button {{
    background-color: transparent !important;
    color: var(--ot-amber) !important;
    border: 1px solid var(--ot-amber) !important;
    border-radius: 0 !important;
    font-family: {THEME.font_mono} !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px !important;
    transition: all 0.12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: var(--ot-amber) !important;
    color: #000 !important;
}}

/* ---------- Tabs ------------------------------------------------------ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background-color: var(--ot-panel);
    border-bottom: 1px solid var(--ot-border);
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: var(--ot-muted) !important;
    border-radius: 0 !important;
    font-size: 11px !important;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 16px;
    border-right: 1px solid var(--ot-border);
}}
.stTabs [aria-selected="true"] {{
    background-color: var(--ot-raised) !important;
    color: var(--ot-amber) !important;
    border-bottom: 2px solid var(--ot-amber) !important;
}}

/* ---------- Metrics --------------------------------------------------- */
[data-testid="stMetric"] {{
    background-color: var(--ot-panel);
    border: 1px solid var(--ot-border);
    border-left: 3px solid var(--ot-amber);
    padding: 8px 12px;
}}
[data-testid="stMetricLabel"] {{
    color: var(--ot-muted) !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
[data-testid="stMetricValue"] {{
    color: var(--ot-white) !important;
    font-family: {THEME.font_mono} !important;
    font-size: 20px !important;
    font-weight: 700;
}}

/* ---------- Tables ---------------------------------------------------- */
.stDataFrame, [data-testid="stTable"] {{
    background-color: var(--ot-panel) !important;
    border: 1px solid var(--ot-border);
}}
.stDataFrame [role="columnheader"] {{
    background-color: var(--ot-raised) !important;
    color: var(--ot-amber) !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.stDataFrame [role="gridcell"] {{
    color: var(--ot-cyan) !important;
    font-family: {THEME.font_mono} !important;
    font-size: 12px !important;
}}

/* ---------- Expander -------------------------------------------------- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    background-color: var(--ot-panel) !important;
    color: var(--ot-amber) !important;
    border: 1px solid var(--ot-border) !important;
    border-radius: 0 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stExpander"] {{
    border: 1px solid var(--ot-border) !important;
    border-radius: 0 !important;
    background-color: var(--ot-panel);
}}

/* ---------- Alerts ---------------------------------------------------- */
.stAlert {{ border-radius: 0 !important; font-size: 12px; }}
[data-baseweb="notification"] {{ border-radius: 0 !important; }}

/* ---------- Custom components ---------------------------------------- */
.ot-tape {{
    background: linear-gradient(180deg, #141414 0%, #0A0A0A 100%);
    border-top: 1px solid var(--ot-border);
    border-bottom: 1px solid var(--ot-amber);
    overflow: hidden;
    white-space: nowrap;
    padding: 6px 0;
    margin-bottom: 8px;
}}
.ot-tape-inner {{
    display: inline-block;
    padding-left: 100%;
    animation: ot-scroll 55s linear infinite;
}}
.ot-tape:hover .ot-tape-inner {{ animation-play-state: paused; }}
@keyframes ot-scroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-100%); }}
}}
.ot-tape-item {{
    display: inline-block;
    padding: 0 26px;
    font-size: 12px;
    letter-spacing: 0.05em;
    border-right: 1px solid var(--ot-grid);
}}
.ot-tape-sym {{ color: var(--ot-amber); font-weight: 700; }}
.ot-tape-px  {{ color: var(--ot-white); margin-left: 8px; }}
.ot-up   {{ color: var(--ot-green) !important; }}
.ot-down {{ color: var(--ot-red) !important; }}
.ot-flat {{ color: var(--ot-muted) !important; }}

.ot-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--ot-panel);
    border: 1px solid var(--ot-border);
    border-left: 3px solid var(--ot-amber);
    padding: 7px 14px;
    margin-bottom: 10px;
}}
.ot-header-title {{
    color: var(--ot-amber);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.14em;
}}
.ot-header-meta {{ color: var(--ot-muted); font-size: 10px; letter-spacing: 0.08em; }}

.ot-panel {{
    background: var(--ot-panel);
    border: 1px solid var(--ot-border);
    padding: 10px 14px;
    margin-bottom: 10px;
}}
.ot-panel-title {{
    color: var(--ot-amber);
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--ot-grid);
    padding-bottom: 4px;
    margin-bottom: 8px;
}}

.ot-tile {{
    background: var(--ot-panel);
    border: 1px solid var(--ot-border);
    border-left: 3px solid var(--ot-amber);
    padding: 8px 12px;
    height: 100%;
}}
.ot-tile-label {{
    color: var(--ot-muted); font-size: 9px;
    letter-spacing: 0.12em; text-transform: uppercase;
}}
.ot-tile-value {{
    color: var(--ot-white); font-size: 19px;
    font-weight: 700; line-height: 1.25;
}}
.ot-tile-delta {{ font-size: 11px; font-weight: 500; }}
.ot-tile-sub   {{ color: var(--ot-muted); font-size: 9px; }}

.ot-news-row {{
    border-bottom: 1px solid var(--ot-grid);
    padding: 6px 8px;
    display: flex;
    gap: 10px;
    align-items: baseline;
}}
.ot-news-row:hover {{ background: var(--ot-raised); }}
.ot-news-time {{ color: var(--ot-muted); font-size: 10px; min-width: 62px; }}
.ot-news-src  {{ color: var(--ot-amber); font-size: 10px; min-width: 118px;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.ot-news-title {{ color: var(--ot-white); font-size: 12px; flex: 1; }}
.ot-news-title a {{ color: var(--ot-white) !important; }}
.ot-news-title a:hover {{ color: var(--ot-amber) !important; }}
.ot-news-score {{ font-size: 10px; min-width: 52px; text-align: right; font-weight: 700; }}

.ot-badge {{
    display: inline-block;
    padding: 1px 7px;
    font-size: 9px;
    letter-spacing: 0.1em;
    border: 1px solid currentColor;
    margin-right: 5px;
    text-transform: uppercase;
}}
.ot-badge-green   {{ color: var(--ot-green); }}
.ot-badge-red     {{ color: var(--ot-red); }}
.ot-badge-amber   {{ color: var(--ot-amber); }}
.ot-badge-cyan    {{ color: var(--ot-cyan); }}
.ot-badge-muted   {{ color: var(--ot-muted); }}
.ot-badge-magenta {{ color: var(--ot-magenta); }}

.ot-alert {{
    border: 1px solid var(--ot-red);
    border-left: 4px solid var(--ot-red);
    background: rgba(255, 59, 59, 0.07);
    color: var(--ot-red);
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 12px;
    letter-spacing: 0.04em;
}}
.ot-alert-warn {{
    border-color: var(--ot-amber);
    background: rgba(255, 176, 0, 0.07);
    color: var(--ot-amber);
}}
.ot-alert-ok {{
    border-color: var(--ot-green);
    background: rgba(0, 255, 102, 0.06);
    color: var(--ot-green);
}}

.ot-statusbar {{
    background: var(--ot-panel);
    border-top: 1px solid var(--ot-border);
    padding: 4px 12px;
    font-size: 9px;
    color: var(--ot-muted);
    letter-spacing: 0.1em;
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
}}

.ot-dot {{
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 5px;
}}
.ot-dot-live {{ background: var(--ot-green); box-shadow: 0 0 5px var(--ot-green);
                animation: ot-pulse 2s infinite; }}
.ot-dot-warn {{ background: var(--ot-amber); }}
.ot-dot-off  {{ background: #444; }}
@keyframes ot-pulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.35; }}
}}

/* Scrollbars */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--ot-bg); }}
::-webkit-scrollbar-thumb {{ background: var(--ot-border); }}
::-webkit-scrollbar-thumb:hover {{ background: var(--ot-amber); }}

/* Tighten Streamlit's default vertical rhythm */
[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
hr {{ border-color: var(--ot-border) !important; margin: 0.6rem 0 !important; }}
</style>
"""


def apply_theme() -> None:
    """Inject the CSS layer. Call once, immediately after set_page_config."""
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
    _register_plotly_template()


# ==========================================================================
# PLOTLY
# ==========================================================================
def _register_plotly_template() -> None:
    """Register 'openterm' as a Plotly template and make it the default."""
    template = go.layout.Template()

    template.layout = go.Layout(
        paper_bgcolor=THEME.bg,
        plot_bgcolor=THEME.bg,
        font=dict(family=THEME.font_mono, size=11, color=THEME.cyan),
        title=dict(font=dict(color=THEME.amber, size=14, family=THEME.font_mono),
                   x=0.01, xanchor="left"),
        xaxis=dict(
            gridcolor=THEME.grid, zerolinecolor=THEME.border,
            linecolor=THEME.border, tickfont=dict(color=THEME.muted, size=10),
            title=dict(font=dict(color=THEME.muted, size=10)),
            showspikes=True, spikecolor=THEME.amber, spikethickness=1,
            spikedash="dot", spikemode="across",
        ),
        yaxis=dict(
            gridcolor=THEME.grid, zerolinecolor=THEME.border,
            linecolor=THEME.border, tickfont=dict(color=THEME.muted, size=10),
            title=dict(font=dict(color=THEME.muted, size=10)),
            showspikes=True, spikecolor=THEME.amber, spikethickness=1,
            spikedash="dot",
        ),
        legend=dict(
            bgcolor="rgba(17,18,20,0.85)", bordercolor=THEME.border,
            borderwidth=1, font=dict(color=THEME.cyan, size=10),
        ),
        hoverlabel=dict(
            bgcolor=THEME.bg_raised, bordercolor=THEME.amber,
            font=dict(family=THEME.font_mono, color=THEME.cyan, size=11),
        ),
        colorway=[THEME.amber, THEME.cyan, THEME.green, THEME.magenta,
                  "#FF8C00", "#8A2BE2", "#00CED1", "#ADFF2F"],
        margin=dict(l=54, r=22, t=38, b=38),
        hovermode="x unified",
        dragmode="pan",
    )

    pio.templates["openterm"] = template
    pio.templates.default = "openterm"


# Diverging scale for heatmaps: red -> black -> green.
HEATMAP_SCALE = [
    [0.00, "#8B0000"], [0.25, THEME.red], [0.48, "#1A1A1A"],
    [0.52, "#1A1A1A"], [0.75, "#00A845"], [1.00, THEME.green],
]

# Sequential amber scale for density/intensity layers.
AMBER_SCALE = [
    [0.0, "#1A1A1A"], [0.3, "#4A3000"], [0.6, "#B37A00"], [1.0, THEME.amber],
]


def style_figure(fig: go.Figure, height: int = 420,
                 title: Optional[str] = None,
                 showlegend: bool = True) -> go.Figure:
    """Apply the standard terminal chart layout to any figure."""
    fig.update_layout(
        template="openterm",
        height=height,
        showlegend=showlegend,
        title=title,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=THEME.muted,
                     activecolor=THEME.amber),
    )
    return fig


def color_for_change(value: Optional[float]) -> str:
    """Green up, red down, muted flat/unknown."""
    if value is None:
        return THEME.muted
    if value > 0:
        return THEME.green
    if value < 0:
        return THEME.red
    return THEME.muted


def css_class_for_change(value: Optional[float]) -> str:
    if value is None:
        return "ot-flat"
    return "ot-up" if value > 0 else "ot-down" if value < 0 else "ot-flat"


PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2,
                             "filename": "open-terminal"},
}


__all__ = [
    "apply_theme", "style_figure", "color_for_change", "css_class_for_change",
    "HEATMAP_SCALE", "AMBER_SCALE", "PLOTLY_CONFIG", "TERMINAL_CSS",
]

"""
config.py :: Open-Terminal central configuration.

Every credential here is optional and free. The application is designed to
degrade gracefully: if a key is missing, the relevant fetcher falls back to a
keyless public endpoint or a scraper rather than raising.

Credential sources (all free, no card required):
    FRED_API_KEY        https://fredaccount.stlouisfed.org/apikeys
    OPENSKY_CLIENT_ID   https://opensky-network.org/  -> Account -> API Client
    OPENSKY_CLIENT_SECRET
    AISSTREAM_API_KEY   https://aisstream.io/  (free live AIS websocket)

Put them in a `.env` file next to this module:

    FRED_API_KEY=abcdef0123456789abcdef0123456789
    OPENSKY_CLIENT_ID=myuser-api-client
    OPENSKY_CLIENT_SECRET=...
    AISSTREAM_API_KEY=...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# .env loading (optional dependency - never fatal)
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except Exception:  # pragma: no cover - dotenv is a convenience, not a need
    pass


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".openterm"
DATA_DIR.mkdir(exist_ok=True)

CACHE_DB = DATA_DIR / "cache.sqlite"
HTTP_CACHE_DB = DATA_DIR / "http_cache"  # requests-cache appends .sqlite

# User-authored state. Deliberately outside the cache: PURGE empties the
# cache, and a button that also deleted your positions would be a trap.
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
BRIEF_DIR = DATA_DIR / "briefs"


# ==========================================================================
# CREDENTIALS
# ==========================================================================
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "").strip()
OPENSKY_CLIENT_ID: str = os.getenv("OPENSKY_CLIENT_ID", "").strip()
OPENSKY_CLIENT_SECRET: str = os.getenv("OPENSKY_CLIENT_SECRET", "").strip()
AISSTREAM_API_KEY: str = os.getenv("AISSTREAM_API_KEY", "").strip()

# SEC EDGAR *requires* a descriptive User-Agent with a contact address or it
# will hard-block your IP. This is their published fair-access policy.
SEC_USER_AGENT: str = os.getenv(
    "SEC_USER_AGENT", "Open-Terminal/1.0 (opensource-research; contact@example.com)"
)

# Generic desktop UA for public pages that gate on obvious bot signatures.
BROWSER_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Some endpoints behave WORSE with a browser UA. FRED's fredgraph.csv is the
# notable one: sent a Chrome User-Agent it accepts the connection and then
# never responds (read timeout), but answers in ~0.3s with a plain client UA.
# Verified empirically - do not "fix" this by making it match the browser UA.
PLAIN_USER_AGENT: str = "python-requests/2.32 (Open-Terminal)"


# ==========================================================================
# CACHE TTLs (seconds)
# ==========================================================================
@dataclass(frozen=True)
class CacheTTL:
    """Per-domain cache lifetimes. Tuned to stay well under free rate limits."""

    quote: int = 60           # Live-ish quotes / ticker tape
    intraday: int = 300       # 1m-1h bars
    daily_bars: int = 3600    # Daily OHLCV  (spec: 1 hour)
    fundamentals: int = 86400 # Balance sheet / income / cash flow
    sec_filings: int = 21600  # EDGAR submissions index (6h)
    scrape: int = 900         # Any HTML scrape        (spec: 15 min)
    ais: int = 300            # Vessel positions
    aviation: int = 30        # OpenSky state vectors (they update ~5-10s)
    macro: int = 43200        # FRED series (most are monthly/daily releases)
    news: int = 600           # RSS / GDELT
    fx: int = 900             # FX crosses used to convert the book
    listing: int = 604800     # A listing's quote currency (7d - it never moves)


TTL = CacheTTL()


# ==========================================================================
# MORNING BRIEF
# ==========================================================================
# The edition cutover. Singapore is UTC+8 with no daylight saving, so this is
# a wall-clock hour that never shifts.
BRIEF_HOUR: int = 8
BRIEF_MAX_SYMBOLS: int = 25       # positions to research per edition
BRIEF_STORIES_PER_POSITION: int = 4


# ==========================================================================
# PORTFOLIO CURRENCY
# ==========================================================================
# Yahoo quotes a listing in its exchange's currency - D05.SI in SGD, VOO in
# USD - and cost basis is entered in that same listing currency. A book that
# spans exchanges has no currency of its own until one is chosen, and adding
# SGD figures straight onto USD ones produces a total in no currency at all.
# Every market value, cost and P&L is converted into this before it is
# summed. Override with OPENTERM_BASE_CURRENCY=SGD in .env.
BASE_CURRENCY: str = (os.getenv("OPENTERM_BASE_CURRENCY", "USD").strip().upper()
                      or "USD")


# ==========================================================================
# NETWORK BEHAVIOUR
# ==========================================================================
@dataclass(frozen=True)
class NetConfig:
    request_timeout: int = 20
    max_retries: int = 4
    backoff_base: float = 1.5      # seconds; delay = base * (2 ** attempt)
    backoff_max: float = 30.0
    jitter: float = 0.35           # +/- fraction randomised into each sleep
    playwright_timeout_ms: int = 35_000
    playwright_headless: bool = True


NET = NetConfig()


# ==========================================================================
# UI THEME - classic Bloomberg palette
# ==========================================================================
@dataclass(frozen=True)
class Theme:
    bg: str = "#0A0A0A"
    bg_panel: str = "#111214"
    bg_raised: str = "#17191C"
    grid: str = "#242629"
    border: str = "#2E3135"

    amber: str = "#FFB000"      # Primary accent / headers
    green: str = "#00FF66"      # Up / bullish
    red: str = "#FF3B3B"        # Down / bearish
    cyan: str = "#00FFFF"       # Body text / data
    white: str = "#E8E8E8"
    muted: str = "#7A7F86"
    magenta: str = "#FF00AA"    # Alerts

    font_mono: str = (
        "'JetBrains Mono','IBM Plex Mono','Consolas','SF Mono',"
        "'Courier New',monospace"
    )


THEME = Theme()


# ==========================================================================
# MODULE A :: EQUITIES
# ==========================================================================
DEFAULT_TICKER = "AAPL"

# Ticker-tape instruments for the scrolling banner.
TAPE_SYMBOLS: List[Tuple[str, str]] = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ"),
    ("^DJI", "DOW"),
    ("^VIX", "VIX"),
    ("CL=F", "WTI CRUDE"),
    ("BZ=F", "BRENT"),
    ("GC=F", "GOLD"),
    ("SI=F", "SILVER"),
    ("^TNX", "US10Y"),
    ("DX-Y.NYB", "DXY"),
    ("BTC-USD", "BITCOIN"),
    ("ETH-USD", "ETHER"),
]


# ==========================================================================
# MODULE B :: MARITIME
# ==========================================================================
# Bounding boxes: (min_lat, min_lon, max_lat, max_lon)
# Drawn generously so vessels queuing at anchorages are counted, not just
# ships physically inside the canal.
#
# There is deliberately no "normal vessel count" field here. An earlier
# version carried a hand-written one per corridor and divided the live count
# by it to label traffic SEVERE or LIGHT - which meant the headline verdict
# on the maritime page came from a number somebody made up, scaled by
# whatever AIS coverage happened to be that day. The baseline is now measured
# from this installation's own observation history (`utils/observations.py`),
# and until enough history exists the terminal says so instead of judging.
@dataclass(frozen=True)
class Chokepoint:
    name: str
    code: str
    bbox: Tuple[float, float, float, float]
    description: str


CHOKEPOINTS: Dict[str, Chokepoint] = {
    "SUEZ": Chokepoint(
        name="Suez Canal",
        code="SUEZ",
        bbox=(29.20, 32.20, 31.60, 33.20),
        description="Egypt. ~12% of global trade; Red Sea <-> Mediterranean.",
    ),
    "PANAMA": Chokepoint(
        name="Panama Canal",
        code="PANAMA",
        bbox=(8.60, -80.20, 9.65, -79.30),
        description="Atlantic <-> Pacific. Draft-limited by Gatun Lake levels.",
    ),
    "MALACCA": Chokepoint(
        name="Strait of Malacca",
        code="MALACCA",
        bbox=(1.00, 98.50, 6.20, 104.60),
        description="Indian Ocean <-> South China Sea. ~25% of traded goods.",
    ),
    "BABELMANDEB": Chokepoint(
        name="Bab-el-Mandeb",
        code="BABELMANDEB",
        bbox=(11.80, 42.40, 13.80, 44.20),
        description="Red Sea southern gate. Houthi threat corridor.",
    ),
    "HORMUZ": Chokepoint(
        name="Strait of Hormuz",
        code="HORMUZ",
        bbox=(25.30, 55.20, 27.30, 57.60),
        description="~20% of global petroleum liquids consumption transits here.",
    ),
    "BOSPHORUS": Chokepoint(
        name="Bosphorus Strait",
        code="BOSPHORUS",
        bbox=(40.90, 28.80, 41.35, 29.30),
        description="Black Sea grain & Russian crude export route.",
    ),
    "GIBRALTAR": Chokepoint(
        name="Strait of Gibraltar",
        code="GIBRALTAR",
        bbox=(35.70, -6.10, 36.30, -5.20),
        description="Mediterranean <-> Atlantic.",
    ),
}

# AIS ship-type code -> human label (ITU-R M.1371 Table 53, collapsed).
AIS_SHIP_TYPES: Dict[int, str] = {
    0: "Not available",
    20: "WIG", 30: "Fishing", 31: "Towing", 32: "Towing (large)",
    33: "Dredging", 34: "Diving ops", 35: "Military", 36: "Sailing",
    37: "Pleasure craft", 40: "High-speed craft", 50: "Pilot vessel",
    51: "SAR", 52: "Tug", 53: "Port tender", 54: "Anti-pollution",
    55: "Law enforcement", 58: "Medical transport", 60: "Passenger",
    70: "Cargo", 71: "Cargo (haz A)", 72: "Cargo (haz B)",
    73: "Cargo (haz C)", 74: "Cargo (haz D)", 80: "Tanker",
    81: "Tanker (haz A)", 82: "Tanker (haz B)", 83: "Tanker (haz C)",
    84: "Tanker (haz D)", 90: "Other",
}

# Local AIS receiver (RTL-SDR + rtl-ais, OpenCPN, AISdispatcher, ...).
# If you run one, point Open-Terminal at its UDP output for true real-time data
# with zero rate limits.
AIS_UDP_HOST: str = os.getenv("AIS_UDP_HOST", "0.0.0.0")
AIS_UDP_PORT: int = int(os.getenv("AIS_UDP_PORT", "10110"))

# Fallback scrape targets, tried in order. See data_fetchers/maritime.py.
MARITIME_SOURCE_ORDER: List[str] = ["aisstream", "vesselfinder", "marinetraffic"]


# ==========================================================================
# MODULE C :: AVIATION
# ==========================================================================
OPENSKY_BASE = "https://opensky-network.org/api"
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)

# Fleet watchlists, keyed by ICAO three-letter operator designator.
#
# WHAT CHANGED AND WHY: this used to be a dict of specific ICAO24 hex
# transponder addresses mapped to descriptions like "FedEx B777F" or
# "Corporate G650 (energy sector, registry-listed)". Those labels were
# written by hand and never verified against a registry, so the terminal was
# asserting the identity and operator of specific airframes on no authority
# at all - and hex assignments do change hands. The list also went stale
# invisibly: a retired airframe simply never appears, which looks identical
# to one parked with its transponder off.
#
# What is here instead is reference data rather than assertion. ICAO Doc 8585
# three-letter designators are a published standard, and every aircraft
# broadcasts its callsign live. Selecting a fleet now filters live state
# vectors by callsign prefix, so the terminal reports only aircraft it can
# currently see, labelled with the callsign they are themselves transmitting.
#
# Deliberately institutional only - national carriers, cargo fleets and state
# transports. Open-Terminal ships no private-individual watchlist. Tracking a
# named private person's aircraft raises real safety and legal issues in
# several jurisdictions, and OpenSky's terms restrict it.
OPERATOR_FLEETS: Dict[str, Dict[str, str]] = {
    "CARGO / FREIGHT": {
        "FDX": "FedEx Express",
        "UPS": "UPS Airlines",
        "GTI": "Atlas Air",
        "CLX": "Cargolux",
        "CKS": "Kalitta Air",
        "ABX": "ABX Air",
        "GEC": "Lufthansa Cargo",
        "CAO": "Air China Cargo",
        "CKK": "China Cargo Airlines",
        "SQC": "Singapore Airlines Cargo",
        "MPH": "Martinair",
        "PAC": "Polar Air Cargo",
        "NCA": "Nippon Cargo Airlines",
        "GSS": "Atlas Air (Giant)",
        "BOX": "AeroLogic",
        "TAY": "ASL Airlines Belgium",
    },
    "FLAG / MAJOR PASSENGER": {
        "UAL": "United Airlines",
        "AAL": "American Airlines",
        "DAL": "Delta Air Lines",
        "SWA": "Southwest Airlines",
        "BAW": "British Airways",
        "DLH": "Lufthansa",
        "AFR": "Air France",
        "KLM": "KLM",
        "UAE": "Emirates",
        "QTR": "Qatar Airways",
        "ETD": "Etihad Airways",
        "SIA": "Singapore Airlines",
        "ANA": "All Nippon Airways",
        "JAL": "Japan Airlines",
        "CPA": "Cathay Pacific",
        "THY": "Turkish Airlines",
        "ETH": "Ethiopian Airlines",
        "RYR": "Ryanair",
        "EZY": "easyJet",
        "IBE": "Iberia",
        "AZA": "ITA Airways",
        "CCA": "Air China",
        "CES": "China Eastern",
        "CSN": "China Southern",
        "AIC": "Air India",
        "QFA": "Qantas",
    },
    "STATE / MILITARY": {
        "RCH": "USAF Air Mobility Command (Reach)",
        "SAM": "USAF Special Air Mission",
        "RRR": "RAF Ascot",
        "GAF": "German Air Force",
        "CFC": "Canadian Forces (Canforce)",
        "IAM": "Italian Air Force",
        "CTM": "French Air Force transport (COTAM)",
    },
}

# Every key above is a three-letter ICAO Doc 8585 operator designator, kept
# short on purpose: an entry that is merely plausible is the same failure
# this table was written to remove. Adding one means checking Doc 8585, not
# guessing from the airline's name.
OPERATOR_NAMES: Dict[str, str] = {
    designator: name
    for fleet in OPERATOR_FLEETS.values()
    for designator, name in fleet.items()
}

# Preset map regions: (min_lat, max_lat, min_lon, max_lon)
AVIATION_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "GLOBAL": (-85.0, 85.0, -180.0, 180.0),
    "NORTH AMERICA": (18.0, 72.0, -168.0, -52.0),
    "EUROPE": (34.0, 72.0, -25.0, 45.0),
    "MIDDLE EAST": (12.0, 40.0, 25.0, 65.0),
    "EAST ASIA": (18.0, 54.0, 100.0, 148.0),
    "SOUTH ASIA": (5.0, 38.0, 60.0, 98.0),
    "OCEANIA": (-48.0, -8.0, 110.0, 180.0),
    "SOUTH AMERICA": (-56.0, 14.0, -82.0, -34.0),
    "AFRICA": (-36.0, 38.0, -19.0, 52.0),
}


# ==========================================================================
# MODULE D :: MACRO
# ==========================================================================
# FRED series IDs. All are free and public.
YIELD_CURVE_SERIES: Dict[str, str] = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# Tenor -> years, for plotting the curve on a sane x-axis.
TENOR_YEARS: Dict[str, float] = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
    "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
}

MACRO_SERIES: Dict[str, Dict[str, str]] = {
    "INFLATION": {
        "CPIAUCSL": "CPI, All Urban Consumers (SA, Index)",
        "CPILFESL": "Core CPI (ex Food & Energy)",
        "PCEPILFE": "Core PCE Price Index (Fed's target)",
        "T10YIE": "10Y Breakeven Inflation Rate",
        "T5YIFR": "5Y-5Y Forward Inflation Expectation",
        "PPIACO": "Producer Price Index, All Commodities",
    },
    "RATES": {
        "FEDFUNDS": "Effective Federal Funds Rate",
        "DFEDTARU": "Fed Funds Target Range - Upper",
        "SOFR": "Secured Overnight Financing Rate",
        "MORTGAGE30US": "30Y Fixed Mortgage Average",
    },
    "GROWTH": {
        "GDPC1": "Real GDP (Chained 2017 $, SAAR)",
        "A191RL1Q225SBEA": "Real GDP % Change QoQ (SAAR)",
        "INDPRO": "Industrial Production Index",
        "RSAFS": "Retail Sales (Advance, SA)",
        "CFNAI": "Chicago Fed National Activity Index",
    },
    "LABOR": {
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls (Total, thousands)",
        "ICSA": "Initial Jobless Claims (weekly)",
        "CIVPART": "Labor Force Participation Rate",
        "CES0500000003": "Average Hourly Earnings, Private (SA)",
    },
    "LIQUIDITY": {
        "M2SL": "M2 Money Stock",
        "WALCL": "Fed Balance Sheet - Total Assets",
        "RRPONTSYD": "Overnight Reverse Repo Facility",
        "WTREGEN": "Treasury General Account",
    },
    "STRESS": {
        "STLFSI4": "St. Louis Fed Financial Stress Index",
        "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
        "NFCI": "Chicago Fed National Financial Conditions",
        "T10Y2Y": "10Y minus 2Y Treasury Spread",
        "T10Y3M": "10Y minus 3M Treasury Spread",
    },
}

# --------------------------------------------------------------------------
# REGIME MATRIX INPUTS
# --------------------------------------------------------------------------
# The composite that decides which macro quadrant we are in. Weights are
# exposed here for the same reason get_recession_indicators exposes its
# signal weights: the classification is a judgement, and a judgement whose
# inputs are buried in code cannot be argued with.
#
# `transform` is not cosmetic - see utils/macro_analytics. "pct" series are
# index levels or counts whose change is a percentage; "level" series are
# already percentages or standardised indices whose change is a difference in
# percentage points. CFNAI in particular oscillates around zero, so treating
# it as "pct" divides by ~0 and produces nonsense.
#
# `invert` marks series that RISE as their axis WEAKENS. Initial claims going
# up is a deteriorating labour market; high-yield spreads widening is
# tightening financial conditions. Without the flag both would be read as
# growth accelerating, which inverts the regime call in exactly the
# conditions where getting it right matters most.
@dataclass(frozen=True)
class RegimeInput:
    series_id: str
    label: str
    axis: str            # "growth" | "inflation"
    transform: str       # "pct" | "level"
    invert: bool
    weight: float


REGIME_INPUTS: List[RegimeInput] = [
    # --- Growth ----------------------------------------------------------
    RegimeInput("CFNAI", "Chicago Fed Activity Index", "growth",
                "level", False, 1.0),
    RegimeInput("PAYEMS", "Nonfarm Payrolls", "growth",
                "pct", False, 1.0),
    RegimeInput("ICSA", "Initial Jobless Claims", "growth",
                "pct", True, 0.7),
    # Financial conditions, NOT the high-yield OAS. FRED serves
    # BAMLH0A0HYM2 on a rolling ~3-year window only (ICE BofA licence the
    # index that way), which is too little history for the 36-month z-score
    # every other contributor uses - it would drop in and out of the vote
    # depending on the month. NFCI measures the same tightening impulse, is
    # standardised around zero by construction, and goes back to 1971. The
    # HY spread is still shown, as context, below.
    RegimeInput("NFCI", "Financial Conditions (NFCI)", "growth",
                "level", True, 0.7),
    # --- Inflation -------------------------------------------------------
    RegimeInput("PCEPILFE", "Core PCE", "inflation", "pct", False, 1.0),
    RegimeInput("CPILFESL", "Core CPI", "inflation", "pct", False, 1.0),
    RegimeInput("T10YIE", "10Y Breakeven", "inflation", "level", False, 0.7),
    RegimeInput("PPIACO", "PPI, All Commodities", "inflation",
                "pct", False, 0.7),
]

# Context series shown in the regime table but not voting on the quadrant -
# they describe the environment rather than the growth/inflation axes.
REGIME_CONTEXT: List[RegimeInput] = [
    RegimeInput("FEDFUNDS", "Fed Funds Rate", "context", "level", False, 0.0),
    RegimeInput("UNRATE", "Unemployment Rate", "context", "level", True, 0.0),
    RegimeInput("CES0500000003", "Avg Hourly Earnings", "context",
                "pct", False, 0.0),
    RegimeInput("T10Y2Y", "10Y-2Y Spread", "context", "level", False, 0.0),
    RegimeInput("BAMLH0A0HYM2", "High-Yield OAS", "context",
                "level", True, 0.0),
]

# An axis needs this many resolved contributors before it is scored at all.
# Below it, classify() returns no regime rather than calling the quadrant off
# a single series that happened to load.
REGIME_MIN_CONTRIBUTORS: int = 2

# Quadrant -> (label, posture, theme colour attribute).
REGIME_QUADRANTS: Dict[Tuple[bool, bool], Tuple[str, str, str]] = {
    # (growth accelerating, inflation accelerating)
    (True, False): ("GOLDILOCKS", "Risk-On / Equity Bullish", "green"),
    (True, True): ("OVERHEATING", "Hike Risk / Cash, Real Assets", "amber"),
    (False, True): ("STAGFLATION", "Defensive / Real Assets", "red"),
    (False, False): ("DEFLATIONARY BUST", "Bond Bullish / Risk-Off", "cyan"),
}

# Fed net liquidity = balance sheet - Treasury General Account - reverse repo.
#
# THE UNIT TRAP: these three are not all published on the same scale. FRED
# reports WALCL in millions and the other two in billions, so subtracting the
# raw series leaves net liquidity within a rounding error of the balance
# sheet itself - wrong by a factor of 1000 on two of three legs, and the
# chart still looks entirely reasonable. macro.get_net_liquidity normalises
# every leg to billions before subtracting and records how it worked the
# scale out.
NET_LIQUIDITY_SERIES: Dict[str, str] = {
    "walcl": "WALCL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
}

# Yahoo tickers used when FRED is unreachable entirely.
YAHOO_YIELD_PROXIES: Dict[str, str] = {
    "3M": "^IRX",   # 13-week T-bill
    "5Y": "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% labor force)",
    "NE.EXP.GNFS.ZS": "Exports of goods & services (% of GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt (% of GDP)",
}


# ==========================================================================
# MODULE H :: FUNDAMENTAL ANALYSIS
# ==========================================================================
# A rule-based screen over filed financials. Descriptive and mechanical, in
# the same register as get_recession_indicators - not advice, and the UI says
# so on every surface.
#
# Everything that decides a verdict lives in this section rather than in the
# scoring code, because a judgement whose inputs are buried is a judgement
# nobody can argue with. Anchors, weights and gates are all editable here.


# --------------------------------------------------------------------------
# DCF assumptions
# --------------------------------------------------------------------------
# Every field here is a FORECAST ASSUMPTION, never a fact, and is rendered as
# such wherever a fair value derived from it is shown.
@dataclass(frozen=True)
class DCFDefaults:
    # Long-run US equity risk premium. Damodaran's implied ERP has spent the
    # last two decades roughly in the 4-6% band; 5% is the midpoint, not a
    # forecast. Overridable in the UI.
    equity_risk_premium: float = 0.05

    horizon_years: int = 10

    # Growth fades linearly from the company's own trailing rate to this over
    # the horizon. No business compounds at its trailing rate forever, and a
    # flat-growth DCF quietly asserts that it does.
    fade_to: float = 0.03

    # Held below long-run nominal GDP on purpose: a terminal rate above it
    # implies the company eventually becomes the entire economy.
    terminal_growth: float = 0.025

    # Trailing growth above this is treated as unsustainable and capped. A
    # 60% FCF CAGR compounded for a decade produces a fair value that is
    # arithmetically correct and worthless.
    growth_cap: float = 0.25

    # Discount to base-case fair value required before BUY TRIGGER fires.
    required_margin_of_safety: float = 0.25


DCF = DCFDefaults()


# --------------------------------------------------------------------------
# Score anchors
# --------------------------------------------------------------------------
# (metric_value, points) in ascending metric order. Points may descend, which
# is how "lower is better" metrics are expressed without an inverted code
# path. See utils/fundamental_math.score_band.
SCORE_ANCHORS: Dict[str, List[Tuple[float, float]]] = {
    # --- Business quality (decimals, not percent) -------------------------
    "gross_margin": [(0.10, 10), (0.25, 35), (0.40, 60), (0.60, 85), (0.80, 100)],
    "operating_margin": [(0.0, 5), (0.08, 35), (0.15, 60), (0.25, 85), (0.40, 100)],
    "net_margin": [(0.0, 5), (0.05, 30), (0.12, 60), (0.20, 85), (0.30, 100)],
    "fcf_margin": [(0.0, 5), (0.05, 30), (0.12, 62), (0.20, 85), (0.30, 100)],
    "roe": [(0.0, 5), (0.08, 35), (0.15, 65), (0.25, 88), (0.40, 100)],
    "roic": [(0.0, 0), (0.08, 40), (0.15, 70), (0.25, 90), (0.40, 100)],

    # --- Growth -----------------------------------------------------------
    "revenue_cagr_3y": [(-0.10, 0), (0.0, 25), (0.05, 50), (0.12, 75), (0.25, 100)],
    "revenue_cagr_5y": [(-0.10, 0), (0.0, 25), (0.05, 50), (0.12, 75), (0.25, 100)],
    "eps_cagr_3y": [(-0.15, 0), (0.0, 25), (0.08, 55), (0.15, 80), (0.30, 100)],
    "fcf_cagr_3y": [(-0.15, 0), (0.0, 25), (0.08, 55), (0.15, 80), (0.30, 100)],

    # --- Financial health -------------------------------------------------
    # Net debt / EBITDA: negative means net cash, which is the best case.
    "net_debt_to_ebitda": [(-2.0, 100), (0.0, 92), (1.5, 75), (3.0, 50),
                           (4.5, 25), (6.0, 5)],
    "interest_coverage": [(1.0, 5), (2.5, 30), (5.0, 60), (10.0, 85), (20.0, 100)],
    "current_ratio": [(0.6, 10), (1.0, 40), (1.5, 75), (2.5, 95), (4.0, 85)],
    "cash_to_debt": [(0.0, 15), (0.25, 45), (0.75, 75), (1.5, 92), (3.0, 100)],
    # Diluted share count CAGR. Negative is buybacks; positive is dilution.
    "share_count_cagr": [(-0.05, 100), (-0.01, 82), (0.0, 65), (0.02, 40),
                         (0.05, 15), (0.10, 0)],
    "sbc_to_revenue": [(0.0, 100), (0.02, 82), (0.05, 60), (0.10, 30), (0.20, 0)],
    # Normalised slope of total debt across the filed years. Falling is good.
    "debt_trend": [(-0.25, 95), (-0.05, 80), (0.0, 65), (0.10, 40), (0.30, 10)],

    # --- Earnings quality -------------------------------------------------
    "fcf_to_net_income": [(0.0, 0), (0.5, 25), (0.8, 55), (1.0, 80), (1.4, 100)],
    # (Net income - operating cash flow) / total assets. High = income the
    # cash flow statement does not corroborate.
    "accruals_ratio": [(-0.10, 95), (0.0, 80), (0.05, 50), (0.10, 25), (0.20, 0)],

    # --- Valuation (higher points = cheaper) ------------------------------
    "fcf_yield": [(0.0, 0), (0.02, 25), (0.04, 50), (0.07, 78), (0.12, 100)],
    "upside_to_base": [(-0.50, 0), (-0.20, 20), (0.0, 45), (0.25, 72),
                       (0.60, 92), (1.00, 100)],
    # Current multiple divided by the company's own 5-year median. Below 1.0
    # means cheaper than its own history.
    "pe_vs_own_history": [(0.5, 100), (0.8, 82), (1.0, 60), (1.3, 35), (1.8, 10)],
    "ev_ebitda_vs_own_history": [(0.5, 100), (0.8, 82), (1.0, 60), (1.3, 35),
                                 (1.8, 10)],
    "pe_vs_peers": [(0.5, 95), (0.8, 78), (1.0, 58), (1.3, 32), (1.8, 8)],
    "price_to_book": [(0.5, 95), (1.0, 78), (1.8, 55), (3.0, 32), (6.0, 8)],

    # --- Risk (higher points = MORE risk) ---------------------------------
    "risk_leverage": [(-1.0, 5), (0.0, 12), (2.0, 30), (3.5, 55), (5.0, 80),
                      (7.0, 100)],
    "risk_earnings_quality": [(-0.05, 10), (0.0, 20), (0.05, 50), (0.10, 75),
                              (0.20, 100)],
    "risk_valuation": [(-0.30, 5), (0.0, 20), (0.30, 45), (0.75, 75), (1.50, 100)],
    "risk_dilution": [(-0.03, 8), (0.0, 25), (0.03, 55), (0.07, 80), (0.12, 100)],
    "risk_beta": [(0.4, 10), (0.8, 25), (1.2, 45), (1.8, 72), (2.5, 95)],
    "risk_margin_trend": [(-0.25, 95), (-0.08, 70), (0.0, 40), (0.08, 20),
                          (0.20, 8)],
}


# --------------------------------------------------------------------------
# Industry profiles
# --------------------------------------------------------------------------
# The fix for "do not use identical metrics for every company". A bank has no
# meaningful Debt/EBITDA, a REIT's net income is buried under depreciation,
# and a loss-making software company has no usable P/E. Each profile picks
# which metrics apply, how they are weighted and what the fair value is
# anchored on.
#
# `unavailable` is the honest half: the metrics that profile genuinely wants
# but free data does not carry. CET1 and insurance solvency live in
# regulatory filings, not SEC company facts; occupancy and per-unit
# production cost are narrative disclosures with no tag at all. They are
# named in the report and they cost confidence, rather than being quietly
# dropped or approximated into something that looks sourced.
@dataclass(frozen=True)
class FundamentalProfile:
    key: str
    label: str
    valuation_anchor: str            # dcf | book | affo | midcycle | growth
    axis_weights: Dict[str, float]   # business_quality/growth/financial_health
    skip_metrics: Tuple[str, ...]
    unavailable: Tuple[str, ...]
    note: str


_GENERIC_AXIS = {"business_quality": 1.0, "growth": 0.8, "financial_health": 0.8}

FUNDAMENTAL_PROFILES: Dict[str, FundamentalProfile] = {
    "GENERIC": FundamentalProfile(
        key="GENERIC", label="General corporate", valuation_anchor="dcf",
        axis_weights=_GENERIC_AXIS, skip_metrics=(), unavailable=(),
        note="Standard corporate metric set: DCF anchored, cross-checked "
             "against the company's own and its peers' multiples.",
    ),
    "BANK": FundamentalProfile(
        key="BANK", label="Bank / lender", valuation_anchor="book",
        axis_weights={"business_quality": 1.2, "growth": 0.5,
                      "financial_health": 1.0},
        # A bank funds itself with deposits and debt by design. Net
        # debt/EBITDA, interest coverage and the current ratio are not weak
        # readings for a bank, they are category errors, and free cash flow
        # is not a meaningful concept on a balance sheet that IS the product.
        skip_metrics=("net_debt_to_ebitda", "interest_coverage",
                      "current_ratio", "cash_to_debt", "fcf_margin",
                      "fcf_cagr_3y", "fcf_to_net_income", "fcf_yield",
                      "risk_leverage"),
        unavailable=("CET1 capital ratio", "Net interest margin (true, on "
                     "earning assets)", "Non-performing loan ratio",
                     "Loan loss reserve coverage"),
        note="Anchored on price/book against ROE, the standard lens for a "
             "lender. Regulatory capital is not in SEC company facts.",
    ),
    "INSURANCE": FundamentalProfile(
        key="INSURANCE", label="Insurance", valuation_anchor="book",
        axis_weights={"business_quality": 1.2, "growth": 0.5,
                      "financial_health": 1.0},
        skip_metrics=("net_debt_to_ebitda", "interest_coverage",
                      "current_ratio", "fcf_margin", "fcf_cagr_3y",
                      "fcf_to_net_income", "risk_leverage"),
        unavailable=("Solvency / RBC ratio", "Reserve development",
                     "Catastrophe exposure"),
        note="Anchored on price/book against ROE. A combined ratio is "
             "computed where the underwriting tags are filed, and reported "
             "as unavailable where they are not.",
    ),
    "REIT": FundamentalProfile(
        key="REIT", label="Real estate / REIT", valuation_anchor="affo",
        axis_weights={"business_quality": 0.9, "growth": 0.7,
                      "financial_health": 1.2},
        # Depreciation dominates a REIT's income statement, so net-income
        # derived margins and EPS growth describe accounting rather than the
        # business. FFO replaces them.
        skip_metrics=("net_margin", "eps_cagr_3y", "fcf_to_net_income",
                      "gross_margin"),
        unavailable=("Net asset value (NAV)", "Occupancy rate",
                     "Same-store NOI growth", "Lease expiry schedule"),
        note="Anchored on price/AFFO, and scored on FFO margin rather than "
             "net margin. Return on equity is reported but reads low for "
             "every REIT by construction - depreciation on property held at "
             "cost. NAV and occupancy are not tagged anywhere.",
    ),
    "COMMODITY": FundamentalProfile(
        key="COMMODITY", label="Energy / materials", valuation_anchor="midcycle",
        axis_weights={"business_quality": 0.9, "growth": 0.5,
                      "financial_health": 1.2},
        # Trailing growth on a cyclical is a statement about where in the
        # cycle the window happened to fall, not about the business.
        skip_metrics=("revenue_cagr_3y", "eps_cagr_3y"),
        unavailable=("Per-unit production cost", "Reserve life / replacement",
                     "Realised price vs benchmark", "Hedge book"),
        note="Anchored on mid-cycle margins rather than trailing earnings: "
             "a cyclical priced off peak or trough profits is mispriced by "
             "construction.",
    ),
    "HIGH_GROWTH_TECH": FundamentalProfile(
        key="HIGH_GROWTH_TECH", label="High-growth technology",
        valuation_anchor="growth",
        axis_weights={"business_quality": 1.0, "growth": 1.2,
                      "financial_health": 0.7},
        # Trailing P/E on a company reinvesting everything into growth is
        # either negative or meaninglessly large; neither is informative.
        skip_metrics=("pe_vs_own_history", "pe_vs_peers"),
        unavailable=("Net revenue retention", "Customer acquisition cost",
                     "Backlog / RPO"),
        note="Anchored on EV/sales and price/FCF with dilution weighted "
             "heavily: stock-based compensation is a real cost to holders "
             "even when it never touches the cash flow statement.",
    ),
}

# Yahoo sectorKey/industryKey fragments -> profile. Matched against the
# issuer's own classification, never against a hand-written ticker list.
PROFILE_SECTOR_MAP: Dict[str, str] = {
    "banks": "BANK",
    "banks-diversified": "BANK",
    "banks-regional": "BANK",
    "capital-markets": "BANK",
    "financial-data-stock-exchanges": "GENERIC",
    "insurance": "INSURANCE",
    "insurance-life": "INSURANCE",
    "insurance-property-casualty": "INSURANCE",
    "insurance-brokers": "GENERIC",
    "insurance-reinsurance": "INSURANCE",
    "insurance-specialty": "INSURANCE",
    "real-estate": "REIT",
    "energy": "COMMODITY",
    "basic-materials": "COMMODITY",
    "oil-gas-integrated": "COMMODITY",
    "oil-gas-e-p": "COMMODITY",
}

# A technology or communication-services name growing revenue faster than
# this is scored on the high-growth profile instead of the generic one.
HIGH_GROWTH_REVENUE_CAGR = 0.20


# --------------------------------------------------------------------------
# Verdict gates
# --------------------------------------------------------------------------
# Deliberately NOT a single summed score. The brief these were written for is
# explicit that a low P/E must not by itself produce BUY and a high P/E must
# not by itself produce SELL - and one number cannot express "cheap but
# deteriorating", which is the case that matters most. So the verdict is a
# gate on three axes evaluated in order, first match wins.
#
# quality = weighted(business quality, growth, financial health)
# value   = valuation score, multi-method, vs the company's own history
# risk    = risk score, where HIGHER IS WORSE
VERDICT_RULES: List[Tuple[str, Dict[str, float]]] = [
    ("STRONG SELL", {"max_quality": 35.0, "max_value": 15.0}),
    ("SELL", {"max_quality": 45.0}),
    ("SELL", {"max_value": 20.0}),
    ("STRONG BUY", {"min_quality": 75.0, "min_value": 70.0, "max_risk": 40.0}),
    ("BUY", {"min_quality": 60.0, "min_value": 55.0, "max_risk": 55.0}),
]
VERDICT_DEFAULT = "HOLD"

# Two red flags on a business already scoring under this force STRONG SELL
# regardless of how cheap it looks. Cheapness caused by deteriorating
# accounting is not an opportunity.
RED_FLAG_STRONG_SELL_QUALITY = 50.0
RED_FLAG_STRONG_SELL_COUNT = 2

VERDICT_COLOURS: Dict[str, str] = {
    "STRONG BUY": "green", "BUY": "green", "HOLD": "amber",
    "SELL": "red", "STRONG SELL": "red", "INSUFFICIENT DATA": "muted",
}

# Below this confidence the report shows INSUFFICIENT DATA instead of a
# verdict. A recommendation off half-missing financials is worse than none.
CONFIDENCE_FLOOR = 40.0

# ==========================================================================
# MODULE E :: NEWS / OSINT
# ==========================================================================
RSS_FEEDS: Dict[str, List[Tuple[str, str]]] = {
    "MARKETS": [
        ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch Top", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("MarketWatch RT Headlines", "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ],
    "MACRO / POLICY": [
        ("Federal Reserve Press", "https://www.federalreserve.gov/feeds/press_all.xml"),
        # Treasury retired its native RSS endpoints (both /rss/press.xml and
        # /news/press-releases/feed now 404). Proxied via Google News instead.
        ("US Treasury Press",
         "https://news.google.com/rss/search?q=site:home.treasury.gov+when:7d&hl=en-US&gl=US&ceid=US:en"),
        ("ECB Press", "https://www.ecb.europa.eu/rss/press.html"),
        ("BLS Releases", "https://www.bls.gov/feed/bls_latest.rss"),
    ],
    "REGULATORY": [
        ("SEC Press Releases", "https://www.sec.gov/news/pressreleases.rss"),
        # litreleases.xml is gone; admin.xml (administrative proceedings) is
        # the live enforcement feed.
        ("SEC Enforcement", "https://www.sec.gov/rss/litigation/admin.xml"),
        ("CFTC Press", "https://www.cftc.gov/RSS/RSSGP/rssgp.xml"),
    ],
    "ENERGY / COMMODITY": [
        ("OilPrice.com", "https://oilprice.com/rss/main"),
        ("EIA Today in Energy", "https://www.eia.gov/rss/todayinenergy.xml"),
        ("Reuters Commodities (Google)", "https://news.google.com/rss/search?q=commodities+when:1d&hl=en-US&gl=US&ceid=US:en"),
    ],
    "SHIPPING / TRADE": [
        ("gCaptain", "https://gcaptain.com/feed/"),
        ("Splash247", "https://splash247.com/feed/"),
        ("Maritime Executive", "https://maritime-executive.com/articles.rss"),
    ],
    "GEOPOLITICS": [
        ("Reuters World (Google)", "https://news.google.com/rss/search?q=site:reuters.com+world+when:1d&hl=en-US&gl=US&ceid=US:en"),
        ("AP Top (Google)", "https://news.google.com/rss/search?q=site:apnews.com+when:1d&hl=en-US&gl=US&ceid=US:en"),
    ],
}

# --------------------------------------------------------------------------
# SOCIAL SENTIMENT
# --------------------------------------------------------------------------
# Three streams, fused into one score. Institutional framing from Yahoo
# headlines, retail conviction from StockTwits (where users tag their own
# posts Bullish or Bearish), and speculative positioning from Reddit.
#
# ENDPOINT NOTES, both learned the hard way:
#
#   StockTwits sits behind Cloudflare and REJECTS PLAIN_USER_AGENT with a
#   "Just a moment..." challenge page. It needs a descriptive or browser UA.
#   That is the exact opposite of FRED's fredgraph.csv, which hangs forever
#   when sent a browser UA - see PLAIN_USER_AGENT above. Two endpoints in
#   this codebase with contradictory requirements; neither is negotiable.
#
#   Reddit's .json API returns 403 to everything without OAuth. The .rss
#   endpoints still answer, but rate-limit hard - three quick probes during
#   development were enough to earn a 429. Hence the slowest token bucket in
#   utils/rate_limiter and a long cache TTL.
STOCKTWITS_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

# Reddit search, per subreddit, as RSS. `restrict_sr` keeps results inside the
# subreddit; `t=week` bounds the window so a dead ticker does not return
# three-year-old posts as current sentiment.
REDDIT_SEARCH_URL = (
    "https://www.reddit.com/r/{subreddit}/search.rss"
    "?q={query}&restrict_sr=1&sort=new&t=week"
)

SOCIAL_SUBREDDITS: Tuple[str, ...] = ("wallstreetbets", "stocks", "investing")

# Platform weights for the fused score. Institutional and retail are equal by
# design: neither leads the other reliably, and the interesting signal is
# usually where they disagree.
SENTIMENT_WEIGHTS: Dict[str, float] = {
    "institutional": 0.35,   # Yahoo Finance headlines
    "retail": 0.35,          # StockTwits, including explicit Bull/Bear tags
    "reddit": 0.30,          # r/wallstreetbets, r/stocks, r/investing
}

# StockTwits users tag their own posts. A self-declared Bull/Bear tag is a
# stronger signal than a lexicon's reading of the same text, so tags carry
# most of the platform score when they exist - but not all of it, because
# only about half of posts carry one.
STOCKTWITS_TAG_WEIGHT = 0.70

# (lower_bound, label). Evaluated top down, first match wins.
SENTIMENT_BANDS: List[Tuple[float, str]] = [
    (0.60, "EXTREMELY BULLISH"),
    (0.20, "MODERATELY BULLISH"),
    (-0.19, "NEUTRAL / MIXED"),
    (-0.59, "MODERATELY BEARISH"),
    (-1.00, "EXTREMELY BEARISH"),
]

SENTIMENT_BAND_COLOURS: Dict[str, str] = {
    "EXTREMELY BULLISH": "green",
    "MODERATELY BULLISH": "green",
    "NEUTRAL / MIXED": "amber",
    "MODERATELY BEARISH": "red",
    "EXTREMELY BEARISH": "red",
}

# Sample count at which a platform's volume contribution to confidence is
# considered full. Below it, confidence scales down proportionally.
SENTIMENT_VOLUME_TARGET: int = 25

# Gap between two platform scores beyond which they are reported as
# diverging. 0.5 on a -1..+1 scale is roughly a full band apart.
SENTIMENT_DIVERGENCE_THRESHOLD: float = 0.50

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Extra finance-specific terms VADER doesn't know. Scores are on VADER's
# -4..+4 scale and get merged into the lexicon at load time.
FINANCE_LEXICON: Dict[str, float] = {
    "beat": 2.0, "beats": 2.0, "outperform": 2.4, "upgrade": 2.6,
    "upgraded": 2.6, "rally": 2.4, "surge": 2.8, "soar": 3.0, "soars": 3.0,
    "jumps": 2.2, "record high": 3.0, "buyback": 1.8, "dividend hike": 2.2,
    "profit": 1.8, "guidance raise": 2.8, "bullish": 2.6, "expansion": 1.4,
    "miss": -2.0, "misses": -2.0, "downgrade": -2.6, "downgraded": -2.6,
    "underperform": -2.4, "plunge": -3.0, "plunges": -3.0, "slump": -2.4,
    "tumble": -2.6, "selloff": -2.4, "sell-off": -2.4, "bearish": -2.6,
    "recession": -3.0, "default": -3.0, "bankruptcy": -3.4, "layoffs": -2.4,
    "probe": -1.8, "lawsuit": -1.8, "fraud": -3.2, "warning": -1.6,
    "profit warning": -3.0, "guidance cut": -2.8, "halted": -2.0,
    "delisting": -3.0, "writedown": -2.4, "impairment": -2.0,
    "hawkish": -1.2, "dovish": 1.2, "inflation": -0.8, "tariff": -1.4,
    "sanctions": -1.8, "shortage": -1.6, "glut": -1.4, "stagflation": -2.8,
}


# ==========================================================================
# COMMAND BAR
# ==========================================================================
# Bloomberg-style function suffixes -> internal module route.
# ==========================================================================
# MODULE J :: ALLOCATION & REBALANCING
# ==========================================================================
# The eleven GICS sectors, in the standard order.
GICS_SECTORS: Tuple[str, ...] = (
    "Information Technology",
    "Financials",
    "Health Care",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
)

# Yahoo does NOT publish GICS. It ships its own eleven-sector taxonomy that
# lines up one-for-one but names six of them differently, and it spells them
# two ways: title case in `info["sector"]` ("Consumer Cyclical") and
# snake_case in an ETF's `funds_data.sector_weightings` ("consumer_cyclical").
#
# Both forms are normalised through this table. Treating "Technology" and
# "Information Technology" as different sectors would split one exposure
# across two rows and make every benchmark comparison wrong - quietly, since
# each row on its own looks perfectly reasonable.
YAHOO_TO_GICS: Dict[str, str] = {
    "technology": "Information Technology",
    "informationtechnology": "Information Technology",
    "financialservices": "Financials",
    "financial": "Financials",
    "financials": "Financials",
    "healthcare": "Health Care",
    "consumercyclical": "Consumer Discretionary",
    "consumerdiscretionary": "Consumer Discretionary",
    "communicationservices": "Communication Services",
    "industrials": "Industrials",
    "consumerdefensive": "Consumer Staples",
    "consumerstaples": "Consumer Staples",
    "energy": "Energy",
    "utilities": "Utilities",
    "realestate": "Real Estate",
    "basicmaterials": "Materials",
    "materials": "Materials",
}


# ==========================================================================
# SECTOR DRILL-DOWN
# ==========================================================================
# The home heatmap's tiles: SPDR sector fund -> (tile label, Yahoo sector
# key). Both sides are fixed classifications rather than data that drifts -
# each fund tracks one sector by mandate, and Yahoo's eleven sectors line up
# one-for-one with GICS through YAHOO_TO_GICS. The keys are Yahoo's URL slugs
# ("real-estate"); the display names yfinance's own constants carry ("Real
# Estate") all 404. SPY is the whole market, not a sector, so it has no key.
SECTOR_ETFS: Dict[str, Tuple[str, Optional[str]]] = {
    "XLK": ("TECH", "technology"),
    "XLF": ("FINANCIALS", "financial-services"),
    "XLE": ("ENERGY", "energy"),
    "XLV": ("HEALTH", "healthcare"),
    "XLI": ("INDUSTRIAL", "industrials"),
    "XLY": ("CONS DISC", "consumer-cyclical"),
    "XLP": ("CONS STAPLE", "consumer-defensive"),
    "XLU": ("UTILITIES", "utilities"),
    "XLB": ("MATERIALS", "basic-materials"),
    "XLRE": ("REAL ESTATE", "real-estate"),
    "XLC": ("COMM SVCS", "communication-services"),
    "SPY": ("S&P 500", None),
}

# Quotes are fetched for at most this many of the companies a filter leaves,
# largest first. A batch of a hundred takes about two seconds; a whole
# industrials list runs to several hundred names, too slow to reprice every
# time a filter changes. Narrowing the filter prices the rest.
SECTOR_PRICE_LIMIT: int = 150


# Benchmark strategies. Each names a real, liquid fund whose CURRENT sector
# weights are fetched live rather than typed in here.
#
# This matters more than it looks. Index sector weights move constantly -
# technology has run from roughly a quarter to well over a third of the S&P
# 500 in a few years - so a hardcoded target table is wrong the day after it
# is written and gets more wrong every month, while continuing to render a
# confident-looking "drift vs benchmark" column. Reading the weights off the
# fund means the benchmark is whatever the benchmark actually is today.
@dataclass(frozen=True)
class AllocationBenchmark:
    key: str
    label: str
    proxy: str          # the fund whose published sector weights define it
    description: str


ALLOCATION_BENCHMARKS: Dict[str, AllocationBenchmark] = {
    "SP500": AllocationBenchmark(
        key="SP500", label="S&P 500", proxy="SPY",
        description="Broad US large-cap. The default reference for a "
                    "diversified equity book."),
    "TECH_GROWTH": AllocationBenchmark(
        key="TECH_GROWTH", label="Tech Growth", proxy="QQQ",
        description="Nasdaq-100. Concentrated in technology and "
                    "communication services by construction - drift against "
                    "it is not the same as drift against the market."),
    "INCOME_DEFENSIVE": AllocationBenchmark(
        key="INCOME_DEFENSIVE", label="Income / Defensive", proxy="SCHD",
        description="Dividend-quality tilt. Overweight staples, health care "
                    "and energy; structurally light on high-multiple tech."),
    "VALUE": AllocationBenchmark(
        key="VALUE", label="Large-Cap Value", proxy="VTV",
        description="Value factor. Financials and health care heavy."),
}

DEFAULT_BENCHMARK = "SP500"

# Drift below this many percentage points is treated as noise and produces no
# action. Rebalancing a 0.4pp gap costs spread and tax to fix a rounding
# difference, and a list of thirty trivial "actions" buries the two that
# matter.
REBALANCE_MIN_DRIFT_PCT: float = 2.0

# Absolute weight beyond which a single position is flagged regardless of
# sector. Concentration risk is a property of the position, not the sector.
POSITION_CONCENTRATION_PCT: float = 20.0

COMMAND_FUNCTIONS: Dict[str, str] = {
    # Equities
    "EQUITY": "equity", "EQ": "equity", "GP": "equity", "DES": "equity",
    "FA": "equity", "CN": "equity",
    # Supply chain & counterparty risk
    "SPLC": "supply_chain", "SUPPLY": "supply_chain", "SC": "supply_chain",
    "SPLY": "supply_chain",
    # Maritime
    "SHIP": "maritime", "AIS": "maritime", "PORT": "maritime",
    # Aviation
    "FLY": "aviation", "FLIGHT": "aviation", "AIR": "aviation",
    # Macro
    "MACRO": "macro", "ECO": "macro", "YCRV": "macro", "CURVE": "macro",
    # Fundamental analysis
    "FA": "fundamentals", "FUND": "fundamentals", "VAL": "fundamentals",
    "REGIME": "macro", "RGM": "macro",
    # News
    "NEWS": "news", "N": "news", "TOP": "news", "OSINT": "news",
    # Social sentiment fusion
    "SOCIAL": "news", "SENT": "news", "BUZZ": "news",
    # Portfolio. Bloomberg's mnemonic for this is PORT, but PORT is already
    # bound to the maritime module here and rebinding it would break muscle
    # memory that already exists.
    "PF": "portfolio", "PORTFOLIO": "portfolio", "HOLD": "portfolio",
    "HOLDINGS": "portfolio", "WATCH": "portfolio", "WL": "portfolio",
    "BRIEF": "portfolio", "AM": "portfolio",
    "ALLOC": "portfolio", "ALLOCATION": "portfolio", "REBAL": "portfolio",
    # Home
    "HOME": "home", "MENU": "home", "DASH": "home",
}

HELP_TEXT = """
COMMAND SYNTAX:  <SUBJECT> <FUNCTION>

  AAPL EQUITY      Equity analytics: chart, indicators, fundamentals, EDGAR
  NVDA GP          Same - GP is the Bloomberg price-graph mnemonic
  AAPL SPLC        Supply chain: counterparties, geography, commodity, credit
  SPLC             Supply chain for the ticker already loaded
  SUEZ SHIP        Maritime chokepoint monitor
  636019825 AIS    Track a specific vessel by MMSI (9 digits) or IMO
  FLY              Live global aircraft state vectors
  EUROPE FLY       Aviation map filtered to a region
  US10Y MACRO      Treasury yield curve + recession signals
  YCRV             Yield curve direct
  CPI ECO          Macro series browser, jumps to inflation
  NEWS             OSINT news terminal with sentiment scoring
  ENERGY TOP       News filtered to the energy/commodity desk
  PF               Portfolio: holdings, watchlist, 08:00 SGT holdings brief
  BRIEF            Same page, opens on the morning brief
  NVDA WATCH       Add a symbol to the watchlist without opening the editor
  HOME             Return to the overview dashboard
  HELP             This screen
""".strip()


# ==========================================================================
# RUNTIME FLAGS
# ==========================================================================
DEBUG: bool = os.getenv("OPENTERM_DEBUG", "0") == "1"
# Set to "1" to disable every network call and render from cache only.
OFFLINE: bool = os.getenv("OPENTERM_OFFLINE", "0") == "1"


def credential_status() -> Dict[str, bool]:
    """Used by the UI to render a green/amber credential health strip."""
    return {
        "FRED": bool(FRED_API_KEY),
        "OpenSky": bool(OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET),
        "AISStream": bool(AISSTREAM_API_KEY),
    }

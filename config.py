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
from typing import Dict, List, Tuple

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


TTL = CacheTTL()


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

# Sector peer sets used by the comparables table when the user doesn't supply
# their own list. Keep these short - each name costs one yfinance round trip.
PEER_GROUPS: Dict[str, List[str]] = {
    "megacap_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "semis": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "MU"],
    "banks": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
    "energy": ["XOM", "CVX", "COP", "SLB", "OXY", "PSX"],
    "shipping": ["ZIM", "MATX", "GOGL", "SBLK", "FRO", "DHT"],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "defense": ["LMT", "RTX", "NOC", "GD", "BA"],
}

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
@dataclass(frozen=True)
class Chokepoint:
    name: str
    code: str
    bbox: Tuple[float, float, float, float]
    description: str
    # Rough "normal" concurrent vessel count, used to colour the congestion
    # gauge. These are order-of-magnitude reference values, not gospel.
    baseline_vessels: int


CHOKEPOINTS: Dict[str, Chokepoint] = {
    "SUEZ": Chokepoint(
        name="Suez Canal",
        code="SUEZ",
        bbox=(29.20, 32.20, 31.60, 33.20),
        description="Egypt. ~12% of global trade; Red Sea <-> Mediterranean.",
        baseline_vessels=90,
    ),
    "PANAMA": Chokepoint(
        name="Panama Canal",
        code="PANAMA",
        bbox=(8.60, -80.20, 9.65, -79.30),
        description="Atlantic <-> Pacific. Draft-limited by Gatun Lake levels.",
        baseline_vessels=70,
    ),
    "MALACCA": Chokepoint(
        name="Strait of Malacca",
        code="MALACCA",
        bbox=(1.00, 98.50, 6.20, 104.60),
        description="Indian Ocean <-> South China Sea. ~25% of traded goods.",
        baseline_vessels=250,
    ),
    "BABELMANDEB": Chokepoint(
        name="Bab-el-Mandeb",
        code="BABELMANDEB",
        bbox=(11.80, 42.40, 13.80, 44.20),
        description="Red Sea southern gate. Houthi threat corridor.",
        baseline_vessels=45,
    ),
    "HORMUZ": Chokepoint(
        name="Strait of Hormuz",
        code="HORMUZ",
        bbox=(25.30, 55.20, 27.30, 57.60),
        description="~20% of global petroleum liquids consumption transits here.",
        baseline_vessels=110,
    ),
    "BOSPHORUS": Chokepoint(
        name="Bosphorus Strait",
        code="BOSPHORUS",
        bbox=(40.90, 28.80, 41.35, 29.30),
        description="Black Sea grain & Russian crude export route.",
        baseline_vessels=35,
    ),
    "GIBRALTAR": Chokepoint(
        name="Strait of Gibraltar",
        code="GIBRALTAR",
        bbox=(35.70, -6.10, 36.30, -5.20),
        description="Mediterranean <-> Atlantic.",
        baseline_vessels=60,
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

# Named watchlists of ICAO24 hex transponder addresses.
#
# NOTE ON SOURCING: these are illustrative examples of *publicly registered,
# institutionally-owned* airframes (national carriers, cargo fleets, government
# transports) taken from open registry data. Open-Terminal deliberately ships
# no private-individual watchlist. Tracking a named private person's aircraft
# raises real safety and legal issues in several jurisdictions - if you add
# your own entries, keep them to corporate and state-operated fleets.
AIRCRAFT_WATCHLISTS: Dict[str, Dict[str, str]] = {
    "GOVERNMENT / STATE": {
        "adfdf8": "USAF VC-25A  82-8000 (AF1 airframe)",
        "adfdf9": "USAF VC-25A  92-9000 (AF1 airframe)",
        "ae0439": "USAF C-32A    (SAM / VIP transport)",
        "43c6e1": "RAF Voyager  ZZ336 (UK VIP)",
        "3f8ff4": "Luftwaffe A350 (German govt)",
    },
    "CARGO / FREIGHT": {
        "a0f1bb": "FedEx B777F",
        "a1cd7a": "UPS B747-8F",
        "48c223": "Cargolux B747-8F",
        "4baa8f": "Turkish Cargo A330F",
        "76ce31": "Singapore Airlines Cargo B747F",
    },
    "ENERGY / COMMODITY CORPORATE": {
        "a3f9d2": "Corporate G650 (energy sector, registry-listed)",
        "a52d19": "Corporate GLEX (commodities trading house)",
    },
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
    },
    "LABOR": {
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls (Total, thousands)",
        "ICSA": "Initial Jobless Claims (weekly)",
        "CIVPART": "Labor Force Participation Rate",
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
COMMAND_FUNCTIONS: Dict[str, str] = {
    # Equities
    "EQUITY": "equity", "EQ": "equity", "GP": "equity", "DES": "equity",
    "FA": "equity", "CN": "equity",
    # Maritime
    "SHIP": "maritime", "AIS": "maritime", "PORT": "maritime",
    # Aviation
    "FLY": "aviation", "FLIGHT": "aviation", "AIR": "aviation",
    # Macro
    "MACRO": "macro", "ECO": "macro", "YCRV": "macro", "CURVE": "macro",
    # News
    "NEWS": "news", "N": "news", "TOP": "news", "OSINT": "news",
    # Home
    "HOME": "home", "MENU": "home", "DASH": "home",
}

HELP_TEXT = """
COMMAND SYNTAX:  <SUBJECT> <FUNCTION>

  AAPL EQUITY      Equity analytics: chart, indicators, fundamentals, EDGAR
  NVDA GP          Same - GP is the Bloomberg price-graph mnemonic
  SUEZ SHIP        Maritime chokepoint monitor
  636019825 AIS    Track a specific vessel by MMSI (9 digits) or IMO
  FLY              Live global aircraft state vectors
  EUROPE FLY       Aviation map filtered to a region
  US10Y MACRO      Treasury yield curve + recession signals
  YCRV             Yield curve direct
  CPI ECO          Macro series browser, jumps to inflation
  NEWS             OSINT news terminal with sentiment scoring
  ENERGY TOP       News filtered to the energy/commodity desk
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

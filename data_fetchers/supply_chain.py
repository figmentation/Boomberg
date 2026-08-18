"""
data_fetchers/supply_chain.py :: Module G - Supply Chain & Counterparty Risk.

Bloomberg equivalent: <EQUITY> SPLC (supply chain analysis).

WHAT THIS IS, AND HONESTLY WHAT IT IS NOT
-----------------------------------------
Bloomberg's SPLC is built on a proprietary relationship database: analyst-
curated supplier/customer links, plus Bloomberg's own revenue-exposure
estimates for pairs that nobody discloses. None of that is purchasable at
zero cost, and no free dataset reproduces it.

So this module does not pretend to. Every edge it draws is traceable to a
primary source, and the gaps are shown as gaps:

  * Named counterparties come from the issuer's own 10-K. US GAAP (ASC 280-10-50)
    forces disclosure of any single customer above 10% of revenue, but it does
    NOT force naming them - so Skyworks names Apple at 67% of revenue while
    Qualcomm writes "Customer/licensee (x), 21%". We surface both, and we never
    guess at an anonymised name.
  * Suppliers are disclosed far more rarely than customers - there is no
    reporting requirement at all. Expect thin upstream coverage. That is a
    property of US disclosure law, not a bug here.
  * Geographic exposure is real reported revenue by region, parsed out of the
    XBRL-derived tables SEC generates for each filing.
  * Commodity dependency is a *derived statistic* - the correlation of the
    equity's returns to commodity futures - not a disclosed input cost. It is
    labelled as such everywhere it appears.
  * Default risk is an Altman Z-score computed from the filed balance sheet.
    A published formula on public inputs, not a rating.
  * ESG metrics are absent. yfinance's sustainability feed now returns empty
    and there is no free replacement worth the name. An empty panel beats an
    invented score.

Data sources, all free:
  * SEC EDGAR  - 10-K documents for text mining; the R*.htm "Financial Report"
                 tables SEC renders from each filing's XBRL, indexed by
                 FilingSummary.xml. Those tables carry the dimensional
                 breakdowns (by region, by segment) that the companyconcept
                 JSON API strips out.
  * yfinance   - Peers, prices, balance sheet inputs for the Z-score.
"""

from __future__ import annotations

import io
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_fetchers import equities
from utils.cache import cached
from utils.rate_limiter import retry_with_backoff, throttled

log = logging.getLogger("openterm.supply_chain")

try:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    BS4_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    log.error("beautifulsoup4 import failed: %s", exc)
    BeautifulSoup = None  # type: ignore[assignment]
    BS4_AVAILABLE = False


# ==========================================================================
# TEXT MINING PATTERNS
# ==========================================================================
# Percentages appear both as "67%" and as "91 percent" - Cirrus Logic uses the
# spelled-out form throughout, and matching only the symbol silently loses the
# single most concentrated customer relationship on the US market.
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)

# A sentence needs BOTH a concentration verb and a counterparty noun to be a
# candidate. Either alone produces mostly noise from the MD&A margin tables.
_CUE = re.compile(
    r"\b(accounted for|represented|comprised|constituted|derived|generated|"
    r"contributed|made up|approximated)\b", re.I)

_DOWNSTREAM = re.compile(
    r"\b(customer|client|purchaser|end.customer|licensee|buyer|reseller|"
    r"distributor)s?\b", re.I)
_UPSTREAM = re.compile(
    r"\b(supplier|vendor|contract manufactur\w+|foundr(y|ies)|"
    r"subcontractor|sole source|single source)s?\b", re.I)

# Issuers that decline to name a partner use these placeholders.
_ANONYMOUS = re.compile(
    r"\b(one|two|three|four|five|a single|our largest|the largest|certain|"
    r"customer\s*[a-z(]|customer/licensee)\b", re.I)

# A disclosure about SEVERAL counterparties combined. Critical to separate:
# Cirrus Logic reports both "Apple ... 91%" (one customer) and "our ten largest
# end customers ... 96%" (an aggregate). Treating the second as a single
# counterparty would claim a 96% single-customer dependence that the filing
# never asserts.
_AGGREGATE = re.compile(
    r"\b((two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(of\s+our\s+)?(largest|significant|major|principal|top)?\s*"
    r"(end[\s-]?)?(customers|clients|suppliers|vendors)"
    r"|(customers|clients|suppliers|vendors)\s+(in\s+the\s+)?aggregate"
    r"|(largest|top|principal|significant|major)\s+"
    r"(end[\s-]?)?(customers|clients|suppliers|vendors))\b", re.I)

_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Legal-form noise to strip when reducing a registered name to a brand.
_LEGAL_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|limited|plc|llc|l\.l\.c|lp|"
    r"holdings?|group|the|sa|nv|ag|se|ab|oyj|as|spa|class\s+[abc])\b", re.I)

# Brand tokens too generic to match on - they appear in ordinary prose and
# would produce a flood of false counterparties.
_TOO_GENERIC = {
    "THE", "AND", "FOR", "NEW", "FIRST", "GENERAL", "NATIONAL", "UNITED",
    "GLOBAL", "AMERICA", "AMERICAN", "US", "USA", "ONE", "NEXT", "OPEN",
    "MAIN", "PRIME", "CORE", "UNION", "CENTRAL", "SUMMIT", "PIONEER",
    "LIBERTY", "FRANKLIN", "SELECT", "PREMIER", "ALLIANCE", "CAPITAL",
    "PARTNERS", "TRUST", "FUND", "INCOME", "GROWTH", "VALUE", "TARGET",
    "SERVICE", "SERVICES", "SYSTEMS", "SOLUTIONS", "TECHNOLOGY",
    "TECHNOLOGIES", "INTERNATIONAL", "INDUSTRIES", "ENTERPRISE", "MATERIALS",
    "RESOURCES", "PROPERTIES", "REALTY", "ENERGY", "POWER", "HEALTH", "CARE",
    "MEDIA", "BANK", "FINANCIAL", "INSURANCE", "MOTORS", "FOODS", "BRANDS",
    "SPORT", "SPORTS", "TRAVEL", "MARKET", "MARKETS", "EXCHANGE", "DIGITAL",
    "DATA", "CLOUD", "NETWORK", "NETWORKS", "MOBILE", "WIRELESS", "SEMI",
}

# Region label -> representative (lat, lon) for the exposure map. SEC filers
# use a small, stable vocabulary for reportable geographies.
REGION_COORDS: Dict[str, Tuple[float, float]] = {
    "UNITED STATES": (39.8, -98.6), "U.S.": (39.8, -98.6), "US": (39.8, -98.6),
    "AMERICAS": (12.0, -77.0), "NORTH AMERICA": (48.0, -100.0),
    "SOUTH AMERICA": (-15.0, -60.0), "LATIN AMERICA": (-10.0, -60.0),
    "CANADA": (56.0, -106.0), "MEXICO": (23.6, -102.5), "BRAZIL": (-14.2, -51.9),
    "EUROPE": (50.0, 10.0), "EMEA": (30.0, 20.0), "GERMANY": (51.2, 10.5),
    "UNITED KINGDOM": (55.4, -3.4), "FRANCE": (46.2, 2.2), "IRELAND": (53.4, -8.2),
    "NETHERLANDS": (52.1, 5.3), "SWITZERLAND": (46.8, 8.2), "ISRAEL": (31.0, 34.9),
    "GREATER CHINA": (35.9, 104.2), "CHINA": (35.9, 104.2),
    "CHINA (INCLUDING HONG KONG)": (35.9, 104.2), "HONG KONG": (22.3, 114.2),
    "TAIWAN": (23.7, 121.0), "JAPAN": (36.2, 138.3), "KOREA": (35.9, 127.8),
    "SOUTH KOREA": (35.9, 127.8), "SINGAPORE": (1.35, 103.8), "INDIA": (20.6, 79.0),
    "ASIA": (34.0, 100.0), "ASIA PACIFIC": (10.0, 115.0),
    "REST OF ASIA PACIFIC": (10.0, 115.0), "APAC": (10.0, 115.0),
    "AUSTRALIA": (-25.3, 133.8), "AFRICA": (0.0, 20.0), "MIDDLE EAST": (29.0, 45.0),
    "OTHER": (0.0, 0.0), "OTHER COUNTRIES": (0.0, 0.0), "REST OF WORLD": (0.0, 0.0),
    "INTERNATIONAL": (0.0, 0.0), "ALL OTHER": (0.0, 0.0),
}

# Futures we test equity returns against. These are the contracts the terminal
# already fetches for the tape, so no new upstream dependency.
COMMODITY_SYMBOLS: Dict[str, str] = {
    "CL=F": "WTI CRUDE", "BZ=F": "BRENT", "NG=F": "NAT GAS",
    "GC=F": "GOLD", "SI=F": "SILVER", "HG=F": "COPPER",
    "ZC=F": "CORN", "ZW=F": "WHEAT", "ZS=F": "SOYBEAN",
    "PL=F": "PLATINUM", "PA=F": "PALLADIUM",
}


# ==========================================================================
# COMPANY-NAME UNIVERSE (for resolving mined names to tickers)
# ==========================================================================
_UNIVERSE: Optional[Dict[str, str]] = None
_UNIVERSE_RE: Optional[re.Pattern] = None


def _brand(title: str) -> str:
    """'Apple Inc.' -> 'APPLE'. Strips legal form and punctuation."""
    cleaned = re.sub(r"[,.()]", " ", title or "")
    cleaned = _LEGAL_SUFFIX.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().upper()


def _company_universe() -> Tuple[Dict[str, str], Optional[re.Pattern]]:
    """
    {BRAND: TICKER} over every SEC registrant, plus one compiled alternation.

    Why a dictionary rather than a general-purpose NER model: the names we care
    about must resolve to a *ticker* to be worth drawing on a graph, and the SEC
    already publishes the authoritative name->ticker list. It also sidesteps the
    failure mode that killed the first attempt here - a proper-noun regex needs
    a legal suffix to fire, so it matched "Apple Inc." but missed the plain
    "Apple" that Skyworks actually writes.
    """
    global _UNIVERSE, _UNIVERSE_RE
    if _UNIVERSE is not None:
        return _UNIVERSE, _UNIVERSE_RE

    universe: Dict[str, str] = {}
    try:
        mapping = equities.get_sec_ticker_map()  # TICKER -> CIK, cached weekly
        resp = equities._sec_session().get(
            equities.SEC_TICKER_MAP_URL, timeout=config.NET.request_timeout)
        resp.raise_for_status()
        for row in resp.json().values():
            ticker = str(row.get("ticker", "")).upper()
            brand = _brand(str(row.get("title", "")))
            if not ticker or len(brand) < 4 or brand in _TOO_GENERIC:
                continue
            universe.setdefault(brand, ticker)
        _ = mapping  # touched only to warm the shared cache
    except Exception as exc:
        log.warning("Could not build company universe: %s", exc)
        _UNIVERSE, _UNIVERSE_RE = {}, None
        return _UNIVERSE, _UNIVERSE_RE

    # Longest-first so "ADVANCED MICRO DEVICES" wins over "ADVANCED".
    names = sorted((n for n in universe if 4 <= len(n) <= 40), key=len, reverse=True)
    try:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
    except Exception as exc:  # pathological size guard
        log.warning("Universe regex failed to compile: %s", exc)
        pattern = None

    _UNIVERSE, _UNIVERSE_RE = universe, pattern
    log.info("Company universe ready: %d names", len(universe))
    return _UNIVERSE, _UNIVERSE_RE


# ==========================================================================
# FILING ACCESS
# ==========================================================================
@cached(ttl=config.TTL.sec_filings, namespace="splc_filing_text")
@throttled("sec")
@retry_with_backoff(on_giveup=lambda exc: "")
def _filing_text(url: str) -> str:
    """Flatten a filing document to whitespace-normalised text."""
    if not BS4_AVAILABLE:
        return ""
    resp = equities._sec_session().get(url, timeout=config.NET.request_timeout * 2)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" "))


@cached(ttl=config.TTL.sec_filings, namespace="splc_reports")
@throttled("sec")
@retry_with_backoff(on_giveup=lambda exc: [])
def _filing_reports(base_url: str) -> List[Dict[str, str]]:
    """
    The R*.htm tables SEC renders from a filing's XBRL, from FilingSummary.xml.

    These carry the dimensional breakdowns (revenue by region, by segment) that
    data.sec.gov's companyconcept endpoint drops - it returns consolidated
    totals only, with no segment axis.
    """
    if not BS4_AVAILABLE:
        return []
    resp = equities._sec_session().get(
        f"{base_url}/FilingSummary.xml", timeout=config.NET.request_timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "xml")

    reports = []
    for report in soup.find_all("Report"):
        short = report.find("ShortName")
        html = report.find("HtmlFileName") or report.find("XmlFileName")
        if short and html and html.get_text().strip():
            reports.append({"name": short.get_text().strip(),
                            "file": html.get_text().strip()})
    return reports


def _latest_10k(ticker: str) -> Optional[Dict[str, Any]]:
    """Metadata for the most recent 10-K: url, base directory, filing date."""
    filings = equities.get_sec_filings(ticker, ("10-K",), 3)
    if filings.empty:
        return None
    row = filings.iloc[0]
    return {
        "url": row["url"],
        "base": str(row["url"]).rsplit("/", 1)[0],
        "filed": row["filing_date"],
        "company": row.get("company", ticker),
    }


# ==========================================================================
# 1. COUNTERPARTIES  (named + anonymised concentration disclosures)
# ==========================================================================
def _counterparty_count(match: Optional[re.Match]) -> Optional[int]:
    """How many counterparties an aggregate disclosure covers, if it says."""
    if not match:
        return None
    token = re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
                      match.group(0), re.I)
    if not token:
        return None
    value = token.group(1).lower()
    if value in _COUNT_WORDS:
        return _COUNT_WORDS[value]
    try:
        number = int(value)
    except ValueError:
        return None
    return number if 1 < number <= 50 else None


def _classify(sentence: str) -> Optional[str]:
    """Which side of the chain is this sentence about?"""
    down, up = _DOWNSTREAM.search(sentence), _UPSTREAM.search(sentence)
    if down and not up:
        return "customer"
    if up and not down:
        return "supplier"
    if up and down:
        # Both nouns present: the earlier one usually governs the sentence.
        return "customer" if down.start() < up.start() else "supplier"
    return None


@cached(ttl=config.TTL.fundamentals, namespace="splc_counterparties")
def get_counterparties(ticker: str, max_rows: int = 40) -> pd.DataFrame:
    """
    Concentration disclosures mined from the latest 10-K.

    Returns:
        DataFrame: counterparty, counterparty_ticker, relationship,
                   pct_of_revenue, named, quote, source_url, filed
        `named=False` rows are real disclosures where the issuer withheld the
        name ("one customer accounted for 18%") - kept because the exposure is
        the risk, whether or not the partner is identified.
    """
    filing = _latest_10k(ticker)
    if not filing:
        return pd.DataFrame()

    text = _filing_text(filing["url"])
    if not text:
        return pd.DataFrame()

    universe, pattern = _company_universe()
    self_brands = {_brand(filing.get("company", "")), ticker.upper()}

    rows: List[Dict[str, Any]] = []
    seen: set = set()

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) > 900 or not _CUE.search(sentence):
            continue
        relationship = _classify(sentence)
        if not relationship:
            continue

        percentages = [float(p) for p in _PCT.findall(sentence)]
        percentages = [p for p in percentages if 0 < p <= 100]
        if not percentages:
            continue

        # Take the FIRST figure, not the largest. These sentences list the
        # current year followed by the two comparatives ("67%, 69% and 66%"),
        # so max() silently reports a prior year - for Skyworks that meant
        # showing FY2024's 69% in place of FY2025's 67%.
        pct = percentages[0]

        names = set()
        if pattern:
            for match in pattern.finditer(sentence):
                brand = match.group(1).upper()
                mapped = universe.get(brand)
                if mapped and mapped != ticker.upper() and brand not in self_brands:
                    names.add((brand, mapped))

        quote = re.sub(r"\s+", " ", sentence).strip()
        if len(quote) > 400:
            quote = quote[:397] + "..."

        if names:
            for brand, mapped in names:
                key = (mapped, relationship)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "counterparty": brand.title(),
                    "counterparty_ticker": mapped,
                    "relationship": relationship,
                    "pct_of_revenue": pct,
                    "named": True,
                    "is_aggregate": bool(_AGGREGATE.search(sentence)),
                    "quote": quote,
                    "source_url": filing["url"],
                    "filed": filing["filed"],
                })
        elif _ANONYMOUS.search(sentence) or _AGGREGATE.search(sentence):
            aggregate_match = _AGGREGATE.search(sentence)
            count = _counterparty_count(aggregate_match)
            if aggregate_match:
                label = (f"Top {count} {relationship}s (combined)" if count
                         else f"Largest {relationship}s (combined)")
            else:
                label = f"Undisclosed {relationship}"

            key = (label, relationship, pct)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "counterparty": label,
                "counterparty_ticker": None,
                "relationship": relationship,
                "pct_of_revenue": pct,
                "named": False,
                "is_aggregate": bool(aggregate_match),
                "quote": quote,
                "source_url": filing["url"],
                "filed": filing["filed"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["named", "pct_of_revenue"], ascending=[False, False])
    return df.head(max_rows).reset_index(drop=True)


# ==========================================================================
# 2. GEOGRAPHIC EXPOSURE  (reported revenue by region)
# ==========================================================================
_GEO_REPORT = re.compile(
    r"(geograph|by region|revenue.*(region|countr)|(region|countr).*revenue)",
    re.I)
_GEO_DETAIL = re.compile(r"\(detail", re.I)

# Some filers break out only property/plant/equipment by geography, not
# revenue - Skyworks is one. Those tables have the same shape as a revenue
# table, so a report about assets alone is skipped rather than risk
# presenting a balance-sheet figure as revenue.
_ASSETS_ONLY = re.compile(
    r"\b(ppe|property|plant|equipment|long.lived|assets?)\b", re.I)
_MENTIONS_REVENUE = re.compile(r"\b(revenue|sales)\b", re.I)


def _numeric(value: Any) -> Optional[float]:
    """Parse SEC table cells: '$ 178,353', '(95,699)', '—'."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"—", "-", "$", ""}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^\d.]", "", text)
    if not text or text == ".":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


@cached(ttl=config.TTL.fundamentals, namespace="splc_geography")
def get_geographic_revenue(ticker: str) -> pd.DataFrame:
    """
    Revenue by reportable geography, as filed.

    Parsed from the SEC-rendered XBRL tables rather than the companyconcept
    API, which strips the geographic axis and returns consolidated revenue only.

    Returns:
        DataFrame: region, revenue, pct, lat, lon   (most recent period)
    """
    filing = _latest_10k(ticker)
    if not filing:
        return pd.DataFrame()

    reports = _filing_reports(filing["base"])
    if not reports:
        return pd.DataFrame()

    # Prefer the narrow "(Details)" tables - they are one clean matrix, whereas
    # the parent note bundles the whole disclosure into a single text blob.
    def usable(report: Dict[str, str]) -> bool:
        name = report["name"]
        if not _GEO_REPORT.search(name):
            return False
        # "PPE by Geographic Area" is about assets, not revenue.
        return not (_ASSETS_ONLY.search(name) and not _MENTIONS_REVENUE.search(name))

    candidates = [r for r in reports if usable(r) and _GEO_DETAIL.search(r["name"])]
    candidates += [r for r in reports if usable(r) and r not in candidates]
    if not candidates:
        return pd.DataFrame()

    # Score every candidate and keep the richest breakdown rather than the
    # first hit: Apple files both a 5-region segment table (Americas, Europe,
    # Greater China, Japan, Rest of Asia Pacific) and a thinner 3-row
    # "countries over 10%" table, and the segment one is the better map.
    best: Optional[pd.DataFrame] = None
    for report in candidates[:6]:
        url = f"{filing['base']}/{report['file']}"
        try:
            resp = equities._sec_session().get(
                url, timeout=config.NET.request_timeout)
            resp.raise_for_status()
            frame = _geo_from_report(resp.text)
        except Exception as exc:
            log.debug("Geo report %s failed: %s", report["file"], exc)
            continue

        if frame is None or frame.empty:
            continue
        frame["source"] = report["name"]
        frame["source_url"] = url
        if best is None or len(frame) > len(best):
            best = frame

    if best is None:
        return pd.DataFrame()

    # How much of consolidated revenue these regions actually account for.
    # Tesla, for instance, breaks out only its two largest markets, and
    # reporting "US 69%" off a partial base would overstate it. The caller
    # shows this so a partial disclosure is never read as a full one.
    best["coverage_pct"] = np.nan
    try:
        income = equities.get_financial_statements(ticker).get(
            "income_statement", pd.DataFrame())
        consolidated = _first_row(income, "Total Revenue", "Operating Revenue")
        if consolidated and consolidated > 0:
            # Filings render these tables in millions; statements are in units.
            scale = 1e6 if consolidated / max(best["revenue"].sum(), 1) > 1e4 else 1.0
            best["coverage_pct"] = (
                best["revenue"].sum() * scale / consolidated * 100)
    except Exception as exc:
        log.debug("Coverage check failed for %s: %s", ticker, exc)

    return best


def _cell_text(cell: Any) -> str:
    """Cell text with footnote superscripts removed."""
    clone = cell
    for sup in clone.find_all(["sup"]):
        sup.decompose()
    return re.sub(r"\s+", " ", clone.get_text(" ", strip=True))


def _region_key(text: str) -> str:
    """Collapse a region label to a comparable form: no punctuation, no case."""
    text = re.sub(r"[^\w\s]", " ", str(text or "").upper())
    return re.sub(r"\s+", " ", text).strip()


# REGION_COORDS keyed by the collapsed form, so "China (including Hong Kong)"
# in a filing matches the entry written with parentheses.
_REGION_INDEX: Dict[str, str] = {_region_key(k): k for k in REGION_COORDS}


def _normalise_region(label: str) -> Optional[str]:
    """'China (1)' / 'China (including Hong Kong)' -> a REGION_COORDS key."""
    text = re.sub(r"\s*\(\d+\)\s*", " ", str(label or ""))
    key = _region_key(text)
    if not key or key.startswith("TOTAL"):
        return None
    return _REGION_INDEX.get(key)


# The row label that actually carries a revenue figure. Long-lived assets and
# headcount rows use the same layout and must not be mistaken for revenue.
_REVENUE_ROW = re.compile(
    r"\b(net sales|net revenue|total revenue|revenues?|sales)\b", re.I)


def _row_values(row: Any) -> List[float]:
    """Numeric cells of a row, in column order. Footnote spans excluded."""
    values = []
    for cell in row.find_all("td"):
        if "num" not in " ".join(cell.get("class") or []):
            continue
        number = _numeric(_cell_text(cell))
        if number is not None:
            values.append(number)
    return values


def _geo_from_report(html: str) -> Optional[pd.DataFrame]:
    """
    Region -> revenue from one SEC-rendered XBRL report.

    Parses the markup directly rather than via pandas.read_html: the R-files
    tag numeric cells with class="num"/"nump" and put footnote markers in their
    own spans, which read_html flattens into the same row. That is how Apple's
    China revenue first came out as 1.0 (the footnote marker) instead of 64,377.

    Three layouts occur in practice and all are handled:
      A  region and figures on one row        "U.S.  151,790  142,196"
      B  regions across the column headers    (the segment matrix)
      C  region on a bare row, figures on the NEXT "Net sales" row - this is
         how SEC renders a stacked dimension, and it is what Apple files.
    """
    if not BS4_AVAILABLE:
        return None

    soup = BeautifulSoup(html, "lxml")

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue

        # ---- layouts A and C share one pass ------------------------------
        found: List[Dict[str, Any]] = []
        pending: Optional[str] = None

        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            label = _cell_text(cells[0])
            region = _normalise_region(label)
            values = _row_values(row)

            if region and values:
                # A: label and figures together.
                found.append({"region": region, "revenue": abs(values[0])})
                pending = None
            elif region and not values:
                # C: region marker; its figures are on a following row.
                pending = region
            elif pending and values and _REVENUE_ROW.search(label):
                found.append({"region": pending, "revenue": abs(values[0])})
                pending = None

        if len({r["region"] for r in found}) >= 2:
            frame = _finalise_geo(found)
            if frame is not None:
                return frame

        # ---- layout B: regions across the header -------------------------
        header_cells: List[Any] = []
        for row in rows[:2]:
            header_cells.extend(row.find_all(["th", "td"]))
        regions = [r for r in (_normalise_region(_cell_text(c))
                               for c in header_cells) if r]
        if len(regions) < 2:
            continue

        for row in rows:
            first = row.find("td")
            if not first or not _REVENUE_ROW.search(_cell_text(first)):
                continue
            values = _row_values(row)
            if len(values) < len(regions):
                continue
            found = [{"region": region, "revenue": abs(values[position])}
                     for position, region in enumerate(regions)]
            frame = _finalise_geo(found)
            if frame is not None:
                return frame

    return None


def _finalise_geo(rows: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Deduplicate, attach coordinates and compute shares."""
    if len(rows) < 2:
        return None
    frame = pd.DataFrame(rows).drop_duplicates(subset="region", keep="first")
    frame = frame[frame["revenue"] > 0]
    if len(frame) < 2:
        return None

    total = frame["revenue"].sum()
    if total <= 0:
        return None

    frame["pct"] = frame["revenue"] / total * 100
    frame["lat"] = frame["region"].map(lambda r: REGION_COORDS[r][0])
    frame["lon"] = frame["region"].map(lambda r: REGION_COORDS[r][1])
    frame["region"] = frame["region"].str.title()
    return frame.sort_values("revenue", ascending=False).reset_index(drop=True)


# ==========================================================================
# 3. COMMODITY DEPENDENCY  (derived, not disclosed)
# ==========================================================================
@cached(ttl=config.TTL.daily_bars, namespace="splc_commodity")
def get_commodity_exposure(ticker: str, period: str = "2y",
                           min_abs_corr: float = 0.15) -> pd.DataFrame:
    """
    Correlation and beta of the equity's daily returns against commodity futures.

    This is a STATISTICAL association, not a disclosed input-cost dependency.
    A high copper correlation may mean a miner sells copper, a manufacturer buys
    it, or simply that both track global industrial demand. The UI must say so.

    Returns:
        DataFrame: commodity, symbol, correlation, beta, r_squared, observations
    """
    history = equities.get_history(ticker, period, "1d")
    if history.empty or "Close" not in history.columns:
        return pd.DataFrame()

    equity_returns = history["Close"].pct_change().dropna()
    if len(equity_returns) < 60:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for symbol, name in COMMODITY_SYMBOLS.items():
        try:
            commodity = equities.get_history(symbol, period, "1d")
        except Exception as exc:
            log.debug("Commodity %s failed: %s", symbol, exc)
            continue
        if commodity.empty or "Close" not in commodity.columns:
            continue

        commodity_returns = commodity["Close"].pct_change().dropna()
        joined = pd.concat([equity_returns, commodity_returns], axis=1,
                           join="inner").dropna()
        joined.columns = ["equity", "commodity"]
        if len(joined) < 60 or joined["commodity"].std() == 0:
            continue

        correlation = float(joined["equity"].corr(joined["commodity"]))
        if not np.isfinite(correlation):
            continue

        variance = float(joined["commodity"].var())
        beta = (float(joined["equity"].cov(joined["commodity"])) / variance
                if variance else None)

        rows.append({
            "commodity": name, "symbol": symbol,
            "correlation": correlation,
            "beta": beta,
            "r_squared": correlation ** 2,
            "observations": len(joined),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["correlation"].abs() >= min_abs_corr]
    return df.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)


# ==========================================================================
# 4. DEFAULT RISK  (Altman Z-score)
# ==========================================================================
def _first_row(frame: pd.DataFrame, *candidates: str) -> Optional[float]:
    """Most recent value of the first matching row label in a statement."""
    if frame is None or frame.empty:
        return None
    index = {str(i).strip().lower(): i for i in frame.index}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in index:
            series = frame.loc[index[key]].dropna()
            if not series.empty:
                try:
                    return float(series.iloc[0])
                except (TypeError, ValueError):
                    continue
    return None


@cached(ttl=config.TTL.fundamentals, namespace="splc_credit")
def get_credit_risk(ticker: str) -> Dict[str, Any]:
    """
    Altman Z-score from the filed balance sheet and income statement.

    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E, where
        A working capital / total assets
        B retained earnings / total assets
        C EBIT / total assets
        D market cap / total liabilities
        E revenue / total assets

    Bands are Altman's originals for public manufacturers: >2.99 safe,
    1.81-2.99 grey, <1.81 distress. They are a poor fit for banks, insurers and
    asset-light software - the caller is told which sector it is looking at so
    it can caveat accordingly.
    """
    statements = equities.get_financial_statements(ticker)
    if not statements:
        return {}

    balance = statements.get("balance_sheet", pd.DataFrame())
    income = statements.get("income_statement", pd.DataFrame())
    if balance.empty:
        return {}

    total_assets = _first_row(balance, "Total Assets")
    total_liabilities = _first_row(
        balance, "Total Liabilities Net Minority Interest", "Total Liabilities")
    current_assets = _first_row(balance, "Current Assets", "Total Current Assets")
    current_liabilities = _first_row(
        balance, "Current Liabilities", "Total Current Liabilities")
    retained_earnings = _first_row(balance, "Retained Earnings")
    ebit = _first_row(income, "EBIT", "Operating Income", "Total Operating Income As Reported")
    revenue = _first_row(income, "Total Revenue", "Operating Revenue")

    info = equities.get_company_info(ticker)
    market_cap = info.get("marketCap")

    if not total_assets or total_assets <= 0:
        return {}

    def ratio(numerator: Optional[float]) -> Optional[float]:
        return numerator / total_assets if numerator is not None else None

    working_capital = (current_assets - current_liabilities
                       if current_assets is not None and current_liabilities is not None
                       else None)

    a = ratio(working_capital)
    b = ratio(retained_earnings)
    c = ratio(ebit)
    d = (market_cap / total_liabilities
         if market_cap and total_liabilities and total_liabilities > 0 else None)
    e = ratio(revenue)

    components = {"working_capital_to_assets": a, "retained_earnings_to_assets": b,
                  "ebit_to_assets": c, "equity_to_liabilities": d,
                  "revenue_to_assets": e}

    if sum(1 for v in components.values() if v is None) > 1:
        return {"components": components, "z_score": None,
                "note": "Insufficient balance-sheet detail for a Z-score."}

    z = (1.2 * (a or 0) + 1.4 * (b or 0) + 3.3 * (c or 0)
         + 0.6 * (d or 0) + 1.0 * (e or 0))

    if z > 2.99:
        band, risk = "SAFE", "LOW"
    elif z >= 1.81:
        band, risk = "GREY", "MODERATE"
    else:
        band, risk = "DISTRESS", "ELEVATED"

    sector = info.get("sector") or ""
    unreliable = sector in {"Financial Services", "Real Estate", "Utilities"}

    return {
        "z_score": float(z), "band": band, "risk": risk,
        "components": components, "sector": sector,
        "model_fit": "POOR" if unreliable else "REASONABLE",
        "note": ("Altman's bands were calibrated on manufacturers; for "
                 f"{sector.lower()} the score is not meaningful."
                 if unreliable else
                 "Altman Z-score on the latest filed statements."),
    }


# ==========================================================================
# 5. INDUSTRY COMPARABLES
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="splc_industry_peers")
def get_industry_peers(ticker: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Comparable companies, taken from the issuer's own industry classification.

    Yahoo assigns every listed company an industry key and publishes the
    constituents of each industry ranked by market weight. That is a real
    classification of this specific company, not a hand-written sector list -
    the terminal never asserts a peer it did not derive.

    Returns:
        [{ticker, name, weight}] ordered by market weight, focal name removed.
        Empty when the classification or its constituent list is unavailable;
        the caller draws no comparables rather than inventing them.
    """
    if not equities.YFINANCE_AVAILABLE:
        return []

    ticker = ticker.upper()
    industry_key = (equities.get_company_info(ticker) or {}).get("industryKey")
    if not industry_key:
        return []

    try:
        table = equities.yf.Industry(industry_key).top_companies
    except Exception as exc:
        log.warning("industry peers unavailable for %s: %s", ticker, exc)
        return []

    if table is None or table.empty:
        return []

    peers: List[Dict[str, Any]] = []
    for symbol, row in table.iterrows():
        symbol = str(symbol).upper()
        if symbol == ticker:
            continue
        peers.append({
            "ticker": symbol,
            "name": str(row.get("name") or symbol),
            "weight": float(row.get("market weight") or 0.0),
        })
        if len(peers) >= limit:
            break
    return peers


def industry_label(ticker: str) -> str:
    """Human-readable industry name used to caption the comparables row."""
    info = equities.get_company_info(ticker) or {}
    return str(info.get("industry") or info.get("sector") or "").strip()


# ==========================================================================
# 6. NETWORK ASSEMBLY
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="splc_network")
def build_network(ticker: str) -> Dict[str, Any]:
    """
    Assemble the graph the SPLC map draws.

    Returns:
        {nodes: [...], edges: [...], stats: {...}}
        Node: {id, label, tier, pct, named, sector, source_url}
              tier is "focal" | "upstream" | "downstream" | "peer".

    Counterparties trace to a disclosure in the issuer's own filings.
    Comparables are the issuer's own industry classification ranked by market
    weight - derived per company, never a hand-written sector list. When the
    classification is unavailable the row is simply absent.
    """
    info = equities.get_company_info(ticker)
    focal_name = info.get("longName") or info.get("shortName") or ticker

    nodes: List[Dict[str, Any]] = [{
        "id": ticker.upper(), "label": focal_name, "tier": "focal",
        "pct": None, "named": True, "sector": info.get("sector"),
        "source_url": None,
    }]
    edges: List[Dict[str, Any]] = []

    counterparties = get_counterparties(ticker)
    for position, (_, row) in enumerate(counterparties.iterrows()):
        # Undisclosed counterparties all share the label "Undisclosed
        # customer", so the label alone collides every one of them onto a
        # single node. Qualify unticketed rows by position to keep each
        # disclosure its own node.
        node_id = row["counterparty_ticker"] or f"{row['counterparty']}#{position}"
        tier = "upstream" if row["relationship"] == "supplier" else "downstream"
        nodes.append({
            "id": node_id, "label": row["counterparty"], "tier": tier,
            "pct": row["pct_of_revenue"], "named": bool(row["named"]),
            "sector": None, "source_url": row["source_url"],
        })
        # Goods flow upstream->focal->downstream; the edge follows that.
        edges.append({
            "source": node_id if tier == "upstream" else ticker.upper(),
            "target": ticker.upper() if tier == "upstream" else node_id,
            "weight": row["pct_of_revenue"], "kind": row["relationship"],
            "named": bool(row["named"]), "quote": row["quote"],
        })

    for peer in get_industry_peers(ticker, 6):
        nodes.append({
            "id": peer["ticker"], "label": peer["name"], "tier": "peer",
            "pct": None, "named": True, "sector": info.get("sector"),
            "weight": peer["weight"], "source_url": None,
        })

    if counterparties.empty:
        named = downstream = single = aggregates = pd.DataFrame()
    else:
        named = counterparties[counterparties["named"]]
        downstream = counterparties[counterparties["relationship"] == "customer"]
        # Single-counterparty disclosures only. An aggregate ("our ten largest
        # customers were 96%") says nothing about any one customer's share.
        single = downstream[~downstream["is_aggregate"]]
        aggregates = downstream[downstream["is_aggregate"]]

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "counterparties": len(counterparties),
            "peers": sum(1 for n in nodes if n["tier"] == "peer"),
            "industry": industry_label(ticker),
            "named": len(named),
            "customers": len(downstream),
            "suppliers": len(counterparties) - len(downstream)
            if not counterparties.empty else 0,
            # Largest SINGLE customer. Deliberately excludes aggregates.
            "max_customer_pct": float(single["pct_of_revenue"].max())
            if not single.empty else None,
            "max_named_customer_pct": float(
                single[single["named"]]["pct_of_revenue"].max())
            if not single.empty and single["named"].any() else None,
            # Reported separately, never summed with the above: overlapping
            # disclosures would add to a meaningless number above 100%.
            "top_group_pct": float(aggregates["pct_of_revenue"].max())
            if not aggregates.empty else None,
            "top_group_label": str(aggregates.sort_values(
                "pct_of_revenue", ascending=False).iloc[0]["counterparty"])
            if not aggregates.empty else None,
        },
    }


def refresh(ticker: str) -> None:
    """
    Drop this issuer's cached supply-chain data and refetch it.

    Backs the SPLC page's REBUILD control. Every fetcher here memoises into
    the project's SQLite cache, so `st.cache_data.clear()` does not touch
    them - the `_refresh` kwarg the @cached wrapper exposes does.

    Order matters: build_network reads get_counterparties and
    get_industry_peers, so those are refreshed first. Refreshing the network
    alone would reassemble it from exactly the same stale rows.
    """
    ticker = ticker.upper()
    for fetcher in (get_counterparties, get_industry_peers,
                    get_geographic_revenue, get_commodity_exposure,
                    get_credit_risk, build_network):
        try:
            fetcher(ticker, _refresh=True)
        except Exception as exc:               # one dead upstream must not
            log.warning("refresh %s failed for %s: %s",                # strand
                        getattr(fetcher, "__name__", fetcher), ticker, exc)


def concentration_verdict(stats: Dict[str, Any]) -> Tuple[str, str]:
    """(verdict, explanation) for the headline banner."""
    top = stats.get("max_customer_pct")
    group = stats.get("top_group_pct")
    group_label = (stats.get("top_group_label") or "the largest customers").lower()

    if not top:
        if group:
            return ("GROUP CONCENTRATION ONLY",
                    f"No single customer is broken out, but {group_label} are "
                    f"{group:.0f}% of revenue between them.")
        return ("NO DISCLOSED CONCENTRATION",
                "No single customer crossed the 10% threshold that forces "
                "disclosure, or the issuer files no 10-K.")

    group_note = (f" Separately, {group_label} are {group:.0f}% combined."
                  if group else "")

    if top >= 50:
        return ("CRITICAL SINGLE-CUSTOMER DEPENDENCE",
                f"One customer is {top:.0f}% of revenue. Losing it would be an "
                f"existential event, and it hands that customer enormous "
                f"pricing power.{group_note}")
    if top >= 25:
        return ("HIGH CONCENTRATION",
                f"Largest single customer is {top:.0f}% of revenue.{group_note}")
    return ("MODERATE CONCENTRATION",
            f"Largest single customer is {top:.0f}% of revenue.{group_note}")


__all__ = [
    "get_counterparties", "get_geographic_revenue", "get_commodity_exposure",
    "get_credit_risk", "get_industry_peers", "build_network",
    "refresh", "concentration_verdict",
    "COMMODITY_SYMBOLS", "REGION_COORDS",
]

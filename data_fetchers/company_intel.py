"""
data_fetchers/company_intel.py :: Module F - People & Corporate Identity.

Bloomberg equivalents: <EQUITY> MGMT (management roster), PEOPLE, CN (contacts).

Answers two questions about a listed company:

  1. Who runs it?      Named executive officers with title, age and the
                       compensation disclosed in the proxy statement.
  2. Where is it?      Official social presence for the company itself and,
                       where it exists, for the individual executives.

Data sources, all free:
  * yfinance   - `info["companyOfficers"]`, itself derived from the DEF 14A
                 summary compensation table. Covers the named executive
                 officers only (typically CEO, CFO + the next three highest
                 paid), which is what the SEC requires firms to disclose.
  * Wikidata   - The entity API (`wbsearchentities` + `wbgetentities`).
                 Social handles live on structured properties: P2002 X/Twitter,
                 P4264 LinkedIn company, P6634 LinkedIn person, P2013 Facebook,
                 P2003 Instagram, P2397 YouTube. These are community-curated
                 and sourced, which is why we prefer them to guessing a handle
                 from the company name - a wrong handle is worse than none.

On what is deliberately NOT here: no scraping of social platforms, no
follower counts, no engagement metrics. Those need authenticated APIs that
are no longer free, and inventing them would be worse than omitting them.
Individual executives are public figures in their corporate capacity; this
module surfaces only officially-listed accounts already published by the
company or curated on Wikidata.

Everything degrades gracefully: a failed lookup yields an empty structure so
the page still renders.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from utils.cache import cached, get_session
from utils.rate_limiter import retry_with_backoff, throttled

log = logging.getLogger("openterm.company_intel")

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    log.error("yfinance import failed: %s", exc)
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False


WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Wikidata property -> (our key, URL template). The template turns a bare
# identifier into something clickable; `None` means the value is already a URL.
SOCIAL_PROPERTIES: Dict[str, tuple] = {
    "P2002": ("twitter", "https://x.com/{}"),
    "P4264": ("linkedin", "https://www.linkedin.com/company/{}"),
    "P6634": ("linkedin", "https://www.linkedin.com/in/{}"),
    "P2013": ("facebook", "https://www.facebook.com/{}"),
    "P2003": ("instagram", "https://www.instagram.com/{}"),
    "P2397": ("youtube", "https://www.youtube.com/channel/{}"),
    "P11245": ("mastodon", None),
    "P7085": ("tiktok", "https://www.tiktok.com/@{}"),
    "P856": ("website", None),
}

# Ordering for the roster: rank by seniority rather than by pay, so an
# underpaid founder-CEO still sorts first.
#
# Two traps make a flat pattern list wrong here:
#   * "Senior Vice President" contains "President", which would rank every SVP
#     above the CFO.
#   * "Executive VP & CEO of Commercial Business" contains "CEO", but that
#     person runs a division, not the company. A trailing "of <something>"
#     is the tell.
_VP = re.compile(r"\bv(ice\s+)?p(resident)?\b|\bvp\b", re.I)
_CEO = re.compile(r"\b(chief\s+exec\w*|ceo)\b(?!\s+of\s+)", re.I)
_CHAIR = re.compile(r"\b(chair(man|woman|person)?|founder)\b", re.I)
_PRESIDENT = re.compile(r"\bpresident\b", re.I)
_CFO = re.compile(r"\b(chief\s+financ\w*|cfo)\b", re.I)
_COO = re.compile(r"\b(chief\s+operat\w*|coo)\b", re.I)
_CTO = re.compile(r"\b(chief\s+(technol\w*|scien\w*|info\w*)|cto|cio)\b", re.I)
_CHIEF = re.compile(r"\bchief\b", re.I)
_EVP = re.compile(r"\b(exec\w*\s+v(ice\s+)?p(resident)?|evp)\b", re.I)
_SVP = re.compile(r"\b(senior\s+v(ice\s+)?p(resident)?|svp)\b", re.I)


def _seniority(title: str) -> int:
    """Rank a job title 0 (chief executive) to 99 (unrecognised)."""
    title = title or ""

    if _CEO.search(title):
        return 0
    if _CHAIR.search(title):
        return 1
    # Bare "President" only - a Vice President of any flavour is not one.
    if _PRESIDENT.search(title) and not _VP.search(title):
        return 2
    if _CFO.search(title):
        return 3
    if _COO.search(title):
        return 4
    if _CTO.search(title):
        return 5
    if _CHIEF.search(title):
        return 6
    if _EVP.search(title):
        return 7
    if _SVP.search(title):
        return 8
    if _VP.search(title):
        return 9
    return 99


# Wikidata descriptions for issuers vary a lot ("company", "corporation",
# "manufacturer", "bank"...). Matching a single word rejected most real hits,
# so the guard tests a family of business words and only downgrades the entity
# to "unverified" - it never discards it.
_BUSINESS_WORDS = re.compile(
    r"\b(compan\w+|corporat\w+|business|enterprise|manufactur\w+|retail\w+|"
    r"bank\w*|insur\w+|conglomerate|holding\w*|firm|producer|airline|"
    r"automaker|brand|chain|group|multinational|supplier|operator)\b", re.I)

# People are a different matter. A search for "Sabih Khan" returns whoever
# ranks highest, and Wikidata described that hit as a "researcher" - a
# different person entirely. A wrong bio is bad; a wrong social handle
# attached to a named executive is worse, so a person must actually read as a
# business figure or we attach nothing at all.
_EXECUTIVE_WORDS = re.compile(
    r"\b(executive|business\w*|entrepreneur|ceo|cfo|coo|cto|chair\w*|"
    r"director|manager|financier|investor|banker|industrialist|founder|"
    r"president|magnate|tycoon|administrator)\b", re.I)


def _clean_name(name: str) -> str:
    """
    yfinance pads names with honorifics and double spaces:
        "Mr. Timothy D. Cook"  ->  "Timothy D. Cook"
        "Mr. Kevan  Parekh"    ->  "Kevan Parekh"
    """
    cleaned = re.sub(r"^(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Hon)\.?\s+", "", (name or "").strip())
    return re.sub(r"\s+", " ", cleaned)


# ==========================================================================
# WIKIDATA
# ==========================================================================
def _wikidata_session():
    """
    Wikidata asks for a descriptive User-Agent identifying the tool; generic
    library UAs get throttled or blocked outright under their policy.
    """
    return get_session("wikidata", expire_after=86400 * 7,
                       user_agent=config.SEC_USER_AGENT)


@cached(ttl=86400 * 7, namespace="wikidata_search")
@throttled("wikidata")
@retry_with_backoff(on_giveup=lambda exc: None)
def _search_entity(name: str, kind: str = "item") -> Optional[str]:
    """
    Resolve a name to a Wikidata QID.

    We take the top hit rather than trying to disambiguate. For company and
    executive names that is reliable in practice; the caller sanity-checks the
    result against the entity's own claims before trusting it.
    """
    if not name or not name.strip():
        return None

    resp = _wikidata_session().get(
        WIKIDATA_API,
        params={"action": "wbsearchentities", "search": name.strip(),
                "language": "en", "uselang": "en", "format": "json",
                "limit": 5, "type": kind},
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()
    hits = resp.json().get("search", [])
    return hits[0]["id"] if hits else None


@cached(ttl=86400 * 7, namespace="wikidata_entity")
@throttled("wikidata")
@retry_with_backoff(on_giveup=lambda exc: {})
def _entity_claims(qid: str) -> Dict[str, Any]:
    """
    Fetch an entity's label, description and the social properties we care
    about. Returns {} rather than raising so a missing person doesn't break a
    whole roster.
    """
    if not qid:
        return {}

    resp = _wikidata_session().get(
        WIKIDATA_API,
        params={"action": "wbgetentities", "ids": qid,
                "props": "claims|labels|descriptions", "format": "json"},
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()
    entity = resp.json().get("entities", {}).get(qid, {})
    if not entity:
        return {}

    out: Dict[str, Any] = {
        "qid": qid,
        "label": entity.get("labels", {}).get("en", {}).get("value"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
        "social": {},
    }

    claims = entity.get("claims", {})
    for prop, (key, template) in SOCIAL_PROPERTIES.items():
        for statement in claims.get(prop, []):
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if not isinstance(value, str) or not value:
                continue
            # First value wins. Wikidata often lists regional variants of a
            # website and secondary support accounts; the primary is first.
            if key not in out["social"]:
                out["social"][key] = {
                    "handle": value,
                    "url": template.format(value) if template else value,
                }
            break

    return out


def _lookup_social(name: str, kind: str = "item",
                   require_business: bool = False) -> Dict[str, Any]:
    """
    Name -> social profile.

    `require_business` is a cheap guard against matching the wrong entity -
    searching "Apple" should not return the fruit. It never discards a result,
    it only sets `unverified` so the UI can mark the row.
    """
    qid = _search_entity(name, kind)
    if not qid:
        return {}

    entity = _entity_claims(qid)
    if not entity:
        return {}

    if require_business:
        description = entity.get("description") or ""
        if not _BUSINESS_WORDS.search(description):
            log.debug("Wikidata %s for '%s' looks non-corporate: %r",
                      qid, name, description)
            entity["unverified"] = True

    return entity


# ==========================================================================
# PUBLIC API
# ==========================================================================
@cached(ttl=config.TTL.fundamentals, namespace="company_social")
def get_company_social(ticker: str, company_name: str = "") -> Dict[str, Any]:
    """
    Official social accounts for the issuer itself.

    Args:
        ticker:       Used only for logging and cache keying.
        company_name: The long name from yfinance. Searching Wikidata by
                      company name beats searching by ticker - ticker
                      properties (P249) are sparsely populated and the
                      SPARQL endpoint that could query them times out.

    Returns:
        {qid, label, description, wikidata_url, social: {platform: {handle, url}}}
    """
    if not company_name:
        return {}

    profile = _lookup_social(company_name, "item", require_business=True)
    if profile:
        log.debug("Wikidata %s -> %s (%d socials)", ticker, profile.get("qid"),
                  len(profile.get("social", {})))
    return profile or {}


@cached(ttl=config.TTL.fundamentals, namespace="company_executives")
def get_executives(ticker: str, with_social: bool = True) -> pd.DataFrame:
    """
    Named executive officers, ranked by seniority.

    Args:
        with_social: Look each officer up on Wikidata. Costs one or two extra
                     requests per person, so the caller can turn it off.

    Returns:
        DataFrame: name, title, age, year_born, total_pay, exercised_value,
                   unexercised_value, fiscal_year, seniority, twitter,
                   linkedin, wikidata_url, bio
        Empty DataFrame if the issuer files no proxy or yfinance has no data.
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("Officer fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    officers = info.get("companyOfficers") or []
    if not officers:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for officer in officers:
        name = _clean_name(officer.get("name", ""))
        if not name:
            continue
        title = (officer.get("title") or "").strip()
        rows.append({
            "name": name,
            "title": title,
            "age": officer.get("age"),
            "year_born": officer.get("yearBorn"),
            "total_pay": officer.get("totalPay"),
            "exercised_value": officer.get("exercisedValue") or None,
            "unexercised_value": officer.get("unexercisedValue") or None,
            "fiscal_year": officer.get("fiscalYear"),
            "seniority": _seniority(title),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["seniority", "total_pay"], ascending=[True, False]
    ).reset_index(drop=True)

    if not with_social:
        return df

    # Only look up the top of the roster - down to SVP. A full 10-person sweep
    # is 20 requests, and unranked officers are almost never on Wikidata.
    handles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        profile: Dict[str, Any] = {}
        if row["seniority"] <= 8:
            try:
                profile = _lookup_social(row["name"], "item")
            except Exception as exc:
                log.debug("Social lookup failed for %s: %s", row["name"], exc)

            # Only trust the match if the entity reads like a business figure.
            # Same-name collisions are common and silently misattribute an
            # unrelated person's accounts to a named officer.
            description = profile.get("description") or ""
            if profile and not _EXECUTIVE_WORDS.search(description):
                log.debug("Discarding Wikidata match for %s - described as %r",
                          row["name"], description)
                profile = {}

        social = profile.get("social", {})
        handles.append({
            "twitter": social.get("twitter", {}).get("url"),
            "twitter_handle": social.get("twitter", {}).get("handle"),
            "linkedin": social.get("linkedin", {}).get("url"),
            "wikidata_url": profile.get("wikidata_url"),
            "bio": profile.get("description"),
        })

    return pd.concat([df, pd.DataFrame(handles)], axis=1)


def compensation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Headline pay statistics for the metric tiles.

    CEO pay ratio here is CEO vs. the *other named officers*, not the SEC's
    CEO-to-median-employee ratio - we don't have the median employee figure.
    Labelled accordingly in the UI so the two aren't confused.
    """
    if df.empty or "total_pay" not in df.columns:
        return {}

    paid = df[df["total_pay"].notna() & (df["total_pay"] > 0)]
    if paid.empty:
        return {"officers": len(df)}

    ceo_row = df[df["seniority"] == 0]
    ceo_pay = None
    if not ceo_row.empty:
        ceo_pay = ceo_row.iloc[0].get("total_pay")

    others = paid[paid["seniority"] != 0]["total_pay"]

    summary: Dict[str, Any] = {
        "officers": len(df),
        "disclosed": len(paid),
        "total_comp": float(paid["total_pay"].sum()),
        "median_comp": float(paid["total_pay"].median()),
        "ceo_pay": float(ceo_pay) if pd.notna(ceo_pay) else None,
        "mean_age": float(df["age"].dropna().mean()) if df["age"].notna().any() else None,
    }
    if ceo_pay and len(others) and others.median() > 0:
        summary["ceo_vs_peers"] = float(ceo_pay) / float(others.median())
    return summary


__all__ = [
    "get_executives",
    "get_company_social",
    "compensation_summary",
    "SOCIAL_PROPERTIES",
]

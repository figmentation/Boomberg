"""
data_fetchers/aviation.py :: Module C - Aviation & Fleet Tracking.

Bloomberg equivalent: FLY.

Primary source: OpenSky Network (https://opensky-network.org) - a research
network of volunteer ADS-B receivers that publishes global state vectors for
free under a research-friendly licence.

AUTHENTICATION NOTE
-------------------
OpenSky migrated from HTTP Basic auth to OAuth2 client-credentials. Basic auth
against /api/states/all is deprecated. This module:
  * uses OAuth2 when OPENSKY_CLIENT_ID/SECRET are set (4000 credits/day),
  * falls back to anonymous access when they aren't (400 credits/day, and
    anonymous callers get 10-second-resolution data instead of 5-second).

Credits are consumed per request and weighted by bounding-box area, so a
global query is far more expensive than a regional one. The module defaults
to regional queries and throttles hard.

SCOPE NOTE
----------
Aircraft positions are broadcast unencrypted by the aircraft themselves and
are legally receivable. This module ships watchlists of institutional
airframes only - national carriers, cargo fleets, government transports.
It does not include, and you should think carefully before adding, aircraft
associated with private individuals: several jurisdictions treat sustained
tracking of a named person's movements very differently from aggregate fleet
analysis, and OpenSky's own terms restrict targeting individuals.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config
from utils.cache import cached, get_session
from utils.rate_limiter import circuit_breaker, retry_with_backoff, throttled

log = logging.getLogger("openterm.aviation")


# OpenSky's /states/all response is a positional array. Index -> field name,
# per their API documentation.
STATE_FIELDS: List[str] = [
    "icao24",           # 0  Unique 24-bit transponder address (hex)
    "callsign",         # 1  Flight callsign, 8 chars, may be blank
    "origin_country",   # 2  Country of registration
    "time_position",    # 3  Unix ts of last position report
    "last_contact",     # 4  Unix ts of last message of any type
    "longitude",        # 5  WGS-84 degrees
    "latitude",         # 6  WGS-84 degrees
    "baro_altitude",    # 7  Barometric altitude, metres
    "on_ground",        # 8  Bool - surface position report
    "velocity",         # 9  Ground speed, m/s
    "true_track",       # 10 Heading, degrees clockwise from north
    "vertical_rate",    # 11 m/s, positive = climbing
    "sensors",          # 12 Receiver IDs (null unless requested)
    "geo_altitude",     # 13 GNSS altitude, metres
    "squawk",           # 14 Transponder code
    "spi",              # 15 Special purpose indicator
    "position_source",  # 16 0=ADS-B 1=ASTERIX 2=MLAT 3=FLARM
]

POSITION_SOURCES = {0: "ADS-B", 1: "ASTERIX", 2: "MLAT", 3: "FLARM"}

# Emergency/alert squawk codes worth surfacing.
SQUAWK_ALERTS = {
    "7500": ("HIJACK", "Unlawful interference"),
    "7600": ("RADIO FAIL", "Lost communications"),
    "7700": ("EMERGENCY", "General emergency"),
}

_token_cache: Dict[str, Any] = {"token": None, "expires_at": 0.0}


# ==========================================================================
# AUTH
# ==========================================================================
@retry_with_backoff(max_retries=2, on_giveup=lambda exc: None)
def _get_oauth_token() -> Optional[str]:
    """
    Fetch (and memoise) an OpenSky OAuth2 access token.

    Tokens last 30 minutes; we refresh at the 25-minute mark to avoid racing
    the expiry on a slow request.
    """
    if not (config.OPENSKY_CLIENT_ID and config.OPENSKY_CLIENT_SECRET):
        return None

    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    import requests

    resp = requests.post(
        config.OPENSKY_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": config.OPENSKY_CLIENT_ID,
            "client_secret": config.OPENSKY_CLIENT_SECRET,
        },
        timeout=config.NET.request_timeout,
    )
    resp.raise_for_status()
    payload = resp.json()

    token = payload.get("access_token")
    if not token:
        return None

    expires_in = int(payload.get("expires_in", 1800))
    _token_cache["token"] = token
    _token_cache["expires_at"] = time.time() + max(60, expires_in - 300)
    log.info("OpenSky OAuth2 token acquired (valid %ds)", expires_in)
    return token


def _auth_headers() -> Dict[str, str]:
    token = _get_oauth_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_authenticated() -> bool:
    """True when OAuth2 credentials are configured (higher daily quota)."""
    return bool(config.OPENSKY_CLIENT_ID and config.OPENSKY_CLIENT_SECRET)


# ==========================================================================
# LIVE STATE VECTORS
# ==========================================================================
@cached(ttl=config.TTL.aviation, namespace="opensky_states")
@circuit_breaker("opensky", failure_threshold=3, recovery_timeout=300.0,
                 on_open=lambda: pd.DataFrame())
@throttled("opensky")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_states(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    icao24: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    Live aircraft state vectors.

    Args:
        bbox:   (min_lat, max_lat, min_lon, max_lon). Strongly recommended -
                a global query costs 4x the credits of a bounded one.
        icao24: Tuple of lowercase hex transponder addresses to filter to.
                Must be a tuple (hashable) for the cache key.

    Returns:
        DataFrame, one row per aircraft, with the STATE_FIELDS columns plus
        derived altitude_ft, speed_kts, vertical_rate_fpm and a UTC timestamp.
    """
    params: Dict[str, Any] = {}
    if bbox:
        min_lat, max_lat, min_lon, max_lon = bbox
        params.update({
            "lamin": min_lat, "lamax": max_lat,
            "lomin": min_lon, "lomax": max_lon,
        })
    if icao24:
        params["icao24"] = [c.lower() for c in icao24]

    session = get_session("opensky", expire_after=config.TTL.aviation)
    resp = session.get(
        f"{config.OPENSKY_BASE}/states/all",
        params=params,
        headers=_auth_headers(),
        timeout=config.NET.request_timeout,
    )

    if resp.status_code == 429:
        from utils.rate_limiter import RateLimitExceeded

        raise RateLimitExceeded(
            "OpenSky daily credit budget exhausted. Anonymous access allows "
            "400 credits/day; register free OAuth2 credentials for 4000."
        )
    resp.raise_for_status()

    payload = resp.json() or {}
    states = payload.get("states") or []
    if not states:
        return pd.DataFrame(columns=STATE_FIELDS)

    df = pd.DataFrame(states, columns=STATE_FIELDS[: len(states[0])])
    return _enrich_states(df, payload.get("time"))


def _enrich_states(df: pd.DataFrame, api_time: Optional[int]) -> pd.DataFrame:
    """Add aviation-standard units and derived columns to the raw frame."""
    if df.empty:
        return df

    df = df.copy()

    # Drop rows with no usable position - they can't be mapped.
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return df

    for col in ("latitude", "longitude", "baro_altitude", "geo_altitude",
                "velocity", "true_track", "vertical_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()

    # Metric -> aviation units.
    df["altitude_ft"] = (df["baro_altitude"] * 3.28084).round(0)
    df["geo_altitude_ft"] = (df.get("geo_altitude", pd.Series(dtype=float)) * 3.28084).round(0)
    df["speed_kts"] = (df["velocity"] * 1.94384).round(0)
    df["vertical_rate_fpm"] = (df["vertical_rate"] * 196.85).round(0)

    df["phase"] = df.apply(_flight_phase, axis=1)
    df["source"] = df.get("position_source", pd.Series(dtype=int)).map(POSITION_SOURCES).fillna("UNKNOWN")

    # Staleness - a position report from 5 minutes ago is not "live".
    now = api_time or int(time.time())
    if "time_position" in df.columns:
        df["age_seconds"] = (now - pd.to_numeric(df["time_position"], errors="coerce")).round(0)
    df["fetched_at"] = datetime.now(timezone.utc)

    # Emergency squawks.
    if "squawk" in df.columns:
        df["alert"] = df["squawk"].astype(str).map(
            lambda code: SQUAWK_ALERTS.get(code, ("", ""))[0]
        )
    else:
        df["alert"] = ""

    return df.reset_index(drop=True)


def _flight_phase(row: pd.Series) -> str:
    """Classify each aircraft's phase from altitude and vertical rate."""
    if row.get("on_ground"):
        return "GROUND"

    vertical = row.get("vertical_rate")
    altitude = row.get("baro_altitude")

    if vertical is not None and not pd.isna(vertical):
        if vertical > 2.5:      # ~500 fpm
            return "CLIMB"
        if vertical < -2.5:
            return "DESCENT"

    if altitude is not None and not pd.isna(altitude):
        if altitude > 8000:     # above ~FL260
            return "CRUISE"
        return "MANEUVER"

    return "UNKNOWN"


def get_states_by_region(region: str = "GLOBAL") -> pd.DataFrame:
    """Convenience wrapper over the named regions in config.AVIATION_REGIONS."""
    bbox = config.AVIATION_REGIONS.get(region.upper())
    return get_states(bbox=bbox)


# ==========================================================================
# WATCHLIST TRACKING
# ==========================================================================
def track_watchlist(watchlist: Optional[str] = None) -> pd.DataFrame:
    """
    Positions for the configured fleets of interest.

    OpenSky's icao24 filter accepts many addresses in one request, so an
    entire watchlist costs a single API credit.

    Args:
        watchlist: Key of config.AIRCRAFT_WATCHLISTS, or None for all of them.

    Returns:
        DataFrame of airborne/tracked aircraft plus `watchlist` and `label`
        columns. Aircraft that are parked with transponders off simply don't
        appear - absence of a row is not evidence the aircraft doesn't exist.
    """
    if watchlist:
        groups = {watchlist: config.AIRCRAFT_WATCHLISTS.get(watchlist, {})}
    else:
        groups = config.AIRCRAFT_WATCHLISTS

    label_map: Dict[str, Tuple[str, str]] = {}
    for group_name, entries in groups.items():
        for hex_code, label in entries.items():
            label_map[hex_code.lower()] = (group_name, label)

    if not label_map:
        return pd.DataFrame()

    df = get_states(icao24=tuple(sorted(label_map)))
    if df.empty:
        return df

    df["watchlist"] = df["icao24"].map(lambda h: label_map.get(h, ("", ""))[0])
    df["label"] = df["icao24"].map(lambda h: label_map.get(h, ("", ""))[1])
    return df


@cached(ttl=300, namespace="opensky_track")
@throttled("opensky")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_aircraft_track(icao24: str, timestamp: int = 0) -> pd.DataFrame:
    """
    Historical waypoint trail for one aircraft.

    Args:
        icao24:    Lowercase hex transponder address.
        timestamp: Unix time inside the desired flight; 0 = the live/most
                   recent track.

    Note: OpenSky's /tracks endpoint is officially experimental and is
    frequently unavailable to anonymous users. An empty frame is normal.
    """
    resp = get_session("opensky", expire_after=300).get(
        f"{config.OPENSKY_BASE}/tracks/all",
        params={"icao24": icao24.lower(), "time": timestamp},
        headers=_auth_headers(),
        timeout=config.NET.request_timeout,
    )

    if resp.status_code in (403, 404):
        log.info("No track available for %s (HTTP %d)", icao24, resp.status_code)
        return pd.DataFrame()
    resp.raise_for_status()

    payload = resp.json() or {}
    path = payload.get("path") or []
    if not path:
        return pd.DataFrame()

    # Waypoint: [time, latitude, longitude, baro_altitude, true_track, on_ground]
    df = pd.DataFrame(path, columns=["time", "latitude", "longitude",
                                     "baro_altitude", "true_track", "on_ground"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["altitude_ft"] = (pd.to_numeric(df["baro_altitude"], errors="coerce") * 3.28084).round(0)
    df["icao24"] = icao24.lower()
    df["callsign"] = payload.get("callsign", "")
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


@cached(ttl=600, namespace="opensky_flights")
@throttled("opensky")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_flights_by_aircraft(icao24: str, days_back: int = 7) -> pd.DataFrame:
    """
    Departure/arrival history for one airframe.

    OpenSky caps each /flights/aircraft query at a 30-day window.

    Returns: DataFrame with departure/arrival airport ICAO codes and times.
    """
    end = int(time.time())
    begin = end - min(days_back, 30) * 86400

    resp = get_session("opensky", expire_after=600).get(
        f"{config.OPENSKY_BASE}/flights/aircraft",
        params={"icao24": icao24.lower(), "begin": begin, "end": end},
        headers=_auth_headers(),
        timeout=config.NET.request_timeout,
    )

    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    flights = resp.json() or []
    if not flights:
        return pd.DataFrame()

    df = pd.DataFrame(flights)
    for col, out in (("firstSeen", "departure_time"), ("lastSeen", "arrival_time")):
        if col in df.columns:
            df[out] = pd.to_datetime(df[col], unit="s", utc=True)

    rename = {
        "estDepartureAirport": "departure_airport",
        "estArrivalAirport": "arrival_airport",
        "callsign": "callsign",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if {"departure_time", "arrival_time"}.issubset(df.columns):
        df["duration_min"] = (
            (df["arrival_time"] - df["departure_time"]).dt.total_seconds() / 60
        ).round(0)

    keep = [c for c in ("callsign", "departure_airport", "arrival_airport",
                        "departure_time", "arrival_time", "duration_min", "icao24")
            if c in df.columns]
    return df[keep].sort_values("departure_time", ascending=False).reset_index(drop=True)


@cached(ttl=600, namespace="opensky_airport")
@throttled("opensky")
@retry_with_backoff(on_giveup=lambda exc: pd.DataFrame())
def get_airport_traffic(
    airport_icao: str, hours: int = 12, direction: str = "arrival"
) -> pd.DataFrame:
    """
    Arrivals or departures for an airport - a proxy for cargo/hub throughput.

    Args:
        airport_icao: 4-letter ICAO code ("KJFK", "EGLL", "OMDB", "VHHH").
        hours:        Lookback window, capped at 7 days by the API.
        direction:    "arrival" | "departure"
    """
    end = int(time.time())
    begin = end - min(hours, 168) * 3600
    endpoint = "arrival" if direction == "arrival" else "departure"

    resp = get_session("opensky", expire_after=600).get(
        f"{config.OPENSKY_BASE}/flights/{endpoint}",
        params={"airport": airport_icao.upper(), "begin": begin, "end": end},
        headers=_auth_headers(),
        timeout=config.NET.request_timeout,
    )

    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    flights = resp.json() or []
    if not flights:
        return pd.DataFrame()

    df = pd.DataFrame(flights)
    df["departure_time"] = pd.to_datetime(df.get("firstSeen"), unit="s", utc=True)
    df["arrival_time"] = pd.to_datetime(df.get("lastSeen"), unit="s", utc=True)
    df = df.rename(columns={
        "estDepartureAirport": "from",
        "estArrivalAirport": "to",
    })

    keep = [c for c in ("icao24", "callsign", "from", "to",
                        "departure_time", "arrival_time") if c in df.columns]
    return df[keep].sort_values("arrival_time", ascending=False).reset_index(drop=True)


# ==========================================================================
# ANALYTICS
# ==========================================================================
def summarize_traffic(df: pd.DataFrame) -> Dict[str, Any]:
    """Headline counts and distributions for the aviation dashboard tiles."""
    if df is None or df.empty:
        return {"total": 0}

    out: Dict[str, Any] = {
        "total": len(df),
        "airborne": int((~df["on_ground"].fillna(False).astype(bool)).sum())
        if "on_ground" in df else len(df),
    }
    out["on_ground"] = out["total"] - out["airborne"]

    if "altitude_ft" in df:
        alt = df["altitude_ft"].dropna()
        if len(alt):
            out["mean_altitude_ft"] = int(alt.mean())
            out["max_altitude_ft"] = int(alt.max())

    if "speed_kts" in df:
        spd = df["speed_kts"].dropna()
        if len(spd):
            out["mean_speed_kts"] = int(spd.mean())
            out["max_speed_kts"] = int(spd.max())

    if "origin_country" in df:
        out["top_countries"] = df["origin_country"].value_counts().head(10).to_dict()

    if "phase" in df:
        out["phases"] = df["phase"].value_counts().to_dict()

    if "alert" in df:
        alerts = df[df["alert"].astype(bool)]
        out["emergency_count"] = len(alerts)
        if len(alerts):
            out["emergencies"] = alerts[
                [c for c in ("icao24", "callsign", "squawk", "alert",
                             "origin_country") if c in alerts.columns]
            ].to_dict("records")

    return out


def identify_operator(callsign: str) -> str:
    """
    Map a callsign prefix to its operator.

    Callsigns follow ICAO 3-letter designators (FDX1234 = FedEx). Covers the
    freight and flag carriers most relevant to trade-flow analysis.
    """
    if not callsign:
        return "UNKNOWN"

    prefix = str(callsign).strip().upper()[:3]
    operators = {
        # Freight
        "FDX": "FedEx Express", "UPS": "UPS Airlines", "GTI": "Atlas Air",
        "CLX": "Cargolux", "CKS": "Kalitta Air", "ABX": "ABX Air",
        "GEC": "Lufthansa Cargo", "CAO": "Air China Cargo",
        "CKK": "China Cargo", "SQC": "Singapore Cargo", "MPH": "Martinair",
        "PAC": "Polar Air Cargo", "NCA": "Nippon Cargo", "ETH": "Ethiopian",
        # Flag / major passenger
        "UAL": "United", "AAL": "American", "DAL": "Delta",
        "SWA": "Southwest", "BAW": "British Airways", "DLH": "Lufthansa",
        "AFR": "Air France", "KLM": "KLM", "UAE": "Emirates",
        "QTR": "Qatar Airways", "SIA": "Singapore Airlines",
        "ANA": "All Nippon", "JAL": "Japan Airlines", "CPA": "Cathay Pacific",
        "THY": "Turkish Airlines", "RYR": "Ryanair", "EZY": "easyJet",
        # State / military
        "RCH": "USAF Air Mobility (Reach)", "SAM": "USAF Special Air Mission",
        "AF1": "US Air Force One", "RRR": "RAF Ascot", "GAF": "German Air Force",
        "CFC": "Canadian Forces", "IAM": "Italian Air Force",
    }
    return operators.get(prefix, "UNKNOWN")


def add_operator_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a resolved operator name to a state-vector frame."""
    if df is None or df.empty or "callsign" not in df.columns:
        return df
    out = df.copy()
    out["operator"] = out["callsign"].map(identify_operator)
    return out


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * radius_nm * math.asin(math.sqrt(a))


def filter_near(df: pd.DataFrame, lat: float, lon: float,
                radius_nm: float = 100.0) -> pd.DataFrame:
    """Aircraft within `radius_nm` of a point, sorted nearest-first."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["distance_nm"] = out.apply(
        lambda r: haversine_nm(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    return out[out["distance_nm"] <= radius_nm].sort_values("distance_nm").reset_index(drop=True)


__all__ = [
    "get_states", "get_states_by_region", "track_watchlist",
    "get_aircraft_track", "get_flights_by_aircraft", "get_airport_traffic",
    "summarize_traffic", "identify_operator", "add_operator_column",
    "filter_near", "haversine_nm", "is_authenticated",
    "STATE_FIELDS", "SQUAWK_ALERTS",
]

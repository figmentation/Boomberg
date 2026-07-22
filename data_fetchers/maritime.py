"""
data_fetchers/maritime.py :: Module B - Maritime & Supply Chain OSINT.

Bloomberg equivalents: AIS, SHIP, PORT.

SOURCE STRATEGY (and an honest note about it)
--------------------------------------------
Vessel AIS positions are broadcast in the clear over VHF and are legally
receivable. Getting them into a terminal is the hard part, and the sources
differ a lot in how welcome you are:

  1. LOCAL RECEIVER (best)     - An RTL-SDR dongle running rtl-ais/AISdispatcher
                                 emits NMEA over UDP. `pyais` decodes it here.
                                 Real-time, unlimited, entirely yours. If you
                                 care about this data, spend the $25 on a dongle.

  2. aisstream.io (recommended) - Free API key, websocket firehose, explicitly
                                 intended for this use. No scraping, no ToS
                                 grey area. This is the default remote source.

  3. AISHub                    - Free, but you must contribute a feed to get one.

  4. Playwright scrapers       - MarineTraffic / VesselFinder. IMPLEMENTED, but
                                 read this first: both sites' Terms of Service
                                 prohibit automated collection, and both sit
                                 behind Cloudflare, so these paths are brittle
                                 by design and will break without warning.
                                 They are last-resort fallbacks, are rate
                                 limited to a crawl, and are DISABLED by
                                 default. Set OPENTERM_ALLOW_SCRAPERS=1 to
                                 enable them and accept that decision yourself.

The chokepoint tracker aggregates whatever source is live into vessel counts
per corridor. With no source configured it returns empty frames and the UI
says so, rather than inventing positions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import config
from utils.cache import cached, get_session
from utils.rate_limiter import retry_with_backoff, throttled

log = logging.getLogger("openterm.maritime")

# Opt-in flag for the ToS-restricted scraper paths. Off by default.
ALLOW_SCRAPERS: bool = os.getenv("OPENTERM_ALLOW_SCRAPERS", "0") == "1"

AISSTREAM_WS = "wss://stream.aisstream.io/v0/stream"

# AIS navigational status codes (ITU-R M.1371).
NAV_STATUS: Dict[int, str] = {
    0: "Under way (engine)", 1: "At anchor", 2: "Not under command",
    3: "Restricted manoeuvrability", 4: "Constrained by draught",
    5: "Moored", 6: "Aground", 7: "Fishing", 8: "Under way (sailing)",
    9: "Reserved (HSC)", 10: "Reserved (WIG)", 11: "Towing astern",
    12: "Towing alongside", 13: "Reserved", 14: "AIS-SART/MOB/EPIRB",
    15: "Undefined",
}

# Statuses that mean "this ship is not moving cargo right now".
IDLE_STATUSES = {1, 5, 6}


# ==========================================================================
# DATA MODEL
# ==========================================================================
@dataclass
class VesselPosition:
    """One AIS position report, normalised across every source."""

    mmsi: str
    latitude: float
    longitude: float
    timestamp: datetime
    name: Optional[str] = None
    imo: Optional[str] = None
    callsign: Optional[str] = None
    ship_type: Optional[int] = None
    ship_type_label: Optional[str] = None
    sog: Optional[float] = None          # Speed over ground, knots
    cog: Optional[float] = None          # Course over ground, degrees
    heading: Optional[float] = None      # True heading, degrees
    nav_status: Optional[int] = None
    nav_status_label: Optional[str] = None
    draught: Optional[float] = None      # Metres
    destination: Optional[str] = None
    eta: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi, "name": self.name, "imo": self.imo,
            "callsign": self.callsign, "latitude": self.latitude,
            "longitude": self.longitude, "sog_kts": self.sog,
            "cog_deg": self.cog, "heading_deg": self.heading,
            "nav_status": self.nav_status_label, "nav_status_code": self.nav_status,
            "ship_type": self.ship_type_label, "ship_type_code": self.ship_type,
            "draught_m": self.draught, "destination": self.destination,
            "eta": self.eta, "length_m": self.length, "width_m": self.width,
            "timestamp": self.timestamp, "source": self.source,
        }


def _ship_type_label(code: Optional[int]) -> Optional[str]:
    """Resolve an AIS ship-type code, rounding down to the decade bucket."""
    if code is None:
        return None
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    if code in config.AIS_SHIP_TYPES:
        return config.AIS_SHIP_TYPES[code]
    return config.AIS_SHIP_TYPES.get((code // 10) * 10, f"Type {code}")


def _positions_to_frame(positions: List[VesselPosition]) -> pd.DataFrame:
    """Vessel list -> DataFrame, deduplicated to the newest fix per MMSI."""
    if not positions:
        return pd.DataFrame()

    df = pd.DataFrame([p.to_dict() for p in positions])
    df = df.sort_values("timestamp").drop_duplicates(subset=["mmsi"], keep="last")
    return df.reset_index(drop=True)


# ==========================================================================
# SOURCE 1: aisstream.io websocket
# ==========================================================================
def _aisstream_subscription(
    bbox: Optional[Tuple[float, float, float, float]],
    mmsi_filter: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Build the subscription payload.

    aisstream wants bounding boxes as [[lat1, lon1], [lat2, lon2]] pairs.
    Our internal bbox convention is (min_lat, min_lon, max_lat, max_lon).
    """
    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        boxes = [[[min_lat, min_lon], [max_lat, max_lon]]]
    else:
        boxes = [[[-90.0, -180.0], [90.0, 180.0]]]

    sub: Dict[str, Any] = {
        "APIKey": config.AISSTREAM_API_KEY,
        "BoundingBoxes": boxes,
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }
    if mmsi_filter:
        sub["FiltersShipMMSI"] = [str(m) for m in mmsi_filter]
    return sub


async def _aisstream_collect(
    bbox: Optional[Tuple[float, float, float, float]],
    duration: float,
    mmsi_filter: Optional[List[str]],
    max_vessels: int,
) -> List[VesselPosition]:
    """
    Listen on the aisstream websocket for `duration` seconds and collect fixes.

    Position reports and static-data reports arrive as separate messages, so
    we merge them per-MMSI: positions give lat/lon/speed, static data gives
    IMO, name, draught, destination and dimensions.
    """
    import websockets

    vessels: Dict[str, VesselPosition] = {}
    static: Dict[str, Dict[str, Any]] = {}
    deadline = time.monotonic() + duration

    async with websockets.connect(AISSTREAM_WS, ping_interval=20,
                                  close_timeout=5) as ws:
        await ws.send(json.dumps(_aisstream_subscription(bbox, mmsi_filter)))

        while time.monotonic() < deadline and len(vessels) < max_vessels:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 10.0))
            except asyncio.TimeoutError:
                continue

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            if "error" in msg:
                log.error("aisstream error: %s", msg["error"])
                break

            meta = msg.get("MetaData") or {}
            mmsi = str(meta.get("MMSI") or meta.get("MMSI_String") or "").strip()
            if not mmsi:
                continue

            msg_type = msg.get("MessageType")
            body = (msg.get("Message") or {}).get(msg_type) or {}

            if msg_type == "ShipStaticData":
                dim = body.get("Dimension") or {}
                static[mmsi] = {
                    "imo": str(body.get("ImoNumber")) if body.get("ImoNumber") else None,
                    "callsign": (body.get("CallSign") or "").strip() or None,
                    "ship_type": body.get("Type"),
                    "draught": body.get("MaximumStaticDraught"),
                    "destination": (body.get("Destination") or "").strip() or None,
                    "length": (dim.get("A", 0) or 0) + (dim.get("B", 0) or 0) or None,
                    "width": (dim.get("C", 0) or 0) + (dim.get("D", 0) or 0) or None,
                }
                # Enrich an existing position if we already have one.
                if mmsi in vessels:
                    _apply_static(vessels[mmsi], static[mmsi])
                continue

            if msg_type != "PositionReport":
                continue

            lat = meta.get("latitude") if meta.get("latitude") is not None else body.get("Latitude")
            lon = meta.get("longitude") if meta.get("longitude") is not None else body.get("Longitude")
            if lat is None or lon is None:
                continue

            nav = body.get("NavigationalStatus")
            position = VesselPosition(
                mmsi=mmsi,
                latitude=float(lat),
                longitude=float(lon),
                timestamp=_parse_ts(meta.get("time_utc")),
                name=(meta.get("ShipName") or "").strip() or None,
                sog=_num(body.get("Sog")),
                cog=_num(body.get("Cog")),
                heading=_num(body.get("TrueHeading")),
                nav_status=nav,
                nav_status_label=NAV_STATUS.get(nav) if nav is not None else None,
                source="aisstream",
            )
            if mmsi in static:
                _apply_static(position, static[mmsi])
            vessels[mmsi] = position

    return list(vessels.values())


def _apply_static(position: VesselPosition, static: Dict[str, Any]) -> None:
    """Merge a ShipStaticData record into a position report in place."""
    position.imo = position.imo or static.get("imo")
    position.callsign = position.callsign or static.get("callsign")
    position.draught = position.draught or static.get("draught")
    position.destination = position.destination or static.get("destination")
    position.length = position.length or static.get("length")
    position.width = position.width or static.get("width")
    if static.get("ship_type") is not None:
        position.ship_type = static["ship_type"]
        position.ship_type_label = _ship_type_label(static["ship_type"])


def fetch_aisstream(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    duration: float = 20.0,
    mmsi_filter: Optional[List[str]] = None,
    max_vessels: int = 800,
) -> List[VesselPosition]:
    """
    Synchronous wrapper around the aisstream collector.

    Args:
        duration: Seconds to listen. Busy corridors fill up in 10-15s; quiet
                  ones need 30s+. This blocks, so keep it short in the UI.

    Returns [] when no API key is configured - never raises into the page.
    """
    if not config.AISSTREAM_API_KEY:
        log.info("AISSTREAM_API_KEY not set - skipping aisstream source")
        return []

    try:
        # Streamlit script threads have no running loop, so asyncio.run is safe.
        # If one *is* running (notebook, async host), fall back to a worker thread.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                _aisstream_collect(bbox, duration, mmsi_filter, max_vessels)
            )

        result: List[VesselPosition] = []

        def runner() -> None:
            result.extend(asyncio.run(
                _aisstream_collect(bbox, duration, mmsi_filter, max_vessels)
            ))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=duration + 15)
        return result

    except Exception as exc:
        log.error("aisstream collection failed: %s", exc)
        return []


# ==========================================================================
# SOURCE 2: local AIS receiver over UDP (pyais)
# ==========================================================================
def fetch_local_ais(
    duration: float = 15.0,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> List[VesselPosition]:
    """
    Decode NMEA AIS sentences from a local SDR receiver's UDP output.

    This is the highest-quality path available: your own antenna, no rate
    limits, no terms of service, sub-second latency. Range is line-of-sight
    (~20-40nm), so it only covers your local waters.

    Typical setup:
        rtl_ais -n -h 127.0.0.1 -P 10110
    then set AIS_UDP_PORT=10110.
    """
    try:
        from pyais import decode
        from pyais.exceptions import InvalidNMEAMessageException
    except ImportError:
        log.info("pyais not installed - local AIS receiver disabled")
        return []

    host = host if host is not None else config.AIS_UDP_HOST
    port = port if port is not None else config.AIS_UDP_PORT

    vessels: Dict[str, VesselPosition] = {}
    static: Dict[str, Dict[str, Any]] = {}
    sock: Optional[socket.socket] = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(2.0)
        sock.bind((host, port))

        deadline = time.monotonic() + duration
        # Multi-part messages (types 5/24) arrive as fragments keyed by channel.
        fragments: Dict[str, List[bytes]] = {}

        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue

            for line in data.decode("ascii", errors="ignore").splitlines():
                line = line.strip()
                if not line.startswith(("!AIVDM", "!AIVDO")):
                    continue

                try:
                    parts = line.split(",")
                    total, seq = int(parts[1]), int(parts[2])
                    channel = parts[3] or "0"

                    if total > 1:
                        buffer = fragments.setdefault(channel, [])
                        buffer.append(line.encode())
                        if seq < total:
                            continue
                        message = decode(*buffer)
                        fragments.pop(channel, None)
                    else:
                        message = decode(line)

                    _ingest_pyais(message, vessels, static)

                except (InvalidNMEAMessageException, ValueError, IndexError):
                    continue
                except Exception as exc:
                    log.debug("AIS decode error: %s", exc)

    except OSError as exc:
        log.info("Local AIS UDP listener unavailable on %s:%s (%s)", host, port, exc)
    finally:
        if sock is not None:
            sock.close()

    return list(vessels.values())


def _ingest_pyais(message: Any, vessels: Dict[str, VesselPosition],
                  static: Dict[str, Dict[str, Any]]) -> None:
    """Route a decoded pyais message into the vessel/static registries."""
    msg = message.asdict() if hasattr(message, "asdict") else dict(message)
    msg_type = msg.get("msg_type")
    mmsi = str(msg.get("mmsi") or "").strip()
    if not mmsi:
        return

    # Types 1/2/3 = Class A position report; 18/19 = Class B.
    if msg_type in (1, 2, 3, 18, 19):
        lat, lon = msg.get("lat"), msg.get("lon")
        # 91/181 are the AIS "not available" sentinels.
        if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
            return

        nav = msg.get("status")
        nav_code = int(nav) if isinstance(nav, (int, float)) else None
        position = VesselPosition(
            mmsi=mmsi,
            latitude=float(lat),
            longitude=float(lon),
            timestamp=datetime.now(timezone.utc),
            sog=_num(msg.get("speed")),
            cog=_num(msg.get("course")),
            heading=_num(msg.get("heading")),
            nav_status=nav_code,
            nav_status_label=NAV_STATUS.get(nav_code) if nav_code is not None else None,
            source="local_ais",
        )
        if mmsi in static:
            _apply_static(position, static[mmsi])
        vessels[mmsi] = position

    # Type 5 = Class A static & voyage data; 24 = Class B static.
    elif msg_type in (5, 24):
        record = {
            "imo": str(msg["imo"]) if msg.get("imo") else None,
            "callsign": (msg.get("callsign") or "").strip() or None,
            "ship_type": msg.get("ship_type"),
            "draught": _num(msg.get("draught")),
            "destination": (msg.get("destination") or "").strip() or None,
            "length": (msg.get("to_bow", 0) or 0) + (msg.get("to_stern", 0) or 0) or None,
            "width": (msg.get("to_port", 0) or 0) + (msg.get("to_starboard", 0) or 0) or None,
        }
        static[mmsi] = record
        name = (msg.get("shipname") or "").strip()
        if mmsi in vessels:
            _apply_static(vessels[mmsi], record)
            if name:
                vessels[mmsi].name = name


# ==========================================================================
# SOURCE 3: Playwright scrapers (opt-in, ToS-restricted)
# ==========================================================================
_SCRAPER_WARNING = (
    "Scraper sources are disabled. MarineTraffic and VesselFinder both "
    "prohibit automated collection in their Terms of Service and sit behind "
    "Cloudflare. Prefer aisstream.io (free API key) or a local SDR receiver. "
    "To enable anyway, set OPENTERM_ALLOW_SCRAPERS=1."
)


def _scrapers_enabled() -> bool:
    if not ALLOW_SCRAPERS:
        log.info(_SCRAPER_WARNING)
        return False
    return True


@cached(ttl=config.TTL.scrape, namespace="vesselfinder_scrape")
@throttled("scrape")
@retry_with_backoff(max_retries=2, on_giveup=lambda exc: [])
def fetch_vesselfinder(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_vessels: int = 300,
) -> List[VesselPosition]:
    """
    Read vessel markers off VesselFinder's public map tiles via Playwright.

    Approach: load the map at the target viewport and intercept the XHR the
    page itself makes for marker data, rather than parsing rendered pixels.
    That is far more stable, but it is still scraping - see _SCRAPER_WARNING.

    Returns [] on any failure. Never raises into the UI.
    """
    if not _scrapers_enabled():
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("playwright not installed - scraper source unavailable")
        return []

    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        zoom = _zoom_for_bbox(bbox)
    else:
        center_lat, center_lon, zoom = 30.0, 32.5, 8

    url = f"https://www.vesselfinder.com/?zoom={zoom}&lat={center_lat}&lon={center_lon}"
    captured: List[Dict[str, Any]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=config.NET.playwright_headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        try:
            context = browser.new_context(
                user_agent=config.BROWSER_USER_AGENT,
                viewport={"width": 1600, "height": 1000},
                locale="en-US",
            )
            page = context.new_page()

            def on_response(response) -> None:
                # The map fetches marker batches from an internal endpoint.
                if any(token in response.url for token in ("/api/", "vesselsonmap", "/ss/")):
                    try:
                        if "json" in (response.headers.get("content-type") or ""):
                            payload = response.json()
                            if isinstance(payload, (list, dict)):
                                captured.append(payload)
                    except Exception:
                        pass

            page.on("response", on_response)
            page.goto(url, timeout=config.NET.playwright_timeout_ms,
                      wait_until="domcontentloaded")
            # Let the map settle and issue its marker requests.
            page.wait_for_timeout(9000)
        finally:
            browser.close()

    positions = _parse_scraped_markers(captured, "vesselfinder", max_vessels)
    if not positions:
        raise ValueError("VesselFinder scrape captured no vessel markers "
                         "(layout changed or request was blocked)")
    return positions


@cached(ttl=config.TTL.scrape, namespace="marinetraffic_scrape")
@throttled("scrape")
@retry_with_backoff(max_retries=2, on_giveup=lambda exc: [])
def fetch_marinetraffic(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_vessels: int = 300,
) -> List[VesselPosition]:
    """
    MarineTraffic public map, same interception approach as VesselFinder.

    Expect this to fail more often than not: MarineTraffic runs aggressive
    bot mitigation and this path exists only as the final fallback in the
    chain. See _SCRAPER_WARNING.
    """
    if not _scrapers_enabled():
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        zoom = _zoom_for_bbox(bbox)
    else:
        center_lat, center_lon, zoom = 30.0, 32.5, 8

    url = f"https://www.marinetraffic.com/en/ais/home/centerx:{center_lon}/centery:{center_lat}/zoom:{zoom}"
    captured: List[Dict[str, Any]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=config.NET.playwright_headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        try:
            context = browser.new_context(
                user_agent=config.BROWSER_USER_AGENT,
                viewport={"width": 1600, "height": 1000},
            )
            page = context.new_page()

            def on_response(response) -> None:
                if "getData" in response.url or "/station/" in response.url:
                    try:
                        if "json" in (response.headers.get("content-type") or ""):
                            captured.append(response.json())
                    except Exception:
                        pass

            page.on("response", on_response)
            page.goto(url, timeout=config.NET.playwright_timeout_ms,
                      wait_until="domcontentloaded")
            page.wait_for_timeout(10000)

            # Cloudflare interstitial detection - bail rather than retry-loop.
            content = (page.content() or "").lower()
            if "just a moment" in content or "cf-challenge" in content:
                raise RuntimeError("Blocked by Cloudflare challenge")
        finally:
            browser.close()

    positions = _parse_scraped_markers(captured, "marinetraffic", max_vessels)
    if not positions:
        raise ValueError("MarineTraffic scrape captured no vessel markers")
    return positions


def _parse_scraped_markers(
    payloads: List[Any], source: str, max_vessels: int
) -> List[VesselPosition]:
    """
    Best-effort extraction of vessel records from intercepted JSON.

    Both sites use short, undocumented field names that change periodically,
    so this searches for any dict carrying a plausible lat/lon/MMSI triple
    rather than assuming a fixed schema.
    """
    out: List[VesselPosition] = []
    seen: set = set()

    lat_keys = ("LAT", "lat", "latitude", "y", "Y")
    lon_keys = ("LON", "lon", "lng", "longitude", "x", "X")
    mmsi_keys = ("MMSI", "mmsi", "SHIP_ID", "ship_id", "id")

    def walk(node: Any) -> None:
        if len(out) >= max_vessels:
            return

        if isinstance(node, dict):
            lat = _first(node, lat_keys)
            lon = _first(node, lon_keys)
            mmsi = _first(node, mmsi_keys)

            if lat is not None and lon is not None and mmsi is not None:
                try:
                    lat_f, lon_f = float(lat), float(lon)
                    # Some endpoints send coordinates scaled by 1e6 or 600000.
                    if abs(lat_f) > 90:
                        lat_f /= 600000.0
                        lon_f /= 600000.0
                    if abs(lat_f) <= 90 and abs(lon_f) <= 180:
                        key = str(mmsi)
                        if key not in seen:
                            seen.add(key)
                            ship_type = _first(node, ("SHIPTYPE", "type", "ship_type"))
                            out.append(VesselPosition(
                                mmsi=key,
                                latitude=lat_f,
                                longitude=lon_f,
                                timestamp=datetime.now(timezone.utc),
                                name=_str_or_none(_first(node, ("SHIPNAME", "name", "NAME"))),
                                sog=_num(_first(node, ("SPEED", "sog", "speed"))),
                                cog=_num(_first(node, ("COURSE", "cog", "course"))),
                                heading=_num(_first(node, ("HEADING", "heading"))),
                                ship_type=_int_or_none(ship_type),
                                ship_type_label=_ship_type_label(_int_or_none(ship_type)),
                                destination=_str_or_none(_first(node, ("DESTINATION", "destination"))),
                                source=source,
                            ))
                except (TypeError, ValueError):
                    pass

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    for payload in payloads:
        walk(payload)

    return out


# ==========================================================================
# UNIFIED FETCH WITH FALLBACK CHAIN
# ==========================================================================
@cached(ttl=config.TTL.ais, namespace="vessels_in_area")
def get_vessels_in_area(
    bbox: Tuple[float, float, float, float],
    max_vessels: int = 500,
    stream_seconds: float = 20.0,
) -> pd.DataFrame:
    """
    Vessels inside a bounding box, from whichever source answers first.

    Chain: local receiver -> aisstream -> VesselFinder -> MarineTraffic.
    Each step is tried in turn; the first non-empty result wins. Results are
    clipped to the bbox because streaming sources can overshoot slightly.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon)

    Returns:
        DataFrame of vessel positions, empty if every source is unavailable.
        `df.attrs["source"]` records which source produced it.
    """
    attempts: List[Tuple[str, Any]] = [
        ("local_ais", lambda: fetch_local_ais(duration=min(stream_seconds, 12.0))),
        ("aisstream", lambda: fetch_aisstream(bbox=bbox, duration=stream_seconds,
                                              max_vessels=max_vessels)),
        ("vesselfinder", lambda: fetch_vesselfinder(bbox=bbox, max_vessels=max_vessels)),
        ("marinetraffic", lambda: fetch_marinetraffic(bbox=bbox, max_vessels=max_vessels)),
    ]

    for name, fetcher in attempts:
        try:
            positions = fetcher()
            if positions:
                df = _positions_to_frame(positions)
                df = _clip_to_bbox(df, bbox)
                if not df.empty:
                    df.attrs["source"] = name
                    log.info("Maritime: %d vessels from %s", len(df), name)
                    return df
        except Exception as exc:
            log.warning("Maritime source %s failed: %s", name, exc)

    log.warning("All maritime sources exhausted for bbox %s", bbox)
    empty = pd.DataFrame()
    empty.attrs["source"] = "none"
    return empty


def _clip_to_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Drop rows outside the requested box."""
    if df.empty:
        return df
    min_lat, min_lon, max_lat, max_lon = bbox
    return df[
        df["latitude"].between(min_lat, max_lat)
        & df["longitude"].between(min_lon, max_lon)
    ].reset_index(drop=True)


def track_vessel(identifier: str, stream_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Look up a single vessel by MMSI or IMO.

    Args:
        identifier: 9-digit MMSI (e.g. 636019825) or IMO number, with or
                    without an "IMO" prefix.

    Returns:
        {"found": bool, "vessel": {...}, "source": str, "note": str}

    MMSI lookups are direct - the AIS filter takes MMSI natively. IMO lookups
    require finding a matching static-data broadcast, which is slower and can
    miss if the vessel hasn't transmitted static data during the listen window.
    """
    ident = str(identifier).strip().upper().replace("IMO", "").strip()

    if not ident.isdigit():
        return {"found": False, "note": f"'{identifier}' is not a numeric MMSI or IMO."}

    is_mmsi = len(ident) == 9
    result: Dict[str, Any] = {"found": False, "identifier": ident,
                              "id_type": "MMSI" if is_mmsi else "IMO"}

    # --- MMSI: direct filter ----------------------------------------------
    if is_mmsi and config.AISSTREAM_API_KEY:
        positions = fetch_aisstream(bbox=None, duration=stream_seconds,
                                    mmsi_filter=[ident], max_vessels=5)
        match = next((p for p in positions if p.mmsi == ident), None)
        if match:
            result.update({"found": True, "vessel": match.to_dict(),
                           "source": "aisstream"})
            result["flag"] = mmsi_to_flag(ident)
            return result

    # --- Local receiver ----------------------------------------------------
    for position in fetch_local_ais(duration=min(stream_seconds, 15.0)):
        if position.mmsi == ident or (position.imo and position.imo == ident):
            result.update({"found": True, "vessel": position.to_dict(),
                           "source": "local_ais"})
            result["flag"] = mmsi_to_flag(position.mmsi)
            return result

    result["note"] = (
        "Vessel not observed during the listening window. It may be out of "
        "AIS range, transmitting infrequently, or have AIS switched off. "
        "Try a longer window, or configure AISSTREAM_API_KEY for global "
        "coverage."
        if not config.AISSTREAM_API_KEY
        else "Vessel not observed during the listening window."
    )
    return result


# ==========================================================================
# CHOKEPOINT TRACKER
# ==========================================================================
def get_chokepoint_status(
    chokepoint_code: str, stream_seconds: float = 20.0
) -> Dict[str, Any]:
    """
    Traffic and congestion read for one maritime corridor.

    Computes:
      * total vessel count in the corridor bbox
      * how many are anchored/moored (the queue) vs under way (the flow)
      * a congestion ratio versus the configured baseline
      * fleet composition by ship type
      * mean draught, a proxy for whether transits are laden or in ballast

    Interpretation caveat: counts depend entirely on AIS coverage in that
    box. A low count can mean light traffic OR poor receiver coverage. The
    `coverage_note` field records which.
    """
    chokepoint = config.CHOKEPOINTS.get(chokepoint_code.upper())
    if not chokepoint:
        return {"error": f"Unknown chokepoint '{chokepoint_code}'"}

    df = get_vessels_in_area(chokepoint.bbox, stream_seconds=stream_seconds)
    source = df.attrs.get("source", "none") if hasattr(df, "attrs") else "none"

    out: Dict[str, Any] = {
        "code": chokepoint.code,
        "name": chokepoint.name,
        "description": chokepoint.description,
        "bbox": chokepoint.bbox,
        "baseline_vessels": chokepoint.baseline_vessels,
        "source": source,
        "timestamp": datetime.now(timezone.utc),
        "vessels": df,
    }

    if df.empty:
        out.update({
            "vessel_count": 0,
            "status": "NO DATA",
            "coverage_note": (
                "No AIS source returned data for this corridor. Configure "
                "AISSTREAM_API_KEY or a local receiver."
            ),
        })
        return out

    total = len(df)
    out["vessel_count"] = total

    # Queue vs flow.
    if "nav_status_code" in df.columns:
        idle_mask = df["nav_status_code"].isin(IDLE_STATUSES)
        out["anchored_count"] = int(idle_mask.sum())
        out["underway_count"] = int((~idle_mask).sum())
    elif "sog_kts" in df.columns:
        # Fall back to speed: under 0.5kt is effectively stationary.
        slow = df["sog_kts"].fillna(0) < 0.5
        out["anchored_count"] = int(slow.sum())
        out["underway_count"] = int((~slow).sum())

    # Congestion index vs baseline.
    ratio = total / max(chokepoint.baseline_vessels, 1)
    out["congestion_ratio"] = round(ratio, 2)
    out["status"] = (
        "SEVERE CONGESTION" if ratio >= 1.6
        else "ELEVATED" if ratio >= 1.25
        else "NORMAL" if ratio >= 0.6
        else "LIGHT TRAFFIC"
    )

    # Fleet mix.
    if "ship_type" in df.columns:
        composition = df["ship_type"].dropna().value_counts().head(8)
        out["composition"] = composition.to_dict()
        types = df["ship_type"].fillna("")
        out["tanker_count"] = int(types.str.contains("Tanker", case=False).sum())
        out["cargo_count"] = int(types.str.contains("Cargo", case=False).sum())

    # Draught: laden ships sit deeper. A falling mean can signal more ballast
    # legs, i.e. weakening backhaul demand.
    if "draught_m" in df.columns:
        draughts = pd.to_numeric(df["draught_m"], errors="coerce").dropna()
        draughts = draughts[(draughts > 0) & (draughts < 30)]
        if len(draughts):
            out["mean_draught_m"] = round(float(draughts.mean()), 2)
            out["max_draught_m"] = round(float(draughts.max()), 2)

    if "sog_kts" in df.columns:
        speeds = pd.to_numeric(df["sog_kts"], errors="coerce").dropna()
        speeds = speeds[speeds < 40]  # filter obvious AIS noise
        if len(speeds):
            out["mean_speed_kts"] = round(float(speeds.mean()), 1)

    return out


def get_all_chokepoints(stream_seconds: float = 12.0) -> pd.DataFrame:
    """
    Summary row per configured corridor - the maritime overview table.

    Uses a short listen window per corridor since this hits every one of them.
    """
    rows: List[Dict[str, Any]] = []

    for code in config.CHOKEPOINTS:
        try:
            status = get_chokepoint_status(code, stream_seconds=stream_seconds)
            rows.append({
                "Chokepoint": status.get("name", code),
                "Code": code,
                "Vessels": status.get("vessel_count", 0),
                "Anchored": status.get("anchored_count"),
                "Under Way": status.get("underway_count"),
                "Tankers": status.get("tanker_count"),
                "Cargo": status.get("cargo_count"),
                "Baseline": status.get("baseline_vessels"),
                "Congestion": status.get("congestion_ratio"),
                "Status": status.get("status", "NO DATA"),
                "Mean Draught (m)": status.get("mean_draught_m"),
                "Source": status.get("source", "none"),
            })
        except Exception as exc:
            log.error("Chokepoint %s failed: %s", code, exc)
            rows.append({"Chokepoint": code, "Status": "ERROR", "Vessels": 0})

    return pd.DataFrame(rows)


# ==========================================================================
# REFERENCE DATA / HELPERS
# ==========================================================================
# MMSI Maritime Identification Digits -> flag state. Abridged to the flags
# that actually matter for commercial shipping analysis.
MID_FLAGS: Dict[str, str] = {
    "201": "Albania", "205": "Belgium", "209": "Cyprus", "210": "Cyprus",
    "211": "Germany", "212": "Cyprus", "215": "Malta", "219": "Denmark",
    "224": "Spain", "226": "France", "227": "France", "228": "France",
    "232": "United Kingdom", "233": "United Kingdom", "235": "United Kingdom",
    "236": "Gibraltar", "238": "Croatia", "239": "Greece", "240": "Greece",
    "241": "Greece", "244": "Netherlands", "245": "Netherlands",
    "246": "Netherlands", "247": "Italy", "248": "Malta", "249": "Malta",
    "255": "Madeira", "256": "Malta", "257": "Norway", "258": "Norway",
    "259": "Norway", "261": "Poland", "263": "Portugal", "265": "Sweden",
    "266": "Sweden", "271": "Turkey", "272": "Ukraine", "273": "Russia",
    "310": "Bermuda", "311": "Bahamas", "312": "Belize", "314": "Barbados",
    "316": "Canada", "319": "Cayman Islands", "329": "Guadeloupe",
    "338": "United States", "351": "Panama", "352": "Panama",
    "353": "Panama", "354": "Panama", "355": "Panama", "356": "Panama",
    "357": "Panama", "366": "United States", "367": "United States",
    "368": "United States", "369": "United States", "370": "Panama",
    "371": "Panama", "372": "Panama", "373": "Panama", "374": "Panama",
    "375": "St Vincent", "376": "St Vincent", "377": "St Vincent",
    "412": "China", "413": "China", "414": "China", "416": "Taiwan",
    "419": "India", "422": "Iran", "423": "Azerbaijan", "431": "Japan",
    "432": "Japan", "440": "South Korea", "441": "South Korea",
    "445": "North Korea", "457": "Mongolia", "463": "Pakistan",
    "466": "Qatar", "470": "UAE", "471": "UAE", "473": "Yemen",
    "477": "Hong Kong", "525": "Indonesia", "533": "Malaysia",
    "563": "Singapore", "564": "Singapore", "565": "Singapore",
    "566": "Singapore", "567": "Thailand", "574": "Vietnam",
    "576": "Vanuatu", "577": "Vanuatu", "601": "South Africa",
    "605": "Algeria", "612": "Cameroon", "613": "Congo", "621": "Djibouti",
    "622": "Egypt", "636": "Liberia", "637": "Liberia", "642": "Libya",
    "654": "Mauritania", "655": "Morocco", "664": "Seychelles",
    "667": "Sierra Leone", "671": "Togo", "672": "Tunisia",
    "677": "Tanzania", "710": "Brazil", "725": "Chile", "730": "Colombia",
    "735": "Ecuador", "760": "Peru", "770": "Uruguay", "775": "Venezuela",
}


def mmsi_to_flag(mmsi: str) -> str:
    """
    Flag state from an MMSI's leading Maritime Identification Digits.

    Caveat worth knowing: flag of registry is a legal convenience, not a
    statement of beneficial ownership. Liberia, Panama and the Marshall
    Islands are flags of convenience - a Liberian-flagged tanker is very
    unlikely to be Liberian-owned.
    """
    mmsi = str(mmsi).strip()
    if len(mmsi) < 3:
        return "Unknown"
    return MID_FLAGS.get(mmsi[:3], f"MID {mmsi[:3]}")


def _zoom_for_bbox(bbox: Tuple[float, float, float, float]) -> int:
    """Pick a slippy-map zoom level that frames the bbox."""
    min_lat, min_lon, max_lat, max_lon = bbox
    span = max(abs(max_lat - min_lat), abs(max_lon - min_lon))
    for threshold, zoom in ((0.5, 11), (1.0, 10), (2.5, 9), (5.0, 8),
                            (10.0, 7), (20.0, 6)):
        if span < threshold:
            return zoom
    return 5


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    min_lat, min_lon, max_lat, max_lon = bbox
    return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * radius_nm * math.asin(math.sqrt(a))


def source_availability() -> Dict[str, Dict[str, Any]]:
    """What the maritime module can actually reach right now, for the UI."""
    return {
        "local_ais": {
            "available": _probe_udp(config.AIS_UDP_HOST, config.AIS_UDP_PORT),
            "quality": "BEST",
            "note": f"UDP {config.AIS_UDP_HOST}:{config.AIS_UDP_PORT} "
                    "(rtl-ais / AISdispatcher)",
        },
        "aisstream": {
            "available": bool(config.AISSTREAM_API_KEY),
            "quality": "GOOD",
            "note": "Free API key at aisstream.io - global coverage, no ToS issues",
        },
        "vesselfinder": {
            "available": ALLOW_SCRAPERS,
            "quality": "FRAGILE",
            "note": "Scraper. ToS-restricted, Cloudflare-protected. Opt-in.",
        },
        "marinetraffic": {
            "available": ALLOW_SCRAPERS,
            "quality": "FRAGILE",
            "note": "Scraper. ToS-restricted, Cloudflare-protected. Opt-in.",
        },
    }


def _probe_udp(host: str, port: int, timeout: float = 0.5) -> bool:
    """Can we bind the AIS UDP port? A bind failure means nothing is feeding it."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout)
        sock.bind((host, port))
        try:
            sock.recvfrom(2048)
            return True
        except socket.timeout:
            return False
    except OSError:
        return False
    finally:
        if sock is not None:
            sock.close()


def _num(value: Any) -> Optional[float]:
    """float() tolerating None, NaN and AIS 'not available' sentinels."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    # AIS sentinels: 511 = heading unavailable, 1023 = SOG unavailable,
    # 3600 (=360.0 deg) = COG unavailable.
    if out in (511.0, 1023.0, 360.0):
        return None
    return out


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(node: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in node and node[key] is not None:
            return node[key]
    return None


def _parse_ts(raw: Any) -> datetime:
    """Parse aisstream's timestamp, falling back to now()."""
    if not raw:
        return datetime.now(timezone.utc)
    try:
        text = str(raw).replace(" UTC", "").replace("Z", "+00:00").strip()
        # aisstream emits nanosecond precision; Python handles 6 digits.
        if "." in text:
            head, _, tail = text.partition(".")
            frac = "".join(c for c in tail if c.isdigit())[:6]
            offset = tail[len(frac):] if len(tail) > len(frac) else ""
            text = f"{head}.{frac}{offset}"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


__all__ = [
    "VesselPosition", "get_vessels_in_area", "track_vessel",
    "get_chokepoint_status", "get_all_chokepoints",
    "fetch_aisstream", "fetch_local_ais",
    "fetch_vesselfinder", "fetch_marinetraffic",
    "mmsi_to_flag", "bbox_center", "haversine_nm",
    "source_availability", "NAV_STATUS", "ALLOW_SCRAPERS",
]

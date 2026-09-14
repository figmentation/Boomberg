"""
ui/maps.py :: Dark-mode geographic rendering for AIS and flight data.

Two backends, chosen per use case:

  * Folium  - Leaflet under the hood. Best for vessel/aircraft markers where
              you want popups, rotated icons and layer control. Rendered via
              streamlit-folium.
  * Plotly  - Scattergeo/Scattermap. Best for dense global scatter (thousands
              of aircraft) where Folium's DOM-per-marker approach crawls.

Both use free tile sources: CartoDB Dark Matter (Leaflet) and Plotly's
built-in "carto-darkmatter" style. Neither needs a Mapbox token - Plotly's
`scattermap` trace replaced the token-gated `scattermapbox` in Plotly 5.24.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import CHOKEPOINTS, THEME
from ui.terminal_theme import AMBER_SCALE, style_figure

log = logging.getLogger("openterm.maps")

try:
    import folium
    from folium.plugins import MarkerCluster, MiniMap

    FOLIUM_AVAILABLE = True
except ImportError:
    folium = None  # type: ignore[assignment]
    FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import st_folium

    ST_FOLIUM_AVAILABLE = True
except ImportError:
    st_folium = None  # type: ignore[assignment]
    ST_FOLIUM_AVAILABLE = False


# CartoDB Dark Matter - free, no key, and the only basemap that doesn't fight
# an amber-on-black theme.
DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
DARK_ATTR = "&copy; OpenStreetMap contributors &copy; CARTO"


# ==========================================================================
# FOLIUM: BASE
# ==========================================================================
def base_map(
    center: Tuple[float, float] = (25.0, 30.0),
    zoom: int = 4,
    minimap: bool = False,
) -> Optional[Any]:
    """Dark Leaflet map with the terminal's tile styling."""
    if not FOLIUM_AVAILABLE:
        return None

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=DARK_TILES,
        attr=DARK_ATTR,
        control_scale=True,
        prefer_canvas=True,   # canvas rendering handles hundreds of markers
    )

    if minimap:
        try:
            MiniMap(tile_layer=folium.TileLayer(DARK_TILES, attr=DARK_ATTR),
                    toggle_display=True, position="bottomleft").add_to(fmap)
        except Exception:
            pass

    return fmap


# ==========================================================================
# FOLIUM: VESSELS
# ==========================================================================
# Ship type -> marker colour. Tankers amber (energy), cargo cyan (trade),
# passenger magenta, everything else grey.
VESSEL_COLORS: Dict[str, str] = {
    "Tanker": THEME.amber,
    "Cargo": THEME.cyan,
    "Passenger": THEME.magenta,
    "Fishing": "#7A7F86",
    "Tug": "#8A8F96",
    "Military": THEME.red,
}


def _vessel_color(ship_type: Optional[str]) -> str:
    if not ship_type:
        return THEME.muted
    for key, color in VESSEL_COLORS.items():
        if key.lower() in str(ship_type).lower():
            return color
    return THEME.muted


def vessel_map(
    df: pd.DataFrame,
    center: Optional[Tuple[float, float]] = None,
    zoom: int = 7,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cluster: bool = False,
    show_headings: bool = True,
) -> Optional[Any]:
    """
    Plot AIS vessel positions on a dark Leaflet map.

    Each vessel is a heading-rotated arrow when a course/heading is known and
    a circle when it isn't, so a static anchored ship reads differently from
    one making way. Popups carry the full AIS record.

    Args:
        df:   Frame from maritime.get_vessels_in_area().
        bbox: Draws the corridor boundary and auto-fits the view to it.
    """
    if not FOLIUM_AVAILABLE:
        return None

    if center is None:
        if bbox:
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        elif df is not None and not df.empty:
            center = (float(df["latitude"].mean()), float(df["longitude"].mean()))
        else:
            center = (25.0, 30.0)

    fmap = base_map(center=center, zoom=zoom)
    if fmap is None:
        return None

    # Corridor outline.
    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color=THEME.amber, weight=1.4, fill=False, dash_array="6,6",
            tooltip="Monitored corridor",
        ).add_to(fmap)
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    if df is None or df.empty:
        return fmap

    target = MarkerCluster(name="Vessels").add_to(fmap) if cluster else fmap

    for _, vessel in df.iterrows():
        lat, lon = vessel.get("latitude"), vessel.get("longitude")
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue

        color = _vessel_color(vessel.get("ship_type"))
        heading = vessel.get("heading_deg")
        if heading is None or pd.isna(heading):
            heading = vessel.get("cog_deg")

        popup_html = _vessel_popup(vessel)
        name = str(vessel.get("name") or vessel.get("mmsi") or "Unknown")

        if show_headings and heading is not None and not pd.isna(heading):
            # Rotated triangle: instantly shows which way traffic is flowing.
            icon = folium.DivIcon(html=(
                f'<div style="transform: rotate({float(heading)}deg);'
                f'width:0;height:0;'
                f'border-left:5px solid transparent;'
                f'border-right:5px solid transparent;'
                f'border-bottom:13px solid {color};'
                f'filter: drop-shadow(0 0 2px {color});"></div>'
            ), icon_size=(10, 13), icon_anchor=(5, 7))
            folium.Marker(
                location=[float(lat), float(lon)], icon=icon,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=name,
            ).add_to(target)
        else:
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=3.5, color=color, fill=True, fillColor=color,
                fillOpacity=0.8, weight=1,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=name,
            ).add_to(target)

    return fmap


def _vessel_popup(vessel: pd.Series) -> str:
    """HTML popup for one vessel, dark-styled to match the map."""
    from data_fetchers.maritime import mmsi_to_flag

    def field(label: str, value: Any, suffix: str = "") -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return ""
        return (
            f'<tr><td style="color:{THEME.muted};padding-right:9px;">{label}</td>'
            f'<td style="color:{THEME.cyan};">{value}{suffix}</td></tr>'
        )

    mmsi = vessel.get("mmsi", "")
    rows = "".join([
        field("MMSI", mmsi),
        field("IMO", vessel.get("imo")),
        field("Flag", mmsi_to_flag(str(mmsi)) if mmsi else None),
        field("Type", vessel.get("ship_type")),
        field("Status", vessel.get("nav_status")),
        field("Speed", _round(vessel.get("sog_kts"), 1), " kts"),
        field("Course", _round(vessel.get("cog_deg"), 0), "°"),
        field("Draught", _round(vessel.get("draught_m"), 1), " m"),
        field("Length", _round(vessel.get("length_m"), 0), " m"),
        field("Destination", vessel.get("destination")),
        field("Position", f"{vessel.get('latitude'):.4f}, {vessel.get('longitude'):.4f}"),
        field("Source", vessel.get("source")),
    ])

    name = str(vessel.get("name") or f"MMSI {mmsi}")
    return (
        f'<div style="font-family:monospace;font-size:11px;background:{THEME.bg_panel};'
        f'color:{THEME.cyan};padding:7px;min-width:210px;">'
        f'<div style="color:{THEME.amber};font-weight:bold;font-size:12px;'
        f'border-bottom:1px solid {THEME.border};padding-bottom:3px;margin-bottom:5px;">'
        f'{name}</div><table>{rows}</table></div>'
    )


def chokepoint_overview_map(statuses: List[Dict[str, Any]]) -> Optional[Any]:
    """
    World map with one congestion marker per corridor.

    Marker radius scales with vessel count, colour with congestion status -
    a fast visual triage of where trade is backing up.
    """
    if not FOLIUM_AVAILABLE:
        return None

    fmap = base_map(center=(20.0, 40.0), zoom=2)
    if fmap is None:
        return None

    status_colors = {
        "SEVERE CONGESTION": THEME.red,
        "ELEVATED": THEME.amber,
        "NORMAL": THEME.green,
        "LIGHT TRAFFIC": THEME.cyan,
        "NO DATA": THEME.muted,
    }

    for status in statuses:
        bbox = status.get("bbox")
        if not bbox:
            continue

        lat = (bbox[0] + bbox[2]) / 2
        lon = (bbox[1] + bbox[3]) / 2
        label = status.get("status", "NO DATA")
        color = status_colors.get(label, THEME.muted)
        count = status.get("vessel_count", 0)

        folium.Rectangle(
            bounds=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
            color=color, weight=1.2, fill=True, fillOpacity=0.12,
        ).add_to(fmap)

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(7, min(26, math.sqrt(max(count, 1)) * 2.6)),
            color=color, fill=True, fillColor=color, fillOpacity=0.55, weight=2,
            tooltip=f"{status.get('name')}: {count} vessels — {label}",
            popup=folium.Popup(
                f'<div style="font-family:monospace;font-size:11px;">'
                f'<b style="color:{THEME.amber};">{status.get("name")}</b><br>'
                f'Vessels: {count}<br>'
                f'Anchored: {status.get("anchored_count", "—")}<br>'
                f'Under way: {status.get("underway_count", "—")}<br>'
                f'Congestion: {status.get("congestion_ratio") or "—"}x '
                f'measured baseline '
                f'({status.get("baseline_vessels") or "not yet measured"})<br>'
                f'Status: <b style="color:{color};">{label}</b><br>'
                f'<i>{status.get("description", "")}</i></div>',
                max_width=300,
            ),
        ).add_to(fmap)

    return fmap


# ==========================================================================
# PLOTLY: AIRCRAFT
# ==========================================================================
# Every aircraft on the live traffic map is drawn in this one light blue.
# They used to sit on AMBER_SCALE, whose low end is #1A1A1A: on the
# carto-darkmatter basemap anything low or slow - every parked aircraft -
# was near-black and effectively invisible. Light blue reads against the
# dark map and stays distinct from the cyan used for data text and from
# the amber of vessels and flight tracks.
AIRCRAFT_COLOR = "#7DD3FC"


def flight_map(
    df: pd.DataFrame,
    title: str = "LIVE AIRCRAFT",
    height: int = 640,
    show_labels: bool = False,
) -> go.Figure:
    """
    Global aircraft scatter on a dark basemap.

    Plotly rather than Folium here because a global OpenSky query returns
    5-10k aircraft and Leaflet's per-marker DOM nodes make that unusable.
    Every aircraft is a circle in AIRCRAFT_COLOR; callsign, altitude, speed
    and heading are in its hover card.
    """
    fig = go.Figure()

    if df is None or df.empty:
        fig.add_annotation(
            text="NO AIRCRAFT DATA<br><span style='font-size:11px'>"
                 "OpenSky may be rate-limiting or the region may be empty</span>",
            showarrow=False, font=dict(color=THEME.muted, size=15),
        )
        return _style_geo(fig, height, title)

    data = df.dropna(subset=["latitude", "longitude"]).copy()
    if data.empty:
        return _style_geo(fig, height, title)

    hover = data.apply(_flight_hover, axis=1)

    # Circles, not heading-rotated triangles. On a map trace any symbol other
    # than "circle" is drawn from the basemap style's icon sprite and ignores
    # marker.color - carto-darkmatter's triangles render black whatever colour
    # is set, which is why aircraft were near-invisible. Heading is in the
    # hover card instead.
    marker: Dict[str, Any] = {
        "size": 7,
        "symbol": "circle",
        "allowoverlap": True,
        "color": AIRCRAFT_COLOR,
    }

    fig.add_trace(go.Scattermap(
        lat=data["latitude"], lon=data["longitude"],
        mode="markers+text" if show_labels else "markers",
        marker=marker,
        text=data["callsign"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=8, color=THEME.cyan),
        hovertext=hover, hoverinfo="text",
        name="Aircraft",
    ))

    # Frame the view on where the traffic actually is.
    center_lat = float(data["latitude"].median())
    center_lon = float(data["longitude"].median())
    lat_span = float(data["latitude"].max() - data["latitude"].min())
    zoom = _zoom_for_span(max(lat_span, 1.0))

    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return _style_geo(fig, height, title)


def _flight_hover(row: pd.Series) -> str:
    """Hover card for one aircraft."""
    parts = [
        f"<b>{row.get('callsign') or row.get('icao24', '')}</b>",
        f"ICAO24: {row.get('icao24', '')}",
    ]
    if row.get("label"):
        parts.insert(1, f"<i>{row['label']}</i>")
    if row.get("operator") and row["operator"] != "UNKNOWN":
        parts.append(f"Operator: {row['operator']}")
    if row.get("origin_country"):
        parts.append(f"Origin: {row['origin_country']}")
    if pd.notna(row.get("altitude_ft")):
        parts.append(f"Altitude: {row['altitude_ft']:,.0f} ft")
    if pd.notna(row.get("speed_kts")):
        parts.append(f"Speed: {row['speed_kts']:,.0f} kts")
    if pd.notna(row.get("true_track")):
        parts.append(f"Heading: {row['true_track']:.0f}°")
    if row.get("phase"):
        parts.append(f"Phase: {row['phase']}")
    if row.get("squawk"):
        parts.append(f"Squawk: {row['squawk']}")
    if row.get("alert"):
        parts.append(f"<b style='color:red'>⚠ {row['alert']}</b>")
    return "<br>".join(parts)


def flight_track_map(track: pd.DataFrame, title: str = "FLIGHT TRACK",
                     height: int = 520) -> go.Figure:
    """Waypoint trail for a single aircraft, coloured by altitude."""
    fig = go.Figure()

    if track is None or track.empty:
        fig.add_annotation(text="NO TRACK DATA AVAILABLE", showarrow=False,
                           font=dict(color=THEME.muted, size=14))
        return _style_geo(fig, height, title)

    fig.add_trace(go.Scattermap(
        lat=track["latitude"], lon=track["longitude"],
        mode="lines", line=dict(width=2.4, color=THEME.amber),
        name="Track", hoverinfo="skip",
    ))

    fig.add_trace(go.Scattermap(
        lat=track["latitude"], lon=track["longitude"],
        mode="markers",
        marker=dict(size=5, color=track.get("altitude_ft", pd.Series(dtype=float)),
                    colorscale=AMBER_SCALE, showscale=True,
                    colorbar=dict(title=dict(text="ALT FT",
                                             font=dict(color=THEME.muted, size=9)),
                                  tickfont=dict(color=THEME.muted, size=9),
                                  thickness=10, len=0.6)),
        hovertext=track.apply(
            lambda r: f"{r.get('timestamp')}<br>{r.get('altitude_ft', 0):,.0f} ft",
            axis=1,
        ),
        hoverinfo="text", name="Waypoints",
    ))

    # Origin and current position markers.
    for index, (color, label) in ((0, (THEME.green, "START")),
                                 (-1, (THEME.red, "CURRENT"))):
        point = track.iloc[index]
        fig.add_trace(go.Scattermap(
            lat=[point["latitude"]], lon=[point["longitude"]],
            mode="markers", marker=dict(size=13, color=color),
            name=label, hovertext=label, hoverinfo="text",
        ))

    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=float(track["latitude"].mean()),
                             lon=float(track["longitude"].mean())),
                 zoom=_zoom_for_span(
                     float(track["latitude"].max() - track["latitude"].min()) or 1
                 )),
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return _style_geo(fig, height, title)


# ==========================================================================
# PLOTLY: VESSELS (dense fallback)
# ==========================================================================
def vessel_map_plotly(
    df: pd.DataFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title: str = "AIS VESSEL POSITIONS",
    height: int = 600,
) -> go.Figure:
    """
    Plotly vessel scatter - the fallback when folium/streamlit-folium aren't
    installed, and the better option above ~800 vessels.
    """
    fig = go.Figure()

    if df is None or df.empty:
        fig.add_annotation(
            text="NO AIS DATA<br><span style='font-size:11px'>"
                 "Configure AISSTREAM_API_KEY or a local receiver</span>",
            showarrow=False, font=dict(color=THEME.muted, size=15),
        )
    else:
        data = df.dropna(subset=["latitude", "longitude"]).copy()
        colors = data.get("ship_type", pd.Series([""] * len(data))).map(_vessel_color)

        fig.add_trace(go.Scattermap(
            lat=data["latitude"], lon=data["longitude"],
            mode="markers",
            marker=dict(
                size=8, color=colors.tolist(), symbol="triangle",
                angle=data.get("heading_deg",
                               data.get("cog_deg", pd.Series(0, index=data.index))
                               ).fillna(0).tolist(),
                allowoverlap=True,
            ),
            hovertext=data.apply(_vessel_hover, axis=1),
            hoverinfo="text", name="Vessels",
        ))

    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        fig.add_trace(go.Scattermap(
            lat=[min_lat, min_lat, max_lat, max_lat, min_lat],
            lon=[min_lon, max_lon, max_lon, min_lon, min_lon],
            mode="lines", line=dict(width=1.4, color=THEME.amber),
            name="Corridor", hoverinfo="skip",
        ))
        center = ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
        zoom = _zoom_for_span(max_lat - min_lat)
    elif df is not None and not df.empty:
        center = (float(df["latitude"].mean()), float(df["longitude"].mean()))
        zoom = 6
    else:
        center, zoom = (25.0, 30.0), 3

    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=center[0], lon=center[1]), zoom=zoom),
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return _style_geo(fig, height, title)


def _vessel_hover(row: pd.Series) -> str:
    parts = [f"<b>{row.get('name') or row.get('mmsi')}</b>",
             f"MMSI: {row.get('mmsi', '')}"]
    for label, key, fmt in (
        ("Type", "ship_type", "{}"),
        ("Status", "nav_status", "{}"),
        ("Speed", "sog_kts", "{:.1f} kts"),
        ("Course", "cog_deg", "{:.0f}°"),
        ("Draught", "draught_m", "{:.1f} m"),
        ("Destination", "destination", "{}"),
    ):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and value != "":
            try:
                parts.append(f"{label}: {fmt.format(value)}")
            except (ValueError, TypeError):
                parts.append(f"{label}: {value}")
    return "<br>".join(parts)


# ==========================================================================
# DENSITY
# ==========================================================================
def density_map(
    df: pd.DataFrame,
    title: str = "TRAFFIC DENSITY",
    height: int = 560,
    radius: int = 18,
) -> go.Figure:
    """Heat layer over positions - useful for spotting anchorage clusters."""
    fig = go.Figure()

    if df is None or df.empty:
        fig.add_annotation(text="NO DATA", showarrow=False,
                           font=dict(color=THEME.muted, size=14))
        return _style_geo(fig, height, title)

    data = df.dropna(subset=["latitude", "longitude"])

    fig.add_trace(go.Densitymap(
        lat=data["latitude"], lon=data["longitude"],
        radius=radius, colorscale=AMBER_SCALE, showscale=True,
        colorbar=dict(title=dict(text="DENSITY", font=dict(color=THEME.muted, size=9)),
                      tickfont=dict(color=THEME.muted, size=9),
                      thickness=10, len=0.6),
    ))

    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=float(data["latitude"].median()),
                             lon=float(data["longitude"].median())),
                 zoom=_zoom_for_span(
                     float(data["latitude"].max() - data["latitude"].min()) or 1
                 )),
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return _style_geo(fig, height, title)


def exposure_map(
    df: pd.DataFrame,
    title: str = "REVENUE BY REPORTED GEOGRAPHY",
    height: int = 380,
) -> go.Figure:
    """
    Revenue exposure bubbles over a world map.

    Expects the frame from supply_chain.get_geographic_revenue: region, revenue,
    pct, lat, lon. Regions a filer reports as a residual ("Other countries",
    "Rest of world") carry no meaningful location and are dropped rather than
    pinned at (0, 0) in the Gulf of Guinea.
    """
    fig = go.Figure()

    if df is None or df.empty:
        fig.add_annotation(text="NO GEOGRAPHIC DISCLOSURE", showarrow=False,
                           font=dict(color=THEME.muted, size=14))
        return _style_geo(fig, height, title)

    data = df.dropna(subset=["lat", "lon"])
    data = data[(data["lat"] != 0) | (data["lon"] != 0)]
    if data.empty:
        fig.add_annotation(
            text="ONLY UNLOCATABLE RESIDUAL REGIONS DISCLOSED",
            showarrow=False, font=dict(color=THEME.muted, size=12))
        return _style_geo(fig, height, title)

    largest = float(data["pct"].max()) or 1.0
    fig.add_trace(go.Scattergeo(
        lat=data["lat"], lon=data["lon"],
        text=[f"{row.region}<br>{row.pct:.1f}% of disclosed revenue"
              for row in data.itertuples()],
        hoverinfo="text",
        marker=dict(
            size=[14 + (pct / largest) * 46 for pct in data["pct"]],
            color=data["pct"], colorscale=AMBER_SCALE,
            line=dict(color=THEME.amber, width=1),
            opacity=0.82,
            colorbar=dict(title=dict(text="% REV",
                                     font=dict(color=THEME.muted, size=9)),
                          tickfont=dict(color=THEME.muted, size=9),
                          thickness=10, len=0.6),
        ),
        name="REVENUE",
    ))

    fig.update_geos(
        projection_type="natural earth",
        bgcolor=THEME.bg, landcolor=THEME.bg_raised,
        oceancolor=THEME.bg, showocean=True,
        lakecolor=THEME.bg, coastlinecolor=THEME.border,
        countrycolor=THEME.grid, showcountries=True, showland=True,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=32, b=0), showlegend=False)
    return _style_geo(fig, height, title)


# ==========================================================================
# HELPERS
# ==========================================================================
def _style_geo(fig: go.Figure, height: int, title: str) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=THEME.bg,
        plot_bgcolor=THEME.bg,
        title=dict(text=title, font=dict(color=THEME.amber, size=13,
                                         family=THEME.font_mono),
                   x=0.01, xanchor="left"),
        font=dict(family=THEME.font_mono, color=THEME.cyan, size=11),
        legend=dict(bgcolor="rgba(17,18,20,0.85)", bordercolor=THEME.border,
                    borderwidth=1, font=dict(color=THEME.cyan, size=10),
                    x=0.01, y=0.99),
        hoverlabel=dict(bgcolor=THEME.bg_raised, bordercolor=THEME.amber,
                        font=dict(family=THEME.font_mono, color=THEME.cyan,
                                  size=11)),
        showlegend=True,
    )
    return fig


def _zoom_for_span(lat_span: float) -> float:
    """Map a latitude span in degrees to a slippy-map zoom level."""
    if lat_span <= 0:
        return 8
    for threshold, zoom in ((0.5, 10), (1.5, 8.5), (4, 7), (10, 5.5),
                            (25, 4), (60, 2.8)):
        if lat_span < threshold:
            return zoom
    return 1.4


def _round(value: Any, digits: int = 1) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def render_folium(fmap: Any, height: int = 600, key: Optional[str] = None) -> None:
    """
    Display a Folium map inside Streamlit.

    `returned_objects=[]` matters: without it, every pan/zoom triggers a full
    script rerun and refetches data. We only want the map rendered, not
    round-tripping interaction state.
    """
    import streamlit as st

    if fmap is None:
        st.warning(
            "Folium is not installed - install `folium` and `streamlit-folium`, "
            "or use the Plotly map view."
        )
        return

    if not ST_FOLIUM_AVAILABLE:
        # Fall back to raw HTML embedding.
        st.components.v1.html(fmap._repr_html_(), height=height)
        return

    st_folium(fmap, height=height, use_container_width=True,
              returned_objects=[], key=key)


__all__ = [
    "base_map", "vessel_map", "vessel_map_plotly", "chokepoint_overview_map",
    "flight_map", "flight_track_map", "density_map", "render_folium",
    "FOLIUM_AVAILABLE", "ST_FOLIUM_AVAILABLE",
]

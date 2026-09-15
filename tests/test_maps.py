"""
Aircraft and vessel markers on the Plotly maps.

REGRESSION CONTEXT
------------------
On a Plotly map trace, marker.color only applies to the "circle" symbol.
Any other symbol is an icon from the basemap style's sprite, so the
heading-rotated triangles both maps used rendered black on carto-darkmatter
whatever colour was set:

  * Aircraft also sat on the amber intensity scale, whose low end is
    #1A1A1A - even with the colour applied, anything low or slow would have
    been near-black. Every aircraft is now one light blue.

  * Vessels were coloured by ship type (tankers amber, cargo cyan, passenger
    magenta), and every one of them drew black anyway.

Both maps now use circles, with heading moved into the hover card.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ui import maps


@pytest.fixture(autouse=True)
def _template():
    """style_figure references the "openterm" template, which app.py
    registers via apply_theme(). Tests never call that."""
    from ui.terminal_theme import _register_plotly_template
    _register_plotly_template()


def _states(**overrides):
    """A frame shaped like aviation.get_states_by_region output."""
    frame = pd.DataFrame({
        "icao24": ["abc123", "def456", "0a0b0c"],
        "callsign": ["SIA21", "BAW11", "UAE5"],
        "latitude": [1.35, 51.47, 25.25],
        "longitude": [103.99, -0.45, 55.36],
        "true_track": [90.0, 270.0, None],
        "altitude_ft": [0.0, 36000.0, None],
        "speed_kts": [12.0, 480.0, 250.0],
    })
    return frame.assign(**overrides)


def _aircraft(figure):
    return next(trace for trace in figure.data if trace.name == "Aircraft")


class TestFlightMapMarkers:
    def test_every_aircraft_is_light_blue(self):
        assert _aircraft(maps.flight_map(_states())).marker.color == maps.AIRCRAFT_COLOR

    def test_colour_does_not_depend_on_altitude_or_speed(self):
        # A parked aircraft is exactly the one the amber scale drew near-black.
        parked = _aircraft(maps.flight_map(_states(altitude_ft=0.0, speed_kts=0.0)))
        cruising = _aircraft(maps.flight_map(_states(altitude_ft=40000.0)))
        assert parked.marker.color == cruising.marker.color == maps.AIRCRAFT_COLOR

    def test_no_colour_scale_or_colour_bar(self):
        marker = _aircraft(maps.flight_map(_states())).marker
        assert marker.colorscale is None
        assert not marker.showscale

    def test_light_blue_reads_on_the_dark_basemap(self):
        """Bright against the basemap, blue-dominant, and not the data-text cyan."""
        from config import THEME

        red, green, blue = (int(maps.AIRCRAFT_COLOR.lstrip("#")[i:i + 2], 16) / 255
                            for i in (0, 2, 4))
        assert 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5
        assert blue >= green >= red
        assert maps.AIRCRAFT_COLOR.upper() != THEME.cyan.upper()

    def test_markers_are_circles_so_the_colour_applies(self):
        """
        A map trace only honours marker.color for "circle". Any other symbol
        is an icon from the basemap's sprite: the triangles this map used
        rendered black on carto-darkmatter even with the colour set.
        """
        marker = _aircraft(maps.flight_map(_states())).marker
        assert marker.symbol == "circle"
        assert marker.angle is None

    def test_heading_stays_in_the_hover_card(self):
        hover = list(_aircraft(maps.flight_map(_states())).hovertext)
        assert "Heading: 90°" in hover[0]

    def test_empty_frame_renders_a_notice_not_a_crash(self):
        figure = maps.flight_map(pd.DataFrame())
        assert not [trace for trace in figure.data if trace.name == "Aircraft"]


def _vessels():
    """A frame shaped like the maritime module's AIS vessel output."""
    return pd.DataFrame({
        "mmsi": [111, 222, 333, 444],
        "name": ["GULF TANKER", "BOX CARRIER", "RED SEA FERRY", ""],
        "latitude": [30.00, 30.10, 30.20, 30.30],
        "longitude": [32.50, 32.55, 32.60, 32.65],
        "ship_type": ["Tanker", "Cargo", "Passenger", None],
        "heading_deg": [10.0, 200.0, None, 90.0],
        "cog_deg": [12.0, 198.0, 45.0, 88.0],
        "sog_kts": [11.0, 14.5, 18.0, 0.0],
    })


def _vessel_trace(figure):
    return next(trace for trace in figure.data if trace.name == "Vessels")


class TestVesselMapMarkers:
    def test_markers_are_circles_so_type_colours_apply(self):
        marker = _vessel_trace(maps.vessel_map_plotly(_vessels())).marker
        assert marker.symbol == "circle"
        assert marker.angle is None

    def test_each_vessel_keeps_its_ship_type_colour(self):
        from config import THEME

        marker = _vessel_trace(maps.vessel_map_plotly(_vessels())).marker
        assert list(marker.color) == [THEME.amber, THEME.cyan, THEME.magenta,
                                      THEME.muted]

    def test_heading_and_course_are_in_the_hover_card(self):
        hover = list(_vessel_trace(maps.vessel_map_plotly(_vessels())).hovertext)
        assert "Heading: 10°" in hover[0]
        assert "Course: 12°" in hover[0]
        # No heading reported: the line is left out rather than shown as 0°.
        assert "Heading" not in hover[2]

    def test_corridor_outline_is_still_drawn(self):
        figure = maps.vessel_map_plotly(_vessels(), bbox=(29.9, 32.3, 30.5, 32.9))
        corridor = next(trace for trace in figure.data if trace.name == "Corridor")
        assert corridor.mode == "lines"

    def test_empty_frame_renders_a_notice_not_a_crash(self):
        figure = maps.vessel_map_plotly(pd.DataFrame())
        assert not [trace for trace in figure.data if trace.name == "Vessels"]

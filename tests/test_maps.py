"""
Aircraft markers on the live traffic map.

REGRESSION CONTEXT
------------------
Aircraft were coloured on the amber intensity scale, whose low end is
#1A1A1A - near-black on the carto-darkmatter basemap. Anything low or slow,
which includes every aircraft on the ground, was effectively invisible, and
the legend swatch rendered black. Every aircraft is now drawn in one light
blue that reads against the dark map.
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

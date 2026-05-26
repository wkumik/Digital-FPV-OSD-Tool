import unittest

from _widget_primitives import _format_value
from osd_decoder import extract_value, extract_gps_coords
from osd_parser import OsdFile, OsdFrame
from widgets import TelemetryFrame, Widget, _get_widget_value


def _place(rows, cols, *placements):
    """Build an OsdFrame from (row, col, [codes]) placements onto a blank grid."""
    grid = [0] * (rows * cols)
    for r, c, codes in placements:
        for i, code in enumerate(codes):
            grid[r * cols + c + i] = code
    return OsdFrame(0, 0, grid, cols, rows)


def speed_frame(index, time_ms, text, cols=5):
    digits = [ord(ch) for ch in text]
    grid = [0] * (cols - 1 - len(digits)) + digits + [0xA1]
    return OsdFrame(index, time_ms, grid, cols, 1)


class WidgetSmoothingTest(unittest.TestCase):
    def test_ardupilot_native_speed_decodes(self):
        frame = speed_frame(0, 0, "42")

        self.assertEqual(extract_value(frame, "osd_speed_kmh", "ArduPilot"), 42)

    def test_ardupilot_native_decimal_glyph_does_not_misalign(self):
        # Current "4.2 A" in ArduPilot's native glyphs: digit glyph 0xC4 ('4'),
        # decimal glyph 0xD2 ('.2'), then the current anchor 0x9A. The two-char
        # ".2" occupies a single cell and must not shift the digit run.
        grid = [0, 0, 0xC4, 0xD2, 0x9A]
        frame = OsdFrame(0, 0, grid, 5, 1)

        self.assertAlmostEqual(
            extract_value(frame, "osd_current_a", "ArduPilot"), 4.2, places=6)

    def test_zero_percent_uses_unsmoothed_current_frame_value(self):
        osd = OsdFile(
            frames=[
                speed_frame(0, 0, "0"),
                speed_frame(1, 100, "100"),
                speed_frame(2, 200, "200"),
            ],
            timestamps=[0, 100, 200],
            grid_cols=5,
            grid_rows=1,
        )
        telemetry = TelemetryFrame(
            osd_frame=osd.frames[0],
            firmware="ArduPilot",
            osd_file=osd,
            osd_time_ms=50,
        )

        self.assertEqual(telemetry.get_osd_visual_value("osd_speed_kmh", 0.0), 0.0)

    def test_alpha_beta_smoothing_uses_lookahead(self):
        osd = OsdFile(
            frames=[
                speed_frame(0, 0, "0"),
                speed_frame(1, 100, "100"),
                speed_frame(2, 200, "200"),
            ],
            timestamps=[0, 100, 200],
            grid_cols=5,
            grid_rows=1,
        )
        telemetry = TelemetryFrame(
            osd_frame=osd.frames[0],
            firmware="ArduPilot",
            osd_file=osd,
            osd_time_ms=50,
        )

        self.assertEqual(telemetry._lookahead_ms(3.0, 100.0), 405)
        self.assertGreater(telemetry.get_osd_visual_value("osd_speed_kmh", 3.0), 50.0)

    def test_widget_text_stays_raw_while_geometry_smooths(self):
        osd = OsdFile(
            frames=[
                speed_frame(0, 0, "0"),
                speed_frame(1, 100, "100"),
                speed_frame(2, 200, "200"),
            ],
            timestamps=[0, 100, 200],
            grid_cols=5,
            grid_rows=1,
        )
        telemetry = TelemetryFrame(
            osd_frame=osd.frames[0],
            firmware="ArduPilot",
            osd_file=osd,
            osd_time_ms=50,
        )
        widget = Widget(
            type="gauge",
            source="osd_speed_kmh",
            style={"smoothness": 3.0},
        )

        geom_value, text_value, _unit, fmt = _get_widget_value(telemetry, widget)

        # Geometry uses the smoothed/lookahead value...
        self.assertGreater(float(geom_value), 50.0)
        # ...but the numeric label stays on the exact current OSD frame value.
        self.assertEqual(float(text_value), 0.0)
        self.assertEqual(_format_value(text_value, fmt, fmt), "0")

    def test_disabled_smoothing_text_and_geometry_match_raw(self):
        osd = OsdFile(
            frames=[speed_frame(0, 0, "10"), speed_frame(1, 100, "90")],
            timestamps=[0, 100],
            grid_cols=5,
            grid_rows=1,
        )
        telemetry = TelemetryFrame(
            osd_frame=osd.frames[0],
            firmware="ArduPilot",
            osd_file=osd,
            osd_time_ms=50,
        )
        # No smoothness key -> default 0.0 -> no smoothing (back-compat).
        widget = Widget(type="gauge", source="osd_speed_kmh", style={})

        geom_value, text_value, _unit, _fmt = _get_widget_value(telemetry, widget)

        self.assertEqual(float(geom_value), 10.0)
        self.assertEqual(float(text_value), 10.0)

    def test_visual_value_falls_back_without_osd_file_or_time(self):
        frame = speed_frame(0, 0, "42")
        no_file = TelemetryFrame(osd_frame=frame, firmware="ArduPilot")
        no_time = TelemetryFrame(osd_frame=frame, firmware="ArduPilot",
                                 osd_file=OsdFile(frames=[frame], timestamps=[0],
                                                  grid_cols=5, grid_rows=1),
                                 osd_time_ms=None)

        # Both fall back to the exact current-frame value regardless of smoothness.
        self.assertEqual(no_file.get_osd_visual_value("osd_speed_kmh", 3.0), 42.0)
        self.assertEqual(no_time.get_osd_visual_value("osd_speed_kmh", 3.0), 42.0)

    def test_track_cache_is_per_osd_file(self):
        def make(values):
            frames = [speed_frame(i, i * 100, v) for i, v in enumerate(values)]
            return OsdFile(frames=frames,
                           timestamps=[i * 100 for i in range(len(values))],
                           grid_cols=5, grid_rows=1)

        slow = make(["0", "0", "0", "0"])
        fast = make(["50", "60", "70", "80"])
        t = 200

        slow_tf = TelemetryFrame(osd_frame=slow.frame_at_time(t),
                                 firmware="ArduPilot", osd_file=slow, osd_time_ms=t)
        fast_tf = TelemetryFrame(osd_frame=fast.frame_at_time(t),
                                 firmware="ArduPilot", osd_file=fast, osd_time_ms=t)

        slow_v = slow_tf.get_osd_visual_value("osd_speed_kmh", 3.0)
        fast_v = fast_tf.get_osd_visual_value("osd_speed_kmh", 3.0)

        # A separate OsdFile must not read another file's cached track.
        self.assertAlmostEqual(slow_v, 0.0, places=6)
        self.assertGreater(fast_v, slow_v)

    def test_track_cache_evicts_when_osd_file_collected(self):
        import gc

        from widgets import _VISUAL_TRACK_CACHE

        osd = OsdFile(
            frames=[speed_frame(0, 0, "10"), speed_frame(1, 100, "90")],
            timestamps=[0, 100],
            grid_cols=5,
            grid_rows=1,
        )
        file_id = id(osd)
        tf = TelemetryFrame(osd_frame=osd.frames[0], firmware="ArduPilot",
                            osd_file=osd, osd_time_ms=50)
        tf.get_osd_visual_value("osd_speed_kmh", 3.0)
        self.assertIn(file_id, _VISUAL_TRACK_CACHE)

        del osd, tf
        gc.collect()
        # The finalizer must drop the bucket so a reused id() can't hit it.
        self.assertNotIn(file_id, _VISUAL_TRACK_CACHE)

    def test_alpha_beta_visual_track_avoids_sparse_sample_bounce(self):
        osd = OsdFile(
            frames=[
                speed_frame(0, 0, "40"),
                speed_frame(1, 75, "38"),
                speed_frame(2, 150, "43"),
                speed_frame(3, 225, "52"),
                speed_frame(4, 300, "52"),
                speed_frame(5, 375, "50"),
            ],
            timestamps=[0, 75, 150, 225, 300, 375],
            grid_cols=5,
            grid_rows=1,
        )
        values = []
        for t in range(0, 260, 16):
            frame = osd.frame_at_time(t)
            telemetry = TelemetryFrame(
                osd_frame=frame,
                firmware="ArduPilot",
                osd_file=osd,
                osd_time_ms=t,
            )
            values.append(telemetry.get_osd_visual_value("osd_speed_kmh", 10.0))

        for prev, current, nxt in zip(values, values[1:], values[2:]):
            self.assertFalse(
                abs(current - prev) > 0.05
                and abs(nxt - current) > 0.05
                and (current - prev) * (nxt - current) < 0
            )


class BetaflightOsdDecodeTest(unittest.TestCase):
    """Regression coverage for the Betaflight glyph IDs and blink-bit masking
    (see the DJI HD .osd sample with raw lat/lon)."""

    def test_speed_uses_kph_glyph_not_inav(self):
        # "47<KPH>" — SYM_KPH 0x9E, digits to the left. INAV's inherited 0x88
        # would instead latch the artificial-horizon centre line.
        frame = _place(20, 53, (9, 14, [0x34, 0x37, 0x9E]))
        self.assertEqual(extract_value(frame, "osd_speed_kmh", "Betaflight"), 47)

    def test_altitude_anchors_on_altitude_glyph(self):
        # "<ALT>23<M>" — SYM_ALTITUDE 0x7F prefix, digits to the right.
        frame = _place(20, 53, (9, 36, [0x7F, 0x32, 0x33, 0x0C]))
        self.assertEqual(extract_value(frame, "osd_altitude_m", "Betaflight"), 23)

    def test_blinking_sat_count_decodes_through_attribute_bits(self):
        # "<SAT_L><SAT_R> 5" with blink bits set (0x2xx). 0x220 is a blanked
        # tens place; gap=1 steps over it to read the single-digit count.
        frame = _place(20, 53,
                       (0, 2, [0x21E, 0x21F, 0x220, 0x235]))
        self.assertEqual(extract_value(frame, "osd_sats", "Betaflight"), 5)

    def test_vspeed_and_power_decode(self):
        frame = _place(20, 53,
                       (10, 36, [0x75, 0x32, 0x2E, 0x39, 0x9F]),   # "<up>2.9<MPS>"
                       (15, 4,  [0x31, 0x34, 0x39, 0x57]))          # "149W"
        self.assertAlmostEqual(
            extract_value(frame, "osd_vspeed_ms", "Betaflight"), 2.9, places=6)
        self.assertEqual(extract_value(frame, "osd_power_w", "Betaflight"), 149)

    def test_extract_gps_coords_from_direction_glyphs(self):
        lat = [0x89] + [ord(c) for c in "50.0346578"]
        lon = [0x98] + [ord(c) for c in "19.9916804"]
        frame = _place(20, 53, (17, 3, lat), (18, 3, lon))
        pt = extract_gps_coords(frame)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt[0], 50.0346578, places=6)
        self.assertAlmostEqual(pt[1], 19.9916804, places=6)

    def test_extract_gps_coords_ignores_shallow_decimals(self):
        # Altitude "23.4<M>" must not be mistaken for a coordinate.
        frame = _place(20, 53, (9, 36, [0x7F, 0x32, 0x33, 0x2E, 0x34, 0x0C]))
        self.assertIsNone(extract_gps_coords(frame))


class MapTileUnderlayTest(unittest.TestCase):
    """Web-Mercator tile math + offline fallback for the map underlay. No
    network: these exercise projection and the cooldown short-circuit."""

    def test_pixel_projection_directions(self):
        import _widget_map_tiles as T
        x1, y1 = T._lonlat_to_pixel(50.0, 19.0, 14)
        x2, y2 = T._lonlat_to_pixel(50.0, 20.0, 14)   # further east → larger x
        x3, y3 = T._lonlat_to_pixel(51.0, 19.0, 14)   # further north → smaller y
        self.assertGreater(x2, x1)
        self.assertAlmostEqual(y1, y2, places=3)
        self.assertLess(y3, y1)

    def test_choose_zoom_fits_bbox(self):
        import _widget_map_tiles as T
        z = T._choose_zoom(50.03, 50.035, 19.987, 19.993, 600, 600, 19)
        xl, yt = T._lonlat_to_pixel(50.035, 19.987, z)
        xr, yb = T._lonlat_to_pixel(50.03, 19.993, z)
        self.assertLessEqual(xr - xl, 600)
        self.assertLessEqual(yb - yt, 600)

    def test_is_underlay_and_attribution(self):
        import _widget_map_tiles as T
        self.assertTrue(T.is_underlay("street"))
        self.assertTrue(T.is_underlay("satellite"))
        self.assertFalse(T.is_underlay("none"))
        self.assertFalse(T.is_underlay(""))
        self.assertIn("OpenStreetMap", T.attribution("street"))

    def test_offline_build_returns_none(self):
        import time as _time
        import _widget_map_tiles as T
        T._MEM_TILES.clear()
        T._FAILED.clear()
        saved = T._net_cooldown_until
        T._net_cooldown_until = _time.monotonic() + 1000.0   # simulate offline
        try:
            # Coords unlikely to be in the on-disk cache (open ocean).
            lats = [1.2340 + i * 1e-4 for i in range(5)]
            lons = [100.5670 + i * 1e-4 for i in range(5)]
            self.assertIsNone(T.build_basemap(lats, lons, "street", 256, 256))
        finally:
            T._net_cooldown_until = saved


if __name__ == "__main__":
    unittest.main()

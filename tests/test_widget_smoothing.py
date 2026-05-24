import unittest

from _widget_primitives import _format_value
from osd_decoder import extract_value
from osd_parser import OsdFile, OsdFrame
from widgets import TelemetryFrame, Widget, _get_widget_value


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


if __name__ == "__main__":
    unittest.main()

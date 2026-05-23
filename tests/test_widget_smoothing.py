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

    def test_widget_text_uses_smoothed_value_too(self):
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

        value, _unit, fmt = _get_widget_value(telemetry, widget)

        self.assertGreater(float(value), 50.0)
        self.assertNotEqual(_format_value(value, fmt, fmt), "0")

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

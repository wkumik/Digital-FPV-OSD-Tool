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

    def test_widget_visual_downsamples_above_100_percent(self):
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

        self.assertEqual(telemetry.get_osd_visual_value("osd_speed_kmh", 0.0), 50.0)
        self.assertAlmostEqual(
            telemetry.get_osd_visual_value("osd_speed_kmh", 3.0),
            31.25,
        )

    def test_widget_visual_value_keeps_text_value_exact(self):
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

        self.assertAlmostEqual(float(value), 31.25)
        self.assertEqual(_format_value(value, fmt, fmt), "0")


if __name__ == "__main__":
    unittest.main()

import unittest

from _player_common import rawvideo_cmd, rgba_from_rgb24, PIL_OK


class RawvideoCmdTest(unittest.TestCase):
    def test_extraction_no_hwaccel_single_frame(self):
        cmd = rawvideo_cmd("ffmpeg", "in.mp4", 1.5, 640, 360, vframes=1, hwaccel=False)
        self.assertEqual(cmd, [
            "ffmpeg", "-ss", "1.5", "-i", "in.mp4",
            "-vf", "scale=640:360",
            "-vframes", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-v", "quiet", "pipe:1",
        ])

    def test_playback_hwaccel_no_vframes(self):
        cmd = rawvideo_cmd("ffmpeg", "in.mp4", 0.0, 1280, 720, vframes=None, hwaccel=True)
        self.assertEqual(cmd, [
            "ffmpeg", "-hwaccel", "auto", "-ss", "0.0", "-i", "in.mp4",
            "-vf", "scale=1280:720",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-v", "quiet", "pipe:1",
        ])

    def test_color_vf_precedes_scale_joined_by_comma(self):
        cmd = rawvideo_cmd("ffmpeg", "in.mp4", 0.0, 100, 100,
                           color_vf="eq=gamma=1.2", vframes=1, hwaccel=False)
        vf_idx = cmd.index("-vf")
        self.assertEqual(cmd[vf_idx + 1], "eq=gamma=1.2,scale=100:100")


class RgbaFromRgb24Test(unittest.TestCase):
    def test_short_data_returns_none(self):
        self.assertIsNone(rgba_from_rgb24(b"", 2, 2))
        self.assertIsNone(rgba_from_rgb24(b"\x00" * 5, 2, 2))

    def test_exact_size_data_returns_rgba_image(self):
        if not PIL_OK:
            self.skipTest("PIL not installed")
        data = bytes(range(2 * 2 * 3))
        img = rgba_from_rgb24(data, 2, 2)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (2, 2))
        self.assertEqual(img.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()

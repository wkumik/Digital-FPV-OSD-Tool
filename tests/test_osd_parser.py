import struct
import unittest

from osd_parser import (GRID_COLS, GRID_ROWS, HEADER_SIZE,
                        _dims_credible, _parse_ruby, _ts_plausible)


def _ruby(cols, rows, n_frames, *, declare=None, truncate=0,
          ts_step=16, glyph=0x20, fc=b'BTFL'):
    """Build a synthetic Ruby-format .osd file.

    declare  — dims to write into header bytes 36-39 (None = leave zeroed,
               as a standard 53x20 recording does)
    truncate — bytes to lop off the end, simulating a recording cut off
               mid-write
    """
    hdr = bytearray(HEADER_SIZE)
    hdr[0:4] = fc
    if declare is not None:
        struct.pack_into('<HH', hdr, 36, *declare)

    cells = cols * rows
    body  = bytearray()
    for i in range(n_frames):
        body += struct.pack('<I', i * ts_step)
        body += struct.pack(f'<{cells}H', *([glyph] * cells))
    if truncate:
        del body[len(body) - truncate:]

    return bytes(hdr) + bytes(body)


class HeaderGridDetectionTest(unittest.TestCase):
    def test_undeclared_dims_use_the_default_grid(self):
        osd = _parse_ruby(_ruby(GRID_COLS, GRID_ROWS, 10))
        self.assertEqual((osd.grid_cols, osd.grid_rows), (GRID_COLS, GRID_ROWS))
        self.assertEqual(len(osd.frames), 10)

    def test_declared_dims_are_honoured_on_an_exact_fit(self):
        osd = _parse_ruby(_ruby(60, 22, 10, declare=(60, 22)))
        self.assertEqual((osd.grid_cols, osd.grid_rows), (60, 22))
        self.assertEqual(len(osd.frames), 10)

    def test_partial_final_frame_does_not_lose_the_declared_dims(self):
        # The regression this guards: a recording cut off mid-write used to
        # fail the exact-multiple check, fall back to 53x20 and misread every
        # frame in the file.
        raw = _ruby(60, 22, 10, declare=(60, 22), truncate=500)
        osd = _parse_ruby(raw)
        self.assertEqual((osd.grid_cols, osd.grid_rows), (60, 22))
        self.assertEqual(len(osd.frames), 9)          # stub frame dropped
        self.assertEqual(osd.frames[-1].time_ms, 8 * 16)

    def test_implausible_declared_dims_fall_back_to_the_default(self):
        # A real 53x20 file whose header bytes 36-39 claim 30x10.  Reading it
        # at that stride lands mid-glyph, so the "timestamps" jump by minutes.
        raw = bytearray(_ruby(GRID_COLS, GRID_ROWS, 10))
        struct.pack_into('<HH', raw, 36, 30, 10)
        osd = _parse_ruby(bytes(raw))
        self.assertEqual((osd.grid_cols, osd.grid_rows), (GRID_COLS, GRID_ROWS))
        self.assertEqual(len(osd.frames), 10)

    def test_out_of_range_dims_are_rejected(self):
        self.assertFalse(_dims_credible(_ruby(60, 22, 4, declare=(60, 22)), 8, 4))
        self.assertFalse(_dims_credible(_ruby(60, 22, 4, declare=(60, 22)), 200, 22))

    def test_declared_dims_larger_than_the_file_are_rejected(self):
        raw = _ruby(60, 22, 1, declare=(60, 22))
        self.assertFalse(_dims_credible(raw, 120, 60))


class TsPlausibleTest(unittest.TestCase):
    def _stride(self, cols, rows):
        return 4 + cols * rows * 2

    def test_accepts_a_real_timestamp_track(self):
        raw = _ruby(60, 22, 8, declare=(60, 22))
        self.assertTrue(_ts_plausible(raw, self._stride(60, 22), 8))

    def test_rejects_a_stride_that_reads_glyph_data_as_timestamps(self):
        raw = _ruby(60, 22, 8, declare=(60, 22))
        self.assertFalse(_ts_plausible(raw, self._stride(53, 20), 8))

    def test_rejects_a_track_that_never_advances(self):
        raw = _ruby(60, 22, 8, declare=(60, 22), ts_step=0)
        self.assertFalse(_ts_plausible(raw, self._stride(60, 22), 8))

    def test_rejects_implausibly_large_gaps(self):
        raw = _ruby(60, 22, 8, declare=(60, 22), ts_step=60_000)
        self.assertFalse(_ts_plausible(raw, self._stride(60, 22), 8))

    def test_rejects_backwards_timestamps(self):
        raw = bytearray(_ruby(60, 22, 8, declare=(60, 22)))
        stride = self._stride(60, 22)
        struct.pack_into('<I', raw, HEADER_SIZE + 3 * stride, 0)
        self.assertFalse(_ts_plausible(bytes(raw), stride, 8))

    def test_single_frame_cannot_be_corroborated(self):
        raw = _ruby(60, 22, 1, declare=(60, 22))
        self.assertFalse(_ts_plausible(raw, self._stride(60, 22), 1))


if __name__ == '__main__':
    unittest.main()

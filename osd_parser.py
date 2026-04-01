# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
osd_parser.py  –  Parse VueOSD .osd binary files.

FORMAT (confirmed from Ruby firmware source: rx_video_recording_data.cpp,
function rx_video_recording_data_add_osd_frame):

  Header (40 bytes, written once on first call):
    bytes 0-3:   FC type string null-terminated:
                   "BTFL" = Betaflight
                   "INAV" = INAV
                   "PITL" = PitLab
                   "ARDU" = ArduPilot
    bytes 4-39:  null padding

  Each OSD frame (2124 bytes, written every MSP screen update):
    bytes 0-3:   u32 timestamp_ms  (g_TimeNow - s_uTimeStartedRecordingOSDData)
    bytes 4-...: DEFAULT_MSPOSD_RECORDING_COLS × DEFAULT_MSPOSD_RECORDING_ROWS
                 = 53 × 20 = 1060 u16 MSP char codes  (little-endian, 2120 bytes)

  Each frame is a COMPLETE INDEPENDENT SNAPSHOT of mspState.uScreenChars.
  There is NO differential encoding.
    char_code == 0    → empty / transparent cell
    char_code == 0x20 → space (explicitly blank)
    char_code != 0    → draw this MSP glyph

  Timestamp sync: for video position T_ms, find the latest OSD frame
  whose timestamp ≤ T_ms (binary search on timestamps list).

  Grid → pixel mapping (auto-fit, horizontally centred):
    scale = video_h / (GRID_ROWS * tile_h)
    x_off = (video_w − GRID_COLS * tile_w * scale) / 2
    y_off = 0
"""

from __future__ import annotations
import struct, re, bisect
from dataclasses import dataclass, field
from typing import Optional, List

HEADER_SIZE     = 40
GRID_COLS       = 53       # default; overridden per-file when header supplies dims
GRID_ROWS       = 20
CHARS_PER_FRAME = GRID_COLS * GRID_ROWS   # 1060
FRAME_SIZE      = 4 + CHARS_PER_FRAME * 2 # 2124  (u32 ts + 1060×u16)

FC_TYPES: dict[bytes, str] = {
    b'BTFL': 'Betaflight',
    b'INAV': 'INAV',
    b'PITL': 'PitLab',
    b'ARDU': 'ArduPilot',
}


@dataclass
class FlightStats:
    fc_type:        Optional[str]   = None
    total_arm_time: Optional[str]   = None
    min_battery_v:  Optional[float] = None
    min_rssi_pct:   Optional[int]   = None
    max_current_a:  Optional[float] = None
    used_mah:       Optional[int]   = None
    efficiency:     Optional[str]   = None   # e.g. "45 mAh/km"
    blackbox_pct:   Optional[str]   = None


@dataclass
class OsdFrame:
    """Complete snapshot of the MSP OSD screen at this timestamp."""
    index:     int
    time_ms:   int
    grid:      List[int]   # flat; 0 = transparent
    grid_cols: int = GRID_COLS   # actual columns for this frame

    def char_at(self, row: int, col: int) -> int:
        return self.grid[row * self.grid_cols + col]

    def non_empty(self) -> list[tuple[int, int, int]]:
        """Return [(row, col, char_code), ...] for all visible (non-zero) cells."""
        cols = self.grid_cols
        return [(i // cols, i % cols, c)
                for i, c in enumerate(self.grid) if c != 0]


@dataclass
class OsdFile:
    stats:      FlightStats       = field(default_factory=FlightStats)
    frames:     List[OsdFrame]    = field(default_factory=list)
    timestamps: List[int]         = field(default_factory=list)  # ms, ascending
    grid_cols:  int               = GRID_COLS
    grid_rows:  int               = GRID_ROWS

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> int:
        return self.timestamps[-1] if self.timestamps else 0

    def frame_at_index(self, index: int) -> Optional[OsdFrame]:
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def frame_at_time(self, time_ms: int) -> Optional[OsdFrame]:
        """Return the most recent OSD frame at or before time_ms."""
        if not self.timestamps:
            return None
        # bisect_right gives insertion point AFTER any matches.
        # Subtract 1 to get the last frame whose timestamp <= time_ms.
        idx = bisect.bisect_right(self.timestamps, time_ms) - 1
        idx = max(0, min(idx, len(self.frames) - 1))
        return self.frames[idx]


# ── Stats extraction from first (stats-screen) OSD frame ──────────────────────

def _clean(s: str) -> str:
    return re.sub(r'[^\x20-\x7E.]', '', s).strip()

def _extract_stats(frame: OsdFrame, grid_cols: int = GRID_COLS,
                    grid_rows: int = GRID_ROWS) -> FlightStats:
    """Read flight stats text from the first OSD frame (post-flight stats screen)."""
    s = FlightStats()

    def row_text(r: int) -> str:
        row = frame.grid[r * grid_cols:(r + 1) * grid_cols]
        return ''.join(chr(c) if 32 <= c < 127 else ' ' for c in row)

    def after_colon(line: str) -> str:
        idx = line.find(':')
        return line[idx + 1:].strip() if idx >= 0 else ''

    for r in range(grid_rows):
        line = row_text(r)
        if ('TOTAL' in line and 'ARM' in line) or \
           ('FLY'   in line and 'TIME' in line) or \
           ('FLIGHT' in line and 'TIME' in line):
            s.total_arm_time = _clean(after_colon(line)) or None
        elif 'MIN' in line and 'BATTERY' in line:
            try:
                s.min_battery_v = float(_clean(after_colon(line)).split()[0])
            except Exception:
                pass
        elif 'MIN' in line and 'RSSI' in line:
            try:
                s.min_rssi_pct = int(
                    float(_clean(after_colon(line)).replace('%', '').split()[0]))
            except Exception:
                pass
        elif 'CURRENT' in line and 'MIN' not in line:
            try:
                raw = _clean(after_colon(line)).split()[0].rstrip('aA')
                s.max_current_a = round(float(raw), 2)
            except Exception:
                pass
        elif ('USED' in line and 'MAH' in line) or \
             ('USED' in line and 'CAPACITY' in line):
            try:
                s.used_mah = int(float(_clean(after_colon(line)).split()[0]))
            except Exception:
                pass
        elif 'EFF' in line:
            s.efficiency = _clean(after_colon(line)) or None
        elif 'BLACKBOX' in line:
            s.blackbox_pct = _clean(after_colon(line)) or None
    return s


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_osd(path: str) -> OsdFile:
    with open(path, 'rb') as f:
        raw = f.read()

    if len(raw) < HEADER_SIZE:
        raise ValueError("File too small to be a valid OSD file")

    fc_tag  = raw[:4]
    fc_type = FC_TYPES.get(fc_tag, fc_tag.decode('ascii', errors='replace').rstrip('\x00'))
    if not fc_type or fc_type not in FC_TYPES.values():
        # Accept any 4-char ASCII tag — other systems may use different strings
        fc_type = fc_tag.decode('ascii', errors='replace').rstrip('\x00') or 'Unknown'

    # ── Detect grid dimensions from header ────────────────────────────────
    # WTFOS / DJI O3 recordings store grid size at header bytes 36-39 as
    # two little-endian u16 values (cols, rows).  The recording source tag
    # (e.g. "DJO3") appears at bytes 32-35.  When present and valid, these
    # override the default 53×20 grid.
    grid_cols, grid_rows = GRID_COLS, GRID_ROWS
    if len(raw) >= 40:
        hdr_cols = struct.unpack_from('<H', raw, 36)[0]
        hdr_rows = struct.unpack_from('<H', raw, 38)[0]
        if 30 <= hdr_cols <= 120 and 10 <= hdr_rows <= 60:
            # Sanity-check: the file must divide evenly with these dimensions
            chars  = hdr_cols * hdr_rows
            fsize  = 4 + chars * 2
            remainder = (len(raw) - HEADER_SIZE) % fsize
            if remainder == 0 and (len(raw) - HEADER_SIZE) // fsize > 0:
                grid_cols, grid_rows = hdr_cols, hdr_rows

    chars_per_frame = grid_cols * grid_rows
    frame_size      = 4 + chars_per_frame * 2

    n_frames = (len(raw) - HEADER_SIZE) // frame_size
    if n_frames == 0:
        raise ValueError("OSD file contains no frames")

    osd  = OsdFile(grid_cols=grid_cols, grid_rows=grid_rows)
    fmt  = f'<{chars_per_frame}H'

    for i in range(n_frames):
        off   = HEADER_SIZE + i * frame_size
        ts_ms = struct.unpack_from('<I', raw, off)[0]
        grid  = list(struct.unpack_from(fmt, raw, off + 4))
        osd.frames.append(OsdFrame(index=i, time_ms=ts_ms, grid=grid,
                                   grid_cols=grid_cols))
        osd.timestamps.append(ts_ms)

    # Pull stats from first frame (FC shows post-flight stats screen at start)
    osd.stats = _extract_stats(osd.frames[0], grid_cols, grid_rows)
    osd.stats.fc_type = fc_type
    return osd

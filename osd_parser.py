# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
osd_parser.py  –  Parse VueOSD .osd binary files.

Supports two formats:

  1. Ruby firmware format (DJI V2 / standard)
     Confirmed from rx_video_recording_data.cpp.

     Header (40 bytes):
       bytes 0-3:   FC type ("BTFL", "INAV", "ARDU", etc.)
       bytes 4-39:  null padding
       bytes 36-37: optional u16 grid_cols (DJI O3 / WTFOS extension)
       bytes 38-39: optional u16 grid_rows

     Frame (variable):
       bytes 0-3:   u32 timestamp_ms
       bytes 4-...: grid_cols×grid_rows u16 MSP char codes (row-major)

  2. MSPOSD format (DJI V1 / WTFOS — fpv-wtf/msp-osd)
     Magic: "MSPOSD\\0" at file start.

     Header v2 (22 bytes):
       0-6:   "MSPOSD\\0"
       7-8:   u16 version (2)
       9:     char_width  (active grid cols, e.g. 50)
       10:    char_height (active grid rows, e.g. 18)
       11:    font_width  (tile px, e.g. 24)
       12:    font_height (tile px, e.g. 36)
       13-14: u16 x_offset
       15-16: u16 y_offset
       17-21: char[5] font_variant ("BTFL", "INAV", "ARDU", etc.)

     Header v1 (18 bytes): same but font_variant is u8 enum
       (1=BTFL, 2=INAV, 3=ARDU, 4=ULTR, 5=QUIC)

     Frame (8 + size*2 bytes):
       0-3:   u32 frame_idx (video frame number, NOT ms)
       4-7:   u32 size (num u16 elements, typically 1320 = 60×22)
       8+:    u16[size] chars in COLUMN-MAJOR order (index = row + 22*col)
"""

from __future__ import annotations
import struct, re, bisect
from dataclasses import dataclass, field
from typing import Optional, List

# ── Ruby (standard) format constants ──────────────────────────────────────────

HEADER_SIZE     = 40
GRID_COLS       = 53       # default; overridden per-file when header supplies dims
GRID_ROWS       = 20
CHARS_PER_FRAME = GRID_COLS * GRID_ROWS   # 1060
FRAME_SIZE      = 4 + CHARS_PER_FRAME * 2 # 2124  (u32 ts + 1060×u16)

# ── MSPOSD format constants ──────────────────────────────────────────────────

MSPOSD_MAGIC        = b'MSPOSD\x00'
MSPOSD_MAX_COLS     = 60
MSPOSD_MAX_ROWS     = 22
MSPOSD_CELLS        = MSPOSD_MAX_COLS * MSPOSD_MAX_ROWS   # 1320
MSPOSD_FRAME_HEADER = 8    # u32 frame_idx + u32 size
MSPOSD_FRAME_SIZE   = MSPOSD_FRAME_HEADER + MSPOSD_CELLS * 2  # 2648

MSPOSD_V1_VARIANTS  = {1: 'Betaflight', 2: 'INAV', 3: 'ArduPilot',
                        4: 'KISS Ultra', 5: 'Quicksilver'}

FC_TYPES: dict[bytes, str] = {
    b'BTFL': 'Betaflight',
    b'INAV': 'INAV',
    b'PITL': 'PitLab',
    b'ARDU': 'ArduPilot',
    b'ULTR': 'KISS Ultra',
    b'QUIC': 'Quicksilver',
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
    grid:      List[int]   # flat row-major; 0 = transparent
    grid_cols: int = GRID_COLS
    grid_rows: int = GRID_ROWS

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
        idx = bisect.bisect_right(self.timestamps, time_ms) - 1
        idx = max(0, min(idx, len(self.frames) - 1))
        return self.frames[idx]


# ── Stats extraction from first (stats-screen) OSD frame ──────────────────────

def _clean(s: str) -> str:
    return re.sub(r'[^\x20-\x7E.]', '', s).strip()

def _extract_stats(frame: OsdFrame) -> FlightStats:
    """Read flight stats text from the first OSD frame (post-flight stats screen)."""
    s = FlightStats()
    cols = frame.grid_cols
    rows = frame.grid_rows

    def row_text(r: int) -> str:
        row = frame.grid[r * cols:(r + 1) * cols]
        return ''.join(chr(c) if 32 <= c < 127 else ' ' for c in row)

    def after_colon(line: str) -> str:
        idx = line.find(':')
        return line[idx + 1:].strip() if idx >= 0 else ''

    for r in range(rows):
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


# ── MSPOSD parser ────────────────────────────────────────────────────────────

def _parse_msposd(raw: bytes, video_fps: float = 60.0) -> OsdFile:
    """Parse an MSPOSD-format .osd file (WTFOS / DJI V1)."""
    if len(raw) < 18:
        raise ValueError("File too small for MSPOSD header")

    version = struct.unpack_from('<H', raw, 7)[0]

    if version == 1:
        hdr_size = 18
        char_w   = raw[9]
        char_h   = raw[10]
        font_w   = raw[11]
        font_h   = raw[12]
        x_off    = struct.unpack_from('<H', raw, 13)[0]
        y_off    = struct.unpack_from('<H', raw, 15)[0]
        variant_enum = raw[17]
        fc_type = MSPOSD_V1_VARIANTS.get(variant_enum, 'Unknown')
    elif version >= 2:
        hdr_size = 22
        if len(raw) < hdr_size:
            raise ValueError("File too small for MSPOSD v2 header")
        char_w   = raw[9]
        char_h   = raw[10]
        font_w   = raw[11]
        font_h   = raw[12]
        x_off    = struct.unpack_from('<H', raw, 13)[0]
        y_off    = struct.unpack_from('<H', raw, 15)[0]
        fc_tag   = raw[17:22].rstrip(b'\x00')
        fc_type  = FC_TYPES.get(fc_tag, fc_tag.decode('ascii', errors='replace'))
    else:
        raise ValueError(f"Unsupported MSPOSD version: {version}")

    # Validate grid dimensions
    if char_w == 0 or char_h == 0:
        raise ValueError(f"Invalid MSPOSD grid: {char_w}x{char_h}")

    data     = raw[hdr_size:]
    data_len = len(data)
    osd      = OsdFile(grid_cols=char_w, grid_rows=char_h)

    # Parse variable-size frames
    off = 0
    frame_idx = 0
    while off + MSPOSD_FRAME_HEADER <= data_len:
        vidframe = struct.unpack_from('<I', data, off)[0]
        size     = struct.unpack_from('<I', data, off + 4)[0]
        frame_bytes = size * 2
        if off + MSPOSD_FRAME_HEADER + frame_bytes > data_len:
            break  # truncated final frame

        # Read column-major uint16 chars
        col_major = list(struct.unpack_from(f'<{size}H', data,
                                            off + MSPOSD_FRAME_HEADER))

        # Convert column-major (index = row + MAX_ROWS * col) to row-major
        # Only keep the active region (char_w x char_h)
        flat = [0] * (char_w * char_h)
        for col in range(min(char_w, MSPOSD_MAX_COLS)):
            for row in range(min(char_h, MSPOSD_MAX_ROWS)):
                src_idx = row + MSPOSD_MAX_ROWS * col
                if src_idx < size:
                    flat[row * char_w + col] = col_major[src_idx]

        time_ms = int(vidframe * 1000.0 / video_fps) if video_fps > 0 else 0
        osd.frames.append(OsdFrame(index=frame_idx, time_ms=time_ms,
                                   grid=flat, grid_cols=char_w,
                                   grid_rows=char_h))
        osd.timestamps.append(time_ms)

        off += MSPOSD_FRAME_HEADER + frame_bytes
        frame_idx += 1

    if not osd.frames:
        raise ValueError("MSPOSD file contains no frames")

    # Sort frames by timestamp — the first frame (stats screen) may have a
    # higher frame_idx than the flight data that follows it.
    osd.frames.sort(key=lambda f: f.time_ms)
    osd.timestamps.sort()
    # Re-index after sort
    for i, fr in enumerate(osd.frames):
        fr.index = i

    osd.stats = _extract_stats(osd.frames[0])
    osd.stats.fc_type = fc_type
    return osd


# ── Ruby format parser ────────────────────────────────────────────────────────

def _parse_ruby(raw: bytes) -> OsdFile:
    """Parse a Ruby-firmware-format .osd file (DJI V2 / standard)."""
    if len(raw) < HEADER_SIZE:
        raise ValueError("File too small to be a valid OSD file")

    fc_tag  = raw[:4]
    fc_type = FC_TYPES.get(fc_tag, fc_tag.decode('ascii', errors='replace').rstrip('\x00'))
    if not fc_type or fc_type not in FC_TYPES.values():
        fc_type = fc_tag.decode('ascii', errors='replace').rstrip('\x00') or 'Unknown'

    # ── Detect grid dimensions from header ────────────────────────────────
    # WTFOS / DJI O3 recordings store grid size at header bytes 36-39 as
    # two little-endian u16 values (cols, rows).  The recording source tag
    # (e.g. "DJO3") appears at bytes 32-35.  When present and valid, these
    # override the default 53x20 grid.
    grid_cols, grid_rows = GRID_COLS, GRID_ROWS
    if len(raw) >= 40:
        hdr_cols = struct.unpack_from('<H', raw, 36)[0]
        hdr_rows = struct.unpack_from('<H', raw, 38)[0]
        if 30 <= hdr_cols <= 120 and 10 <= hdr_rows <= 60:
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
                                   grid_cols=grid_cols, grid_rows=grid_rows))
        osd.timestamps.append(ts_ms)

    osd.stats = _extract_stats(osd.frames[0])
    osd.stats.fc_type = fc_type
    return osd


# ── Main entry point ─────────────────────────────────────────────────────────

def parse_osd(path: str, video_fps: float = 60.0) -> OsdFile:
    with open(path, 'rb') as f:
        raw = f.read()

    # Auto-detect format by magic bytes
    if raw[:7] == MSPOSD_MAGIC:
        return _parse_msposd(raw, video_fps)

    return _parse_ruby(raw)

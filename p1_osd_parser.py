"""
BetaFPV P1 OSD Parser
=====================
Extracts OSD character data embedded in H.264 SEI NAL units inside P1 MP4 files.

FORMAT NOTES (reverse-engineered from Video122.mp4):
  Container : MP4, single H.264 track, no audio
  OSD embed : H.264 SEI NAL type 6 → SEI payload type 5 (user_data_unregistered)
  UUID      : bde945dc-b748-d9e6-20d8-2c96efee23d9
  Grid      : 50 cols × 18 rows  (Walksnail Avatar HD compatible)
  Update    : ~16 Hz (one SEI per video frame for first ~10 frames, then every 3-5 frames)

  After the 16-byte UUID, the SEI user_data payload is:
    Every data byte is separated by 0xFF delimiter bytes (strip them to get clean stream).

  Clean stream layout:
    Byte 0:  segment count  (number of changed regions this frame)
    Then repeating segments, each starting with the 2-byte magic [0xb6][0x03]:
      [0xb6][0x03]   — segment header
      [row]          — grid row  (0-based, 0 = top)
      [col]          — grid column (0-based, 0 = left)
      [chars ...]    — Betaflight MAX7456 font character bytes,
                       runs until the next [0xb6][0x03] header or end of stream

  Character encoding: Betaflight MAX7456 font (same as used by .osd files)

  Frame 0 SEI is a small config frame (~30 bytes clean) and can be skipped.
  It contains a [0xb6][0x03] segment with "ARMED" arming status — decode normally.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os

# ── Constants ─────────────────────────────────────────────────────────────────

P1_SEI_UUID = bytes.fromhex("bde945dcb748d9e620d82c96efee23d9")

P1_GRID_COLS = 50
P1_GRID_ROWS = 18

_SEGMENT_MAGIC = (0xb6, 0x03)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class P1OsdFrame:
    """One decoded OSD frame from the P1 SEI stream."""
    time_ms: int                           # timestamp in milliseconds
    frame_index: int                       # video frame number (0-based)
    # Full grid snapshot — accumulated across frames (only changed cells update)
    grid: List[bytearray] = field(default_factory=lambda:
        [bytearray(P1_GRID_COLS) for _ in range(P1_GRID_ROWS)])

    def get_text(self) -> str:
        """Return the full OSD grid as a human-readable string (ASCII only)."""
        lines = []
        for row in self.grid:
            line = ''.join(chr(b) if 0x20 <= b < 0x7F else ' ' for b in row)
            lines.append(line.rstrip())
        return '\n'.join(lines)


@dataclass
class P1OsdData:
    """Complete parsed OSD data extracted from a P1 MP4 file."""
    frames: List[P1OsdFrame]
    grid_cols: int = P1_GRID_COLS
    grid_rows: int = P1_GRID_ROWS
    video_fps: float = 60.0
    duration_ms: int = 0

    def frame_at(self, time_ms: int) -> Optional[P1OsdFrame]:
        """Return the most recent OSD frame at or before the given time."""
        best = None
        for f in self.frames:
            if f.time_ms <= time_ms:
                best = f
            else:
                break
        return best


# ── MP4 / NAL parsing ────────────────────────────────────────────────────────

def _read_mp4_boxes(data: bytes, offset: int = 0, end: int = None) -> List[Tuple]:
    """Yield (offset, size, box_type) for top-level MP4 boxes."""
    if end is None:
        end = len(data)
    boxes = []
    while offset < end - 8:
        if offset + 8 > len(data):
            break
        size = struct.unpack_from('>I', data, offset)[0]
        box_type = data[offset+4:offset+8].decode('ascii', errors='replace')
        if size == 0:
            size = end - offset
        elif size == 1:
            if offset + 16 > len(data):
                break
            size = struct.unpack_from('>Q', data, offset+8)[0]
        if size < 8:
            break
        boxes.append((offset, size, box_type))
        offset += size
    return boxes


def _collect_sei_payloads(mdat: bytes) -> List[Tuple[int, bytes]]:
    """
    Walk AVC length-prefixed NAL units in mdat, collect SEI type-5 payloads
    that carry the P1 OSD UUID.
    Returns list of (frame_number, user_data_bytes).
    """
    results = []
    offset = 0
    frame_num = 0
    data_len = len(mdat)

    while offset < data_len - 5:
        if offset + 4 > data_len:
            break
        nal_len = struct.unpack_from('>I', mdat, offset)[0]
        if nal_len == 0 or offset + 4 + nal_len > data_len:
            offset += 1
            continue

        nal_header = mdat[offset + 4]
        nal_type   = nal_header & 0x1F

        if nal_type not in (1, 5, 6, 7, 8):
            offset += 1
            continue

        if nal_type in (1, 5):   # coded slice / IDR → increment frame counter
            frame_num += 1

        if nal_type == 6:        # SEI NAL
            payload = mdat[offset + 5 : offset + 4 + nal_len]
            if len(payload) >= 22:
                user_data = _parse_sei_user_data(payload)
                if user_data is not None:
                    results.append((frame_num, user_data))

        offset += 4 + nal_len

    return results


def _parse_sei_user_data(payload: bytes) -> Optional[bytes]:
    """
    Parse an H.264 SEI NAL payload.  Returns user_data bytes if the SEI is
    type 5 (user_data_unregistered) with the P1 UUID, else None.
    """
    pos = 0
    if pos >= len(payload):
        return None

    # SEI payload type (may be multi-byte: sum 0xFF bytes then final byte)
    sei_type = 0
    while pos < len(payload) and payload[pos] == 0xFF:
        sei_type += 255
        pos += 1
    if pos >= len(payload):
        return None
    sei_type += payload[pos]; pos += 1

    # SEI payload size
    sei_size = 0
    while pos < len(payload) and payload[pos] == 0xFF:
        sei_size += 255
        pos += 1
    if pos >= len(payload):
        return None
    sei_size += payload[pos]; pos += 1

    if sei_type != 5:
        return None                         # not user_data_unregistered

    if pos + 16 > len(payload):
        return None
    uuid = payload[pos:pos+16]
    if uuid != P1_SEI_UUID:
        return None                         # different UUID → not P1 OSD

    pos += 16
    return payload[pos:]                    # return user_data portion


# ── Payload decoder ───────────────────────────────────────────────────────────

def _decode_payload(raw: bytes) -> List[Tuple[int, int, bytes]]:
    """
    Decode one P1 SEI user_data payload.
    Returns list of (row, col, char_bytes) segments.
    """
    # Strip all 0xFF delimiter bytes to get the clean data stream
    clean = bytes(b for b in raw if b != 0xFF)
    if len(clean) < 3:
        return []

    segments = []
    i = 1   # skip leading count byte

    while i < len(clean) - 2:
        if clean[i] == _SEGMENT_MAGIC[0] and clean[i+1] == _SEGMENT_MAGIC[1]:
            i += 2
            if i + 2 >= len(clean):
                break
            row = clean[i]; i += 1
            col = clean[i]; i += 1

            if row >= P1_GRID_ROWS or col >= P1_GRID_COLS:
                continue    # out-of-bounds — skip

            # Collect char bytes until next segment header or end
            chars = bytearray()
            while i < len(clean):
                if (i + 1 < len(clean)
                        and clean[i]   == _SEGMENT_MAGIC[0]
                        and clean[i+1] == _SEGMENT_MAGIC[1]):
                    break
                chars.append(clean[i])
                i += 1

            # Each run is framed as [bg_byte][char...][attr_byte].
            # Strip the leading background byte (always 0x00) and the trailing
            # colour-attribute byte so only real OSD character codes remain.
            # NOTE: use len() not bool() — b'\x00' is falsy but IS a valid "clear cell".
            if len(chars) >= 2:
                chars = chars[1:-1]   # drop first (bg) and last (attr/fg)
            if len(chars) > 0:
                segments.append((row, col, bytes(chars)))
        else:
            i += 1

    return segments


def _apply_segments(grid: List[bytearray],
                    segments: List[Tuple[int, int, bytes]]) -> None:
    """Write segment character bytes into the persistent OSD grid (in-place)."""
    for row, col, chars in segments:
        for j, ch in enumerate(chars):
            c = col + j
            if c < P1_GRID_COLS:
                grid[row][c] = ch


# ── Public API ────────────────────────────────────────────────────────────────

def detect_p1(mp4_path: str) -> bool:
    """
    Return True if the file is a BetaFPV P1 MP4 with embedded OSD.
    Fast check — reads only the first 8 MB and looks for the P1 UUID.
    """
    try:
        with open(mp4_path, 'rb') as f:
            chunk = f.read(8 * 1024 * 1024)
        return P1_SEI_UUID in chunk
    except OSError:
        return False


def parse_p1_osd(mp4_path: str) -> Optional[P1OsdData]:
    """
    Extract OSD data from a BetaFPV P1 MP4 file.

    Returns a P1OsdData object (list of timestamped OSD grid snapshots),
    or None if the file doesn't contain P1 OSD data.
    """
    with open(mp4_path, 'rb') as f:
        data = f.read()

    # Find mdat box
    boxes = _read_mp4_boxes(data)
    mdat_offset = mdat_size = None
    fps = 60.0
    duration_ms = 0

    for off, sz, btype in boxes:
        if btype == 'mdat':
            mdat_offset = off + 8   # skip box header
            mdat_size   = sz - 8
        if btype == 'moov':
            # Try to read fps and duration from mvhd
            try:
                moov_children = _read_mp4_boxes(data, off + 8, off + sz)
                for co, cs, ct in moov_children:
                    if ct == 'mvhd':
                        # mvhd v0: timescale at offset 12, duration at 16
                        ts  = struct.unpack_from('>I', data, co + 8 + 12)[0]
                        dur = struct.unpack_from('>I', data, co + 8 + 16)[0]
                        if ts:
                            duration_ms = int(dur * 1000 / ts)
                    if ct == 'trak':
                        # Try to find trak → mdia → mdhd for video fps
                        trak_children = _read_mp4_boxes(data, co + 8, co + cs)
                        for to, ts2, tt in trak_children:
                            if tt == 'mdia':
                                mdia_ch = _read_mp4_boxes(data, to + 8, to + ts2)
                                for mo, ms, mt in mdia_ch:
                                    if mt == 'mdhd':
                                        th = data[mo+8]  # version byte
                                        if th == 0:
                                            mts = struct.unpack_from('>I', data, mo+8+8)[0]
                                            if mts:
                                                fps = float(mts)  # rough — proper fps needs stts
            except Exception:
                pass

    if mdat_offset is None:
        return None

    mdat = data[mdat_offset : mdat_offset + mdat_size]
    raw_seis = _collect_sei_payloads(mdat)

    if not raw_seis:
        return None

    # Build OSD frames — P1 sends a FULL SNAPSHOT each SEI update (not deltas).
    # Cells absent from an SEI are genuinely empty (transparent) for that frame.
    # We reset to a blank grid for each SEI so old cells don't ghost.
    osd_frames: List[P1OsdFrame] = []

    for frame_num, raw_payload in raw_seis:
        segments = _decode_payload(raw_payload)
        if not segments:
            continue

        # Fresh blank grid for every frame — P1 is a full-snapshot protocol
        snapshot = [bytearray(P1_GRID_COLS) for _ in range(P1_GRID_ROWS)]
        _apply_segments(snapshot, segments)
        time_ms  = int(frame_num / fps * 1000)

        osd_frames.append(P1OsdFrame(
            time_ms     = time_ms,
            frame_index = frame_num,
            grid        = snapshot,
        ))

    if not osd_frames:
        return None

    return P1OsdData(
        frames      = osd_frames,
        grid_cols   = P1_GRID_COLS,
        grid_rows   = P1_GRID_ROWS,
        video_fps   = fps,
        duration_ms = duration_ms,
    )


def p1_to_osd_file(p1: "P1OsdData") -> "OsdFile":
    """
    Convert a P1OsdData into a standard OsdFile so it drops straight into the
    existing rendering pipeline without any other changes.

    P1 grid  (50 cols × 18 rows) is written into the standard 53×20 flat grid.
    Cells outside the P1 area stay 0 (transparent).
    """
    from osd_parser import OsdFile, OsdFrame, FlightStats, GRID_COLS, GRID_ROWS

    osd = OsdFile()

    for i, p1f in enumerate(p1.frames):
        flat = [0] * (GRID_COLS * GRID_ROWS)
        for r, row_bytes in enumerate(p1f.grid):
            if r >= GRID_ROWS:
                break
            for c, ch in enumerate(row_bytes):
                if c >= GRID_COLS:
                    break
                flat[r * GRID_COLS + c] = ch
        osd.frames.append(OsdFrame(index=i, time_ms=p1f.time_ms, grid=flat))
        osd.timestamps.append(p1f.time_ms)

    # Extract flight stats from the last frame (P1 shows stats at end of flight)
    if osd.frames:
        from osd_parser import _extract_stats
        osd.stats = _extract_stats(osd.frames[-1])
        osd.stats.fc_type = "BetaFPV P1"

    return osd


def render_p1_frame_to_osd_chars(p1_frame: "P1OsdFrame") -> List[List[int]]:
    """
    Convert a P1OsdFrame grid into a 2D list of integer character codes
    compatible with the existing OSD rendering pipeline.
    Rows × Cols of uint8 Betaflight MAX7456 char indices.
    """
    return [list(row) for row in p1_frame.grid]


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/Video122.mp4'

    print(f"Analysing: {path}")
    print(f"Is P1 file: {detect_p1(path)}")
    print()

    osd = parse_p1_osd(path)
    if osd is None:
        print("No P1 OSD data found.")
        sys.exit(1)

    print(f"Extracted {len(osd.frames)} OSD frames")
    print(f"Grid: {osd.grid_cols} cols × {osd.grid_rows} rows")
    print(f"Duration: {osd.duration_ms} ms")
    print()

    print("=== First OSD frame ===")
    print(osd.frames[0].get_text())
    print()

    print("=== Last OSD frame ===")
    print(osd.frames[-1].get_text())

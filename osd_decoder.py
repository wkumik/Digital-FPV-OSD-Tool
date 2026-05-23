# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
osd_decoder.py - Reverse-parse numeric values from the MSP OSD glyph grid.

Each firmware draws its OSD as a 2D grid of MSP character codes. Unit icons
(km/h, m, V, A, %, m/s, etc.) live in fixed glyph slots. To extract a value
we:

    1. Scan the grid for the unit-icon glyph (the 'anchor').
    2. Walk into the adjacent cells and read digit glyphs (0x30-0x39, plus
       '.' and '-').
    3. Parse the joined digits as a float.

Failure is silent: extract_value() returns None when the anchor isn't found
or the adjacent cells don't look like digits. Widgets observing a None value
hide themselves.

Symbol IDs differ between firmware. Tables here are best-effort starting
points for current versions of:

    Betaflight ~4.4+
    INAV ~6.x+
    ArduPilot

If a specific font pack or firmware version uses different anchors, the
right place to adjust is the FW_TABLES dict below.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, Mapping


# Symbol-table entry shape:
#   (anchor_glyph_id, side, max_digits, scale, max_gap, allow_separator)
#
# anchor_glyph_id - the byte we scan for in the grid.
# side            - "left"  : digits sit BEFORE the anchor (most common -
#                             '123K' where K is the km/h icon).
#                   "right" : digits sit AFTER the anchor (e.g. icon-prefix
#                             layouts).
# max_digits      - upper bound on how far we walk reading digits.
# scale           - multiply parsed value by this (e.g. 0.1 if the firmware
#                   stores tenths without a decimal point).
# max_gap         - number of blank cells (0x00 / 0x20) we tolerate between
#                   the anchor icon and the start of the digit run. Most
#                   anchors butt directly against the digits (gap=0); INAV's
#                   throttle layout draws "T  50%" with a space between the
#                   icon and the number, so it needs gap=2.
# allow_separator - when True, the walker tolerates ONE unknown glyph between
#                   two digit runs and treats it as a decimal point. Needed
#                   for fonts that draw '.' with a non-ASCII slim glyph that
#                   our digit table doesn't otherwise recognise (e.g. the
#                   voltage readout "3.70V" where the dot is a custom glyph).


# ----- INAV ------------------------------------------------------------------
#
# Values below match the symbol set in INAV's src/main/io/osd_symbols.h
# (current as of INAV 6/7). The km/h, m/s, ft/s, V, A, mAh, m, ft icons are
# stable across recent firmware releases.

_INAV_SYMBOLS: dict[str, tuple[int, str, int, float, int]] = {
    # Speed (km/h) - 0x88 is the km/h unit glyph in this INAV font.
    # Confirmed against three sample frames: 43, 59, 72 km/h.
    "osd_speed_kmh":   (0x88, "left", 4, 1.0, 0),

    # Altitude (m) - 0x76 is the metres suffix for altitude.
    # Confirmed across frames: 14m -> 361m -> 9m matches a climb-and-descend.
    "osd_altitude_m":  (0x76, "left", 5, 1.0, 0),

    # Home distance (m) - INAV draws ``[home_icon][direction][digits][0x7A]``,
    # so the only reliable adjacency is the trailing 0x7A suffix.
    "osd_home_dist":   (0x7A, "left", 5, 1.0, 0),

    # Link / GPS - prefix icons, digits sit RIGHT of the icon
    "osd_rssi":        (0x01, "right", 3, 1.0, 0),
    "osd_sats":        (0x09, "right", 3, 1.0, 0),

    # Capacity. 0x99 confirmed against real INAV footage as the mAh suffix
    # glyph (digit run sits LEFT of it).
    "osd_mah_drawn":   (0x99, "left", 6, 1.0, 0),

    # Throttle % - INAV draws ``<throttle_icon>  NN`` with a blank cell
    # between the icon and the digits, so we need a gap-tolerant anchor.
    # Confirmed across 4 sample frames: values 50, 56, 77, 71 (% throttle).
    "osd_throttle_pct": (0x95, "right", 3, 1.0, 2),

    # Voltage (V) - 0x1F is the volts anchor in this INAV font.
    # The display has NO explicit decimal point: the firmware writes integer
    # hundredths (e.g. 378 next to 'V' means 3.78V per cell, or 1610 means
    # 16.10V pack), so we scale by 0.01. max_digits=4 covers both the 3-digit
    # per-cell and the 4-digit pack readouts. allow_separator stays False
    # because the digit string never contains an explicit dot, and any stray
    # unknown glyph between digits would break the integer interpretation.
    "osd_field_1f":     (0x1F, "left",  4, 0.01, 0, False),
}


# ----- Betaflight ------------------------------------------------------------
#
# Betaflight and INAV share most unit-icon IDs in the same hex range. The few
# that diverge (e.g. battery icons used for percent bars) are not used here.

_BTFL_SYMBOLS: dict[str, tuple[int, str, int, float, int]] = dict(_INAV_SYMBOLS)


# ----- ArduPilot -------------------------------------------------------------
#
# ArduPilot ships its own osd font and symbol set; many of the same anchor
# IDs are used as INAV. Worth verifying against real footage.

_ARDU_SYMBOLS: dict[str, tuple[int, str, int, float, int]] = {
    **_INAV_SYMBOLS,
    "osd_speed_kmh":   (0xA1, "left", 4, 1.0, 0),
    "osd_altitude_m":  (0xB1, "left", 5, 1.0, 0),
    "osd_rssi":        (0x01, "right", 3, 1.0, 0),
    "osd_mah_drawn":   (0x07, "left", 6, 1.0, 0),
    "osd_current_a":   (0x9A, "left", 5, 1.0, 0),
    "osd_field_1f":    (0x06, "left", 5, 1.0, 0),
}


FW_TABLES: dict[str, dict[str, tuple[int, str, int, float, int]]] = {
    "INAV":       _INAV_SYMBOLS,
    "Betaflight": _BTFL_SYMBOLS,
    "ArduPilot":  _ARDU_SYMBOLS,
}


# All OSD-derived field keys, in display order. widgets.py advertises these
# in the Source dropdown.
OSD_FIELD_REGISTRY: list[tuple[str, str, str, str]] = [
    # (key, display name, unit string, default fmt)
    ("osd_speed_kmh",    "Speed (OSD)",        "km/h", "{:.0f}"),
    ("osd_altitude_m",   "Altitude (OSD)",     "m",    "{:.0f}"),
    ("osd_home_dist",    "Home distance",      "m",    "{:.0f}"),
    ("osd_rssi",         "RSSI (OSD)",         "",     "{:.0f}"),
    ("osd_sats",         "GPS sats (OSD)",     "",     "{:.0f}"),
    ("osd_mah_drawn",    "mAh drawn (OSD)",    "mAh",  "{:.0f}"),
    ("osd_current_a",    "Current (OSD)",      "A",    "{:.1f}"),
    ("osd_throttle_pct", "Throttle % (OSD)",   "%",    "{:.0f}"),
    ("osd_field_1f",     "Voltage (OSD)",      "V",    "{:.2f}"),
]


# Glyph codes that we treat as digits / decimal sign / minus when reading
# adjacent cells around an anchor. Betaflight, INAV, and ArduPilot fonts all
# map digits to ASCII 0x30-0x39.
_DIGIT_LO, _DIGIT_HI = 0x30, 0x39
_DOT, _MINUS = 0x2E, 0x2D


def _glyph_to_char(code: int,
                   extra: Optional[Mapping[int, str]] = None) -> Optional[str]:
    """Return the character we should treat ``code`` as inside a number run,
    or None if it isn't part of one (terminates the walk).

    ``extra`` is an optional override map (glyph_id -> character). It lets
    callers feed in font-specific slim digit mappings discovered via template
    matching - see ``build_slim_digit_map``. Standard ASCII digits / dot /
    minus always win even if ``extra`` tries to remap them.
    """
    if _DIGIT_LO <= code <= _DIGIT_HI:
        return chr(code)
    if code == _DOT:
        return "."
    if code == _MINUS:
        return "-"
    if 0xC0 <= code <= 0xC9:
        return str(code - 0xC0)
    if 0xD0 <= code <= 0xD9:
        return f".{code - 0xD0}"
    if extra is not None:
        return extra.get(code)
    return None


def _parse_run(digits: list[str], scale: float) -> Optional[float]:
    """Join digits/separators into a float, applying ``scale``. Returns None
    on ambiguity (multiple minus signs, minus not at start, multiple dots)."""
    if not digits:
        return None
    s = "".join(digits)
    minus_count = s.count("-")
    if minus_count > 1 or (minus_count == 1 and not s.startswith("-")):
        return None
    if s.count(".") > 1:
        return None
    if s in ("-", ".", "-."):
        return None
    try:
        return float(s) * scale
    except ValueError:
        return None


def _read_left(grid, row_base: int, anchor_col: int, max_digits: int,
               max_gap: int = 0,
               extra: Optional[Mapping[int, str]] = None,
               allow_separator: bool = False) -> list[str]:
    """Read digit cells walking BACKWARD from the cell at anchor_col - 1.

    ``max_gap`` blank cells (0x00 / 0x20) are tolerated between the anchor
    and the first digit; needed for layouts like "NN  <icon>" where a space
    sits between the value and its unit glyph.

    When ``allow_separator`` is True, one unknown glyph sitting between two
    digit runs is consumed as a decimal point. Lets fields like voltage
    ("3.70V") parse even when the font draws '.' as a custom slim glyph.
    """
    cc = anchor_col - 1
    # Skip up to max_gap blank cells before the digit run starts.
    skipped = 0
    while cc >= 0 and skipped < max_gap:
        code = grid[row_base + cc]
        if code == 0 or code == 0x20:
            cc -= 1
            skipped += 1
        else:
            break
    out: list[str] = []
    separator_used = False
    for _ in range(max_digits):
        if cc < 0:
            break
        ch = _glyph_to_char(grid[row_base + cc], extra)
        if ch is None:
            # Try treating this unknown cell as a decimal separator if we
            # already have at least one digit and the next cell to the left
            # is also a digit.
            if (allow_separator and not separator_used and out
                    and cc - 1 >= 0
                    and _glyph_to_char(grid[row_base + cc - 1], extra) is not None):
                out.insert(0, ".")
                separator_used = True
                cc -= 1
                continue
            break
        out.insert(0, ch)
        cc -= 1
    return out


def _read_right(grid, row_base: int, anchor_col: int, max_digits: int,
                cols: int, max_gap: int = 0,
                extra: Optional[Mapping[int, str]] = None,
                allow_separator: bool = False) -> list[str]:
    """Read digit cells walking FORWARD from the cell at anchor_col + 1.

    See ``_read_left`` for the role of ``max_gap`` and ``allow_separator``.
    """
    cc = anchor_col + 1
    skipped = 0
    while cc < cols and skipped < max_gap:
        code = grid[row_base + cc]
        if code == 0 or code == 0x20:
            cc += 1
            skipped += 1
        else:
            break
    out: list[str] = []
    separator_used = False
    for _ in range(max_digits):
        if cc >= cols:
            break
        ch = _glyph_to_char(grid[row_base + cc], extra)
        if ch is None:
            if (allow_separator and not separator_used and out
                    and cc + 1 < cols
                    and _glyph_to_char(grid[row_base + cc + 1], extra) is not None):
                out.append(".")
                separator_used = True
                cc += 1
                continue
            break
        out.append(ch)
        cc += 1
    return out


def extract_value(osd_frame, field_key: str,
                  firmware: str = "INAV",
                  slim_map: Optional[Mapping[int, str]] = None,
                  ) -> Optional[float]:
    """
    Return the numeric value for ``field_key`` extracted from the given
    OsdFrame, or None when the anchor glyph isn't present (or the adjacent
    cells don't look like a number).

    Some anchor glyphs (notably the artificial-horizon bars in the 0x80-0x88
    range) appear in many cells. We therefore try EVERY occurrence of the
    anchor across the whole grid and accept the first one whose adjacent
    cells actually parse as a number.

    ``firmware`` selects the symbol table; falls back to INAV if unknown.
    """
    if osd_frame is None:
        return None
    table = FW_TABLES.get(firmware) or _INAV_SYMBOLS
    info = table.get(field_key)
    if info is None:
        return None
    # 6-tuple as documented at top of file; max_gap/allow_separator default
    # to 0/False for legacy entries that haven't been converted yet.
    if len(info) >= 6:
        anchor, side, max_digits, scale, max_gap, allow_separator = info[:6]
    elif len(info) == 5:
        anchor, side, max_digits, scale, max_gap = info
        allow_separator = False
    else:
        anchor, side, max_digits, scale = info
        max_gap = 0
        allow_separator = False

    grid = osd_frame.grid
    cols = osd_frame.grid_cols

    for i, code in enumerate(grid):
        if code != anchor:
            continue
        r, c = divmod(i, cols)
        row_base = r * cols
        if side == "right":
            digits = _read_right(grid, row_base, c, max_digits, cols, max_gap,
                                 slim_map, allow_separator)
        else:
            digits = _read_left(grid, row_base, c, max_digits, max_gap,
                                slim_map, allow_separator)
        val = _parse_run(digits, scale)
        if val is not None:
            return val
    return None


# ----- Debug helper ---------------------------------------------------------

def debug_anchors(osd_frame, firmware: str = "INAV") -> dict[str, tuple[int, int] | None]:
    """
    For diagnostics: return {field_key: (row, col) | None} of where each
    known anchor was found in the grid. Useful when a widget is silently
    blank and you want to check whether the symbol table matches the font.
    """
    out: dict[str, tuple[int, int] | None] = {}
    if osd_frame is None:
        return out
    table = FW_TABLES.get(firmware) or _INAV_SYMBOLS
    grid = osd_frame.grid
    cols = osd_frame.grid_cols
    rows = osd_frame.grid_rows
    for key, (anchor, *_rest) in table.items():
        found: Optional[Tuple[int, int]] = None
        for r in range(rows):
            row_base = r * cols
            try:
                c = grid[row_base:row_base + cols].index(anchor)
                found = (r, c)
                break
            except ValueError:
                continue
        out[key] = found
    return out


def _glyph_repr(code: int,
                slim_map: Optional[Mapping[int, str]] = None) -> str:
    """Best-effort printable rendering of a single OSD cell.

    ASCII printables come through verbatim so letter-bearing strings (Plus
    Codes, "T 25%", "GPS 12") read normally. Non-ASCII glyphs - the unit
    icons, attitude bars, etc. - render as ``<XX>`` so their hex IDs stay
    legible inline. If ``slim_map`` resolves the glyph to a digit (font's
    slim / alt digit range), we show that digit instead so the cluster dump
    reads as a number, not a sea of hex bytes. Empty cells (0x00 / 0x20)
    become a space.
    """
    if code == 0 or code == 0x20:
        return " "
    if 0x21 <= code <= 0x7E:
        return chr(code)
    if slim_map is not None:
        ch = slim_map.get(code)
        if ch is not None:
            return ch
    return f"<{code:02X}>"


def find_cell_clusters(osd_frame,
                       slim_map: Optional[Mapping[int, str]] = None,
                       ) -> list[dict]:
    """
    Return every horizontal run of non-empty cells in the OSD grid. Unlike
    ``find_anchor_candidates`` (which only fires near digit runs), this
    surfaces multi-character text elements - Plus Codes, throttle strings,
    altitude home labels - that the digit-adjacency walker misses.

    Each cluster:
      {
        "row":   int,
        "col":   int,
        "len":   int,
        "codes": [int, ...],   # raw glyph IDs
        "text":  str,          # ASCII chars + <XX> for non-ASCII glyphs
        "hex":   str,          # space-separated hex of all codes
      }
    """
    out: list[dict] = []
    if osd_frame is None:
        return out
    grid = osd_frame.grid
    cols = osd_frame.grid_cols
    rows = osd_frame.grid_rows
    for r in range(rows):
        row_base = r * cols
        c = 0
        while c < cols:
            code = grid[row_base + c]
            if code == 0 or code == 0x20:
                c += 1
                continue
            start = c
            codes: list[int] = []
            while c < cols:
                code = grid[row_base + c]
                if code == 0 or code == 0x20:
                    break
                codes.append(code)
                c += 1
            out.append({
                "row":   r,
                "col":   start,
                "len":   len(codes),
                "codes": codes,
                "text":  "".join(_glyph_repr(x, slim_map) for x in codes),
                "hex":   " ".join(f"{x:02X}" for x in codes),
            })
    return out


def find_anchor_candidates(osd_frame) -> list[dict]:
    """
    Scan the grid for digit runs and report every non-digit glyph immediately
    adjacent to one. Those neighbour glyphs ARE the actual unit-icon IDs the
    firmware uses, regardless of which firmware/font is in play.

    Returns a list of dicts:
      {
        "anchor_hex": "0xA5",         # the candidate glyph id
        "side":       "left"|"right", # whether the digits sit before/after
        "row":        int,
        "col":        int,
        "digits":     "123",          # the digit string we read off
      }

    Sort the result by anchor frequency to see which IDs the OSD layout uses
    most often.
    """
    out: list[dict] = []
    if osd_frame is None:
        return out
    grid = osd_frame.grid
    cols = osd_frame.grid_cols
    rows = osd_frame.grid_rows
    for r in range(rows):
        row_base = r * cols
        c = 0
        while c < cols:
            ch = _glyph_to_char(grid[row_base + c])
            if ch is None:
                c += 1
                continue
            # Found the start of a digit run — walk forward to find its end.
            start = c
            chars = [ch]
            c += 1
            while c < cols:
                nxt = _glyph_to_char(grid[row_base + c])
                if nxt is None:
                    break
                chars.append(nxt)
                c += 1
            digit_str = "".join(chars)
            # Right-neighbour candidate
            if c < cols:
                rgl = grid[row_base + c]
                if rgl != 0 and rgl != 0x20:
                    out.append({
                        "anchor_hex": f"0x{rgl:02X}",
                        "side":       "left",   # digits sit LEFT of anchor
                        "row":        r,
                        "col":        c,
                        "digits":     digit_str,
                    })
            # Left-neighbour candidate
            if start - 1 >= 0:
                lgl = grid[row_base + start - 1]
                if lgl != 0 and lgl != 0x20:
                    out.append({
                        "anchor_hex": f"0x{lgl:02X}",
                        "side":       "right",  # digits sit RIGHT of anchor
                        "row":        r,
                        "col":        start - 1,
                        "digits":     digit_str,
                    })
    return out


# ----- Slim digit auto-detection --------------------------------------------
#
# INAV's font sheets pack a second "slim" digit set into the 0x80-0xFF range
# (used by tightly-spaced fields like amp / voltage / timer readouts). The
# exact glyph IDs differ per font pack. Rather than hand-roll a per-font
# table, we ask the loaded font itself: for every non-ASCII glyph, compare
# its bitmap against each ASCII digit glyph (0x30-0x39) via IoU on the alpha
# mask, and accept the match if it's both strong and unambiguous.
#
# Why this is safe to do automatically:
#   - The check is per-font, so a font without a slim digit set just produces
#     an empty map. Decoder behaviour is unchanged in that case.
#   - We require IoU >= 0.70 AND a margin >= 0.10 over the next-best digit,
#     so compound glyphs (horizon ladder with a "3" annotation, GPS icon with
#     "4" badge, etc.) don't accidentally register as digits.
#   - ASCII digits 0x30-0x39 are never overridden by the map.

_SLIM_MATCH_THRESHOLD = 0.70
_SLIM_MATCH_MARGIN    = 0.10

# Cache by font identity so we only template-match once per font load.
_SLIM_CACHE: dict[tuple, dict[int, str]] = {}


def _glyph_mask(font, code: int):
    """Return a binary (uint8 0/1) numpy mask of the glyph's foreground
    pixels, or None when the glyph is blank / unavailable.

    Foreground = any pixel with alpha > 32. If the font sheet has no alpha
    (rare), we fall back to non-black RGB.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    img = font.get_char(code)
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3:
        return None
    if arr.shape[2] >= 4:
        mask = (arr[:, :, 3] > 32)
    else:
        # Fallback: foreground is anything brighter than near-black.
        mask = (arr[:, :, :3].max(axis=2) > 32)
    if not mask.any():
        return None
    return mask.astype("uint8")


def _iou(a, b) -> float:
    """Intersection-over-union of two binary masks of identical shape."""
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    union = int((a | b).sum())
    return inter / union if union else 0.0


def build_slim_digit_map(font, scan_range=range(0x80, 0x100)) -> dict[int, str]:
    """Template-match each non-ASCII glyph against the font's ASCII digits.

    Returns ``{glyph_id: '0'..'9'}`` for every glyph in ``scan_range`` that
    visually matches one of the ASCII digit glyphs strongly and
    unambiguously.

    Result is cached per (font.name, tile_w, tile_h). Calling multiple times
    with the same font is free.
    """
    if font is None:
        return {}
    cache_key = (
        getattr(font, "name", ""),
        getattr(font, "tile_w", 0),
        getattr(font, "tile_h", 0),
        getattr(font, "n_cols", 1),
        id(getattr(font, "image", None)),
    )
    cached = _SLIM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digit_masks: list[tuple[str, "object"]] = []
    for d in range(_DIGIT_LO, _DIGIT_HI + 1):
        m = _glyph_mask(font, d)
        if m is not None:
            digit_masks.append((chr(d), m))
    if not digit_masks:
        _SLIM_CACHE[cache_key] = {}
        return {}

    # All digit masks share the same tile dims (same font), so candidate
    # masks line up pixel-for-pixel without resizing.
    out: dict[int, str] = {}
    for code in scan_range:
        # Skip ASCII range entirely - it's authoritative.
        if _DIGIT_LO <= code <= _DIGIT_HI:
            continue
        m = _glyph_mask(font, code)
        if m is None or m.shape != digit_masks[0][1].shape:
            continue
        best_ch, best_iou = "", 0.0
        second_iou = 0.0
        for ch, dm in digit_masks:
            score = _iou(m, dm)
            if score > best_iou:
                second_iou = best_iou
                best_iou, best_ch = score, ch
            elif score > second_iou:
                second_iou = score
        if (best_iou >= _SLIM_MATCH_THRESHOLD and
                (best_iou - second_iou) >= _SLIM_MATCH_MARGIN):
            out[code] = best_ch

    out = _extrapolate_slim_map(out)
    _SLIM_CACHE[cache_key] = out
    return out


def _extrapolate_slim_map(matched: dict[int, str]) -> dict[int, str]:
    """Fill in slim digit codes that the IoU matcher missed.

    INAV/Betaflight fonts pack their slim digit set as a sequential alphabet
    (e.g. 0xA1='0', 0xA2='1', ..., 0xAA='9'). The per-glyph IoU matcher
    often confidently matches only 3-4 of the 10 because slim '2'/'3'/'5'/etc
    are too narrow / curvy to score 0.70 against their ASCII counterparts.

    When the matches we DO have form a consistent 1:1 code/value progression
    (each +1 in glyph code maps to +1 in digit value), we extrapolate to
    fill the rest of the alphabet. Triggers per cluster, so two unrelated
    digit sets (low slim 0xA0-0xAA and high slim 0xB0-0xBA) don't merge.

    Conservative: only fires when stride is identical in code and value
    direction AND at least 3 points constrain the line. Falls through to
    the original map unchanged when matches don't fit the pattern.
    """
    if len(matched) < 3:
        return dict(matched)

    items = sorted(matched.items())
    # Group matched codes by proximity so a low-slim and high-slim digit set
    # don't accidentally pool. Same group while consecutive matched codes
    # are within MAX_GAP - enough for stride 3 across a 10-digit alphabet.
    MAX_GAP = 9
    groups: list[list[tuple[int, int]]] = []
    bucket: list[tuple[int, int]] = []
    for code, ch in items:
        try:
            v = int(ch)
        except ValueError:
            continue
        if not bucket or code - bucket[-1][0] <= MAX_GAP:
            bucket.append((code, v))
        else:
            groups.append(bucket)
            bucket = [(code, v)]
    if bucket:
        groups.append(bucket)

    out = dict(matched)
    for group in groups:
        if len(group) < 3:
            continue
        cs = [c for c, _ in group]
        vs = [v for _, v in group]
        d_codes = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
        d_vals  = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        if any(dc <= 0 for dc in d_codes):
            continue
        if any(dc != d_codes[0] for dc in d_codes):
            continue
        if any(dv != d_vals[0] for dv in d_vals):
            continue
        if d_codes[0] != d_vals[0]:
            # Code stride != value stride means we can't safely interpolate
            # between two matched codes (would need a non-integer ratio).
            continue
        # 1:1 mapping confirmed. base = code of the slim '0'.
        base = cs[0] - vs[0]
        for v in range(10):
            code = base + v
            if 0 <= code <= 0xFF and code not in out:
                out[code] = str(v)
    return out


# ----- Plus Code (Open Location Code) --------------------------------------
#
# INAV / Betaflight optionally print the pilot location as a Plus Code,
# Google's open-location encoding, in the OSD. This lets the map widget plot
# a flight track from footage that has no SRT GPS data: scan every OSD frame
# for the Plus Code cluster, decode each to (lat, lon), use that as the
# track.
#
# Reference: https://github.com/google/open-location-code
# Algorithm summary:
#   - 20-char alphabet (no vowels or look-alikes).
#   - Standard full code = 8 chars + '+' + 2-3 chars.
#   - First 10 chars encode lat/lon in alternating pairs at progressively
#     finer resolution (20°, 1°, 0.05°, 0.0025°, 0.000125°).
#   - Extra chars beyond the 10th use a 4-row x 5-col grid encoding per char.

_OLC_ALPHABET = "23456789CFGHJMPQRVWX"
_OLC_DECODE = {c: i for i, c in enumerate(_OLC_ALPHABET)}
_OLC_GRID_ROWS = 4
_OLC_GRID_COLS = 5

# Match a full Plus Code embedded in arbitrary text. Pattern is restrictive
# enough that random ASCII rarely collides (8 chars from a 20-char alphabet
# followed by '+').
_PLUS_CODE_RE = re.compile(
    r"[23456789CFGHJMPQRVWX]{8}\+[23456789CFGHJMPQRVWX]{2,4}"
)


def decode_plus_code(code: str) -> Optional[Tuple[float, float]]:
    """Decode a full Plus Code string into (lat, lon) at the cell centre.

    Returns None when the code is malformed or short (we don't currently
    support padded short codes - INAV / Betaflight emit full codes).
    """
    if not code:
        return None
    code = code.upper().strip()
    plus_idx = code.find("+")
    if plus_idx != 8 or len(code) < 10:
        return None
    digits = code[:8] + code[9:]
    if any(c not in _OLC_DECODE for c in digits):
        return None

    lat = -90.0
    lon = -180.0
    # Pair encoding for the first 10 digits.
    res = 20.0
    n_pairs = min(5, len(digits) // 2)
    for i in range(n_pairs):
        lat += _OLC_DECODE[digits[2 * i]]     * res
        lon += _OLC_DECODE[digits[2 * i + 1]] * res
        res /= 20.0

    # Cell size after the pair encoding (5th pair = 0.000125 deg).
    cell_h = 20.0 / (20 ** n_pairs)
    cell_w = cell_h

    # Grid encoding for any digits past the 10th.
    if len(digits) > 10:
        for c in digits[10:]:
            d = _OLC_DECODE[c]
            cell_h /= _OLC_GRID_ROWS
            cell_w /= _OLC_GRID_COLS
            lat += (d // _OLC_GRID_COLS) * cell_h
            lon += (d %  _OLC_GRID_COLS) * cell_w

    # Centre of the smallest decoded cell.
    return lat + cell_h / 2.0, lon + cell_w / 2.0


def extract_plus_code(osd_frame) -> Optional[str]:
    """Search the OSD grid for a cluster containing a full Plus Code.

    Returns the matched substring (e.g. ``849VCWC8+R9``) or None when no
    cluster matches. Uses ``find_cell_clusters`` so any leading icon glyph
    (rendered as ``<XX>``) sitting next to the code is simply ignored by the
    regex search.
    """
    if osd_frame is None:
        return None
    for cl in find_cell_clusters(osd_frame):
        m = _PLUS_CODE_RE.search(cl["text"].upper())
        if m:
            return m.group(0)
    return None

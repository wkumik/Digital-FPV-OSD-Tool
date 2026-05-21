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
from typing import Optional, Tuple


# Symbol-table entry shape:
#   (anchor_glyph_id, side, max_digits, scale)
#
# anchor_glyph_id - the byte we scan for in the grid.
# side            - "left"  : digits sit BEFORE the anchor (most common -
#                             '123K' where K is the km/h icon).
#                   "right" : digits sit AFTER the anchor (e.g. icon-prefix
#                             layouts).
# max_digits      - upper bound on how far we walk reading digits.
# scale           - multiply parsed value by this (e.g. 0.1 if the firmware
#                   stores tenths without a decimal point).


# ----- INAV ------------------------------------------------------------------
#
# Values below match the symbol set in INAV's src/main/io/osd_symbols.h
# (current as of INAV 6/7). The km/h, m/s, ft/s, V, A, mAh, m, ft icons are
# stable across recent firmware releases.

_INAV_SYMBOLS: dict[str, tuple[int, str, int, float]] = {
    # Speed (km/h) - 0x88 is the km/h unit glyph in this INAV font.
    # Confirmed against three sample frames: 43, 59, 72 km/h.
    "osd_speed_kmh":   (0x88, "left", 4, 1.0),

    # Altitude (m) - 0x76 is the metres suffix for altitude.
    # Confirmed across frames: 14m -> 361m -> 9m matches a climb-and-descend.
    "osd_altitude_m":  (0x76, "left", 5, 1.0),

    # Home distance (m) - INAV draws ``[home_icon][direction][digits][0x7A]``,
    # so the only reliable adjacency is the trailing 0x7A suffix.
    "osd_home_dist":   (0x7A, "left", 5, 1.0),

    # Link / GPS - prefix icons, digits sit RIGHT of the icon
    "osd_rssi":        (0x01, "right", 3, 1.0),
    "osd_sats":        (0x09, "right", 3, 1.0),
}


# ----- Betaflight ------------------------------------------------------------
#
# Betaflight and INAV share most unit-icon IDs in the same hex range. The few
# that diverge (e.g. battery icons used for percent bars) are not used here.

_BTFL_SYMBOLS: dict[str, tuple[int, str, int, float]] = dict(_INAV_SYMBOLS)


# ----- ArduPilot -------------------------------------------------------------
#
# ArduPilot ships its own osd font and symbol set; many of the same anchor
# IDs are used as INAV. Worth verifying against real footage.

_ARDU_SYMBOLS: dict[str, tuple[int, str, int, float]] = dict(_INAV_SYMBOLS)


FW_TABLES: dict[str, dict[str, tuple[int, str, int, float]]] = {
    "INAV":       _INAV_SYMBOLS,
    "Betaflight": _BTFL_SYMBOLS,
    "ArduPilot":  _ARDU_SYMBOLS,
}


# All OSD-derived field keys, in display order. widgets.py advertises these
# in the Source dropdown.
OSD_FIELD_REGISTRY: list[tuple[str, str, str, str]] = [
    # (key, display name, unit string, default fmt)
    ("osd_speed_kmh",   "Speed (OSD)",      "km/h", "{:.0f}"),
    ("osd_altitude_m",  "Altitude (OSD)",   "m",    "{:.0f}"),
    ("osd_home_dist",   "Home distance",    "m",    "{:.0f}"),
    ("osd_rssi",        "RSSI (OSD)",       "",     "{:.0f}"),
    ("osd_sats",        "GPS sats (OSD)",   "",     "{:.0f}"),
]


# Glyph codes that we treat as digits / decimal sign / minus when reading
# adjacent cells around an anchor. Betaflight, INAV, and ArduPilot fonts all
# map digits to ASCII 0x30-0x39.
_DIGIT_LO, _DIGIT_HI = 0x30, 0x39
_DOT, _MINUS = 0x2E, 0x2D


def _glyph_to_char(code: int) -> Optional[str]:
    """Return the character we should treat ``code`` as inside a number run,
    or None if it isn't part of one (terminates the walk)."""
    if _DIGIT_LO <= code <= _DIGIT_HI:
        return chr(code)
    if code == _DOT:
        return "."
    if code == _MINUS:
        return "-"
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


def _read_left(grid, row_base: int, anchor_col: int, max_digits: int) -> list[str]:
    """Read digit cells walking BACKWARD from the cell at anchor_col - 1."""
    out: list[str] = []
    for d in range(1, max_digits + 1):
        cc = anchor_col - d
        if cc < 0:
            break
        ch = _glyph_to_char(grid[row_base + cc])
        if ch is None:
            break
        out.insert(0, ch)
    return out


def _read_right(grid, row_base: int, anchor_col: int, max_digits: int,
                cols: int) -> list[str]:
    """Read digit cells walking FORWARD from the cell at anchor_col + 1."""
    out: list[str] = []
    for d in range(1, max_digits + 1):
        cc = anchor_col + d
        if cc >= cols:
            break
        ch = _glyph_to_char(grid[row_base + cc])
        if ch is None:
            break
        out.append(ch)
    return out


def extract_value(osd_frame, field_key: str,
                  firmware: str = "INAV") -> Optional[float]:
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
    anchor, side, max_digits, scale = info

    grid = osd_frame.grid
    cols = osd_frame.grid_cols

    for i, code in enumerate(grid):
        if code != anchor:
            continue
        r, c = divmod(i, cols)
        row_base = r * cols
        if side == "right":
            digits = _read_right(grid, row_base, c, max_digits, cols)
        else:
            digits = _read_left(grid, row_base, c, max_digits)
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

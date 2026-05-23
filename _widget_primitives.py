# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_widget_primitives.py - Shared low-level drawing helpers for widget modules.

Imported by _widget_draw.py and _widget_map.py. Not part of the public API.
"""

from __future__ import annotations

import math
import os
from typing import Any

try:
    from PIL import ImageFont as PILFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

# Bundled widget font: Roboto Mono Medium (500) — clean, even-width digits for
# numeric readouts. Resolved relative to this module so it works in dev and in
# the PyInstaller build (assets/ is shipped alongside, same as icon.png).
_BUNDLED_FONT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "RobotoMono-Medium.ttf")


# ----- Colour parsing --------------------------------------------------------

def _parse_color(spec: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Accept '#RGB', '#RRGGBB', '#RRGGBBAA', or '' (empty -> fully transparent).
    ``alpha`` is only applied when the spec doesn't include its own alpha.
    """
    if not spec:
        return (0, 0, 0, 0)
    s = spec.lstrip("#")
    a = max(0, min(255, int(alpha)))
    try:
        if len(s) == 3:
            r = int(s[0] * 2, 16); g = int(s[1] * 2, 16); b = int(s[2] * 2, 16)
            return (r, g, b, a)
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)
        if len(s) == 8:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
    except ValueError:
        pass
    return (255, 255, 255, a)


# ----- Font cache ------------------------------------------------------------

_FONT_CACHE: dict[int, Any] = {}


def _font(size: int) -> Any:
    """Return a cached TrueType font at the requested pixel size.

    Prefers the bundled Roboto Mono Medium; falls back to system Arial /
    DejaVu and finally PIL's built-in default if none are available.
    """
    size = max(6, int(size))
    if size not in _FONT_CACHE:
        for loader in (
            lambda: PILFont.truetype(_BUNDLED_FONT, size),
            lambda: PILFont.truetype("arial.ttf", size),
            lambda: PILFont.truetype("DejaVuSans.ttf", size),
            PILFont.load_default,
        ):
            try:
                _FONT_CACHE[size] = loader()
                break
            except Exception:
                continue
    return _FONT_CACHE[size]


def _format_value(value: Any, fmt: str, fallback_fmt: str) -> str:
    """Best-effort apply user fmt, falling back to source default, then str()."""
    for f in (fmt, fallback_fmt, "{}"):
        if not f:
            continue
        try:
            return f.format(value)
        except (ValueError, TypeError, KeyError, IndexError):
            continue
    return str(value)


# ----- Drawing helpers -------------------------------------------------------

def _draw_text_with_shadow(draw: Any, xy: tuple, text: str,
                           font: Any, fill: Any, shadow_alpha: int = 140) -> None:
    """Subtle drop shadow: low-alpha black, 1 px offset."""
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, shadow_alpha))
    draw.text((x, y),         text, font=font, fill=fill)


def _polar(cx: float, cy: float, r: float, ang_deg: float) -> tuple[float, float]:
    """PIL convention: 0° = +x axis, angles increase clockwise."""
    rad = math.radians(ang_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)

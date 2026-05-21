# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
widgets.py - Custom OSD widgets composited on top of the glyph grid.

Widgets are user-positioned overlays (digital readouts, bars, gauges) that
visualise a single telemetry value at an arbitrary location on the video.
They complement the MSP OSD glyph grid rather than replace it.

Phase 1 scope:
  - Three widget types: digital readout, linear bar, circular gauge
  - Data source: SRT TelemetryData only (OSD glyph reverse-parsing comes later)
  - Coordinates normalised (0..1) to the SOURCE video area (not the padded
    canvas), so widgets stay pinned to the picture when pillarboxing is added

Two render paths, mirroring osd_renderer.py:
  - render_widgets_pil()    - PIL preview path (single frame, Qt)
  - blend_widgets_numpy()   - export hot path (per-frame, blends into the
                              pre-allocated RGBA buffer with Porter-Duff over)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont as PILFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_decoder import extract_value as _osd_extract, OSD_FIELD_REGISTRY


# ----- Widget model ----------------------------------------------------------

@dataclass
class Widget:
    """
    A single custom overlay.

    x, y, w, h are normalised 0..1 in the SOURCE video area. (x, y) is the
    widget's CENTRE so resizing keeps the visual anchor stable.
    """
    type:    str   = "digital"           # "digital" | "bar" | "gauge"
    source:  str   = "altitude_m"        # telemetry field key (see WIDGET_DATA_SOURCES)
    x:       float = 0.5
    y:       float = 0.5
    w:       float = 0.15
    h:       float = 0.08
    style:   dict  = field(default_factory=dict)
    enabled: bool  = True

    @classmethod
    def from_dict(cls, d: dict) -> "Widget":
        return cls(
            type    = str(d.get("type", "digital")),
            source  = str(d.get("source", "altitude_m")),
            x       = float(d.get("x", 0.5)),
            y       = float(d.get("y", 0.5)),
            w       = float(d.get("w", 0.15)),
            h       = float(d.get("h", 0.08)),
            style   = dict(d.get("style", {})),
            enabled = bool(d.get("enabled", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def widgets_from_list(items: list) -> list[Widget]:
    """Parse a JSON-loaded list of dicts into Widget instances."""
    out: list[Widget] = []
    if not items:
        return out
    for d in items:
        if isinstance(d, dict):
            try:
                out.append(Widget.from_dict(d))
            except (TypeError, ValueError):
                continue
    return out


# ----- Telemetry data source -------------------------------------------------
#
# Each entry: (key, display_name, attribute_on_TelemetryData, unit_string, default_fmt)
# `attr` of None means OSD-derived (looked up via osd_decoder).

# SRT-backed sources first (data comes from the .srt track)
_SRT_SOURCES: list[tuple[str, str, Optional[str], str, str]] = [
    ("altitude_m",  "Altitude (SRT)",       "altitude_m",  "m",    "{:.0f}"),
    ("distance_m",  "Distance (SRT)",       "distance_m",  "m",    "{:.0f}"),
    ("voltage_v",   "Voltage (SRT)",        "voltage_v",   "V",    "{:.1f}"),
    ("sbat_v",      "Sky Battery (SRT)",    "sbat_v",      "V",    "{:.1f}"),
    ("gbat_v",      "Ground Battery (SRT)", "gbat_v",      "V",    "{:.1f}"),
    ("signal",      "Signal",               "signal",      "",     "{:.0f}"),
    ("link_mbps",   "Bitrate",              "link_mbps",   "Mbps", "{:.1f}"),
    ("delay_ms",    "Latency",              "delay_ms",    "ms",   "{:.0f}"),
    ("radio1_dbm",  "Radio 1 dBm",          "radio1_dbm",  "dBm",  "{:+d}"),
    ("radio2_dbm",  "Radio 2 dBm",          "radio2_dbm",  "dBm",  "{:+d}"),
    ("radio1_snr",  "Radio 1 SNR",          "radio1_snr",  "",     "{:.0f}"),
    ("radio2_snr",  "Radio 2 SNR",          "radio2_snr",  "",     "{:.0f}"),
    ("channel",     "RF Channel",           "channel",     "",     "{:.0f}"),
    ("freq_mhz",    "Frequency",            "freq_mhz",    "MHz",  "{:.0f}"),
]

# OSD-glyph-backed sources (data read from the .osd file via osd_decoder).
_OSD_SOURCES: list[tuple[str, str, Optional[str], str, str]] = [
    (key, name, None, unit, fmt) for key, name, unit, fmt in OSD_FIELD_REGISTRY
]

WIDGET_DATA_SOURCES: list[tuple[str, str, Optional[str], str, str]] = (
    _SRT_SOURCES + _OSD_SOURCES
)

_SOURCE_LOOKUP: dict[str, tuple[Optional[str], str, str]] = {
    key: (attr, unit, fmt) for key, _name, attr, unit, fmt in WIDGET_DATA_SOURCES
}


# ----- TelemetryFrame --------------------------------------------------------

class TelemetryFrame:
    """Unified view over SRT telemetry + OSD-glyph values.

    Widgets see a single object regardless of where the data lives. SRT fields
    forward to the underlying TelemetryData via attribute access; OSD-derived
    fields are decoded on demand from the current OsdFrame.

    Decoded OSD values are cached for the lifetime of this frame instance, so
    multiple widgets reading the same field don't re-scan the grid.
    """
    __slots__ = ("_srt", "_osd", "_fw", "_osd_cache")

    def __init__(self, srt_td=None, osd_frame=None, firmware: str = "INAV"):
        self._srt = srt_td
        self._osd = osd_frame
        self._fw  = firmware
        self._osd_cache: dict[str, Optional[float]] = {}

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so we can safely
        # forward unknown attributes onto the underlying SRT TelemetryData.
        if self._srt is None:
            return None
        return getattr(self._srt, name, None)

    def get_osd_value(self, key: str) -> Optional[float]:
        if self._osd is None:
            return None
        if key in self._osd_cache:
            return self._osd_cache[key]
        v = _osd_extract(self._osd, key, self._fw)
        self._osd_cache[key] = v
        return v


def widget_data_source_label(key: str) -> str:
    for k, name, *_ in WIDGET_DATA_SOURCES:
        if k == key:
            return name
    return key


def _get_value(telemetry: Any, source: str) -> tuple[Optional[float], str, str]:
    """
    Resolve a widget's data source. ``telemetry`` is normally a TelemetryFrame
    that bridges SRT + OSD; a bare TelemetryData also works for SRT-only fields.
    Returns (value_or_None, unit_str, default_fmt).
    """
    info = _SOURCE_LOOKUP.get(source)
    if info is None or telemetry is None:
        return None, "", "{}"
    attr, unit, fmt = info
    if attr is not None:
        # SRT-attribute path - works for both TelemetryFrame (via __getattr__)
        # and a bare TelemetryData.
        return getattr(telemetry, attr, None), unit, fmt
    # OSD-derived path - needs TelemetryFrame's decoder bridge.
    get_osd = getattr(telemetry, "get_osd_value", None)
    if get_osd is None:
        return None, unit, fmt
    return get_osd(source), unit, fmt


# ----- Colour parsing --------------------------------------------------------

def _parse_color(spec: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """
    Accept '#RGB', '#RRGGBB', '#RRGGBBAA', or '' (empty -> fully transparent).
    `alpha` is only applied when the spec doesn't include its own alpha.
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

def _font(size: int):
    """Return a cached TrueType font at the requested pixel size."""
    size = max(6, int(size))
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = PILFont.truetype("arial.ttf", size)
        except Exception:
            try:
                _FONT_CACHE[size] = PILFont.truetype("DejaVuSans.ttf", size)
            except Exception:
                _FONT_CACHE[size] = PILFont.load_default()
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


# ----- Widget drawing primitives --------------------------------------------

def _draw_text_with_shadow(draw, xy, text, font, fill, shadow_alpha=140):
    """Subtle drop shadow: low-alpha black, 1 px offset. Soft enough to read
    against video without the hard `outline` look the old default had."""
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, shadow_alpha))
    draw.text((x, y),         text, font=font, fill=fill)


def _draw_digital(img: "Image.Image", w: Widget,
                  value: Any, unit: str, default_fmt: str,
                  rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    color    = _parse_color(style.get("color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("opacity", 1.0))))
    bg_alpha = int(255 * float(style.get("bg_opacity", 0.0)))
    bg_color = _parse_color(style.get("bg_color", "#000000"), alpha=bg_alpha)
    label    = str(style.get("label", ""))
    fmt      = str(style.get("fmt", default_fmt))
    show_unit = bool(style.get("show_unit", True))
    align    = str(style.get("align", "center"))
    shadow   = bool(style.get("shadow", True))

    val_str = _format_value(value, fmt, default_fmt)
    parts: list[str] = []
    if label:
        parts.append(label)
    parts.append(val_str)
    if show_unit and unit:
        parts.append(unit)
    text = " ".join(parts)

    font = _font(int(wh * float(style.get("font_scale", 0.85))))
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    if align == "left":
        tx = rx
    elif align == "right":
        tx = rx + ww - tw
    else:
        tx = rx + (ww - tw) // 2
    ty = ry + (wh - th) // 2 - bb[1]

    # Background pill only when the user explicitly asks for one.
    if bg_color[3] > 0:
        pad = max(2, int(wh * 0.10))
        draw.rounded_rectangle(
            [tx - pad, ty + bb[1] - pad, tx + tw + pad, ty + bb[1] + th + pad],
            radius=max(2, int(wh * 0.12)), fill=bg_color,
        )

    if shadow:
        _draw_text_with_shadow(draw, (tx, ty), text, font, fill=color,
                               shadow_alpha=int(style.get("shadow_alpha", 130)))
    else:
        draw.text((tx, ty), text, font=font, fill=color)


def _draw_bar(img: "Image.Image", w: Widget,
              value: Any, unit: str, default_fmt: str,
              rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    color    = _parse_color(style.get("color", "#3FA9F5"),
                            alpha=int(255 * float(style.get("opacity", 0.80))))
    bg_color = _parse_color(style.get("bg_color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("bg_opacity", 0.18))))
    orient   = str(style.get("orientation", "horizontal"))
    show_value = bool(style.get("show_value", True))

    draw = ImageDraw.Draw(img)

    # Layout: value text on top, thin bar below. Cleaner than the old in-bar
    # text which fought the fill colour for legibility.
    if orient == "vertical":
        # Vertical layout keeps its full-height bar; value sits beneath in a
        # small caption strip.
        cap_h = int(wh * 0.18) if show_value else 0
        bar_top = ry
        bar_bot = ry + wh - cap_h - (2 if show_value else 0)
        bar_left = rx
        bar_right = rx + ww
        bar_h = max(2, bar_bot - bar_top)
        radius = max(1, int(min(ww, bar_h) * 0.20))
        draw.rounded_rectangle([bar_left, bar_top, bar_right, bar_bot],
                               radius=radius, fill=bg_color)
        fill_h = int(bar_h * frac)
        if fill_h > 0:
            draw.rounded_rectangle(
                [bar_left, bar_bot - fill_h, bar_right, bar_bot],
                radius=radius, fill=color,
            )
        cap_y0 = bar_bot + 2
        cap_y1 = ry + wh
        text_anchor = (cap_y0, cap_y1)
    else:
        # Horizontal: split into a top text band and a thin bar at the bottom.
        if show_value:
            text_h = max(10, int(wh * 0.45))
            gap    = max(1, int(wh * 0.08))
            bar_top = ry + text_h + gap
        else:
            text_h = 0
            gap    = 0
            bar_top = ry
        bar_bot  = ry + wh
        bar_h    = max(2, bar_bot - bar_top)
        radius   = max(1, int(min(ww, bar_h) * 0.30))
        draw.rounded_rectangle([rx, bar_top, rx + ww, bar_bot],
                               radius=radius, fill=bg_color)
        fill_w = int(ww * frac)
        if fill_w > 0:
            draw.rounded_rectangle([rx, bar_top, rx + fill_w, bar_bot],
                                   radius=radius, fill=color)
        text_anchor = (ry, ry + text_h)

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        if bool(style.get("show_unit", True)) and unit:
            val_str = f"{val_str} {unit}"
        cap_y0, cap_y1 = text_anchor
        cap_h = max(1, cap_y1 - cap_y0)
        font = _font(int(cap_h * float(style.get("font_scale", 0.85))))
        bb = draw.textbbox((0, 0), val_str, font=font)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        align = str(style.get("align", "left"))
        if align == "right":
            tx = rx + ww - tw
        elif align == "center":
            tx = rx + (ww - tw) // 2
        else:
            tx = rx
        ty = cap_y0 + (cap_h - th) // 2 - bb[1]
        _draw_text_with_shadow(draw, (tx, ty), val_str, font,
                               fill=_parse_color(style.get("text_color", "#FFFFFF")),
                               shadow_alpha=int(style.get("shadow_alpha", 110)))


def _draw_gauge(img: "Image.Image", w: Widget,
                value: Any, unit: str, default_fmt: str,
                rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    # Refined defaults: lower opacity, thinner arc.
    color    = _parse_color(style.get("color", "#3FA9F5"),
                            alpha=int(255 * float(style.get("opacity", 0.90))))
    track    = _parse_color(style.get("bg_color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("bg_opacity", 0.14))))
    start_angle   = float(style.get("start_angle", 135.0))   # PIL: 0 = +x axis, CW
    span_angle    = float(style.get("span_angle", 270.0))
    thickness_pct = float(style.get("thickness", 0.10))      # was 0.18
    show_value    = bool(style.get("show_value", True))
    rounded_caps  = bool(style.get("rounded_caps", True))

    # PIL's draw.arc isn't anti-aliased.  Supersample the gauge: draw to a
    # patch at SS× resolution, then LANCZOS-downscale.  Costs a few × CPU on
    # the gauge area only — still sub-millisecond per widget.
    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    size_s   = min(pw, ph)
    cx_s     = pw // 2
    cy_s     = ph // 2
    half_s   = size_s // 2
    gx0_s    = cx_s - half_s
    gy0_s    = cy_s - half_s
    gx1_s    = gx0_s + size_s
    gy1_s    = gy0_s + size_s
    thick_s  = max(2 * SS, int(size_s * thickness_pct))

    end_angle = start_angle + span_angle
    pdraw.arc([gx0_s, gy0_s, gx1_s, gy1_s], start_angle, end_angle,
              fill=track, width=thick_s)
    v_end = start_angle + span_angle * frac
    if frac > 0.001:
        pdraw.arc([gx0_s, gy0_s, gx1_s, gy1_s], start_angle, v_end,
                  fill=color, width=thick_s)
        # Rounded end caps — draw filled circles at both ends of the value
        # arc so it terminates cleanly instead of a blunt edge.
        if rounded_caps:
            arc_r = ((gx1_s - gx0_s) - thick_s) / 2.0
            cap_r = thick_s / 2.0
            acx   = (gx0_s + gx1_s) / 2.0
            acy   = (gy0_s + gy1_s) / 2.0
            for ang_deg in (start_angle, v_end):
                rad = math.radians(ang_deg)
                px  = acx + arc_r * math.cos(rad)
                py  = acy + arc_r * math.sin(rad)
                pdraw.ellipse([px - cap_r, py - cap_r,
                               px + cap_r, py + cap_r], fill=color)

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        font = _font(int(size_s * float(style.get("font_scale", 0.32))))
        bb = pdraw.textbbox((0, 0), val_str, font=font)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        tx = cx_s - tw // 2
        # Push slightly up so the value visually centres above the unit caption.
        ty = cy_s - th // 2 - bb[1] - int(size_s * (0.04 if unit else 0.0))
        text_color = _parse_color(style.get("text_color", "#FFFFFF"))
        # Plain text — central number sits inside the arc on a transparent
        # background, so the hard shadow it used to have is unnecessary.
        pdraw.text((tx, ty), val_str, font=font, fill=text_color)
        if bool(style.get("show_unit", True)) and unit:
            ufont = _font(int(size_s * float(style.get("unit_scale", 0.13))))
            ubb = pdraw.textbbox((0, 0), unit, font=ufont)
            uw = ubb[2] - ubb[0]
            utx = cx_s - uw // 2
            uty = ty + th + max(2, int(size_s * 0.015))
            pdraw.text((utx, uty), unit, font=ufont,
                       fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)


_DRAW_FUNCS = {
    "digital": _draw_digital,
    "bar":     _draw_bar,
    "gauge":   _draw_gauge,
}


# ----- Geometry --------------------------------------------------------------

def _widget_rect(w: Widget,
                 canvas_w: int, canvas_h: int,
                 source_w: int, source_h: int) -> tuple[int, int, int, int]:
    """
    Resolve a widget's normalised geometry against the canvas + source rect.
    Returns (rx, ry, ww, wh) in canvas pixels, with the widget centred at
    (w.x, w.y) of the source area.
    """
    if source_w <= 0:
        source_w = canvas_w
    if source_h <= 0:
        source_h = canvas_h
    sx = (canvas_w - source_w) // 2
    sy = (canvas_h - source_h) // 2

    ww = max(1, int(round(w.w * source_w)))
    wh = max(1, int(round(w.h * source_h)))
    cx = sx + int(round(w.x * source_w))
    cy = sy + int(round(w.y * source_h))
    rx = cx - ww // 2
    ry = cy - wh // 2
    return rx, ry, ww, wh


# ----- Public entry points ---------------------------------------------------

def render_widgets_pil(img: "Image.Image",
                       widgets: list[Widget],
                       telemetry: Any,
                       source_w: int = 0,
                       source_h: int = 0) -> None:
    """
    Composite widgets onto a PIL RGBA image in-place (preview path).

    `source_w/source_h` describe the source video area inside `img`. Zero means
    img IS the source (no pillar/letterbox padding).
    """
    if not PIL_OK or not widgets or telemetry is None:
        return
    if img.mode != "RGBA":
        return
    canvas_w, canvas_h = img.size
    for w in widgets:
        if not w.enabled:
            continue
        value, unit, default_fmt = _get_value(telemetry, w.source)
        if value is None:
            continue
        rx, ry, ww, wh = _widget_rect(w, canvas_w, canvas_h, source_w, source_h)
        if rx + ww <= 0 or ry + wh <= 0 or rx >= canvas_w or ry >= canvas_h:
            continue
        draw_fn = _DRAW_FUNCS.get(w.type)
        if draw_fn is None:
            continue
        draw_fn(img, w, value, unit, default_fmt, rx, ry, ww, wh)


def blend_widgets_numpy(frame: np.ndarray,
                        widgets: list[Widget],
                        telemetry: Any,
                        source_w: int = 0,
                        source_h: int = 0) -> None:
    """
    Composite widgets onto a numpy RGBA frame buffer in-place (export path).

    For each widget, renders to a small PIL RGBA patch sized to the widget's
    rect, then alpha-blends just that patch into `frame` with Porter-Duff over
    - matching the math used by OsdRenderer.composite().
    """
    if not PIL_OK or not widgets or telemetry is None:
        return
    if frame.ndim != 3 or frame.shape[2] != 4:
        return
    canvas_h, canvas_w = frame.shape[:2]

    for w in widgets:
        if not w.enabled:
            continue
        value, unit, default_fmt = _get_value(telemetry, w.source)
        if value is None:
            continue
        rx, ry, ww, wh = _widget_rect(w, canvas_w, canvas_h, source_w, source_h)
        # Clamp to frame
        px0 = max(rx, 0); py0 = max(ry, 0)
        px1 = min(rx + ww, canvas_w); py1 = min(ry + wh, canvas_h)
        if px1 <= px0 or py1 <= py0:
            continue

        draw_fn = _DRAW_FUNCS.get(w.type)
        if draw_fn is None:
            continue

        # Draw onto a patch sized to the widget; offset coords so widget paints
        # at (0,0) within the patch.
        patch = Image.new("RGBA", (ww, wh), (0, 0, 0, 0))
        draw_fn(patch, w, value, unit, default_fmt, 0, 0, ww, wh)

        # Slice the visible region within the patch
        gx0 = px0 - rx
        gy0 = py0 - ry
        gx1 = gx0 + (px1 - px0)
        gy1 = gy0 + (py1 - py0)

        patch_arr = np.asarray(patch, dtype=np.uint8)[gy0:gy1, gx0:gx1]
        if patch_arr.shape[0] == 0 or patch_arr.shape[1] == 0:
            continue
        src_a = patch_arr[:, :, 3:4].astype(np.float32) / 255.0
        if not (src_a > 0).any():
            continue

        dst = frame[py0:py1, px0:px1]
        dst_a   = dst[:, :, 3:4].astype(np.float32) / 255.0
        out_a   = src_a + dst_a * (1.0 - src_a)
        safe_a  = np.where(out_a > 0, out_a, 1.0)
        out_rgb = (patch_arr[:, :, :3].astype(np.float32) * src_a
                   + dst[:, :, :3].astype(np.float32) * dst_a * (1.0 - src_a)
                  ) / safe_a
        dst[:, :, :3] = out_rgb.astype(np.uint8)
        dst[:, :, 3]  = (out_a[:, :, 0] * 255).astype(np.uint8)

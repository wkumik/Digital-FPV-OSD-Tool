# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
widgets.py - Custom OSD widgets composited on top of the glyph grid.

Public API surface. Drawing logic lives in the private submodules:
  _widget_primitives.py  - shared helpers (colour, font, polar, shadow)
  _widget_draw.py        - per-type draw functions (digital, bar, gauge, …)
  _widget_map.py         - GPS map: projection, smoothing, Catmull-Rom

Two render entry points mirror osd_renderer.py:
  render_widgets_pil()   - PIL preview path (single frame, Qt)
  blend_widgets_numpy()  - export hot path (Porter-Duff over into numpy buffer)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_decoder import (
    extract_value as _osd_extract,
    OSD_FIELD_REGISTRY,
    build_slim_digit_map as _build_slim_digit_map,
)
from _widget_draw import (
    _draw_digital, _draw_bar, _draw_gauge, _draw_semicircle,
    _draw_tickdial, _draw_ring, _draw_analog,
)
from _widget_map import _draw_map, invalidate_track_caches  # noqa: F401 (re-exported)


# ----- Widget model ----------------------------------------------------------

@dataclass
class Widget:
    """A single custom overlay.

    x, y, w, h are normalised 0..1 in the SOURCE video area. (x, y) is the
    widget's CENTRE so resizing keeps the visual anchor stable.
    """
    type:    str   = "digital"
    source:  str   = "altitude_m"
    x:       float = 0.5
    y:       float = 0.5
    w:       float = 0.15
    h:       float = 0.08
    style:   dict  = field(default_factory=dict)
    enabled: bool  = True

    @classmethod
    def from_dict(cls, d: dict) -> "Widget":
        type_   = str(d.get("type", "digital"))
        source_ = str(d.get("source", "altitude_m"))
        # Migrate: older saves stored type="map" with a regular telemetry source.
        # New model: source="gps_track" drives the map type.
        if type_ == "map" and source_ != "gps_track":
            source_ = "gps_track"
        return cls(
            type    = type_,
            source  = source_,
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


# ----- Telemetry data sources ------------------------------------------------
#
# Each entry: (key, display_name, attribute_on_TelemetryData, unit_string, default_fmt)
# attr=None means OSD-derived (decoded via osd_decoder).

_SRT_SOURCES: list[tuple[str, str, Optional[str], str, str]] = [
    ("altitude_m",  "Altitude (SRT)",       "altitude_m",  "m",    "{:.0f}"),
    ("distance_m",  "Distance (SRT)",       "distance_m",  "m",    "{:.0f}"),
    ("voltage_v",   "Voltage (SRT)",        "voltage_v",   "V",    "{:.1f}"),
    ("sbat_v",      "Sky Battery (SRT)",    "sbat_v",      "V",    "{:.1f}"),
    ("gbat_v",      "Ground Battery (SRT)", "gbat_v",      "V",    "{:.1f}"),
    ("signal",      "Signal",               "signal",      "",     "{:.0f}"),
    ("link_mbps",   "Bitrate",              "link_mbps",   "Mbps", "{:.1f}"),
    ("radio1_dbm",  "Radio 1 dBm",          "radio1_dbm",  "dBm",  "{:+d}"),
    ("radio2_dbm",  "Radio 2 dBm",          "radio2_dbm",  "dBm",  "{:+d}"),
    ("radio1_snr",  "Radio 1 SNR",          "radio1_snr",  "",     "{:.0f}"),
    ("radio2_snr",  "Radio 2 SNR",          "radio2_snr",  "",     "{:.0f}"),
    ("freq_mhz",    "Frequency",            "freq_mhz",    "MHz",  "{:.0f}"),
]

_OSD_SOURCES: list[tuple[str, str, Optional[str], str, str]] = [
    (key, name, None, unit, fmt) for key, name, unit, fmt in OSD_FIELD_REGISTRY
]

WIDGET_DATA_SOURCES: list[tuple[str, str, Optional[str], str, str]] = (
    _SRT_SOURCES + _OSD_SOURCES + [("gps_track", "Map (GPS trace)", None, "", "{}")]
)

_SOURCE_LOOKUP: dict[str, tuple[Optional[str], str, str]] = {
    key: (attr, unit, fmt) for key, _name, attr, unit, fmt in WIDGET_DATA_SOURCES
}


def widget_data_source_label(key: str) -> str:
    for k, name, *_ in WIDGET_DATA_SOURCES:
        if k == key:
            return name
    return key


# ----- TelemetryFrame --------------------------------------------------------

class TelemetryFrame:
    """Unified view over SRT telemetry + OSD-glyph values.

    Widgets see a single object regardless of where the data lives. SRT fields
    forward to the underlying TelemetryData via attribute access; OSD-derived
    fields are decoded on demand from the current OsdFrame and cached for the
    lifetime of this frame instance.

    ``srt_file`` / ``osd_file`` expose the full telemetry tracks so widgets
    that need whole-track context (e.g. the map) can reach them.
    """
    __slots__ = ("_srt", "_osd", "_fw", "_osd_cache", "_slim_map",
                 "srt_file", "osd_file")

    def __init__(self, srt_td=None, osd_frame=None, firmware: str = "INAV",
                 osd_font=None, srt_file=None, osd_file=None):
        self._srt = srt_td
        self._osd = osd_frame
        self._fw  = firmware
        self._osd_cache: dict[str, Optional[float]] = {}
        self._slim_map = _build_slim_digit_map(osd_font) if osd_font is not None else {}
        self.srt_file = srt_file
        self.osd_file = osd_file

    @property
    def osd_frame(self):
        return self._osd

    def __getattr__(self, name: str):
        if self._srt is None:
            return None
        return getattr(self._srt, name, None)

    def get_osd_value(self, key: str) -> Optional[float]:
        if self._osd is None:
            return None
        if key in self._osd_cache:
            return self._osd_cache[key]
        v = _osd_extract(self._osd, key, self._fw, slim_map=self._slim_map)
        self._osd_cache[key] = v
        return v


# ----- Value resolution ------------------------------------------------------

def _get_value(telemetry: Any, source: str) -> tuple[Optional[float], str, str]:
    """Resolve a widget's data source to (value, unit, default_fmt)."""
    info = _SOURCE_LOOKUP.get(source)
    if info is None or telemetry is None:
        return None, "", "{}"
    attr, unit, fmt = info
    if attr is not None:
        return getattr(telemetry, attr, None), unit, fmt
    get_osd = getattr(telemetry, "get_osd_value", None)
    if get_osd is None:
        return None, unit, fmt
    return get_osd(source), unit, fmt


# ----- Dispatch --------------------------------------------------------------

_DRAW_FUNCS = {
    "digital":    _draw_digital,
    "bar":        _draw_bar,
    "gauge":      _draw_gauge,
    "semicircle": _draw_semicircle,
    "tickdial":   _draw_tickdial,
    "ring":       _draw_ring,
    "analog":     _draw_analog,
}

# Widgets that need the full TelemetryFrame instead of a scalar value.
_TELEMETRY_AWARE = {"map"}


# ----- Geometry --------------------------------------------------------------

def _widget_rect(w: Widget,
                 canvas_w: int, canvas_h: int,
                 source_w: int, source_h: int) -> tuple[int, int, int, int]:
    """Resolve normalised widget geometry to canvas pixels (rx, ry, ww, wh)."""
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
    """Composite widgets onto a PIL RGBA image in-place (preview path)."""
    if not PIL_OK or not widgets or telemetry is None:
        return
    if img.mode != "RGBA":
        return
    canvas_w, canvas_h = img.size
    for w in widgets:
        if not w.enabled:
            continue
        rx, ry, ww, wh = _widget_rect(w, canvas_w, canvas_h, source_w, source_h)
        if rx + ww <= 0 or ry + wh <= 0 or rx >= canvas_w or ry >= canvas_h:
            continue
        if w.type in _TELEMETRY_AWARE:
            _draw_map(img, w, telemetry, rx, ry, ww, wh)
            continue
        value, unit, default_fmt = _get_value(telemetry, w.source)
        if value is None:
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
    """Composite widgets onto a numpy RGBA frame buffer in-place (export path).

    Renders each widget to a small PIL patch, then alpha-blends into ``frame``
    with Porter-Duff over.
    """
    if not PIL_OK or not widgets or telemetry is None:
        return
    if frame.ndim != 3 or frame.shape[2] != 4:
        return
    canvas_h, canvas_w = frame.shape[:2]

    for w in widgets:
        if not w.enabled:
            continue
        rx, ry, ww, wh = _widget_rect(w, canvas_w, canvas_h, source_w, source_h)
        px0 = max(rx, 0);       py0 = max(ry, 0)
        px1 = min(rx + ww, canvas_w); py1 = min(ry + wh, canvas_h)
        if px1 <= px0 or py1 <= py0:
            continue

        if w.type in _TELEMETRY_AWARE:
            patch = Image.new("RGBA", (ww, wh), (0, 0, 0, 0))
            _draw_map(patch, w, telemetry, 0, 0, ww, wh)
        else:
            value, unit, default_fmt = _get_value(telemetry, w.source)
            if value is None:
                continue
            draw_fn = _DRAW_FUNCS.get(w.type)
            if draw_fn is None:
                continue
            patch = Image.new("RGBA", (ww, wh), (0, 0, 0, 0))
            draw_fn(patch, w, value, unit, default_fmt, 0, 0, ww, wh)

        gx0 = px0 - rx;  gy0 = py0 - ry
        gx1 = gx0 + (px1 - px0); gy1 = gy0 + (py1 - py0)

        patch_arr = np.asarray(patch, dtype=np.uint8)[gy0:gy1, gx0:gx1]
        if patch_arr.shape[0] == 0 or patch_arr.shape[1] == 0:
            continue
        src_a = patch_arr[:, :, 3:4].astype(np.float32) / 255.0
        if not (src_a > 0).any():
            continue

        dst    = frame[py0:py1, px0:px1]
        dst_a  = dst[:, :, 3:4].astype(np.float32) / 255.0
        out_a  = src_a + dst_a * (1.0 - src_a)
        safe_a = np.where(out_a > 0, out_a, 1.0)
        out_rgb = (patch_arr[:, :, :3].astype(np.float32) * src_a
                   + dst[:, :, :3].astype(np.float32) * dst_a * (1.0 - src_a)
                   ) / safe_a
        dst[:, :, :3] = out_rgb.astype(np.uint8)
        dst[:, :, 3]  = (out_a[:, :, 0] * 255).astype(np.uint8)

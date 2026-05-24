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

import bisect
import weakref
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


OSD_WIDGET_MAX_GAP_MS = 1500
OSD_WIDGET_FILTER_STEP_MS = 16
# Per-OsdFile cache of filtered visual tracks, keyed by id(OsdFile). A weakref
# finalizer drops a file's bucket the instant it is garbage-collected, so a new
# OsdFile that reuses a freed address can never read another file's stale track.
# Each value is an inner dict keyed by (firmware, source, smoothness, font).
_VISUAL_TRACK_CACHE: dict[int, dict] = {}


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
    __slots__ = ("_srt", "_osd", "_fw", "_osd_cache", "_visual_cache", "_slim_map",
                 "_slim_key", "osd_time_ms",
                 "srt_file", "osd_file")

    def __init__(self, srt_td=None, osd_frame=None, firmware: str = "INAV",
                 osd_font=None, srt_file=None, osd_file=None,
                 osd_time_ms: int | None = None):
        self._srt = srt_td
        self._osd = osd_frame
        self._fw  = firmware
        self._osd_cache: dict[str, Optional[float]] = {}
        self._visual_cache: dict[tuple[str, float], Optional[float]] = {}
        self._slim_map = _build_slim_digit_map(osd_font) if osd_font is not None else {}
        # Stable, cheap cache token for the slim map. build_slim_digit_map caches
        # per font, so the same map object (and id) recurs across frames; the empty
        # no-font map gets a constant token so its tracks still cache.
        self._slim_key = id(self._slim_map) if self._slim_map else 0
        self.osd_time_ms = osd_time_ms
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

    def get_osd_visual_value(self, key: str, smoothness: float) -> Optional[float]:
        """Return the visual-only OSD value for moving widgets.

        0% is deliberately identical to stock v1.7: use the current OSD frame
        value with no interpolation/filtering. Above 0%, the regular widget
        text still uses get_osd_value(), while geometry uses an alpha-beta
        filter evaluated with enough lookahead to compensate its estimated lag.
        """
        if self.osd_file is None or self.osd_time_ms is None:
            return self.get_osd_value(key)
        smoothness = max(0.0, float(smoothness))
        if smoothness <= 0.0:
            return self.get_osd_value(key)
        cache_key = (key, round(smoothness, 3))
        if cache_key in self._visual_cache:
            return self._visual_cache[cache_key]
        value = self._alpha_beta_visual_value(key, smoothness)
        self._visual_cache[cache_key] = value
        return value

    @staticmethod
    def _sample_decoded(times: list[int], vals: list, time_ms: int) -> Optional[float]:
        """Linearly sample pre-decoded per-frame values at ``time_ms``.

        Mirrors the old per-call decode+interpolate exactly, but reads from an
        already-decoded array instead of re-parsing the glyph grid each step.
        """
        idx = bisect.bisect_right(times, time_ms) - 1
        idx = max(0, min(idx, len(times) - 1))
        if idx + 1 >= len(times):
            return vals[idx]
        gap = times[idx + 1] - times[idx]
        if gap <= 0 or gap > OSD_WIDGET_MAX_GAP_MS:
            return vals[idx]
        prev_v = vals[idx]
        next_v = vals[idx + 1]
        if prev_v is None or next_v is None:
            return None
        alpha = max(0.0, min(1.0, (time_ms - times[idx]) / gap))
        return prev_v + (next_v - prev_v) * alpha

    @staticmethod
    def _alpha_beta_params(smoothness: float) -> tuple[float, float]:
        s = max(0.0, float(smoothness))
        alpha = 1.0 / (1.0 + 1.35 * s)
        beta = alpha * alpha * 0.55
        return max(0.03, min(0.95, alpha)), max(0.002, min(0.45, beta))

    @classmethod
    def _lookahead_ms(cls, smoothness: float, sample_ms: float) -> int:
        if smoothness <= 0.0 or sample_ms <= 0.0:
            return 0
        alpha, _beta = cls._alpha_beta_params(smoothness)
        estimated_delay = sample_ms * (1.0 - alpha) / alpha
        return int(max(0, min(OSD_WIDGET_MAX_GAP_MS, round(estimated_delay))))

    def _alpha_beta_visual_value(self, key: str, smoothness: float) -> Optional[float]:
        frames = getattr(self.osd_file, "frames", None)
        timestamps = getattr(self.osd_file, "timestamps", None)
        if not frames or not timestamps:
            return None
        sample_ms = self._median_sample_ms(timestamps)
        lookahead_ms = self._lookahead_ms(smoothness, sample_ms)
        track = self._visual_track(key, smoothness)
        if track is None:
            return None
        track_times, track_values = track
        target_time = min(int(self.osd_time_ms) + lookahead_ms, track_times[-1])
        return self._sample_track(track_times, track_values, target_time)

    def _visual_track(self, key: str, smoothness: float) -> Optional[tuple[list[int], list[float]]]:
        frames = getattr(self.osd_file, "frames", None)
        timestamps = getattr(self.osd_file, "timestamps", None)
        if not frames or not timestamps:
            return None
        osd_file = self.osd_file
        cache_key = (self._fw, key, round(smoothness, 3), self._slim_key)
        file_id = id(osd_file)
        bucket = _VISUAL_TRACK_CACHE.get(file_id)
        if bucket is not None and cache_key in bucket:
            return bucket[cache_key]

        # Decode each real OSD frame's value exactly once up front, then run the
        # filter at a fixed step sampling from those values. This keeps the
        # one-time pre-pass at O(frame count) decodes instead of one decode per
        # 16 ms tick across the whole flight.
        frame_times = [f.time_ms for f in frames]
        frame_vals = [_osd_extract(f, key, self._fw, slim_map=self._slim_map)
                      for f in frames]

        alpha_gain, beta_gain = self._alpha_beta_params(smoothness)
        start_t = timestamps[0]
        end_t = timestamps[-1]
        track_times: list[int] = []
        track_values: list[float] = []
        x = None
        velocity = 0.0
        last_t = None

        for t in range(start_t, end_t + OSD_WIDGET_FILTER_STEP_MS,
                       OSD_WIDGET_FILTER_STEP_MS):
            t = min(t, end_t)
            measurement = self._sample_decoded(frame_times, frame_vals, t)
            if measurement is None:
                if x is not None:
                    dt_ms = (t - last_t) if last_t is not None else OSD_WIDGET_FILTER_STEP_MS
                    dt_s = max(0.001, dt_ms / 1000.0)
                    x = x + velocity * dt_s
                    last_t = t
                    track_times.append(t)
                    track_values.append(float(x))
                if t >= end_t:
                    break
                continue
            if x is None:
                x = measurement
                velocity = 0.0
                last_t = t
            else:
                dt_s = max(0.001, (t - last_t) / 1000.0)
                x_pred = x + velocity * dt_s
                residual = measurement - x_pred
                x = x_pred + alpha_gain * residual
                velocity = velocity + beta_gain * residual / dt_s
                last_t = t
            track_times.append(t)
            track_values.append(float(x))
            if t >= end_t:
                break

        if not track_times:
            return None
        result = (track_times, track_values)
        if bucket is None:
            bucket = {}
            _VISUAL_TRACK_CACHE[file_id] = bucket
            # Evict the bucket as soon as the OsdFile dies (prevents id reuse hits).
            weakref.finalize(osd_file, _VISUAL_TRACK_CACHE.pop, file_id, None)
        if len(bucket) > 64:
            bucket.clear()
        bucket[cache_key] = result
        return result

    @staticmethod
    def _sample_track(times: list[int], values: list[float], time_ms: int) -> float:
        if time_ms <= times[0]:
            return values[0]
        if time_ms >= times[-1]:
            return values[-1]
        idx = bisect.bisect_right(times, time_ms) - 1
        idx = max(0, min(idx, len(times) - 2))
        t0 = times[idx]
        t1 = times[idx + 1]
        if t1 <= t0:
            return values[idx]
        alpha = (time_ms - t0) / (t1 - t0)
        return values[idx] + (values[idx + 1] - values[idx]) * alpha

    @staticmethod
    def _median_sample_ms(timestamps) -> float:
        if len(timestamps) < 2:
            return 100.0
        end = min(len(timestamps), 48)
        gaps = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, end)
            if timestamps[i] > timestamps[i - 1]
        ]
        if not gaps:
            return 100.0
        gaps.sort()
        return float(gaps[len(gaps) // 2])


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


def _get_widget_value(telemetry: Any,
                      widget: Widget) -> tuple[Optional[Any], Optional[Any], str, str]:
    """Resolve (geometry_value, text_value, unit, fmt) for a widget.

    ``geometry_value`` may be visually smoothed for moving widgets; ``text_value``
    is always the exact current OSD/SRT value so the numeric readout stays
    truthful. They are identical when smoothing is off or not applicable.
    """
    value, unit, fmt = _get_value(telemetry, widget.source)
    if value is None or widget.type == "digital":
        return value, value, unit, fmt
    info = _SOURCE_LOOKUP.get(widget.source)
    is_osd_source = info is not None and info[0] is None and widget.source != "gps_track"
    get_visual = getattr(telemetry, "get_osd_visual_value", None)
    if not is_osd_source or get_visual is None:
        return value, value, unit, fmt
    # Default 0.0: widgets saved before smoothing existed must not start smoothing
    # silently. New widgets get a non-zero default set at creation time.
    smoothness = max(0.0, float(widget.style.get("smoothness", 0.0)))
    visual = get_visual(widget.source, smoothness)
    if visual is None:
        return value, value, unit, fmt
    return float(visual), value, unit, fmt


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
        value, text_value, unit, default_fmt = _get_widget_value(telemetry, w)
        if value is None:
            continue
        draw_fn = _DRAW_FUNCS.get(w.type)
        if draw_fn is None:
            continue
        draw_fn(img, w, value, unit, default_fmt, rx, ry, ww, wh, text_value=text_value)


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
            value, text_value, unit, default_fmt = _get_widget_value(telemetry, w)
            if value is None:
                continue
            draw_fn = _DRAW_FUNCS.get(w.type)
            if draw_fn is None:
                continue
            patch = Image.new("RGBA", (ww, wh), (0, 0, 0, 0))
            draw_fn(patch, w, value, unit, default_fmt, 0, 0, ww, wh, text_value=text_value)

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

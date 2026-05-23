# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_widget_map.py - GPS map widget: projection, smoothing, and rendering.

Track source priority: SRT GPS (1 Hz, sub-metre precision) → OSD Plus Code
fallback (~14 m grid resolution). Projection is cached by object id so the
expensive scan runs only once per SrtFile / OsdFile instance.

Not part of the public API — imported by widgets.py only.
"""

from __future__ import annotations

import math
from typing import Any, Optional

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_decoder import (
    decode_plus_code as _decode_plus_code,
    extract_plus_code as _extract_plus_code,
)
from _widget_primitives import _parse_color


# ----- Projection cache ------------------------------------------------------

_PROJECTED_GPS_CACHE:  dict[int, Any] = {}
_PROJECTED_PLUS_CACHE: dict[int, Any] = {}
# Smoothed + Catmull-Rom trace polyline, keyed by (id(points), pw, ph). The
# polyline depends only on the track and patch size, never the playhead, so
# caching it keeps playback off the per-frame spline recompute that pushed the
# preview into slow motion.
_SMOOTH_TRACE_CACHE:   dict[tuple, list] = {}


def invalidate_track_caches() -> None:
    """Clear projection caches. Call whenever srt_data or osd_data is replaced
    so the map widget picks up the new track on the next render."""
    _PROJECTED_GPS_CACHE.clear()
    _PROJECTED_PLUS_CACHE.clear()
    _SMOOTH_TRACE_CACHE.clear()


# Treat (0, 0) and near-zero coords as "no GPS fix" — avoids a spurious point
# off the African coast when the drone hasn't acquired a lock yet.
_GPS_ZERO_EPS = 1e-3


# ----- Equirectangular projection --------------------------------------------

def _project_latlon(lats: list[float], lons: list[float]):
    """Equirectangular projection with cos(mean_lat) correction.

    Returns ``(points, bbox, cos_lat)`` where points is a list of (x, y) in
    projected units (y negated so north is up) and bbox is
    ``(min_x, max_x, min_y, max_y)``.  Returns ``(None, None, 1.0)`` for
    fewer than two valid points.
    """
    if len(lats) < 2:
        return None, None, 1.0
    mean_lat = sum(lats) / len(lats)
    cos_lat = math.cos(math.radians(mean_lat))
    if cos_lat < 0.05:
        cos_lat = 1.0
    xs = [lon * cos_lat for lon in lons]
    ys = [-lat for lat in lats]
    points = list(zip(xs, ys))
    bbox = (min(xs), max(xs), min(ys), max(ys))
    return points, bbox, cos_lat


def _project_gps_track(srt_file: Any):
    """Return cached projection of the SRT GPS track."""
    if srt_file is None or not getattr(srt_file, "entries", None):
        return None, None, 1.0
    cache_key = id(srt_file)
    cached = _PROJECTED_GPS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lats: list[float] = []
    lons: list[float] = []
    for e in srt_file.entries:
        td = e.telemetry
        lat = getattr(td, "gps_lat", None)
        lon = getattr(td, "gps_lon", None)
        if lat is None or lon is None:
            continue
        if abs(lat) < _GPS_ZERO_EPS and abs(lon) < _GPS_ZERO_EPS:
            continue
        lats.append(lat)
        lons.append(lon)
    result = _project_latlon(lats, lons)
    _PROJECTED_GPS_CACHE[cache_key] = result
    return result


def _project_plus_code_track(osd_file: Any):
    """Return cached projection of the OSD Plus Code track.

    Scans every OsdFrame for a Plus Code, decodes each to (lat, lon), and
    deduplicates consecutive identical codes before projecting.
    """
    if osd_file is None or not getattr(osd_file, "frames", None):
        return None, None, 1.0
    cache_key = id(osd_file)
    cached = _PROJECTED_PLUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lats: list[float] = []
    lons: list[float] = []
    last_code: Optional[str] = None
    for frame in osd_file.frames:
        code = _extract_plus_code(frame)
        if not code or code == last_code:
            if code:
                last_code = code
            continue
        last_code = code
        pt = _decode_plus_code(code)
        if pt is None:
            continue
        lats.append(pt[0])
        lons.append(pt[1])
    result = _project_latlon(lats, lons)
    _PROJECTED_PLUS_CACHE[cache_key] = result
    return result


def _current_position(telemetry: Any) -> Optional[tuple[float, float]]:
    """Resolve the drone's current (lat, lon) from whichever source has it.

    SRT GPS wins (1 Hz, high precision); falls back to the Plus Code in the
    current OSD frame (~14 m precision). Returns None when neither is available.
    """
    cur_lat = getattr(telemetry, "gps_lat", None)
    cur_lon = getattr(telemetry, "gps_lon", None)
    if (cur_lat is not None and cur_lon is not None
            and (abs(cur_lat) >= _GPS_ZERO_EPS or abs(cur_lon) >= _GPS_ZERO_EPS)):
        return cur_lat, cur_lon
    osd_frame = getattr(telemetry, "osd_frame", None)
    if osd_frame is not None:
        code = _extract_plus_code(osd_frame)
        if code:
            pt = _decode_plus_code(code)
            if pt is not None:
                return pt
    return None


# ----- Trace smoothing -------------------------------------------------------

def _smooth_track(pts: list[tuple[float, float]],
                  radius: int = 3) -> list[tuple[float, float]]:
    """Moving-average filter to reduce GPS jitter before spline fitting.

    Catmull-Rom is interpolating: it passes through every control point, so
    raw GPS noise (±2-5 m) survives as small wiggles. Averaging each point
    with its ``radius`` neighbours removes that high-frequency noise while
    preserving the overall flight path shape.
    """
    n = len(pts)
    if n <= 2:
        return pts
    out: list[tuple[float, float]] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n - 1, i + radius)
        w = hi - lo + 1
        out.append((
            sum(pts[j][0] for j in range(lo, hi + 1)) / w,
            sum(pts[j][1] for j in range(lo, hi + 1)) / w,
        ))
    return out


def _catmull_rom_chain(pts: list[tuple[float, float]],
                       steps: int = 8) -> list[tuple[float, float]]:
    """Return a Catmull-Rom spline through *pts*.

    Inserts *steps* interpolated points between every consecutive pair so
    the polyline appears smooth rather than angular. Endpoint tangents are
    extrapolated by duplicating the first/last control point.
    """
    n = len(pts)
    if n < 2:
        return pts
    ext = [pts[0]] + list(pts) + [pts[-1]]
    out: list[tuple[float, float]] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(steps):
            t  = j / steps
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0])
                        + (-p0[0] + p2[0]) * t
                        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1])
                        + (-p0[1] + p2[1]) * t
                        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-1])
    return out


# ----- Draw ------------------------------------------------------------------

def _draw_map(img: Any, w: Any, telemetry: Any,
              rx: int, ry: int, ww: int, wh: int) -> None:
    """Draw the GPS track + current-position marker into the widget rect.

    Unlike the scalar-value draw functions this one needs whole-track context,
    so it receives the full TelemetryFrame. Track source: SRT GPS preferred;
    falls back to scanning the OsdFile for Plus Codes.
    """
    if ww < 8 or wh < 8:
        return
    srt_file = getattr(telemetry, "srt_file", None)
    points, bbox, cos_lat = _project_gps_track(srt_file)
    if not points or bbox is None:
        osd_file = getattr(telemetry, "osd_file", None)
        points, bbox, cos_lat = _project_plus_code_track(osd_file)
    if not points or bbox is None:
        return

    style = w.style
    trace_color  = _parse_color(style.get("color", "#FFFFFF"),
                                alpha=int(255 * float(style.get("opacity", 0.95))))
    marker_color = _parse_color(style.get("marker_color", "#FF3030"))
    bg_color     = _parse_color(style.get("bg_color", "#000000"),
                                alpha=int(255 * float(style.get("bg_opacity", 0.35))))
    border_color = _parse_color(style.get("border_color", "#FFFFFF"),
                                alpha=int(255 * float(style.get("border_opacity", 0.30))))
    padding_frac = max(0.0, min(0.25, float(style.get("padding", 0.08))))
    trace_w_pct  = float(style.get("trace_width", 0.012))
    marker_pct   = float(style.get("marker_size", 0.035))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    if bg_color[3] > 0 or border_color[3] > 0:
        radius = max(1, int(min(pw, ph) * 0.06))
        pdraw.rounded_rectangle(
            [0, 0, pw - 1, ph - 1], radius=radius,
            fill=bg_color if bg_color[3] > 0 else None,
            outline=border_color if border_color[3] > 0 else None,
            width=max(1, SS),
        )

    pad_px_x = int(pw * padding_frac)
    pad_px_y = int(ph * padding_frac)
    inner_w = pw - 2 * pad_px_x
    inner_h = ph - 2 * pad_px_y
    if inner_w < 4 or inner_h < 4:
        return

    min_x, max_x, min_y, max_y = bbox
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min(inner_w / span_x, inner_h / span_y)
    cx_data   = (min_x + max_x) / 2.0
    cy_data   = (min_y + max_y) / 2.0
    cx_screen = pw / 2.0
    cy_screen = ph / 2.0

    def _to_screen(x: float, y: float) -> tuple[float, float]:
        return (cx_screen + (x - cx_data) * scale,
                cy_screen + (y - cy_data) * scale)

    # Project → denoise (moving avg) → Catmull-Rom → draw. Cached per
    # (track, patch size): independent of the playhead, so per-frame playback
    # reuses it instead of recomputing the spline every frame.
    trace_px  = max(1 * SS, int(min(pw, ph) * trace_w_pct))
    cache_key = (id(points), pw, ph)
    smooth_pts = _SMOOTH_TRACE_CACHE.get(cache_key)
    if smooth_pts is None:
        screen_pts = [_to_screen(x, y) for x, y in points]
        smooth_pts = _catmull_rom_chain(_smooth_track(screen_pts, radius=3), steps=8)
        _SMOOTH_TRACE_CACHE[cache_key] = smooth_pts
    pdraw.line(smooth_pts, fill=trace_color, width=trace_px, joint="curve")

    cur = _current_position(telemetry)
    if cur is not None:
        cur_lat, cur_lon = cur
        cur_x = cur_lon * cos_lat
        cur_y = -cur_lat
        mx, my = _to_screen(cur_x, cur_y)
        mr     = max(2 * SS, int(min(pw, ph) * marker_pct))
        halo_r = mr + max(1 * SS, int(mr * 0.35))
        pdraw.ellipse([mx - halo_r, my - halo_r, mx + halo_r, my + halo_r],
                      fill=(255, 255, 255, 180))
        pdraw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=marker_color)

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)

# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_widget_map.py - GPS map widget: projection, smoothing, and rendering.

Track source priority: SRT GPS (1 Hz, sub-metre precision) → raw OSD lat/lon
(SYM_LAT/SYM_LON decimal readout) → OSD Plus Code fallback (~14 m grid
resolution). Projection is cached by object id so the expensive scan runs
only once per SrtFile / OsdFile instance.

Not part of the public API — imported by widgets.py only.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageChops
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_decoder import (
    decode_plus_code as _decode_plus_code,
    extract_plus_code as _extract_plus_code,
    extract_gps_coords as _extract_gps_coords,
)
from _widget_primitives import _parse_color, _font, _polar
from _widget_map_tiles import (
    build_basemap as _build_basemap,
    is_underlay as _is_underlay,
    attribution as _tile_attribution,
)


# ----- Projection cache ------------------------------------------------------

_PROJECTED_GPS_CACHE:    dict[int, Any] = {}
_PROJECTED_OSDLATLON_CACHE: dict[int, Any] = {}
_PROJECTED_PLUS_CACHE:   dict[int, Any] = {}
# Smoothed + Catmull-Rom trace polyline, keyed by (id(points), pw, ph). The
# polyline depends only on the track and patch size, never the playhead, so
# caching it keeps playback off the per-frame spline recompute that pushed the
# preview into slow motion.
_SMOOTH_TRACE_CACHE:   dict[tuple, list] = {}
# Assembled tile basemap + its lat/lon→pixel projector, keyed by
# (provider, pw, ph, rounded-bbox). The basemap is static per flight, so this
# keeps the (network + stitch) work off the per-frame path and the export loop.
_BASEMAP_CACHE:        dict[tuple, Any] = {}
# Monotonic timestamp of the last failed basemap build per key, so an offline
# session doesn't re-spin the tile fetch every frame.
_BASEMAP_FAIL:         dict[tuple, float] = {}
_BASEMAP_RETRY_S = 30.0


def invalidate_track_caches() -> None:
    """Clear projection caches. Call whenever srt_data or osd_data is replaced
    so the map widget picks up the new track on the next render."""
    _PROJECTED_GPS_CACHE.clear()
    _PROJECTED_OSDLATLON_CACHE.clear()
    _PROJECTED_PLUS_CACHE.clear()
    _SMOOTH_TRACE_CACHE.clear()
    _BASEMAP_CACHE.clear()
    _BASEMAP_FAIL.clear()


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


def _project_osd_latlon_track(osd_file: Any):
    """Return cached projection of the raw OSD lat/lon track.

    Scans every OsdFrame for SYM_LAT/SYM_LON coordinate readouts and projects
    the de-duplicated sequence. Higher precision than the Plus Code grid, so
    it ranks above it when there's no SRT GPS.
    """
    if osd_file is None or not getattr(osd_file, "frames", None):
        return None, None, 1.0
    cache_key = id(osd_file)
    cached = _PROJECTED_OSDLATLON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lats: list[float] = []
    lons: list[float] = []
    last: Optional[tuple[float, float]] = None
    for frame in osd_file.frames:
        pt = _extract_gps_coords(frame)
        if pt is None or pt == last:
            continue
        last = pt
        lats.append(pt[0])
        lons.append(pt[1])
    result = _project_latlon(lats, lons)
    _PROJECTED_OSDLATLON_CACHE[cache_key] = result
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
        pt = _extract_gps_coords(osd_frame)
        if pt is not None:
            return pt
        code = _extract_plus_code(osd_frame)
        if code:
            pt = _decode_plus_code(code)
            if pt is not None:
                return pt
    return None


# ----- Heading ----------------------------------------------------------------

def _compute_heading(cur: tuple[float, float],
                     track_ll: list[tuple[float, float]]) -> Optional[float]:
    """Return the bearing (degrees from north, clockwise) of travel at ``cur``.

    Finds the two nearest track points bracketing the current position and
    computes the forward azimuth between them. Returns None when the track
    is too short or the points are too close to determine direction.
    """
    if not track_ll or len(track_ll) < 2:
        return None
    # Find the track point closest to cur.
    cur_lat, cur_lon = cur
    best_i = 0
    best_d2 = float("inf")
    for i, (lat, lon) in enumerate(track_ll):
        d2 = (lat - cur_lat) ** 2 + (lon - cur_lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    # Bracket: one point behind, one ahead (or either side of best_i).
    prev_i = max(0, best_i - 1)
    next_i = min(len(track_ll) - 1, best_i + 1)
    if prev_i == next_i:
        return None
    lat1, lon1 = track_ll[prev_i]
    lat2, lon2 = track_ll[next_i]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Only trust bearing when the points are far enough apart.
    if abs(dlat) < 1e-8 and abs(dlon) < 1e-8:
        if best_i > 0:
            lat1, lon1 = track_ll[best_i - 1]
        elif best_i + 1 < len(track_ll):
            lat1, lon1 = track_ll[best_i]
            lat2, lon2 = track_ll[best_i + 1]
        else:
            return None
        dlat = lat2 - lat1
        dlon = lon2 - lon1
    if abs(dlat) < 1e-8 and abs(dlon) < 1e-8:
        return None
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon_r = math.radians(dlon)
    y = math.sin(dlon_r) * math.cos(lat2_r)
    x = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0


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


# ----- Tile basemap (cached) -------------------------------------------------

def _get_basemap(lats: list[float], lons: list[float], bbox: tuple,
                 provider: str, pw: int, ph: int):
    """Return ``(basemap_img, to_px)`` for the track, or None.

    Caches the assembled basemap per (provider, patch size, rounded bbox) so
    the tile fetch + stitch runs once per flight rather than per frame. After
    a failed build (e.g. offline + uncached) it rate-limits retries so the
    preview/export loop doesn't keep stalling on a dead network.
    """
    if not lats or len(lats) < 2:
        return None
    min_x, max_x, min_y, max_y = bbox
    key = (provider, pw, ph,
           round(min_x, 4), round(max_x, 4), round(min_y, 4), round(max_y, 4))
    cached = _BASEMAP_CACHE.get(key)
    if cached is not None:
        return cached
    last_fail = _BASEMAP_FAIL.get(key)
    if last_fail is not None and (time.monotonic() - last_fail) < _BASEMAP_RETRY_S:
        return None
    result = _build_basemap(lats, lons, provider, pw, ph)
    if result is None:
        _BASEMAP_FAIL[key] = time.monotonic()
        return None
    _BASEMAP_CACHE[key] = result
    _BASEMAP_FAIL.pop(key, None)
    return result


def _draw_attribution(patch: Any, text: str, pw: int, ph: int, SS: int) -> None:
    """Draw the tile provider's required attribution in the bottom-left."""
    d = ImageDraw.Draw(patch)
    font = _font(max(7, int(ph * 0.045)))
    try:
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = len(text) * 6, 11
    pad = max(2, SS)
    x = pad + pad
    y = ph - th - 3 * pad
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(0, 0, 0, 130))
    d.text((x, y), text, fill=(255, 255, 255, 210), font=font)


# ----- Marker drawing ---------------------------------------------------------

def _draw_marker_dot(pdraw: Any, cx: float, cy: float, r: int,
                     color: tuple, SS: int) -> None:
    """Filled circle with a white halo — the original marker style."""
    halo_r = r + max(1 * SS, int(r * 0.35))
    pdraw.ellipse([cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r],
                  fill=(255, 255, 255, 180))
    pdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_marker_quadcopter(pdraw: Any, cx: float, cy: float, r: int,
                            color: tuple, SS: int) -> None:
    """Small quadcopter icon: central body, four arms, four motor circles."""
    arm_len = r * 1.8
    arm_w   = max(1 * SS, int(r * 0.22))
    motor_r = max(1.5 * SS, r * 0.28)
    body_rx = r * 0.42
    body_ry = r * 0.30

    angles = [45, 135, 225, 315]
    # Arms
    for a in angles:
        ex, ey = _polar(cx, cy, arm_len, a)
        pdraw.line([(cx, cy), (ex, ey)], fill=color, width=arm_w)
    # Motors
    motor_color = (min(255, color[0] + 60), min(255, color[1] + 60),
                   min(255, color[2] + 60), color[3])
    for a in angles:
        mx, my = _polar(cx, cy, arm_len, a)
        pdraw.ellipse([mx - motor_r, my - motor_r,
                       mx + motor_r, my + motor_r], fill=motor_color)
    # Body
    pdraw.ellipse([cx - body_rx, cy - body_ry, cx + body_rx, cy + body_ry],
                  fill=color)


_PLANE_ICON: Any = None


def _plane_icon():
    global _PLANE_ICON
    if _PLANE_ICON is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "plane_marker.png")
        try:
            _PLANE_ICON = Image.open(path).convert("RGB")
        except Exception:
            _PLANE_ICON = False
    return _PLANE_ICON if _PLANE_ICON else None


def _draw_marker_plane(pdraw: Any, cx: float, cy: float, r: int,
                       color: tuple, heading: float, SS: int) -> None:
    """Plane marker loaded from assets/plane_marker.png, rotated to heading.

    Nose points UP in the source image.  White background is made transparent
    and the plane is tinted red.
    """
    icon = _plane_icon()
    if icon is None:
        _draw_marker_dot(pdraw, cx, cy, r, color, SS)
        return

    icon_size = max(8 * SS, int(r * 5))
    iw, ih = icon.size
    scale = icon_size / max(iw, ih)
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    scaled = icon.resize((new_w, new_h), Image.LANCZOS)

    # Make white background transparent, tint the plane red.
    arr = np.array(scaled).astype(np.float32)
    whiteness = arr.min(axis=2)  # how white each pixel is
    alpha = np.clip((255.0 - whiteness) / 60.0, 0.0, 1.0)
    alpha = alpha * (color[3] / 255.0)
    red = (220, 40, 40)
    out = np.zeros((new_h, new_w, 4), dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = alpha * red[ch]
    out[:, :, 3] = alpha * 255.0
    tinted = Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")

    angle = -heading
    rotated = tinted.rotate(angle, Image.BICUBIC, expand=True,
                            fillcolor=(0, 0, 0, 0))
    rw, rh = rotated.size
    pdraw._image.paste(rotated, (int(cx - rw / 2.0), int(cy - rh / 2.0)), rotated)


# ----- Draw ------------------------------------------------------------------

def _draw_map(img: Any, w: Any, telemetry: Any,
              rx: int, ry: int, ww: int, wh: int) -> None:
    """Draw the GPS track + current-position marker into the widget rect.

    Unlike the scalar-value draw functions this one needs whole-track context,
    so it receives the full TelemetryFrame. Track source: SRT GPS preferred,
    then raw OSD lat/lon, then OSD Plus Codes. An optional ``map_underlay``
    style ("street"/"satellite") draws a slippy-map basemap beneath the track;
    when tiles are unavailable it transparently falls back to the plain
    equirectangular fit-to-bbox layout.
    """
    if ww < 8 or wh < 8:
        return
    srt_file = getattr(telemetry, "srt_file", None)
    points, bbox, cos_lat = _project_gps_track(srt_file)
    if not points or bbox is None:
        osd_file = getattr(telemetry, "osd_file", None)
        points, bbox, cos_lat = _project_osd_latlon_track(osd_file)
        if not points or bbox is None:
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
    marker_shape = str(style.get("marker_shape", "dot"))
    underlay     = str(style.get("map_underlay", "none"))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)
    corner_r = max(1, int(min(pw, ph) * 0.06))

    if bg_color[3] > 0:
        pdraw.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=corner_r,
                                fill=bg_color)

    # Recover raw lat/lon from the projected track (lat = -y, lon = x/cos_lat)
    # so both the equirectangular and the Web-Mercator (tile) paths share one
    # coordinate source.
    track_ll = [(-y, (x / cos_lat) if cos_lat else x) for x, y in points]

    # ── Resolve a lat/lon → patch-pixel projector ─────────────────────────
    project = None
    attribution = ""
    if _is_underlay(underlay):
        base = _get_basemap([ll[0] for ll in track_ll],
                            [ll[1] for ll in track_ll],
                            bbox, underlay, pw, ph)
        if base is not None:
            basemap, to_px = base
            # Clip the basemap to the widget's rounded rectangle. Combine the
            # basemap's own alpha with the rounded mask and paste through it,
            # WITHOUT mutating the cached basemap (putalpha would re-erode the
            # anti-aliased corners every reused frame).
            mask = Image.new("L", (pw, ph), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                [0, 0, pw - 1, ph - 1], radius=corner_r, fill=255)
            clip = ImageChops.multiply(basemap.split()[3], mask)
            patch.paste(basemap, (0, 0), clip)
            project = to_px
            attribution = _tile_attribution(underlay)

    if project is None:
        # Equirectangular fit-to-bbox (no underlay, or tiles unavailable).
        pad_px_x = int(pw * padding_frac)
        pad_px_y = int(ph * padding_frac)
        inner_w = pw - 2 * pad_px_x
        inner_h = ph - 2 * pad_px_y
        if inner_w < 4 or inner_h < 4:
            return
        min_x, max_x, min_y, max_y = bbox
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        e_scale = min(inner_w / span_x, inner_h / span_y)
        cx_data = (min_x + max_x) / 2.0
        cy_data = (min_y + max_y) / 2.0

        def project(lat: float, lon: float) -> tuple[float, float]:
            x = lon * cos_lat
            y = -lat
            return (pw / 2.0 + (x - cx_data) * e_scale,
                    ph / 2.0 + (y - cy_data) * e_scale)

    # Project → denoise (moving avg) → Catmull-Rom → draw. Cached per
    # (track, patch size, projector mode): independent of the playhead, so
    # per-frame playback reuses it instead of recomputing the spline.
    trace_px  = max(1 * SS, int(min(pw, ph) * trace_w_pct))
    mode_tag  = underlay if attribution else "equi"
    cache_key = (id(points), pw, ph, mode_tag)
    smooth_pts = _SMOOTH_TRACE_CACHE.get(cache_key)
    if smooth_pts is None:
        screen_pts = [project(lat, lon) for lat, lon in track_ll]
        smooth_pts = _catmull_rom_chain(_smooth_track(screen_pts, radius=3), steps=8)
        _SMOOTH_TRACE_CACHE[cache_key] = smooth_pts
    pdraw.line(smooth_pts, fill=trace_color, width=trace_px, joint="curve")

    cur = _current_position(telemetry)
    if cur is not None:
        mx, my = project(cur[0], cur[1])
        mr     = max(2 * SS, int(min(pw, ph) * marker_pct))

        if marker_shape == "quadcopter":
            _draw_marker_quadcopter(pdraw, mx, my, mr, marker_color, SS)
        elif marker_shape == "plane":
            heading = _compute_heading(cur, track_ll)
            if heading is None:
                heading = 0.0
            _draw_marker_plane(pdraw, mx, my, mr, marker_color, heading, SS)
        else:
            _draw_marker_dot(pdraw, mx, my, mr, marker_color, SS)

    # Border last so it frames the basemap cleanly.
    if border_color[3] > 0:
        pdraw.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=corner_r,
                                outline=border_color, width=max(1, SS))

    if attribution:
        _draw_attribution(patch, attribution, pw, ph, SS)

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)

# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_widget_map_tiles.py - Slippy-map tile fetching + Web Mercator basemap for the
GPS map widget underlay.

Providers (free, no API key required):
  "street"    - OpenStreetMap standard raster tiles
  "satellite" - Esri World Imagery

build_basemap() returns a basemap image already sized to the widget patch plus
a ``to_px(lat, lon)`` projector so the GPS track lines up pixel-for-pixel with
the tiles. Tiles are cached in memory and on disk (one PNG per tile under
%APPDATA%/VueOSD/tilecache), so a flight's basemap is fetched once and then
reused across every preview frame, the export pass, and app restarts.

Offline-safe: when a tile isn't cached and the network is unavailable,
build_basemap() returns None and the caller falls back to the plain coloured
map background. A short global cooldown after any failure stops every frame
from re-blocking on a dead network.

Not part of the public API - imported by _widget_map.py only.
"""

from __future__ import annotations

import math
import os
import threading
import time
from io import BytesIO
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.error

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


TILE_SIZE = 256
_HTTP_TIMEOUT = 4.0
_USER_AGENT = "VueOSD/1.7 (+https://github.com/wkumik/Digital-FPV-OSD-Tool)"

# After a network failure, skip live fetches for this long so an offline
# session falls back instantly instead of stalling every frame on timeouts.
_NET_COOLDOWN_S = 30.0

PROVIDERS: dict[str, dict] = {
    "street": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "max_zoom": 19,
        "attribution": "(c) OpenStreetMap contributors",
    },
    "satellite": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "max_zoom": 19,
        "attribution": "Imagery (c) Esri",
    },
}


def is_underlay(mode: Optional[str]) -> bool:
    """True when ``mode`` names a real tile provider (not 'none'/empty)."""
    return bool(mode) and mode in PROVIDERS


def attribution(mode: str) -> str:
    p = PROVIDERS.get(mode)
    return p["attribution"] if p else ""


# ----- Disk + memory tile cache ---------------------------------------------

_MEM_TILES: dict[tuple, Any] = {}
_FAILED: set[tuple] = set()
_LOCK = threading.Lock()
_net_cooldown_until = 0.0


def _cache_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "VueOSD", "tilecache")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _tile_path(provider: str, z: int, x: int, y: int) -> str:
    return os.path.join(_cache_dir(), provider, str(z), str(x), f"{y}.png")


def _fetch_tile(provider: str, z: int, x: int, y: int):
    """Return one RGBA tile from memory → disk → network, or None.

    Network is skipped (returns None immediately) while a failure cooldown is
    active, so an offline session doesn't pay the HTTP timeout on every frame.
    """
    if not PIL_OK:
        return None
    key = (provider, z, x, y)
    with _LOCK:
        if key in _MEM_TILES:
            return _MEM_TILES[key]
        if key in _FAILED:
            return None
        cooling = time.monotonic() < _net_cooldown_until

    path = _tile_path(provider, z, x, y)
    img = None
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
        except Exception:
            img = None

    if img is None:
        if cooling:
            return None
        url = PROVIDERS[provider]["url"].format(z=z, x=x, y=y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                data = r.read()
            img = Image.open(BytesIO(data)).convert("RGBA")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
            except OSError:
                pass
        except Exception:
            with _LOCK:
                _FAILED.add(key)
                globals()["_net_cooldown_until"] = time.monotonic() + _NET_COOLDOWN_S
            return None

    with _LOCK:
        _MEM_TILES[key] = img
    return img


# ----- Web Mercator projection ----------------------------------------------

def _lonlat_to_pixel(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Lat/lon → global Web-Mercator pixel coords at zoom ``z`` (256-px tiles)."""
    n = TILE_SIZE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * n
    return x, y


def _choose_zoom(min_lat: float, max_lat: float, min_lon: float, max_lon: float,
                 out_w: int, out_h: int, max_zoom: int) -> int:
    """Largest zoom whose pixel span for the bbox still fits ``out_w×out_h``."""
    for z in range(max_zoom, 0, -1):
        xl, yt = _lonlat_to_pixel(max_lat, min_lon, z)
        xr, yb = _lonlat_to_pixel(min_lat, max_lon, z)
        if (xr - xl) <= out_w and (yb - yt) <= out_h:
            return z
    return 1


# ----- Basemap assembly ------------------------------------------------------

def build_basemap(lats: list[float], lons: list[float], provider: str,
                  out_w: int, out_h: int, pad_frac: float = 0.12
                  ) -> Optional[tuple[Any, Callable[[float, float],
                                                    tuple[float, float]]]]:
    """Assemble a basemap image of ``out_w×out_h`` for the given track.

    Returns ``(image, to_px)`` where ``to_px(lat, lon)`` maps a coordinate to a
    pixel inside ``image`` (aligned with the tiles), or None when no tiles
    could be obtained (offline + uncached) or inputs are unusable.
    """
    if not PIL_OK or provider not in PROVIDERS:
        return None
    if len(lats) < 2 or len(lons) < 2 or out_w < 8 or out_h < 8:
        return None

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    dlat = (max_lat - min_lat) or 1e-4
    dlon = (max_lon - min_lon) or 1e-4
    min_lat -= dlat * pad_frac
    max_lat += dlat * pad_frac
    min_lon -= dlon * pad_frac
    max_lon += dlon * pad_frac

    max_zoom = PROVIDERS[provider]["max_zoom"]
    z = _choose_zoom(min_lat, max_lat, min_lon, max_lon, out_w, out_h, max_zoom)

    # World-pixel bbox of the (padded) track. North edge (max_lat) → smaller y.
    wx_min, wy_min = _lonlat_to_pixel(max_lat, min_lon, z)
    wx_max, wy_max = _lonlat_to_pixel(min_lat, max_lon, z)
    crop_w = max(1.0, wx_max - wx_min)
    crop_h = max(1.0, wy_max - wy_min)

    # Expand the window to the widget's aspect ratio (around the track centre)
    # so the basemap fills the whole widget — no transparent letterbox bars —
    # while still fully containing the track. The extra area just shows more
    # map context on the longer axis.
    target = out_w / out_h
    cx = (wx_min + wx_max) / 2.0
    cy = (wy_min + wy_max) / 2.0
    if crop_w / crop_h < target:
        crop_w = crop_h * target
    else:
        crop_h = crop_w / target
    wx_min, wx_max = cx - crop_w / 2.0, cx + crop_w / 2.0
    wy_min, wy_max = cy - crop_h / 2.0, cy + crop_h / 2.0

    tx0, tx1 = int(wx_min // TILE_SIZE), int(wx_max // TILE_SIZE)
    ty0, ty1 = int(wy_min // TILE_SIZE), int(wy_max // TILE_SIZE)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if n_tiles > 64:        # sanity guard; zoom choice keeps this small
        return None

    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1)
            for tx in range(tx0, tx1 + 1)]
    nmax = 2 ** z

    def _load(job):
        tx, ty = job
        return job, _fetch_tile(provider, z, tx % nmax, ty % nmax)

    stitch_w = (tx1 - tx0 + 1) * TILE_SIZE
    stitch_h = (ty1 - ty0 + 1) * TILE_SIZE
    stitch = Image.new("RGBA", (stitch_w, stitch_h), (40, 40, 44, 255))
    fetched = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (tx, ty), tile in ex.map(_load, jobs):
            if tile is None:
                continue
            stitch.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))
            fetched += 1
    if fetched == 0:
        return None

    # Resample the exact (sub-pixel) window straight to the widget size. PIL's
    # ``box`` takes a float source region, so the basemap fills out_w×out_h
    # precisely and the track projector below lines up pixel-for-pixel.
    box = (wx_min - tx0 * TILE_SIZE, wy_min - ty0 * TILE_SIZE,
           wx_max - tx0 * TILE_SIZE, wy_max - ty0 * TILE_SIZE)
    basemap = stitch.resize((out_w, out_h), Image.LANCZOS, box=box)

    sx = out_w / crop_w
    sy = out_h / crop_h

    def to_px(lat: float, lon: float) -> tuple[float, float]:
        wx, wy = _lonlat_to_pixel(lat, lon, z)
        return ((wx - wx_min) * sx, (wy - wy_min) * sy)

    return basemap, to_px

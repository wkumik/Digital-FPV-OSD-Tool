# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_common.py — Shared constants and helpers for the player module.

Contains: slider/cache-size constants, the PIL import guard, the FFmpeg
rawvideo command builder, and rgb24 -> RGBA frame decoding.
"""

_SL_MAX = 10_000
_CACHE_MAX = 64
# Playback: how far behind the real-time schedule frame-dropping is allowed
# to chase before we give up on real time and re-anchor the playback clock.
# Beyond this, dropping would starve the display entirely (frozen preview);
# re-anchoring instead plays slightly slower than real time but keeps moving.
_MAX_PLAYBACK_LAG_S = 0.35

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False


def rawvideo_cmd(ffmpeg, video_path, seek_s, scale_w, scale_h,
                 color_vf="", vframes=None, hwaccel=False):
    """FFmpeg command producing rgb24 rawvideo on stdout.

    hwaccel: best-effort GPU decode (playback pipe); FFmpeg falls back to
    software on its own if no hwaccel is available.
    """
    cmd = [ffmpeg]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-ss", str(seek_s), "-i", video_path]
    vf_parts = [f for f in [color_vf, f"scale={scale_w}:{scale_h}"] if f]
    cmd += ["-vf", ",".join(vf_parts)]
    if vframes is not None:
        cmd += ["-vframes", str(vframes)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-v", "quiet", "pipe:1"]
    return cmd


def rgba_from_rgb24(data, w, h):
    """bytes -> RGBA PIL image, or None if data is short or PIL is missing."""
    nbytes = w * h * 3
    if not PIL_OK or not data or len(data) < nbytes:
        return None
    return PILImage.frombytes("RGB", (w, h), data[:nbytes]).convert("RGBA")

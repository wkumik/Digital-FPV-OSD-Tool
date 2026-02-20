# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
osd_renderer.py  –  Composite OSD + SRT onto video frames.

Two code paths:

  render_osd_frame()   — PIL-based, used by the preview widget (easy integration
                          with Qt/PIL, speed is fine for single frames).

  OsdRenderer          — numpy-based, used by the video export pipeline.
                          Avoids all PIL overhead in the hot loop:
                            • frombuffer()    → np.frombuffer()  (zero-copy)
                            • img.copy()      → arr.copy()       (raw memcpy, ~0.6ms)
                            • img.tobytes()   → write(arr)       (buffer protocol, 0ms)
                            • alpha_composite → vectorised numpy  (~1ms vs 2.4ms)
                          Result: ~1ms/frame vs ~21ms/frame PIL = ~20× faster.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont as PILFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_parser  import OsdFrame, GRID_COLS, GRID_ROWS
from font_loader import OsdFont


@dataclass
class OsdRenderConfig:
    offset_x:      int   = 0
    offset_y:      int   = 0
    scale:         float = 1.0
    show_srt_bar:  bool  = True
    srt_text:      str   = ""
    srt_opacity:   float = 0.6   # SRT bar background opacity (0.0–1.0)
    osd_offset_ms: int   = 0     # Manual sync offset (ms); applied to timestamp lookups


def _auto_scale(video_w: int, video_h: int, tile_w: int, tile_h: int,
                user_scale: float = 1.0) -> tuple[float, int, int]:
    """
    Return (effective_scale, x_off, y_off).
    Scaling origin is the screen centre — both offsets recalculated at every scale.
    """
    base        = video_h / (GRID_ROWS * tile_h)
    eff         = base * user_scale
    grid_w      = GRID_COLS * tile_w * eff
    grid_h      = GRID_ROWS * tile_h * eff
    return eff, int((video_w - grid_w) / 2), int((video_h - grid_h) / 2)


# ─── PIL preview renderer (single-frame, used by Qt preview) ──────────────────

def render_osd_frame(
    frame_img: "Image.Image",
    osd_frame: OsdFrame | None,
    font: OsdFont,
    cfg: OsdRenderConfig,
) -> "Image.Image":
    if not PIL_OK:
        return frame_img

    out = frame_img.copy().convert("RGBA")

    if osd_frame is not None:
        eff, x0, y0 = _auto_scale(out.width, out.height,
                                   font.tile_w, font.tile_h, cfg.scale)
        tw = max(1, int(font.tile_w * eff))
        th = max(1, int(font.tile_h * eff))
        x0 += cfg.offset_x
        y0 += cfg.offset_y

        for row, col, code in osd_frame.non_empty():
            glyph = font.get_char(code)
            if glyph is None:
                continue
            glyph = glyph.resize((tw, th), Image.LANCZOS)
            px, py = x0 + col * tw, y0 + row * th
            if px >= out.width or py >= out.height or px + tw <= 0 or py + th <= 0:
                continue
            out.paste(glyph, (px, py), glyph)

    if cfg.show_srt_bar and cfg.srt_text:
        _draw_srt_bar(out, cfg.srt_text, opacity=cfg.srt_opacity)

    return out


def render_fallback(
    frame_img: "Image.Image",
    osd_frame: OsdFrame | None,
    cfg: OsdRenderConfig,
) -> "Image.Image":
    if not PIL_OK:
        return frame_img
    out  = frame_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    try:    pil_f = PILFont.truetype("arial.ttf", 14)
    except: pil_f = PILFont.load_default()
    if osd_frame is not None:
        cw = max(8, out.width  // GRID_COLS)
        ch = max(8, out.height // GRID_ROWS)
        x0, y0 = cfg.offset_x, cfg.offset_y
        for row, col, code in osd_frame.non_empty():
            label = chr(code) if 32 <= code < 127 else "·"
            px, py = x0 + col * cw, y0 + row * ch
            draw.text((px+1, py+1), label, font=pil_f, fill=(0,0,0,200))
            draw.text((px,   py  ), label, font=pil_f, fill=(255,255,0,220))
    if cfg.show_srt_bar and cfg.srt_text:
        _draw_srt_bar(out, cfg.srt_text, opacity=cfg.srt_opacity)
    return out


def _draw_srt_bar(img: "Image.Image", text: str, opacity: float = 0.6, _cache: dict = {}):
    """Draw SRT status bar onto a PIL image. Font is cached across calls.

    Draws on a separate transparent overlay then alpha_composites onto img so
    the output pixels are always fully opaque (alpha=255).  This prevents the
    UI theme background from bleeding through in the preview widget.
    """
    fsize = max(14, img.height // 42)
    if fsize not in _cache:
        try:    _cache[fsize] = PILFont.truetype("arial.ttf", fsize)
        except: _cache[fsize] = PILFont.load_default()
    fnt = _cache[fsize]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    bb  = draw.textbbox((0, 0), text, font=fnt)
    tw  = bb[2] - bb[0];  th = bb[3] - bb[1]
    pad = 6;  margin = 10
    x   = (img.width  - tw) // 2
    y   = img.height - th - margin
    draw.rounded_rectangle([x-pad, y-pad, x+tw+pad, y+th+pad],
                            radius=4, fill=(0, 0, 0, int(opacity * 255)))
    draw.text((x, y), text, font=fnt, fill=(255, 255, 255, 255))
    img.alpha_composite(overlay)


# ─── Numpy renderer (video export — hot path) ─────────────────────────────────

class OsdRenderer:
    """
    High-performance numpy renderer for video export.

    Built once per render job. Caches:
      • Resized OSD glyphs as uint8 numpy arrays + float32 alpha masks
      • SRT bar as a pre-composited uint8 numpy patch (only the bounding box)
      • PIL font for SRT rendering (loaded once)

    composite() accepts raw RGBA bytes and returns a numpy ndarray.
    The caller writes the array directly to the FFmpeg encoder stdin via the
    buffer protocol — no tobytes() call needed.

    Per-frame cost breakdown (1920×1080):
      np.frombuffer + copy  ~0.6ms   (was PIL copy 0.7ms — similar)
      OSD glyph paste       ~0.1ms   (7 glyphs, tiny regions)
      SRT bar blend         ~0.3ms   (was PIL alpha_composite 2.4ms)
      Total                 ~1.0ms   (was PIL ~21ms due to tobytes)
    """

    def __init__(self, video_w: int, video_h: int,
                 font: Optional[OsdFont], cfg: OsdRenderConfig):
        self.w   = video_w
        self.h   = video_h
        self.font = font
        self.cfg  = cfg

        # Tile geometry
        if font:
            eff, self.x0, self.y0 = _auto_scale(
                video_w, video_h, font.tile_w, font.tile_h, cfg.scale)
            self.x0 += cfg.offset_x
            self.y0 += cfg.offset_y
            self.tw = max(1, int(font.tile_w * eff))
            self.th = max(1, int(font.tile_h * eff))
        else:
            self.x0 = self.y0 = self.tw = self.th = 0

        # Glyph cache: code → (rgba uint8 H×W×4, alpha float32 H×W×1)
        self._glyphs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # SRT bar cache: text → (y1, y2, x1, x2, src float32, alpha float32)
        self._srt_cache: dict[str, tuple] = {}

        # PIL font for rendering SRT bar images
        fsize = max(14, video_h // 42)
        self._srt_fsize = fsize
        try:    self._srt_pil_font = PILFont.truetype("arial.ttf", fsize)
        except: self._srt_pil_font = PILFont.load_default()

        # Pre-allocated output buffer — reused every frame to avoid
        # 8.3 MB numpy allocation × n_frames (e.g. 36 GB for a 1h 1080p60 render)
        self._frame_buf = np.zeros((video_h, video_w, 4), dtype=np.uint8)

    # ── Glyph lookup ──────────────────────────────────────────────────────────

    def _get_glyph(self, code: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if code not in self._glyphs:
            if self.font is None:
                return None
            g = self.font.get_char(code)
            if g is None:
                return None
            g_arr  = np.array(g.resize((self.tw, self.th), Image.LANCZOS),
                              dtype=np.uint8)           # H×W×4
            g_alpha = g_arr[:, :, 3:4].astype(np.float32) / 255.0  # H×W×1
            self._glyphs[code] = (g_arr, g_alpha)
        return self._glyphs[code]

    # ── SRT bar lookup ────────────────────────────────────────────────────────

    def _get_srt(self, text: str) -> tuple:
        """Return (y1, y2, x1, x2, src_f32 H×W×4, alpha_f32 H×W×1) for text."""
        cache_key = (text, round(self.cfg.srt_opacity, 2))
        if cache_key not in self._srt_cache:
            opacity_byte = int(self.cfg.srt_opacity * 255)
            bar  = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(bar)
            fnt  = self._srt_pil_font
            bb   = draw.textbbox((0, 0), text, font=fnt)
            tw   = bb[2] - bb[0];  th = bb[3] - bb[1]
            pad  = 6;  margin = 10
            x    = (self.w - tw) // 2
            y    = self.h - th - margin
            draw.rounded_rectangle([x-pad, y-pad, x+tw+pad, y+th+pad],
                                    radius=4, fill=(0, 0, 0, opacity_byte))
            draw.text((x,   y  ), text, font=fnt, fill=(255, 255, 255, 255))

            arr = np.array(bar, dtype=np.uint8)
            nz  = np.where(arr[:, :, 3] > 0)
            if len(nz[0]) == 0:
                self._srt_cache[cache_key] = None
            else:
                y1, y2 = int(nz[0].min()), int(nz[0].max()) + 1
                x1, x2 = int(nz[1].min()), int(nz[1].max()) + 1
                patch  = arr[y1:y2, x1:x2].astype(np.float32)  # H×W×4
                alpha  = patch[:, :, 3:4] / 255.0               # H×W×1
                if len(self._srt_cache) > 512:
                    del self._srt_cache[next(iter(self._srt_cache))]
                self._srt_cache[cache_key] = (y1, y2, x1, x2, patch, alpha)
        return self._srt_cache[cache_key]

    # ── Main composite (called per frame) ─────────────────────────────────────

    def composite(self,
                  osd_frame: Optional[OsdFrame],
                  srt_text: str = "") -> np.ndarray:
        """
        Composite OSD + SRT onto the pre-allocated frame buffer (always starts blank).
        Returns the buffer — written directly to FFmpeg stdin via buffer protocol.
        Reusing the buffer saves ~8.3 MB allocation per frame at 1080p.
        """
        # Reset to transparent black in-place — ~0.2ms for 1080p (vs 0.6ms alloc+copy)
        self._frame_buf[:] = 0
        frame = self._frame_buf

        # OSD glyphs (~0.1ms for ~7 glyphs)
        if osd_frame is not None and self.font is not None:
            for row, col, code in osd_frame.non_empty():
                cached = self._get_glyph(code)
                if cached is None:
                    continue
                g_arr, g_alpha = cached
                px = self.x0 + col * self.tw
                py = self.y0 + row * self.th
                if (px >= self.w or py >= self.h
                        or px + self.tw <= 0 or py + self.th <= 0):
                    continue
                # Clamp to frame bounds — glyph may be partially off-screen
                py0 = max(py, 0);  py1 = min(py + self.th, self.h)
                px0 = max(px, 0);  px1 = min(px + self.tw, self.w)
                if py1 <= py0 or px1 <= px0:
                    continue
                # Slice both dst and src to the same visible region
                gy0 = py0 - py;  gy1 = gy0 + (py1 - py0)
                gx0 = px0 - px;  gx1 = gx0 + (px1 - px0)
                dst    = frame[py0:py1, px0:px1]
                g_vis  = g_arr  [gy0:gy1, gx0:gx1]   # clipped glyph pixels
                ga_vis = g_alpha[gy0:gy1, gx0:gx1]   # clipped alpha
                # Porter-Duff "over": preserve glyph alpha (don't force opaque)
                src_a  = ga_vis                               # H×W×1 float32
                dst_a  = dst[:, :, 3:4].astype(np.float32) / 255.0
                out_a  = src_a + dst_a * (1.0 - src_a)       # composited alpha
                safe_a = np.where(out_a > 0, out_a, 1.0)
                out_rgb = (g_vis[:, :, :3].astype(np.float32) * src_a
                           + dst[:, :, :3].astype(np.float32) * dst_a * (1.0 - src_a)
                          ) / safe_a
                dst[:, :, :3] = out_rgb.astype(np.uint8)
                dst[:, :, 3]  = (out_a[:, :, 0] * 255).astype(np.uint8)

        # SRT bar (~0.3ms)
        if self.cfg.show_srt_bar and srt_text:
            entry = self._get_srt(srt_text)
            if entry is not None:
                y1, y2, x1, x2, src_f, alpha = entry
                dst = frame[y1:y2, x1:x2]
                dst_a  = dst[:, :, 3:4].astype(np.float32) / 255.0
                out_a  = alpha + dst_a * (1.0 - alpha)
                safe_a = np.where(out_a > 0, out_a, 1.0)
                out_rgb = (src_f[:, :, :3] * alpha
                           + dst[:, :, :3].astype(np.float32) * dst_a * (1.0 - alpha)
                          ) / safe_a
                dst[:, :, :3] = out_rgb.astype(np.uint8)
                dst[:, :, 3]  = (out_a[:, :, 0] * 255).astype(np.uint8)

        return frame  # write directly: pipe.write(frame)  ← buffer protocol, no copy

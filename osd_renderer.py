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
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

# A hide region is an inclusive rectangle in OSD grid coordinates:
#   (row_min, col_min, row_max, col_max)
HideRegion = Tuple[int, int, int, int]

try:
    from PIL import Image, ImageDraw, ImageFont as PILFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

from osd_parser  import OsdFrame, GRID_COLS, GRID_ROWS  # defaults; frames carry their own dims
from font_loader import OsdFont


@dataclass
class OsdRenderConfig:
    offset_x:      int   = 0
    offset_y:      int   = 0
    scale:         float = 1.0
    show_srt_bar:  bool  = True
    srt_text:      str   = ""
    srt_opacity:   float = 0.6   # SRT bar background opacity (0.0–1.0)
    srt_scale:     float = 1.0   # SRT bar font/size multiplier (0.50–2.0)
    osd_offset_ms: int   = 0     # Manual sync offset (ms); applied to timestamp lookups
    # Rectangular regions (in OSD grid coordinates, inclusive) whose glyphs
    # should be skipped at composite time. Stored grid-relative so they stay
    # pinned to the OSD elements regardless of scale/offset changes.
    hide_regions:  List[HideRegion] = field(default_factory=list)
    # When the canvas (output buffer) is larger than the original source video
    # — e.g. pillarboxed 4:3→16:9 — these tell the renderer the source area's
    # dimensions so glyphs can be sized & centred against the source rather
    # than the padded canvas. 0 = canvas is the source (no padding).
    source_w:      int   = 0
    source_h:      int   = 0


def cell_in_regions(row: int, col: int, regions: List[HideRegion]) -> bool:
    """Return True if grid cell (row, col) falls inside any hide region."""
    for r0, c0, r1, c1 in regions:
        if r0 <= row <= r1 and c0 <= col <= c1:
            return True
    return False


def _auto_scale(canvas_w: int, canvas_h: int, tile_w: int, tile_h: int,
                user_scale: float = 1.0,
                grid_cols: int = GRID_COLS,
                grid_rows: int = GRID_ROWS,
                source_w: int = 0,
                source_h: int = 0) -> tuple[float, int, int]:
    """
    Return (effective_scale, x_off, y_off).

    Scaling origin is the centre of the SOURCE video area inside the canvas.
    When source_w/source_h are zero (or equal to canvas dims) this collapses to
    the original behaviour: glyphs sized to canvas height, centred in canvas.
    When the canvas is larger than the source (aspect-ratio padding), glyphs
    are still sized against the source and centred on the source area, so the
    user's X/Y offsets push them toward / into the black bars.
    """
    if source_w <= 0:
        source_w = canvas_w
    if source_h <= 0:
        source_h = canvas_h
    base        = source_h / (grid_rows * tile_h)
    eff         = base * user_scale
    grid_w      = grid_cols * tile_w * eff
    grid_h      = grid_rows * tile_h * eff
    src_x       = (canvas_w - source_w) // 2
    src_y       = (canvas_h - source_h) // 2
    x_off       = src_x + int((source_w - grid_w) / 2)
    y_off       = src_y + int((source_h - grid_h) / 2)
    return eff, x_off, y_off


# ─── PIL preview renderer (single-frame, used by Qt preview) ──────────────────

def render_osd_frame(
    frame_img: "Image.Image",
    osd_frame: OsdFrame | None,
    font: OsdFont,
    cfg: OsdRenderConfig,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
) -> "Image.Image":
    if not PIL_OK:
        return frame_img

    out = frame_img.copy().convert("RGBA")

    if osd_frame is not None:
        eff, x0, y0 = _auto_scale(out.width, out.height,
                                   font.tile_w, font.tile_h, cfg.scale,
                                   osd_frame.grid_cols, osd_frame.grid_rows,
                                   source_w=cfg.source_w, source_h=cfg.source_h)
        tw = max(1, int(font.tile_w * eff))
        th = max(1, int(font.tile_h * eff))
        x0 += cfg.offset_x
        y0 += cfg.offset_y

        hide_regions = cfg.hide_regions or []
        for row, col, code in osd_frame.non_empty():
            if hide_regions and cell_in_regions(row, col, hide_regions):
                continue
            glyph = font.get_char(code)
            if glyph is None:
                continue
            glyph = glyph.resize((tw, th), Image.LANCZOS)
            px, py = x0 + col * tw, y0 + row * th
            if px >= out.width or py >= out.height or px + tw <= 0 or py + th <= 0:
                continue
            out.paste(glyph, (px, py), glyph)

    if cfg.show_srt_bar and cfg.srt_text:
        _draw_srt_bar(out, cfg.srt_text, opacity=cfg.srt_opacity,
                      scale=cfg.srt_scale)

    return out


def render_fallback(
    frame_img: "Image.Image",
    osd_frame: OsdFrame | None,
    cfg: OsdRenderConfig,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
) -> "Image.Image":
    if not PIL_OK:
        return frame_img
    out  = frame_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    try:    pil_f = PILFont.truetype("arial.ttf", 14)
    except: pil_f = PILFont.load_default()
    if osd_frame is not None:
        cw = max(8, out.width  // osd_frame.grid_cols)
        ch = max(8, out.height // osd_frame.grid_rows)
        x0, y0 = cfg.offset_x, cfg.offset_y
        hide_regions = cfg.hide_regions or []
        for row, col, code in osd_frame.non_empty():
            if hide_regions and cell_in_regions(row, col, hide_regions):
                continue
            label = chr(code) if 32 <= code < 127 else "·"
            px, py = x0 + col * cw, y0 + row * ch
            draw.text((px+1, py+1), label, font=pil_f, fill=(0,0,0,200))
            draw.text((px,   py  ), label, font=pil_f, fill=(255,255,0,220))
    if cfg.show_srt_bar and cfg.srt_text:
        _draw_srt_bar(out, cfg.srt_text, opacity=cfg.srt_opacity,
                      scale=cfg.srt_scale)
    return out


def _draw_srt_bar(img: "Image.Image", text: str, opacity: float = 0.6,
                  scale: float = 1.0, _cache: dict = {}):
    """Draw SRT status bar onto a PIL image. Font is cached across calls.

    Draws on a separate transparent overlay then alpha_composites onto img so
    the output pixels are always fully opaque (alpha=255).  This prevents the
    UI theme background from bleeding through in the preview widget.
    """
    fsize = max(6, int(img.height // 42 * scale))
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
    alpha_byte = int(opacity * 255)
    draw.rounded_rectangle([x-pad, y-pad, x+tw+pad, y+th+pad],
                            radius=4, fill=(0, 0, 0, alpha_byte))
    draw.text((x, y), text, font=fnt, fill=(255, 255, 255, alpha_byte))
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
                 font: Optional[OsdFont], cfg: OsdRenderConfig,
                 grid_cols: int = GRID_COLS, grid_rows: int = GRID_ROWS):
        self.w   = video_w
        self.h   = video_h
        self.font = font
        self.cfg  = cfg
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        # Tile geometry
        if font:
            eff, self.x0, self.y0 = _auto_scale(
                video_w, video_h, font.tile_w, font.tile_h, cfg.scale,
                grid_cols, grid_rows,
                source_w=cfg.source_w, source_h=cfg.source_h)
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

        # PIL font for rendering SRT bar images (rebuilt when scale changes via cache key)
        fsize = max(14, int(video_h // 42 * cfg.srt_scale))
        self._srt_fsize = fsize
        try:    self._srt_pil_font = PILFont.truetype("arial.ttf", fsize)
        except: self._srt_pil_font = PILFont.load_default()

        # Pre-allocated output buffer — reused every frame to avoid
        # 8.3 MB numpy allocation × n_frames (e.g. 36 GB for a 1h 1080p60 render)
        self._frame_buf = np.zeros((video_h, video_w, 4), dtype=np.uint8)

        # Pre-compute a boolean hide-mask over the grid so per-glyph checks
        # are O(1) in the hot loop, regardless of how many regions exist.
        self._hide_mask: Optional[np.ndarray] = None
        if cfg.hide_regions:
            mask = np.zeros((grid_rows, grid_cols), dtype=bool)
            for r0, c0, r1, c1 in cfg.hide_regions:
                r0c = max(0, min(grid_rows - 1, int(r0)))
                r1c = max(0, min(grid_rows - 1, int(r1)))
                c0c = max(0, min(grid_cols - 1, int(c0)))
                c1c = max(0, min(grid_cols - 1, int(c1)))
                if r1c < r0c or c1c < c0c:
                    continue
                mask[r0c:r1c + 1, c0c:c1c + 1] = True
            if mask.any():
                self._hide_mask = mask

    # ── Glyph lookup ──────────────────────────────────────────────────────────

    def _get_glyph(self, code: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if code not in self._glyphs:
            if self.font is None:
                return None
            g = self.font.get_char(code)
            if g is None:
                return None
            g_arr  = np.array(g.resize((self.tw, self.th), Image.NEAREST),
                              dtype=np.uint8)           # H×W×4
            g_alpha = g_arr[:, :, 3:4].astype(np.float32) / 255.0  # H×W×1
            self._glyphs[code] = (g_arr, g_alpha)
        return self._glyphs[code]

    # ── SRT bar lookup ────────────────────────────────────────────────────────

    def _get_srt(self, text: str) -> tuple:
        """Return (y1, y2, x1, x2, src_f32 H×W×4, alpha_f32 H×W×1) for text."""
        cache_key = (text, round(self.cfg.srt_opacity, 2), round(self.cfg.srt_scale, 2))
        if cache_key not in self._srt_cache:
            opacity_byte = int(self.cfg.srt_opacity * 255)
            bar  = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(bar)
            # Build a scaled font for this cache entry
            fsize = max(14, int(self.h // 42 * self.cfg.srt_scale))
            try:    fnt = PILFont.truetype("arial.ttf", fsize)
            except: fnt = PILFont.load_default()
            bb   = draw.textbbox((0, 0), text, font=fnt)
            tw   = bb[2] - bb[0];  th = bb[3] - bb[1]
            pad  = 6;  margin = 10
            x    = (self.w - tw) // 2
            y    = self.h - th - margin
            draw.rounded_rectangle([x-pad, y-pad, x+tw+pad, y+th+pad],
                                    radius=4, fill=(0, 0, 0, opacity_byte))
            draw.text((x,   y  ), text, font=fnt, fill=(255, 255, 255, opacity_byte))

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
            hide_mask = self._hide_mask
            for row, col, code in osd_frame.non_empty():
                if hide_mask is not None and 0 <= row < hide_mask.shape[0] \
                        and 0 <= col < hide_mask.shape[1] and hide_mask[row, col]:
                    continue
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

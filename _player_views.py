# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_views.py — Passive view widgets for the player module.

Contains: VideoCanvas, Timeline, TransportBar.
"""

import random

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, QRect, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QFont, QPixmap, QImage, QPainter, QColor, QPen, QDesktopServices,
)

from _player_common import PIL_OK, PILImage

# ── Donate ────────────────────────────────────────────────────────────────────

_DONATE_URL = "https://buymeacoffee.com/failsavefpv"
_DONATE_NOTES = [
    "Runs on coffee & failed maiden flights.",
    "Every donation funds one more prop replacement.",
    "This tool is free. My props are not.",
    "If this saved your footage, maybe save my bank account.",
    "Claude wrote the code. I flew into a tree to test it.",
    "100% AI-assisted. 0% AI-responsible for the bugs.",
    "Built with Claude. Crashed with Betaflight.",
    "Blackbox says: prop strike at 0:42. Donate anyway.",
    "The AI did the coding. I did the crashing.",
    "Claude suggested a paywall. I said no. For now.",
    "No props were harmed in the making of this tool. Just all of mine.",
    "Powered by mass regret and full-throttle into concrete.",
    "OSD says 4.2V. Battery says otherwise.",
    "My quad has more rebuilds than this code has commits.",
    "Lost video at 300m. Found it again at 0m. Vertically.",
    "Failsafe engaged. Wallet disengaged.",
    "Tuned PIDs for 3 hours. Flew into a gate on the first pack.",
    "DVR recovered. Dignity not found.",
    "Every flight is a test flight if you believe hard enough.",
    "This app has fewer bugs than my quad has crashes. Barely.",
    "GPS says RTH. Quad says tree.",
    "Buy me a coffee so I can stay up fixing the bugs you found.",
    "Smoke test passed. Motor 3 did not.",
    "If my code crashes as hard as my quad, we're all in trouble.",
    "Free as in freedom. Expensive as in FPV.",
]
_DONATE_NOTE = random.choice(_DONATE_NOTES)


# ─── VideoCanvas ──────────────────────────────────────────────────────────────

class VideoCanvas(QWidget):
    """Two-layer video frame display: base video + OSD overlay.

    The base pixmap (video frame) is resized once on seek/resize and cached.
    The OSD overlay is a separate transparent pixmap that can be updated
    cheaply without re-processing the video frame.
    """

    resized = pyqtSignal()  # emitted after resize so controller can re-render OSD
    regionAdded = pyqtSignal(tuple)     # (r0, c0, r1, c1) in OSD grid coords
    regionRemoveRequested = pyqtSignal(int)  # index into current regions list
    # Widget editing — emit live (continuous) and committed signals so the host
    # can throttle preview refreshes while still updating UI cheaply mid-drag.
    widgetSelected = pyqtSignal(int)             # index in current widget list (-1 = none)
    widgetMoved    = pyqtSignal(int, float, float, float, float)  # (idx, x, y, w, h)
    widgetCommit   = pyqtSignal(int, float, float, float, float)  # same, but drag-end

    def __init__(self, theme_fn, fs_fn, parent=None):
        super().__init__(parent)
        self._T = theme_fn
        self._fs = fs_fn
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._base_pil = None     # full-res video frame (PIL, no OSD)
        self._base_pixmap = None  # base resized to canvas (QPixmap, cached)
        self._osd_pixmap = None   # transparent OSD overlay (QPixmap)
        self._donate_rects = []
        self._show_placeholder = True
        self.setMouseTracking(True)
        # ── Hide-region edit state ────────────────────────────────────────
        # Modes: None = off, "add" = draw-to-add, "remove" = click-to-remove.
        self._region_mode: str | None = None
        self._region_regions: list[tuple[int, int, int, int]] = []
        # Callback: canvas-space pixel (x, y) → (row, col) in OSD grid, or None.
        # Supplied by MainWindow so the canvas stays agnostic of font/OSD state.
        self._region_px_to_cell = None
        # Callback: (row, col) → QRect in canvas coords for painting.
        self._region_cell_to_px = None
        self._region_drag_start: tuple[int, int] | None = None  # canvas px
        self._region_drag_cur: tuple[int, int] | None = None
        self._region_hover_idx: int = -1
        # Aspect-ratio padding: stored as a (aw, ah) ratio so the pad rect
        # can be computed from whatever resolution the cached frame happens
        # to be at (extraction scales preview frames down to <= 1080p).
        # (0, 0) = no padding.
        self._pad_aspect: tuple[int, int] = (0, 0)
        # The unmodified source frame is kept separately so we can re-pad
        # when the aspect ratio changes without re-decoding.
        self._source_pil = None
        # ── Widget edit state ─────────────────────────────────────────────
        # _widget_edit = False -> widgets render via overlay only (no chrome).
        # _widget_edit = True  -> selection chrome drawn, mouse drags widgets.
        # _widget_list mirrors MainWindow's self._widgets (rects in normalised
        # source coords). _widget_src_rect_fn returns the SOURCE area in canvas
        # pixels so we can convert between normalised (0..1) and pixel space.
        self._widget_edit: bool = False
        self._widget_list: list = []          # list[widgets.Widget]
        self._widget_selected: int = -1
        self._widget_hover: int = -1
        self._widget_src_rect_fn = None       # () -> (sx, sy, sw, sh) or None
        self._widget_drag_offset: tuple[int, int] | None = None  # (dx, dy) px within widget
        self._widget_drag_idx: int = -1
        # Edge/corner resize of the selected widget. _resize_edge is a combo of
        # 't'/'b' and 'l'/'r' (e.g. 'tl', 'r'); _resize_bounds is the widget's
        # [left, top, right, bottom] in normalised source coords, with only the
        # dragged edges moving so the opposite edge stays pinned.
        self._widget_resize_idx: int = -1
        self._widget_resize_edge: str | None = None
        self._resize_bounds: list[float] | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def set_base_frame(self, pil_img):
        """Cache a video-only frame. Resizes to canvas and stores as QPixmap."""
        if not PIL_OK or pil_img is None:
            return
        self._show_placeholder = False
        src = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
        self._source_pil = src
        self._base_pil = self._apply_aspect_pad(src)
        self._donate_rects = []
        self._rebuild_base()
        self.update()

    def set_osd_overlay(self, pil_img):
        """Set the OSD overlay (transparent RGBA at display_size). Cheap update."""
        if not PIL_OK or pil_img is None:
            self._osd_pixmap = None
            self.update()
            return
        img = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
        data = img.tobytes("raw", "RGBA")
        qi = QImage(data, img.width, img.height, img.width * 4,
                    QImage.Format.Format_RGBA8888)
        self._osd_pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_frame(self, pil_img):
        """Display a pre-composited frame (used by playback path).
        Sets as base and clears OSD overlay. Already-padded frames bypass
        the aspect-pad re-wrap."""
        if not PIL_OK or pil_img is None:
            return
        self._show_placeholder = False
        src = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
        # Playback path supplies already-composited frames (padded by main),
        # so don't double-pad — but still keep them as the source for re-pad.
        self._source_pil = src
        self._base_pil = src
        self._osd_pixmap = None
        self._donate_rects = []
        self._rebuild_base()
        self.update()

    def set_placeholder(self):
        """Show the Quick Start placeholder."""
        self._base_pil = None
        self._source_pil = None
        self._base_pixmap = None
        self._osd_pixmap = None
        self._show_placeholder = True
        self._donate_rects = []
        self.update()

    def set_aspect_padding(self, aw: int, ah: int):
        """Set or clear letter/pillarbox padding by target aspect ratio.
        Pass (0, 0) to disable. The pad rect is computed from each frame's
        actual size, so this works regardless of the cached frame's
        resolution (ffmpeg scales preview frames down)."""
        self._pad_aspect = (int(aw), int(ah))
        if self._source_pil is not None:
            self._base_pil = self._apply_aspect_pad(self._source_pil)
            self._rebuild_base()
        self._osd_pixmap = None  # stale dims — controller will re-render
        self.update()
        self.resized.emit()

    def _apply_aspect_pad(self, src_pil):
        """Return src_pil wrapped inside a black canvas matching the current
        target aspect ratio, or src_pil unchanged when no padding is set or
        the source already matches the target."""
        if not PIL_OK or src_pil is None:
            return src_pil
        aw, ah = self._pad_aspect
        if aw <= 0 or ah <= 0:
            return src_pil
        sw, sh = src_pil.width, src_pil.height
        target_ratio = aw / ah
        src_ratio    = sw / sh
        if abs(src_ratio - target_ratio) < 1e-3:
            return src_pil
        if src_ratio < target_ratio:
            cw = int(round(sh * target_ratio))
            ch = sh
        else:
            cw = sw
            ch = int(round(sw / target_ratio))
        cw = (cw // 2) * 2
        ch = (ch // 2) * 2
        sx = (cw - sw) // 2
        sy = (ch - sh) // 2
        canvas = PILImage.new("RGBA", (cw, ch), (0, 0, 0, 255))
        canvas.paste(src_pil, (sx, sy))
        return canvas

    def has_frame(self):
        return self._base_pil is not None

    def display_size(self):
        """Return (w, h) of the base pixmap, or canvas size if no frame."""
        if self._base_pixmap is not None:
            return self._base_pixmap.width(), self._base_pixmap.height()
        return max(self.width(), 320), max(self.height(), 180)

    def display_origin(self):
        """Return (x, y) of the top-left of the base pixmap within the canvas."""
        if self._base_pixmap is None:
            return 0, 0
        return ((self.width()  - self._base_pixmap.width())  // 2,
                (self.height() - self._base_pixmap.height()) // 2)

    # ── Hide-region edit API ──────────────────────────────────────────────────

    def set_region_edit_mode(self, mode: str | None):
        """Set region edit mode: None, 'add', or 'remove'.
        When not None the canvas captures mouse events to draw/delete rects."""
        self._region_mode = mode
        self._region_drag_start = None
        self._region_drag_cur = None
        if mode == "add":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "remove":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_region_callbacks(self, px_to_cell, cell_to_px):
        """Install geometry converters owned by MainWindow.

        px_to_cell(x, y)      → (row, col) | None     (canvas px → OSD grid cell)
        cell_to_px(r0,c0,r1,c1) → (x, y, w, h) | None (grid rect → canvas px rect)
        """
        self._region_px_to_cell = px_to_cell
        self._region_cell_to_px = cell_to_px

    def set_regions(self, regions: list[tuple[int, int, int, int]]):
        """Update the list of hide regions drawn on the overlay."""
        self._region_regions = list(regions or [])
        self.update()

    # ── Widget edit API ───────────────────────────────────────────────────────

    def set_widget_edit(self, enabled: bool):
        """Toggle widget edit mode. When on, widgets get selection chrome and
        the canvas captures drag events to reposition them."""
        self._widget_edit = bool(enabled)
        self._widget_drag_offset = None
        self._widget_drag_idx = -1
        if enabled:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_widgets(self, widgets: list):
        """Update the widget list mirror (used for selection chrome + hit-test)."""
        self._widget_list = list(widgets or [])
        # Drop stale selection if list shrank
        if self._widget_selected >= len(self._widget_list):
            self._widget_selected = -1
        self.update()

    def set_widget_selected(self, idx: int):
        """Set which widget index is highlighted (called by the side panel list)."""
        if idx == self._widget_selected:
            return
        self._widget_selected = int(idx) if 0 <= idx < len(self._widget_list) else -1
        self.update()

    def set_widget_src_rect_fn(self, fn):
        """Install a callback returning the SOURCE video rect in canvas px:
        ``() -> (sx, sy, sw, sh) | None``.  ``None`` means no frame loaded yet."""
        self._widget_src_rect_fn = fn

    # ── internal ──────────────────────────────────────────────────────────────

    def _rebuild_base(self):
        """Resize base PIL image to canvas size, cache as QPixmap."""
        if self._base_pil is None:
            self._base_pixmap = None
            return
        cw = max(self.width(), 320)
        ch = max(self.height(), 180)
        iw, ih = self._base_pil.size
        scale = min(cw / iw, ch / ih)
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        tmp = self._base_pil.resize((tw, th), PILImage.BILINEAR)
        data = tmp.tobytes("raw", "RGBA")
        qi = QImage(data, tw, th, tw * 4, QImage.Format.Format_RGBA8888)
        self._base_pixmap = QPixmap.fromImage(qi)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self._T()
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(t["bg2"]))

        if self._show_placeholder or self._base_pixmap is None:
            self._paint_placeholder(p, w, h, t)
        else:
            # Centre the base frame
            px = (w - self._base_pixmap.width()) // 2
            py = (h - self._base_pixmap.height()) // 2
            p.drawPixmap(px, py, self._base_pixmap)
            # OSD overlay on top (same size, same position)
            if self._osd_pixmap is not None:
                p.drawPixmap(px, py, self._osd_pixmap)
            # Hide-region overlay on top of the OSD
            self._paint_regions(p, t)
            # Widget selection chrome on top of everything
            if self._widget_edit:
                self._paint_widget_chrome(p)

        # Border
        p.setPen(QPen(QColor(t["border"]), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(0, 0, w - 1, h - 1, 8, 8)
        p.end()

    def _paint_placeholder(self, p, w, h, t):
        """Draw Quick Start guide + keyboard shortcuts + donation."""
        cx = w // 2
        lw = min(w - 40, 500)
        lx = cx - lw // 2

        TITLE_H = 28
        SEP_ABOVE = 5
        STEP_BELOW = 13
        STEP_H = 22
        STEP_PITCH = 24
        N_STEPS = 4
        qs_h = TITLE_H + SEP_ABOVE + 1 + STEP_BELOW + (N_STEPS - 1) * STEP_PITCH + STEP_H

        # Keyboard shortcuts block
        KB_TITLE_H = 22
        KB_SEP = 5
        KB_LINE_H = 16
        KB_PITCH = 18
        KB_LINES = 3
        kb_h = KB_TITLE_H + KB_SEP + 1 + 8 + (KB_LINES - 1) * KB_PITCH + KB_LINE_H

        # Donation
        px = 3
        heart_w = 13 * px
        heart_h = 11 * px
        don_h = heart_h + 4 + 16 + 8 + 16

        PAD = 12
        GAP = 16
        show_kb = h >= PAD + qs_h + GAP + kb_h + PAD
        show_don = h >= PAD + qs_h + GAP + kb_h + GAP + don_h + PAD

        total_h = qs_h
        if show_kb:
            total_h += GAP + kb_h
        if show_don:
            total_h += GAP + don_h
        top = max(PAD, (h - total_h) // 2)

        # ── Quick Start ───────────────────────────────────────────────────
        title_y = top
        p.setPen(QPen(QColor(t["subtext"])))
        p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        p.drawText(QRect(lx, title_y, lw, TITLE_H),
                   Qt.AlignmentFlag.AlignCenter, "Quick Start")

        sep_y = title_y + TITLE_H + SEP_ABOVE
        p.setPen(QPen(QColor(t["border2"])))
        p.drawLine(lx, sep_y, lx + lw, sep_y)

        step0_y = sep_y + 1 + STEP_BELOW
        steps = [
            "\u2460  Drop a video, .osd or .srt onto the drop zone \u2014 companions auto-load",
            "\u2461  Pick a font style in the left panel",
            "\u2462  Adjust position, opacity and scale; configure the SRT bar",
            "\u2463  Set the output file path, then click  Render",
        ]
        p.setPen(QPen(QColor(t["muted"])))
        p.setFont(QFont("Segoe UI", 10))
        for i, line in enumerate(steps):
            p.drawText(QRect(lx, step0_y + i * STEP_PITCH, lw, STEP_H),
                       Qt.AlignmentFlag.AlignLeft, line)

        cursor_y = top + qs_h

        # ── Keyboard Shortcuts ────────────────────────────────────────────
        if show_kb:
            kb_top = cursor_y + GAP
            p.setPen(QPen(QColor(t["subtext"])))
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.drawText(QRect(lx, kb_top, lw, KB_TITLE_H),
                       Qt.AlignmentFlag.AlignCenter, "Keyboard Shortcuts")

            kb_sep_y = kb_top + KB_TITLE_H + KB_SEP
            p.setPen(QPen(QColor(t["border2"])))
            p.drawLine(lx, kb_sep_y, lx + lw, kb_sep_y)

            kb_lines = [
                "Space  Play / Pause          Left / Right  Step frame",
                "I / O  Set trim in / out      Home / End  Jump to start / end",
                "J  Step back  K  Play/Pause  L  Play/Faster    Shift+\u2190/\u2192  Step 10",
            ]
            p.setPen(QPen(QColor(t["muted"])))
            p.setFont(QFont("Segoe UI", 9))
            kb0_y = kb_sep_y + 1 + 8
            for i, line in enumerate(kb_lines):
                p.drawText(QRect(lx, kb0_y + i * KB_PITCH, lw, KB_LINE_H),
                           Qt.AlignmentFlag.AlignCenter, line)

            cursor_y = kb_top + kb_h

        # ── Donation ──────────────────────────────────────────────────────
        self._donate_rects = []
        if show_don:
            HEART = [
                "00KKK00KKK000",
                "0KRRRK0KRRRK0",
                "KRWWRRRRRRRRK",
                "KRWWRRRRRRRRK",
                "KRRRRRRRRRRRK",
                "0KRRRRRRRRRK0",
                "00KRRRRRRRK00",
                "000KRRRRRK000",
                "0000KRRRK0000",
                "00000KRK00000",
                "000000K000000",
            ]
            heart_x = cx - heart_w // 2
            heart_y = cursor_y + GAP
            _hcol = {"K": QColor("#000000"), "R": QColor("#e60020"), "W": QColor("#ffffff")}
            p.setPen(Qt.PenStyle.NoPen)
            for ri, row in enumerate(HEART):
                for ci, ch in enumerate(row):
                    col = _hcol.get(ch)
                    if col:
                        p.setBrush(col)
                        p.drawRect(heart_x + ci * px, heart_y + ri * px, px, px)

            p.setPen(QPen(QColor(t["muted"])))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRect(lx, heart_y + heart_h + 4, lw, 16),
                       Qt.AlignmentFlag.AlignCenter, _DONATE_NOTE)

            p.setPen(QPen(QColor(t["accent"])))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRect(lx, heart_y + heart_h + 24, lw, 16),
                       Qt.AlignmentFlag.AlignCenter,
                       "buymeacoffee.com/failsavefpv  \u2197")

            self._donate_rects = [
                QRect(heart_x, heart_y, heart_w, heart_h),
                QRect(lx, heart_y + heart_h + 24, lw, 16),
            ]

    def _paint_regions(self, p: QPainter, t: dict):
        """Draw existing hide regions + the in-progress drag rectangle."""
        if not self._region_regions and self._region_drag_start is None \
                and self._region_mode is None:
            return
        cell_to_px = self._region_cell_to_px
        # Saved regions
        if cell_to_px is not None:
            for i, reg in enumerate(self._region_regions):
                rect = cell_to_px(*reg)
                if rect is None:
                    continue
                x, y, w, h = rect
                qr = QRect(x, y, w, h)
                # Translucent red fill + solid border
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 80, 80, 70))
                p.drawRect(qr)
                pen_col = QColor(255, 120, 120,
                                 220 if i == self._region_hover_idx else 180)
                pen = QPen(pen_col, 2 if i == self._region_hover_idx else 1)
                pen.setStyle(Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)
        # In-progress drag rectangle
        if self._region_mode == "add" and self._region_drag_start and self._region_drag_cur:
            x0, y0 = self._region_drag_start
            x1, y1 = self._region_drag_cur
            rx = min(x0, x1); ry = min(y0, y1)
            rw = abs(x1 - x0); rh = abs(y1 - y0)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 200, 80, 60))
            p.drawRect(rx, ry, rw, rh)
            pen = QPen(QColor(255, 200, 80, 230), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rx, ry, rw, rh)

    def _region_hit_test(self, x: int, y: int) -> int:
        """Return index of region whose painted rect contains (x, y), else -1."""
        if not self._region_regions or self._region_cell_to_px is None:
            return -1
        for i, reg in enumerate(self._region_regions):
            rect = self._region_cell_to_px(*reg)
            if rect is None:
                continue
            rx, ry, rw, rh = rect
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return -1

    # ── Widget edit internals ─────────────────────────────────────────────────

    def _widget_src_rect_px(self):
        """Source video rect in canvas px, or None if unknown."""
        fn = self._widget_src_rect_fn
        if fn is None:
            return None
        try:
            r = fn()
        except Exception:
            return None
        if not r or len(r) != 4:
            return None
        sx, sy, sw, sh = r
        if sw <= 0 or sh <= 0:
            return None
        return int(sx), int(sy), int(sw), int(sh)

    def _widget_rect_px(self, w):
        """Resolve a widget's pixel rect on the canvas, or None."""
        src = self._widget_src_rect_px()
        if src is None:
            return None
        sx, sy, sw, sh = src
        ww = max(1, int(round(w.w * sw)))
        wh = max(1, int(round(w.h * sh)))
        cx = sx + int(round(w.x * sw))
        cy = sy + int(round(w.y * sh))
        return cx - ww // 2, cy - wh // 2, ww, wh

    def _widget_hit_test(self, x: int, y: int) -> int:
        """Return index of widget under (x, y) — selected first, then topmost."""
        # Selected widget wins (keeps drag responsive when widgets overlap)
        sel = self._widget_selected
        if 0 <= sel < len(self._widget_list):
            r = self._widget_rect_px(self._widget_list[sel])
            if r is not None:
                rx, ry, rw, rh = r
                if rx <= x < rx + rw and ry <= y < ry + rh:
                    return sel
        for i in range(len(self._widget_list) - 1, -1, -1):
            r = self._widget_rect_px(self._widget_list[i])
            if r is None:
                continue
            rx, ry, rw, rh = r
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return -1

    _RESIZE_MARGIN = 7   # px band around a border that grabs a resize handle

    def _widget_resize_edge_at(self, w, x: int, y: int) -> "str | None":
        """Return the resize edge under (x, y) for widget ``w`` — a combo of
        't'/'b' and 'l'/'r' (corners), a single side, or None when the cursor
        isn't on a border."""
        r = self._widget_rect_px(w)
        if r is None:
            return None
        rx, ry, rw, rh = r
        m = self._RESIZE_MARGIN
        if not (rx - m <= x <= rx + rw + m and ry - m <= y <= ry + rh + m):
            return None
        edge = ""
        if abs(y - ry) <= m:
            edge += "t"
        elif abs(y - (ry + rh)) <= m:
            edge += "b"
        if abs(x - rx) <= m:
            edge += "l"
        elif abs(x - (rx + rw)) <= m:
            edge += "r"
        return edge or None

    @staticmethod
    def _resize_cursor(edge: str):
        if edge in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if edge in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        if edge in ("l", "r"):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def _paint_widget_chrome(self, p: QPainter):
        """Selection border + hover outline for widgets in edit mode."""
        if not self._widget_list:
            return
        for i, w in enumerate(self._widget_list):
            r = self._widget_rect_px(w)
            if r is None:
                continue
            rx, ry, rw, rh = r
            qr = QRect(rx, ry, rw, rh)
            if i == self._widget_selected:
                # Solid accent border + corner ticks
                pen = QPen(QColor(60, 200, 255, 230), 2, Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.setBrush(QColor(60, 200, 255, 30))
                p.drawRect(qr)
                # Corner ticks (small filled squares) for visual affordance
                tick = 5
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(60, 200, 255, 230))
                for cx, cy in ((rx, ry), (rx + rw - tick, ry),
                               (rx, ry + rh - tick),
                               (rx + rw - tick, ry + rh - tick)):
                    p.drawRect(cx, cy, tick, tick)
            elif i == self._widget_hover:
                pen = QPen(QColor(200, 200, 200, 180), 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)
            else:
                pen = QPen(QColor(180, 180, 180, 120), 1, Qt.PenStyle.DotLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._base_pil is not None:
            self._rebuild_base()
            self._osd_pixmap = None  # stale size — controller will re-render
            self.resized.emit()
        self.update()

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if self._widget_edit and event.button() == Qt.MouseButton.LeftButton:
            # Resize takes priority: if the press lands on a border of the
            # already-selected widget, drag that edge instead of moving.
            sel = self._widget_selected
            if 0 <= sel < len(self._widget_list):
                edge = self._widget_resize_edge_at(
                    self._widget_list[sel], pos.x(), pos.y())
                if edge:
                    w = self._widget_list[sel]
                    self._widget_resize_idx = sel
                    self._widget_resize_edge = edge
                    self._resize_bounds = [w.x - w.w / 2, w.y - w.h / 2,
                                           w.x + w.w / 2, w.y + w.h / 2]
                    self.setCursor(self._resize_cursor(edge))
                    self.update()
                    return
            idx = self._widget_hit_test(pos.x(), pos.y())
            if idx != self._widget_selected:
                self._widget_selected = idx
                self.widgetSelected.emit(idx)
            if idx >= 0:
                r = self._widget_rect_px(self._widget_list[idx])
                if r is not None:
                    rx, ry, _rw, _rh = r
                    self._widget_drag_offset = (pos.x() - rx, pos.y() - ry)
                    self._widget_drag_idx = idx
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            return
        if self._region_mode == "add" and event.button() == Qt.MouseButton.LeftButton:
            self._region_drag_start = (pos.x(), pos.y())
            self._region_drag_cur   = (pos.x(), pos.y())
            self.update()
            return
        if self._region_mode == "remove" and event.button() == Qt.MouseButton.LeftButton:
            idx = self._region_hit_test(pos.x(), pos.y())
            if idx >= 0:
                self.regionRemoveRequested.emit(idx)
            return
        if self._show_placeholder and self._donate_rects:
            if any(r.contains(pos) for r in self._donate_rects):
                QDesktopServices.openUrl(QUrl(_DONATE_URL))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._widget_edit:
            # Active resize — drag one edge/corner, keep the opposite pinned.
            if self._widget_resize_idx >= 0 and self._resize_bounds is not None:
                src = self._widget_src_rect_px()
                if src is not None and self._widget_resize_idx < len(self._widget_list):
                    sx, sy, sw, sh = src
                    ux = max(0.0, min(1.0, (pos.x() - sx) / sw))
                    uy = max(0.0, min(1.0, (pos.y() - sy) / sh))
                    l, t, r, b = self._resize_bounds
                    e, MIN = self._widget_resize_edge, 0.03
                    if "l" in e: l = min(ux, r - MIN)
                    if "r" in e: r = max(ux, l + MIN)
                    if "t" in e: t = min(uy, b - MIN)
                    if "b" in e: b = max(uy, t + MIN)
                    self._resize_bounds = [l, t, r, b]
                    w = self._widget_list[self._widget_resize_idx]
                    w.x, w.y, w.w, w.h = (l + r) / 2, (t + b) / 2, r - l, b - t
                    self.widgetMoved.emit(self._widget_resize_idx, w.x, w.y, w.w, w.h)
                    self.update()
                return
            # Active drag — move the widget, normalise back to source coords,
            # emit a continuous move signal so the host can refresh the preview.
            if self._widget_drag_idx >= 0 and self._widget_drag_offset is not None:
                src = self._widget_src_rect_px()
                if src is not None and self._widget_drag_idx < len(self._widget_list):
                    sx, sy, sw, sh = src
                    w = self._widget_list[self._widget_drag_idx]
                    dx, dy = self._widget_drag_offset
                    # New top-left under cursor, then convert top-left -> centre
                    rx = pos.x() - dx
                    ry = pos.y() - dy
                    ww = max(1, int(round(w.w * sw)))
                    wh = max(1, int(round(w.h * sh)))
                    cx = rx + ww / 2
                    cy = ry + wh / 2
                    # Normalise to source coords, clamp so widgets can't escape entirely
                    nx = max(0.0, min(1.0, (cx - sx) / sw))
                    ny = max(0.0, min(1.0, (cy - sy) / sh))
                    # Update local copy so chrome redraws at the new spot immediately
                    w.x = nx
                    w.y = ny
                    self.widgetMoved.emit(self._widget_drag_idx, nx, ny, w.w, w.h)
                    self.update()
                return
            # Hover — resize cursor on the selected widget's border wins, else
            # a light dashed outline on whichever widget is under the cursor.
            sel = self._widget_selected
            if 0 <= sel < len(self._widget_list):
                edge = self._widget_resize_edge_at(
                    self._widget_list[sel], pos.x(), pos.y())
                if edge:
                    self.setCursor(self._resize_cursor(edge))
                    return
            new_idx = self._widget_hit_test(pos.x(), pos.y())
            if new_idx != self._widget_hover:
                self._widget_hover = new_idx
                self.update()
            self.setCursor(Qt.CursorShape.OpenHandCursor if new_idx >= 0
                           else Qt.CursorShape.ArrowCursor)
            return
        if self._region_mode == "add" and self._region_drag_start is not None:
            self._region_drag_cur = (pos.x(), pos.y())
            self.update()
            return
        if self._region_mode == "remove":
            new_idx = self._region_hit_test(pos.x(), pos.y())
            if new_idx != self._region_hover_idx:
                self._region_hover_idx = new_idx
                self.update()
            return
        if self._region_hover_idx != -1:
            self._region_hover_idx = -1
            self.update()
        if self._show_placeholder:
            in_zone = any(r.contains(pos) for r in self._donate_rects)
            self.setCursor(Qt.CursorShape.PointingHandCursor if in_zone
                           else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._widget_edit and self._widget_resize_idx >= 0 \
                and event.button() == Qt.MouseButton.LeftButton:
            idx = self._widget_resize_idx
            self._widget_resize_idx = -1
            self._widget_resize_edge = None
            self._resize_bounds = None
            if 0 <= idx < len(self._widget_list):
                w = self._widget_list[idx]
                self.widgetCommit.emit(idx, w.x, w.y, w.w, w.h)
            self.update()
            return
        if self._widget_edit and self._widget_drag_idx >= 0 \
                and event.button() == Qt.MouseButton.LeftButton:
            idx = self._widget_drag_idx
            self._widget_drag_idx = -1
            self._widget_drag_offset = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if 0 <= idx < len(self._widget_list):
                w = self._widget_list[idx]
                # widgetCommit lets the host persist + do a full-quality refresh
                self.widgetCommit.emit(idx, w.x, w.y, w.w, w.h)
            self.update()
            return
        if self._region_mode == "add" and self._region_drag_start is not None \
                and event.button() == Qt.MouseButton.LeftButton:
            x0, y0 = self._region_drag_start
            pos = event.position().toPoint()
            x1, y1 = pos.x(), pos.y()
            self._region_drag_start = None
            self._region_drag_cur = None
            if abs(x1 - x0) >= 3 and abs(y1 - y0) >= 3 \
                    and self._region_px_to_cell is not None:
                c_a = self._region_px_to_cell(x0, y0)
                c_b = self._region_px_to_cell(x1, y1)
                if c_a is not None and c_b is not None:
                    r0 = min(c_a[0], c_b[0]); r1 = max(c_a[0], c_b[0])
                    c0 = min(c_a[1], c_b[1]); c1 = max(c_a[1], c_b[1])
                    self.regionAdded.emit((r0, c0, r1, c1))
            self.update()
            return
        super().mouseReleaseEvent(event)


# ─── Timeline ─────────────────────────────────────────────────────────────────

class Timeline(QWidget):
    """Combined scrub bar + trim handles + cache indicators."""

    seekRequested = pyqtSignal(float)         # position 0.0-1.0
    trimChanged = pyqtSignal(float, float)    # in_pct, out_pct

    TRACK_H = 6
    HANDLE_W = 8
    PLAYHEAD_W = 2
    MARGIN_X = 12

    def __init__(self, theme_fn, parent=None):
        super().__init__(parent)
        self._T = theme_fn
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._position = 0.0     # 0.0-1.0 (playhead)
        self._in = 0.0           # trim in
        self._out = 1.0          # trim out
        self._cached = set()     # set of float positions that have been cached
        self._dragging = None    # "playhead" | "trim_in" | "trim_out" | None

    # ── public API (compatible with old RangeSelector) ────────────────────────

    @property
    def in_pct(self):
        return self._in

    @property
    def out_pct(self):
        return self._out

    def set_in(self, v):
        self._in = max(0.0, min(v, self._out - 0.01))
        self.update()
        self.trimChanged.emit(self._in, self._out)

    def set_out(self, v):
        self._out = max(self._in + 0.01, min(v, 1.0))
        self.update()
        self.trimChanged.emit(self._in, self._out)

    def reset(self):
        self._in, self._out = 0.0, 1.0
        self.update()
        self.trimChanged.emit(0.0, 1.0)

    def set_position(self, pos):
        """Set playhead position (0.0-1.0) without emitting seekRequested."""
        self._position = max(0.0, min(1.0, pos))
        self.update()

    def set_cached(self, positions):
        """Update the set of cached frame positions."""
        self._cached = set(positions)
        self.update()

    def add_cached(self, pos):
        self._cached.add(pos)
        self.update()

    # ── geometry ──────────────────────────────────────────────────────────────

    def _track_rect(self):
        mx = self.MARGIN_X + self.HANDLE_W
        return (mx, (self.height() - self.TRACK_H) // 2,
                self.width() - mx * 2, self.TRACK_H)

    def _pos_to_x(self, pct):
        tx, _, tw, _ = self._track_rect()
        return int(tx + pct * tw)

    def _x_to_pos(self, x):
        tx, _, tw, _ = self._track_rect()
        if tw <= 0:
            return 0.0
        return max(0.0, min(1.0, (x - tx) / tw))

    def _handle_rect(self, pct):
        """Trim handle — small, hugs the track vertically."""
        hw = self.HANDLE_W
        cx = self._pos_to_x(pct)
        _, ty, _, th = self._track_rect()
        pad = 4
        return QRect(cx - hw // 2, ty - pad, hw, th + pad * 2)

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self._T()
        tx, ty, tw, th = self._track_rect()

        # Full track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["surface"]))
        p.drawRoundedRect(tx, ty, tw, th, 3, 3)

        # Active region between trim handles
        x1 = self._pos_to_x(self._in)
        x2 = self._pos_to_x(self._out)
        active_color = QColor(t["accent"])
        active_color.setAlpha(80)
        p.setBrush(active_color)
        p.drawRect(x1, ty, x2 - x1, th)

        # Dimmed regions outside trim
        dim = QColor(t["bg"])
        dim.setAlpha(160)
        p.setBrush(dim)
        if x1 > tx:
            p.drawRect(tx, ty, x1 - tx, th)
        if x2 < tx + tw:
            p.drawRect(x2, ty, tx + tw - x2, th)

        # Trim handles
        for pct, label in ((self._in, "I"), (self._out, "O")):
            hr = self._handle_rect(pct)
            p.setBrush(QColor(t["text"]))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(hr, 3, 3)
            p.setPen(QColor(t["bg"]))
            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.drawText(hr, Qt.AlignmentFlag.AlignCenter, label)
            p.setPen(Qt.PenStyle.NoPen)

        # Playhead — full height needle with large triangle cap
        ph_x = self._pos_to_x(self._position)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["text"]))
        p.drawRect(ph_x - 1, 0, self.PLAYHEAD_W, self.height())
        # Triangle at top (large, easy to grab)
        tri_h = 7
        tri_w = 7
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtCore import QPointF
        tri = QPolygonF([
            QPointF(ph_x, tri_h),
            QPointF(ph_x - tri_w, 0),
            QPointF(ph_x + tri_w, 0),
        ])
        p.drawPolygon(tri)

        p.end()

    # ── mouse interaction ─────────────────────────────────────────────────────

    def _hit_test(self, x, y):
        """Return what the user clicked: 'trim_in', 'trim_out', or 'playhead'.
        Trim handles only respond near the track; top/bottom = playhead."""
        _, ty, _, th = self._track_rect()
        # Only check trim handles when click is near the track vertically
        track_zone = (ty - 6 <= y <= ty + th + 6)
        if track_zone:
            xi = self._pos_to_x(self._in)
            xo = self._pos_to_x(self._out)
            if abs(x - xi) < 10:
                return "trim_in"
            if abs(x - xo) < 10:
                return "trim_out"
        return "playhead"

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            x = int(e.position().x())
            y = int(e.position().y())
            self._dragging = self._hit_test(x, y)
            if self._dragging == "playhead":
                pos = self._x_to_pos(x)
                self._position = pos
                self.update()
                self.seekRequested.emit(pos)

    def mouseMoveEvent(self, e):
        x = int(e.position().x())
        if self._dragging == "trim_in":
            self.set_in(self._x_to_pos(x))
        elif self._dragging == "trim_out":
            self.set_out(self._x_to_pos(x))
        elif self._dragging == "playhead":
            pos = self._x_to_pos(x)
            self._position = pos
            self.update()
            self.seekRequested.emit(pos)
        else:
            # Cursor hint — only show resize cursor near track zone
            y = int(e.position().y())
            _, ty, _, th = self._track_rect()
            track_zone = (ty - 6 <= y <= ty + th + 6)
            if track_zone:
                xi = self._pos_to_x(self._in)
                xo = self._pos_to_x(self._out)
                near = min(abs(x - xi), abs(x - xo))
                self.setCursor(Qt.CursorShape.SizeHorCursor if near < 10
                               else Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._dragging = None


# ─── TransportBar ─────────────────────────────────────────────────────────────

class TransportBar(QWidget):
    """Playback control buttons + time display."""

    playClicked = pyqtSignal()
    restartClicked = pyqtSignal()
    skipEndClicked = pyqtSignal()
    stepBackClicked = pyqtSignal()
    stepFwdClicked = pyqtSignal()
    refreshClicked = pyqtSignal()
    previewQualityChanged = pyqtSignal(int)  # 0 = cached, 1 = original

    def __init__(self, theme_fn, icon_fn, btn_play_fn, btn_sec_fn, parent=None):
        super().__init__(parent)
        self._T = theme_fn
        self._icon = icon_fn
        self._btn_play = btn_play_fn
        self._btn_sec = btn_sec_fn

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.restart_btn = QPushButton()
        self.restart_btn.setIcon(self._icon("back.png", 20))
        self.restart_btn.setFixedSize(34, 34)
        self.restart_btn.setStyleSheet(self._btn_play())
        self.restart_btn.setToolTip("Go to start (Home)")
        self.restart_btn.clicked.connect(self.restartClicked)

        self.step_back_btn = QPushButton()
        self.step_back_btn.setIcon(self._icon("back.png", 16))
        self.step_back_btn.setFixedSize(30, 34)
        self.step_back_btn.setStyleSheet(self._btn_play())
        self.step_back_btn.setToolTip("Step back 1 frame (\u2190)")
        self.step_back_btn.clicked.connect(self.stepBackClicked)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self._icon("play.png", 22))
        self.play_btn.setFixedSize(44, 34)
        self.play_btn.setStyleSheet(self._btn_play())
        self.play_btn.setToolTip("Play / Pause (Space)")
        self.play_btn.clicked.connect(self.playClicked)

        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(self._icon("next.png", 16))
        self.step_fwd_btn.setFixedSize(30, 34)
        self.step_fwd_btn.setStyleSheet(self._btn_play())
        self.step_fwd_btn.setToolTip("Step forward 1 frame (\u2192)")
        self.step_fwd_btn.clicked.connect(self.stepFwdClicked)

        self.skip_btn = QPushButton()
        self.skip_btn.setIcon(self._icon("next.png", 20))
        self.skip_btn.setFixedSize(34, 34)
        self.skip_btn.setStyleSheet(self._btn_play())
        self.skip_btn.setToolTip("Go to end (End)")
        self.skip_btn.clicked.connect(self.skipEndClicked)

        self.time_lbl = QLabel("0:00 / 0:00")
        t = self._T()
        self.time_lbl.setStyleSheet(
            f"color:{t['text']};font-size:11px;font-weight:bold;"
            f"font-family:'Segoe UI',monospace;")
        self.time_lbl.setFixedWidth(100)
        self.time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.info_lbl = QLabel("t = 0.0s  |  OSD \u2014")
        self.info_lbl.setStyleSheet(f"color:{t['muted']};font-size:10px;")

        self.ref_btn = QPushButton("Refresh Preview")
        self.ref_btn.setFixedHeight(34)
        self.ref_btn.setMinimumWidth(120)
        self.ref_btn.setStyleSheet(self._btn_sec())
        self.ref_btn.clicked.connect(self.refreshClicked)

        self.preview_quality_combo = QComboBox()
        self.preview_quality_combo.addItems(["Cached (fast)", "Original (full-res)"])
        self.preview_quality_combo.setFixedHeight(34)
        self.preview_quality_combo.setMinimumWidth(140)
        self.preview_quality_combo.setStyleSheet(
            f"QComboBox{{background:{t['surface']};color:{t['text']};"
            f"border:1px solid {t['border2']};border-radius:4px;padding:3px 8px;font-size:11px;}}"
            f"QComboBox::drop-down{{border:none;width:18px;}}"
        )
        self.preview_quality_combo.setToolTip(
            "Cached: scaled-down preview frames, snappy seeking.\n"
            "Original: native source resolution — pixel-accurate but\n"
            "noticeably slower to scrub on large videos."
        )
        self.preview_quality_combo.currentIndexChanged.connect(self.previewQualityChanged)

        self.speed_lbl = QLabel("")
        self.speed_lbl.setStyleSheet(f"color:{t['accent']};font-size:10px;font-weight:bold;")
        self.speed_lbl.setFixedWidth(40)

        lay.addWidget(self.time_lbl)
        lay.addWidget(self.speed_lbl)
        lay.addStretch(1)
        lay.addWidget(self.restart_btn)
        lay.addWidget(self.step_back_btn)
        lay.addWidget(self.play_btn)
        lay.addWidget(self.step_fwd_btn)
        lay.addWidget(self.skip_btn)
        lay.addStretch(1)
        lay.addWidget(self.ref_btn)
        lay.addWidget(self.preview_quality_combo)

    def set_playing(self, playing):
        name = "pause.png" if playing else "play.png"
        self.play_btn.setIcon(self._icon(name, 22))

    def set_time(self, current_s, total_s):
        cm, cs = divmod(int(current_s), 60)
        tm, ts = divmod(int(total_s), 60)
        self.time_lbl.setText(f"{cm}:{cs:02d} / {tm}:{ts:02d}")

    def set_info(self, text):
        self.info_lbl.setText(text)

    def set_speed(self, factor):
        if abs(factor - 1.0) < 0.01:
            self.speed_lbl.setText("")
        else:
            self.speed_lbl.setText(f"{factor:.1f}x")

    def refresh_theme(self):
        t = self._T()
        for btn in (self.restart_btn, self.step_back_btn, self.play_btn,
                    self.step_fwd_btn, self.skip_btn):
            btn.setStyleSheet(self._btn_play())
        self.ref_btn.setStyleSheet(self._btn_sec())
        self.time_lbl.setStyleSheet(
            f"color:{t['text']};font-size:11px;font-weight:bold;"
            f"font-family:'Segoe UI',monospace;")
        self.info_lbl.setStyleSheet(f"color:{t['muted']};font-size:10px;")
        self.speed_lbl.setStyleSheet(f"color:{t['accent']};font-size:10px;font-weight:bold;")
        # Retint icons
        self.restart_btn.setIcon(self._icon("back.png", 20))
        self.step_back_btn.setIcon(self._icon("back.png", 16))
        self.step_fwd_btn.setIcon(self._icon("next.png", 16))
        self.skip_btn.setIcon(self._icon("next.png", 20))

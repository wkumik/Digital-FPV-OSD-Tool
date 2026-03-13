# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
player.py — Video player widgets for VueOSD.

Contains: VideoCanvas, Timeline, TransportBar, PlayerController, PlayerPanel.
"""

import os, sys, time, threading, subprocess, random

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QRect, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QFont, QPixmap, QImage, QPainter, QColor, QPen, QIcon,
    QLinearGradient, QDesktopServices,
)

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

from subprocess_utils import _hidden_popen
from video_processor import find_ffmpeg, get_frame_pts
from osd_renderer import OsdRenderConfig, render_osd_frame, render_fallback

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

# Slider resolution
_SL_MAX = 10_000
_CACHE_MAX = 64


# ─── VideoCanvas ──────────────────────────────────────────────────────────────

class VideoCanvas(QWidget):
    """Two-layer video frame display: base video + OSD overlay.

    The base pixmap (video frame) is resized once on seek/resize and cached.
    The OSD overlay is a separate transparent pixmap that can be updated
    cheaply without re-processing the video frame.
    """

    resized = pyqtSignal()  # emitted after resize so controller can re-render OSD

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

    # ── public API ────────────────────────────────────────────────────────────

    def set_base_frame(self, pil_img):
        """Cache a video-only frame. Resizes to canvas and stores as QPixmap."""
        if not PIL_OK or pil_img is None:
            return
        self._show_placeholder = False
        self._base_pil = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
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
        Sets as base and clears OSD overlay."""
        if not PIL_OK or pil_img is None:
            return
        self._show_placeholder = False
        self._base_pil = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
        self._osd_pixmap = None
        self._donate_rects = []
        self._rebuild_base()
        self.update()

    def set_placeholder(self):
        """Show the Quick Start placeholder."""
        self._base_pil = None
        self._base_pixmap = None
        self._osd_pixmap = None
        self._show_placeholder = True
        self._donate_rects = []
        self.update()

    def has_frame(self):
        return self._base_pil is not None

    def display_size(self):
        """Return (w, h) of the base pixmap, or canvas size if no frame."""
        if self._base_pixmap is not None:
            return self._base_pixmap.width(), self._base_pixmap.height()
        return max(self.width(), 320), max(self.height(), 180)

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

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._base_pil is not None:
            self._rebuild_base()
            self._osd_pixmap = None  # stale size — controller will re-render
            self.resized.emit()
        self.update()

    def mousePressEvent(self, event):
        if self._show_placeholder and self._donate_rects:
            pos = event.position().toPoint()
            if any(r.contains(pos) for r in self._donate_rects):
                QDesktopServices.openUrl(QUrl(_DONATE_URL))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._show_placeholder:
            pos = event.position().toPoint()
            in_zone = any(r.contains(pos) for r in self._donate_rects)
            self.setCursor(Qt.CursorShape.PointingHandCursor if in_zone
                           else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)


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


# ─── PlayerController ─────────────────────────────────────────────────────────

class PlayerController:
    """Manages playback state, FFmpeg processes, frame cache, and seeking.

    This is NOT a QObject — it is owned by PlayerPanel and called directly.
    It uses QTimer and threading internally.
    """

    def __init__(self, canvas: VideoCanvas, timeline: Timeline,
                 transport: TransportBar, composite_fn, osd_fn=None):
        self.canvas = canvas
        self.timeline = timeline
        self.transport = transport
        self._composite_fn = composite_fn  # callable(pil_img, pct) -> pil_img (playback)
        self._osd_fn = osd_fn              # callable(pct, w, h) -> PIL RGBA (preview overlay)

        # Video state
        self.video_path = ""
        self.video_dur = 0.0
        self.video_fps = 60.0
        self.video_w = 0
        self.video_h = 0
        self._color_vf = ""   # color correction video filter string

        # Cache
        self.cached_frames = {}   # pct -> PIL Image
        self._cache_max = _CACHE_MAX

        # Seeking
        self._extract_proc = None
        self._scrub_timer = QTimer()
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(30)
        self._scrub_timer.timeout.connect(self._do_scrub)
        self._pending_pct = 0

        # Drag-aware resolution: 480p while dragging, full after 200ms idle
        self._drag_active = False
        self._drag_idle_timer = QTimer()
        self._drag_idle_timer.setSingleShot(True)
        self._drag_idle_timer.setInterval(200)
        self._drag_idle_timer.timeout.connect(self._on_drag_idle)

        # PTS list for accurate OSD sync on videos with packet loss
        self.pts_list = []       # per-frame PTS in seconds
        self._pts_ready = False

        # Prefetch
        self._prefetch_stop = False

        # Playback
        self._playing = False
        self._speed = 1.0
        self._speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
        self._speed_idx = 2  # index into _speed_levels (1.0x)
        self._ffplay_proc = None
        self._ffplay_stop = threading.Event()
        self._ffplay_t0 = 0.0
        self._ffplay_seek = 0.0
        self._ffplay_w = 0
        self._ffplay_h = 0

        # Display timer (pull model) — 60Hz
        self._display_timer = QTimer()
        self._display_timer.setInterval(16)
        self._display_timer.timeout.connect(self._display_tick)

        # Frame buffer (written by reader thread, read by display timer)
        self._buf_lock = threading.Lock()
        self._buf_frame = None   # PIL image
        self._buf_pct = 0.0

        # Preview debounce (30ms — OSD-only rendering is fast)
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(30)
        self._preview_timer.timeout.connect(self._do_refresh)

        # Wire signals
        self.timeline.seekRequested.connect(self._on_seek)
        self.canvas.resized.connect(self._on_canvas_resized)
        self.transport.playClicked.connect(self.toggle_play)
        self.transport.restartClicked.connect(self.seek_start)
        self.transport.skipEndClicked.connect(self.seek_end)
        self.transport.stepBackClicked.connect(lambda: self.step_backward(1))
        self.transport.stepFwdClicked.connect(lambda: self.step_forward(1))

    # ── Video loading ─────────────────────────────────────────────────────────

    def set_color_vf(self, vf: str):
        """Set color correction video filter for preview extraction.

        When set, the filter is prepended to -vf in all FFmpeg extract/play commands.
        Pass "" to disable.
        """
        changed = getattr(self, '_color_vf', '') != vf
        self._color_vf = vf
        if changed and self.video_path:
            # Invalidate cache and refresh
            self.cached_frames.clear()
            self.timeline.set_cached(set())
            self._do_refresh()
            self._start_prefetch()

    def load_video(self, path, duration, fps, width=0, height=0):
        """Called after video info is available."""
        self.stop()
        self.video_path = path
        self.video_dur = duration
        self.video_fps = fps
        self.video_w = width or 0
        self.video_h = height or 0
        self.cached_frames.clear()
        self.pts_list = []
        self._pts_ready = False
        self.timeline.set_position(0.0)
        self.timeline.set_cached(set())
        self.transport.set_time(0, duration)
        self._extract_at_pct(0)
        # Start prefetch after a short delay
        QTimer.singleShot(400, self._start_prefetch)
        # Probe PTS in background for accurate OSD sync
        threading.Thread(target=self._probe_pts, args=(path,), daemon=True).start()

    def clear(self):
        self.stop()
        self.video_path = ""
        self.video_dur = 0.0
        self.cached_frames.clear()
        self.pts_list = []
        self._pts_ready = False
        self.canvas.set_placeholder()
        self.timeline.set_position(0.0)
        self.timeline.set_cached(set())

    def _probe_pts(self, path):
        """Background: probe per-frame PTS for accurate OSD sync."""
        pts = get_frame_pts(path)
        if pts and path == self.video_path:
            self.pts_list = pts
            self._pts_ready = True

    def pct_to_video_time_ms(self, pct):
        """Convert slider pct (0–_SL_MAX) to video time in ms.

        Uses PTS list when available (accurate for videos with dropped frames),
        falls back to linear mapping otherwise.
        """
        if self._pts_ready and self.pts_list:
            idx = int(pct / _SL_MAX * (len(self.pts_list) - 1))
            idx = max(0, min(idx, len(self.pts_list) - 1))
            return int(self.pts_list[idx] * 1000)
        return int(self.video_dur * pct / _SL_MAX * 1000) if self.video_dur > 0 else 0

    def _extraction_dims(self, cap=1080):
        """Return (scale_w, scale_h) for frame extraction, preserving the
        video's native aspect ratio.  Falls back to canvas AR if video
        dimensions are unknown."""
        if self.video_w > 0 and self.video_h > 0:
            ar = self.video_w / self.video_h
        else:
            cw = max(self.canvas.width(), 320)
            ch = max(self.canvas.height(), 180)
            ar = cw / ch
        scale_h = min(max(self.canvas.height(), 180), cap)
        scale_w = int(scale_h * ar)
        # FFmpeg requires even dimensions
        scale_w -= scale_w % 2
        scale_h -= scale_h % 2
        return max(2, scale_w), max(2, scale_h)

    # ── Seeking ───────────────────────────────────────────────────────────────

    def _on_seek(self, pos):
        """Called when user clicks/drags the timeline."""
        if self._playing:
            self.pause()
        pct = int(pos * _SL_MAX)
        self._update_time_labels(pct)

        # Mark drag active; reset idle timer
        self._drag_active = True
        self._drag_idle_timer.start()

        if pct in self.cached_frames:
            self._show_pct(pct)
            return

        # Show nearest cached frame as placeholder while extracting
        if self.cached_frames:
            nearest = min(self.cached_frames, key=lambda k: abs(k - pct))
            self._show_pct(nearest)

        self._pending_pct = pct
        self._scrub_timer.start()

    def _do_scrub(self):
        pct = self._pending_pct
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def _on_drag_idle(self):
        """Drag ended — re-extract current position at full resolution."""
        self._drag_active = False
        pct = int(self.timeline._position * _SL_MAX)
        # Re-extract at full quality (evict the low-res cached version)
        if pct in self.cached_frames:
            del self.cached_frames[pct]
        self._extract_at_pct(pct)

    def _extract_at_pct(self, pct):
        if not self.video_path or not find_ffmpeg():
            return
        ffmpeg = find_ffmpeg()
        t = self.video_dur * pct / _SL_MAX if self.video_dur > 0 else 0.0

        # Kill in-flight extraction
        prev = self._extract_proc
        if prev is not None and prev.poll() is None:
            try:
                prev.kill()
            except Exception:
                pass
        self._extract_proc = None

        cap = 480 if self._drag_active else 1080
        scale_w, scale_h = self._extraction_dims(cap)
        nbytes = scale_w * scale_h * 3

        def _run():
            try:
                color_vf = getattr(self, '_color_vf', '')
                vf_parts = [f for f in [color_vf, f"scale={scale_w}:{scale_h}"] if f]
                proc = _hidden_popen(
                    [ffmpeg, "-ss", str(t), "-i", self.video_path,
                     "-vf", ",".join(vf_parts),
                     "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-v", "quiet", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                self._extract_proc = proc
                data = proc.stdout.read(nbytes)
                proc.wait(timeout=20)
                if data and len(data) >= nbytes and PIL_OK:
                    img = PILImage.frombytes("RGB", (scale_w, scale_h), data[:nbytes])
                    img = img.convert("RGBA")
                    self.cached_frames[pct] = img
                    self._evict_cache(pct)
                    self.timeline.add_cached(pct / _SL_MAX)
                    def _ready(p=pct):
                        self._show_pct(p)
                    QTimer.singleShot(0, _ready)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def _evict_cache(self, current_pct):
        while len(self.cached_frames) > self._cache_max:
            worst = max(self.cached_frames, key=lambda k: abs(k - current_pct))
            del self.cached_frames[worst]

    def _show_pct(self, pct):
        img = self.cached_frames.get(pct)
        if img is None:
            return
        self.canvas.set_base_frame(img)
        self._update_osd(pct)

    def _update_time_labels(self, pct):
        t_s = self.video_dur * pct / _SL_MAX if self.video_dur > 0 else 0.0
        self.transport.set_time(t_s, self.video_dur)

    # ── Prefetch ──────────────────────────────────────────────────────────────

    def _start_prefetch(self):
        if not self.video_path or self.video_dur <= 0 or not find_ffmpeg():
            return
        positions = list(range(0, _SL_MAX + 1, _SL_MAX // 20))
        to_fetch = [p for p in positions if p not in self.cached_frames]
        if not to_fetch:
            return
        self._prefetch_stop = False
        threading.Thread(target=self._prefetch_frames, args=(to_fetch,),
                         daemon=True).start()

    def _prefetch_frames(self, positions):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return
        scale_w, scale_h = self._extraction_dims(1080)
        nbytes = scale_w * scale_h * 3
        color_vf = getattr(self, '_color_vf', '')

        for pct in positions:
            if self._prefetch_stop:
                break
            if pct in self.cached_frames:
                continue
            t = self.video_dur * pct / _SL_MAX
            try:
                vf_parts = [f for f in [color_vf, f"scale={scale_w}:{scale_h}"] if f]
                proc = _hidden_popen(
                    [ffmpeg, "-ss", str(t), "-i", self.video_path,
                     "-vf", ",".join(vf_parts),
                     "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-v", "quiet", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                data = proc.stdout.read(nbytes)
                proc.wait(timeout=15)
                if (not self._prefetch_stop and data and len(data) >= nbytes and PIL_OK):
                    img = PILImage.frombytes("RGB", (scale_w, scale_h), data[:nbytes])
                    img = img.convert("RGBA")
                    self.cached_frames[pct] = img
                    self.timeline.add_cached(pct / _SL_MAX)
            except Exception:
                pass

    # ── Playback ──────────────────────────────────────────────────────────────

    def toggle_play(self):
        if not self.video_path or self.video_dur <= 0:
            return
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self._stop_pipe()
        ffmpeg = find_ffmpeg()
        if not ffmpeg or not self.video_path:
            return

        seek_s = self.video_dur * self.timeline._position
        scale_w, scale_h = self._extraction_dims(1080)

        self._ffplay_stop.clear()
        self._ffplay_seek = seek_s
        self._ffplay_t0 = time.monotonic()
        self._ffplay_w = scale_w
        self._ffplay_h = scale_h
        self._playing = True
        self.transport.set_playing(True)

        color_vf = getattr(self, '_color_vf', '')
        vf_parts = [f for f in [color_vf, f"scale={scale_w}:{scale_h}"] if f]
        cmd = [
            ffmpeg,
            "-ss", str(seek_s),
            "-i", self.video_path,
            "-vf", ",".join(vf_parts),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-an",
            "-v", "quiet",
            "pipe:1",
        ]

        try:
            self._ffplay_proc = _hidden_popen(
                cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
                bufsize=scale_w * scale_h * 3 * 4,
            )
        except Exception:
            self._playing = False
            self.transport.set_playing(False)
            return

        # Snapshot composite settings
        snap = self._build_play_snap(seek_s, scale_w, scale_h)

        self._display_timer.start()
        threading.Thread(target=self._reader_thread, args=(snap,),
                         daemon=True).start()

    def _build_play_snap(self, seek_s, w, h):
        return dict(
            seek_s=seek_s,
            w=w, h=h,
            fps=max(1.0, self.video_fps),
            speed=self._speed,
        )

    def _reader_thread(self, snap):
        """Background: read raw frames, composite OSD, deposit in buffer."""
        proc = self._ffplay_proc
        stop = self._ffplay_stop
        w, h = snap["w"], snap["h"]
        nbytes = w * h * 3
        seek = snap["seek_s"]
        fps = snap["fps"]
        speed = snap["speed"]
        frame_dur = 1.0 / fps
        t0 = self._ffplay_t0

        # Snapshot PTS list for this playback session
        pts = self.pts_list if self._pts_ready else []
        # Find starting frame index in PTS list
        pts_base_idx = 0
        if pts:
            import bisect
            pts_base_idx = bisect.bisect_left(pts, seek)

        frame_idx = 0
        while not stop.is_set():
            try:
                data = proc.stdout.read(nbytes)
            except Exception:
                break
            if not data or len(data) < nbytes:
                break

            try:
                # Use PTS list for accurate time when available
                if pts and (pts_base_idx + frame_idx) < len(pts):
                    video_time_s = pts[pts_base_idx + frame_idx]
                else:
                    video_time_s = seek + frame_idx * frame_dur
                pct_norm = video_time_s / self.video_dur if self.video_dur > 0 else 0.0
                pct_sl = int(pct_norm * _SL_MAX)

                img = PILImage.frombytes("RGB", (w, h), data[:nbytes]).convert("RGBA")

                # Composite OSD
                if self._composite_fn:
                    composited = self._composite_fn(img, pct_sl)
                else:
                    composited = img

                with self._buf_lock:
                    self._buf_frame = composited
                    self._buf_pct = pct_norm

                frame_idx += 1

                # Pace to real-time (adjusted by speed)
                deadline = t0 + frame_idx * frame_dur / speed
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0.002:
                    time.sleep(sleep_s)

            except Exception:
                frame_idx += 1

        if not stop.is_set():
            QTimer.singleShot(0, self._on_playback_finished)

    def _display_tick(self):
        """Main thread: pull latest frame from buffer, display it."""
        if not self._playing:
            return

        # Update timeline position from elapsed time
        elapsed = time.monotonic() - self._ffplay_t0
        video_pos = self._ffplay_seek + elapsed * self._speed
        if video_pos >= self.video_dur:
            self.pause()
            self.timeline.set_position(1.0)
            self.transport.set_time(self.video_dur, self.video_dur)
            return
        pos_norm = video_pos / self.video_dur if self.video_dur > 0 else 0.0
        self.timeline.set_position(pos_norm)
        self.transport.set_time(video_pos, self.video_dur)

        # Pull frame from buffer
        with self._buf_lock:
            frame = self._buf_frame
            self._buf_frame = None

        if frame is not None:
            self.canvas.set_frame(frame)

    def _on_playback_finished(self):
        if self._playing:
            self.pause()
            self.timeline.set_position(1.0)
            self.transport.set_time(self.video_dur, self.video_dur)

    def pause(self):
        self._playing = False
        self._display_timer.stop()
        self.transport.set_playing(False)
        self._stop_pipe()
        # After playback, restore raw base frame for two-layer mode
        # (playback set_frame bakes OSD into the base — must replace it)
        if self._osd_fn and self.cached_frames:
            pct = int(self.timeline._position * _SL_MAX)
            raw = self.cached_frames.get(pct)
            if raw is None:
                nearest = min(self.cached_frames.keys(),
                              key=lambda k: abs(k - pct))
                raw = self.cached_frames.get(nearest)
            if raw is not None:
                self.canvas.set_base_frame(raw)
                self._update_osd(pct)

    def stop(self):
        self.pause()
        self._prefetch_stop = True

    def _stop_pipe(self):
        self._ffplay_stop.set()
        proc = self._ffplay_proc
        if proc:
            try:
                proc.stdout.close()
                proc.kill()
            except Exception:
                pass
            self._ffplay_proc = None

    # ── Seek helpers ──────────────────────────────────────────────────────────

    def seek_start(self):
        self.pause()
        self.timeline.set_position(0.0)
        pct = 0
        self._update_time_labels(pct)
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def seek_end(self):
        self.pause()
        self.timeline.set_position(1.0)
        pct = _SL_MAX
        self._update_time_labels(pct)
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def step_forward(self, n=1):
        if self._playing:
            self.pause()
        if self.video_dur <= 0 or self.video_fps <= 0:
            return
        frame_step = n / (self.video_fps * self.video_dur) if self.video_dur > 0 else 0.01
        new_pos = min(1.0, self.timeline._position + frame_step)
        self.timeline.set_position(new_pos)
        pct = int(new_pos * _SL_MAX)
        self._update_time_labels(pct)
        self._pending_pct = pct
        self._scrub_timer.start()

    def step_backward(self, n=1):
        if self._playing:
            self.pause()
        if self.video_dur <= 0 or self.video_fps <= 0:
            return
        frame_step = n / (self.video_fps * self.video_dur) if self.video_dur > 0 else 0.01
        new_pos = max(0.0, self.timeline._position - frame_step)
        self.timeline.set_position(new_pos)
        pct = int(new_pos * _SL_MAX)
        self._update_time_labels(pct)
        self._pending_pct = pct
        self._scrub_timer.start()

    # ── Shuttle (J/K/L) ──────────────────────────────────────────────────────

    def shuttle_faster(self):
        if self._speed_idx < len(self._speed_levels) - 1:
            self._speed_idx += 1
        self._speed = self._speed_levels[self._speed_idx]
        self.transport.set_speed(self._speed)
        if not self._playing:
            self.play()

    def shuttle_slower(self):
        """J key — step backward (FFmpeg can't pipe reverse frames)."""
        self.step_backward(1)

    def shuttle_reset(self):
        """K key — toggle play/pause and reset speed to 1x."""
        self._speed_idx = 2
        self._speed = 1.0
        self.transport.set_speed(1.0)
        self.toggle_play()

    # ── Preview refresh ───────────────────────────────────────────────────────

    def queue_refresh(self):
        """Debounced preview refresh — called when OSD settings change."""
        self._preview_timer.start()

    def _do_refresh(self):
        """Refresh OSD overlay only — base frame stays cached."""
        pct = int(self.timeline._position * _SL_MAX)
        # Ensure we have a base frame displayed
        if not self.canvas.has_frame() and self.cached_frames:
            nearest = min(self.cached_frames.keys(), key=lambda k: abs(k - pct))
            self.canvas.set_base_frame(self.cached_frames[nearest])
        self._update_osd(pct)

    def _update_osd(self, pct=None):
        """Render OSD overlay at canvas display size and apply it."""
        if pct is None:
            pct = int(self.timeline._position * _SL_MAX)
        if self._osd_fn and self.canvas.has_frame():
            w, h = self.canvas.display_size()
            if w > 0 and h > 0:
                osd_img = self._osd_fn(pct, w, h)
                self.canvas.set_osd_overlay(osd_img)

    def _on_canvas_resized(self):
        """Canvas was resized — re-render OSD at new display size."""
        self._preview_timer.start()

    def refresh_now(self):
        """Immediate preview refresh (no debounce)."""
        self._do_refresh()

    # ── Trim helpers ──────────────────────────────────────────────────────────

    def set_trim_in(self):
        self.timeline.set_in(self.timeline._position)

    def set_trim_out(self):
        self.timeline.set_out(self.timeline._position)


# ─── PlayerPanel ──────────────────────────────────────────────────────────────

class PlayerPanel(QWidget):
    """Container widget: VideoCanvas + Timeline + TransportBar.

    Drop-in replacement for the old centre panel's player area.
    """

    def __init__(self, theme_fn, icon_fn, btn_play_fn, btn_sec_fn,
                 fs_fn, composite_fn, osd_fn=None, parent=None):
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.canvas = VideoCanvas(theme_fn, fs_fn)
        self.timeline = Timeline(theme_fn)
        self.transport = TransportBar(theme_fn, icon_fn, btn_play_fn, btn_sec_fn)
        self.controller = PlayerController(
            self.canvas, self.timeline, self.transport, composite_fn, osd_fn)

        # Trim labels row
        self._trim_row = QWidget()
        trim_lay = QHBoxLayout(self._trim_row)
        trim_lay.setContentsMargins(0, 0, 0, 0)
        trim_lay.setSpacing(4)
        t = theme_fn()
        self.trim_in_lbl = QLabel("In: 0:00")
        self.trim_out_lbl = QLabel("Out: \u2014")
        for lb in (self.trim_in_lbl, self.trim_out_lbl):
            lb.setStyleSheet(f"color:{t['subtext']};font-size:10px;font-weight:bold;")
        self.trim_rst_btn = QPushButton("\u2715")
        self.trim_rst_btn.setFixedSize(20, 20)
        self.trim_rst_btn.setStyleSheet(btn_sec_fn())
        self.trim_rst_btn.setToolTip("Reset trim to full video")
        self.trim_rst_btn.clicked.connect(self.timeline.reset)

        trim_lay.addWidget(self.trim_in_lbl)
        trim_lay.addStretch()
        trim_lay.addWidget(self.trim_out_lbl)
        trim_lay.addWidget(self.trim_rst_btn)

        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.timeline)
        lay.addWidget(self._trim_row)
        lay.addWidget(self.transport)

    def refresh_theme(self):
        t = self.canvas._T()
        self.canvas.update()
        self.timeline.update()
        self.transport.refresh_theme()
        for lb in (self.trim_in_lbl, self.trim_out_lbl):
            lb.setStyleSheet(f"color:{t['subtext']};font-size:10px;font-weight:bold;")

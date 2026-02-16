# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 OnyxFPV — https://github.com/onyxfpv
"""
OnyxFPV OSD Tool
Parse and overlay MSP-OSD data onto FPV DVR video footage.
"""

import sys, os, threading, subprocess, tempfile

# ── Windows: set AppUserModelID so taskbar shows our icon, not Python's ───────
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OnyxFPV.OSDTool.1")
    except Exception:
        pass

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QGroupBox,
    QCheckBox, QSlider, QComboBox, QGridLayout, QMessageBox,
    QSizePolicy, QSplitter, QScrollArea, QSpinBox, QFrame,
    QDialog, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QPen, QIcon


from srt_parser    import parse_srt, SrtFile
from osd_parser    import parse_osd, OsdFile, GRID_COLS, GRID_ROWS
from p1_osd_parser import detect_p1, parse_p1_osd, p1_to_osd_file
from font_loader   import (fonts_by_firmware, load_font, load_font_from_file,
                           OsdFont, FIRMWARE_PREFIXES)
from osd_renderer  import OsdRenderConfig, render_osd_frame, render_fallback
from video_processor import ProcessingConfig, process_video, get_video_info, find_ffmpeg, detect_hw_encoder
from splash_screen   import SplashScreen


try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

# Suppress console window on Windows for ALL subprocess calls.
# Use STARTUPINFO (more reliable than creationflags alone).
def _hidden_popen(*args, **kwargs):
    """subprocess.Popen wrapper that never shows a console window on Windows."""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", si)
        kwargs.setdefault("creationflags", 0x08000000)  # CREATE_NO_WINDOW
    return subprocess.Popen(*args, **kwargs)

def _hidden_run(*args, **kwargs):
    """subprocess.run wrapper that never shows a console window on Windows."""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", si)
        kwargs.setdefault("creationflags", 0x08000000)  # CREATE_NO_WINDOW
    return subprocess.run(*args, **kwargs)


# ─── Theme system ─────────────────────────────────────────────────────────────

import theme as _theme_mod   # single source of truth for all colours

_DARK_THEME = True   # module-level flag; toggled by the theme button

def _T() -> dict:
    """Return the active theme palette (reads live from theme.py)."""
    return _theme_mod.get_dark() if _DARK_THEME else _theme_mod.get_light()


def _build_styles():
    """Rebuild all stylesheet strings from the active theme."""
    global APP_STYLE, GROUP_STYLE, PATH_EMPTY, PATH_FILLED
    global BTN_SEC, BTN_PRIMARY, BTN_PLAY, BTN_STOP, BTN_DANGER
    global COMBO_STYLE, SLIDER_STYLE, PROG_STYLE
    t = _T()
    is_light = not _DARK_THEME

    APP_STYLE = (
        f"QMainWindow,QWidget{{background:{t['bg']};color:{t['text']};"
        f"font-family:'Segoe UI',Arial,sans-serif;font-size:12px;}}"
        f"QLabel{{color:{t['text']};}}"
        f"QCheckBox{{color:{t['text']};}}"
        f"QScrollArea{{border:none;}}"
    )
    # Light: group titles use subtext (softer), dark: keep accent (blue)
    title_col = t['subtext'] if is_light else t['accent']
    GROUP_STYLE = (
        f"QGroupBox{{border:1px solid {t['border']};border-radius:8px;margin-top:8px;"
        f"padding:6px;font-weight:bold;color:{title_col};font-size:11px;}}"
        f"QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}"
    )
    PATH_EMPTY  = (f"background:{t['bg2']};color:{t['muted']};border:1px solid {t['border']};"
                   f"border-radius:4px;padding:3px 8px;font-size:11px;")
    PATH_FILLED = (f"background:{t['bg2']};color:{t['text']};border:1px solid {t['border2']};"
                   f"border-radius:4px;padding:3px 8px;font-size:11px;")

    # Light theme: buttons use a thin border so they read against the near-white bg
    # without being dark slabs. Dark theme: no border needed (surfaces contrast enough).
    btn_border     = f"1px solid {t['border2']}" if is_light else "none"
    btn_border_hov = f"1px solid {t['border2']}" if is_light else "none"

    BTN_SEC  = (f"QPushButton{{background:{t['surface']};color:{t['text']};"
                f"border:{btn_border};border-radius:6px;"
                f"padding:3px 10px;font-size:11px;}}"
                f"QPushButton:hover{{background:{t['surface2']};border:{btn_border_hov};}}"
                f"QPushButton:pressed{{background:{t['surface3']};}}"
                f"QPushButton:disabled{{background:{t['bg']};color:{t['muted']};"
                f"border:1px solid {t['border']};}}"
                f"QPushButton:checked{{background:{t['accent']};color:{'#ffffff' if is_light else t['bg']};border:none;}}")
    BTN_PRIMARY = (
        # Light: blue fill with white text — clear primary action
        # Dark: blue gradient with dark text
        (f"QPushButton{{background:{t['accent']};color:#ffffff;"
         f"border:none;border-radius:8px;font-weight:bold;}}"
         f"QPushButton:hover{{background:{t['accent2']};}}"
         f"QPushButton:pressed{{background:{t['accent2']};}}"
         f"QPushButton:disabled{{background:{t['surface3']};color:{t['muted']};"
         f"border:1px solid {t['border']};}}")
        if is_light else
        (f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
         f"stop:0 {t['accent']},stop:1 {t['accent2']});"
         f"color:{t['bg']};border:none;border-radius:8px;}}"
         f"QPushButton:hover{{background:{t['accent2']};}}"
         f"QPushButton:pressed{{background:{t['accent']};}}"
         f"QPushButton:disabled{{background:{t['surface']};color:{t['muted']};}}")
    )
    BTN_PLAY = (f"QPushButton{{background:{t['surface']};color:{t['text']};"
                f"border:{btn_border};border-radius:8px;font-size:15px;}}"
                f"QPushButton:hover{{background:{t['surface2']};border:{btn_border_hov};}}"
                f"QPushButton:pressed{{background:{t['accent']};color:#ffffff;}}"
                f"QPushButton:disabled{{background:{t['bg']};color:{t['muted']};"
                f"border:1px solid {t['border']};}}")
    BTN_STOP  = (f"QPushButton{{background:{t['red']};color:#ffffff;"
                 f"border:none;border-radius:8px;font-size:16px;font-weight:bold;}}"
                 f"QPushButton:hover{{background:{t['red']}dd;}}"
                 f"QPushButton:pressed{{background:{t['red']};}}"
                 f"QPushButton:disabled{{background:{t['surface']};color:{t['muted']};"
                 f"border:1px solid {t['border']};}}")
    BTN_DANGER = (f"QPushButton{{background:{t['surface']};color:{t['red']};"
                  f"border:{btn_border};border-radius:6px;font-weight:bold;font-size:11px;}}"
                  f"QPushButton:hover{{background:{t['red']};color:#ffffff;border:none;}}")
    COMBO_STYLE = (f"QComboBox{{background:{t['surface']};color:{t['text']};"
                   f"border:1px solid {t['border2']};"
                   f"border-radius:4px;padding:3px 8px;font-size:11px;}}"
                   f"QComboBox::drop-down{{border:none;padding-right:6px;}}"
                   f"QComboBox QAbstractItemView{{background:{t['bg']};color:{t['text']};"
                   f"selection-background-color:{t['surface2']};border:1px solid {t['border2']};}}")
    SLIDER_STYLE = (f"QSlider::groove:horizontal{{background:{t['border']};height:4px;border-radius:2px;}}"
                    f"QSlider::handle:horizontal{{background:{t['accent']};width:14px;height:14px;"
                    f"margin:-5px 0;border-radius:7px;}}"
                    f"QSlider::sub-page:horizontal{{background:{t['accent']};border-radius:2px;}}")
    PROG_STYLE  = (f"QProgressBar{{background:{t['surface']};border-radius:4px;text-align:center;"
                   f"color:{t['text']};font-size:11px;}}"
                   f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                   f"stop:0 {t['accent']},stop:1 {t['accent2']});border-radius:4px;}}")

# Initialise with dark theme
APP_STYLE = GROUP_STYLE = PATH_EMPTY = PATH_FILLED = ""
BTN_SEC = BTN_PRIMARY = BTN_PLAY = BTN_STOP = BTN_DANGER = ""
COMBO_STYLE = SLIDER_STYLE = PROG_STYLE = ""
_build_styles()


# ─── Icon helpers ─────────────────────────────────────────────────────────────

def _icons_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")

def _icon(name: str, size: int = 22) -> QIcon:
    """Load an icon tinted to the active theme's icon colour."""
    import numpy as np
    path = os.path.join(_icons_dir(), name)
    if not os.path.exists(path):
        return QIcon()
    col = QColor(_T()["icon"])
    cr, cg, cb = col.red(), col.green(), col.blue()
    # Load via PIL for fast numpy recolouring — much faster than per-pixel QImage loop
    try:
        from PIL import Image as _PILImg
        img = _PILImg.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        # Replace RGB channels with target colour, preserve alpha
        arr[:, :, 0] = cr
        arr[:, :, 1] = cg
        arr[:, :, 2] = cb
        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
    except Exception:
        # Fallback: plain load without tinting
        pix = QPixmap(path)
    pix = pix.scaled(size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)
    return QIcon(pix)


# ─── Workers ──────────────────────────────────────────────────────────────────

class ProcessWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def run(self):
        try:
            result = process_video(self.cfg, lambda p, m: self.progress.emit(p, m))
            # result is True (no warning) or a warning string
            warning = result if isinstance(result, str) else ""
            self.finished.emit(True, warning)
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True
        self.terminate()


class VideoInfoWorker(QThread):
    result = pyqtSignal(dict)
    def __init__(self, path): super().__init__(); self.path = path
    def run(self): self.result.emit(get_video_info(self.path))


# ─── Widgets ──────────────────────────────────────────────────────────────────

class FileRow(QWidget):
    def __init__(self, label, placeholder, filter_str, save_mode=False, icon=None, parent=None):
        super().__init__(parent)
        self.filter_str = filter_str
        self.save_mode  = save_mode
        self._path      = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Icon + label in a small fixed-width block
        lbl_row = QHBoxLayout()
        lbl_row.setSpacing(4)
        lbl_row.setContentsMargins(0, 0, 0, 0)
        if icon and not icon.isNull():
            icon_lbl = QLabel()
            icon_lbl.setPixmap(icon.pixmap(16, 16))
            icon_lbl.setFixedSize(16, 16)
            lbl_row.addWidget(icon_lbl)
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color:{_T()['subtext']}")
        self._name_lbl = lbl   # stored for theme reapply
        lbl_row.addWidget(lbl)
        lbl_container = QWidget()
        lbl_container.setFixedWidth(72)
        lbl_container.setLayout(lbl_row)

        self.path_lbl = QLabel(placeholder)
        self.path_lbl.setStyleSheet(PATH_EMPTY)
        self.path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.path_lbl.setFixedHeight(28)
        self.path_lbl.setMinimumWidth(60)
        self.path_lbl.setTextFormat(Qt.TextFormat.PlainText)

        self.btn = QPushButton("Save As" if save_mode else "Browse")
        self.btn.setFixedSize(68, 28)
        self.btn.setStyleSheet(BTN_SEC)
        self.btn.clicked.connect(self._browse)

        self.clr = QPushButton("✕")
        self.clr.setFixedSize(28, 28)
        self.clr.setStyleSheet(BTN_DANGER)
        self.clr.clicked.connect(lambda: self.set_path(""))
        self.clr.setVisible(False)

        lay.addWidget(lbl_container)
        lay.addWidget(self.path_lbl, 1)
        lay.addWidget(self.btn)
        lay.addWidget(self.clr)

    def _browse(self):
        if self.save_mode:
            p, _ = QFileDialog.getSaveFileName(self, "Save", "", "MP4 (*.mp4)")
            if p and not p.lower().endswith(".mp4"):
                p += ".mp4"
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select", "", self.filter_str)
        if p:
            self.set_path(p)

    def set_path(self, path):
        self._path = path
        if path:
            name = os.path.basename(path)
            self.path_lbl.setText(name)
            self.path_lbl.setStyleSheet(PATH_FILLED)
            self.path_lbl.setToolTip(path)
            self.clr.setVisible(True)
        else:
            self.path_lbl.setText("No file selected")
            self.path_lbl.setStyleSheet(PATH_EMPTY)
            self.path_lbl.setToolTip("")
            self.clr.setVisible(False)

    @property
    def path(self):
        return self._path


class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, label, lo, hi, val, suffix="", parent=None):
        super().__init__(parent)
        self._s = suffix
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(58)
        lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")

        self.sl = QSlider(Qt.Orientation.Horizontal)
        self.sl.setRange(lo, hi)
        self.sl.setValue(val)
        self.sl.setStyleSheet(SLIDER_STYLE)

        self.vl = QLabel(f"{val}{suffix}")
        self.vl.setFixedWidth(56)
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.vl.setStyleSheet(f"color:{_T()['text']};font-size:11px;font-weight:bold;")

        self.sl.valueChanged.connect(
            lambda v: (self.vl.setText(f"{v}{self._s}"), self.valueChanged.emit(v))
        )
        lay.addWidget(lbl)
        lay.addWidget(self.sl)
        lay.addWidget(self.vl)

    def value(self): return self.sl.value()
    def setValue(self, v): self.sl.setValue(v)


class InfoCard(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(GROUP_STYLE)
        self._g = QGridLayout(self)
        self._g.setColumnStretch(1, 1)
        self._g.setSpacing(2)
        self._g.setContentsMargins(8, 14, 8, 8)
        self._r = 0

    def add_row(self, k, v):
        kl = QLabel(k + ":")
        kl.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        vl = QLabel(str(v))
        vl.setStyleSheet(f"color:{_T()['text']};font-size:10px;font-weight:600;")
        self._g.addWidget(kl, self._r, 0)
        self._g.addWidget(vl, self._r, 1)
        self._r += 1

    def clear(self):
        while self._g.count():
            it = self._g.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._r = 0


class RenderBar(QWidget):
    """Render progress bar — theme-aware, only fills during an active render."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(22)
        self.setMaximumHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value    = 0      # 0-100
        self._active   = False  # True only while rendering

    def setValue(self, v: int):
        self._value = max(0, min(100, v))
        self.update()

    def setActive(self, active: bool):
        """Call setActive(True) when render starts, setActive(False) when done."""
        self._active = active
        if not active:
            self._value = 0
        self.update()

    def value(self) -> int:
        return self._value

    def paintEvent(self, _e):
        t  = _T()
        w, h = self.width(), self.height()
        p  = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r  = 4

        # Background — surface colour (very subtle in light, dark slab in dark)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t['surface']))
        p.drawRoundedRect(0, 0, w, h, r, r)

        # Fill — only when an active render is in progress
        if self._active and self._value > 0:
            fw = int(w * self._value / 100)
            from PyQt6.QtGui import QLinearGradient
            grad = QLinearGradient(0, 0, fw, 0)
            grad.setColorAt(0.0, QColor(t['accent']))
            grad.setColorAt(1.0, QColor(t['accent2']))
            p.setBrush(grad)
            p.drawRoundedRect(0, 0, fw, h, r, r)

        # Text
        p.setPen(QColor(t['text'] if self._active else t['muted']))
        p.setFont(QFont("Segoe UI", 9))
        if self._active and self._value > 0:
            label = f"{self._value}%"
        elif self._active:
            label = "Starting…"
        else:
            label = "Ready"
        p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, label)
        p.end()


class CacheBar(QWidget):
    """Thin progress bar shown below the frame slider while preview frames are being cached."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._total   = 0   # total frames to cache
        self._cached  = 0   # frames cached so far
        self._visible = False
        self.setVisible(False)

    def start(self, total: int):
        self._total   = max(1, total)
        self._cached  = 0
        self._visible = True
        self.setVisible(True)
        self.update()

    def update_count(self, cached: int):
        self._cached = cached
        self.update()
        if self._cached >= self._total:
            self.finish()

    def finish(self):
        self._visible = False
        self.setVisible(False)

    def paintEvent(self, _e):
        t = _T()
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = 3
        # Background track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t['surface']))
        p.drawRoundedRect(0, 0, w, h, r, r)
        # Fill
        if self._total > 0:
            fw = int(w * min(self._cached, self._total) / self._total)
            if fw > 0:
                from PyQt6.QtGui import QLinearGradient
                grad = QLinearGradient(0, 0, fw, 0)
                grad.setColorAt(0.0, QColor(t['accent']))
                grad.setColorAt(1.0, QColor(t['accent2']))
                p.setBrush(grad)
                p.drawRoundedRect(0, 0, fw, h, r, r)
        # Label
        p.setPen(QColor(t['text']))
        p.setFont(QFont("Segoe UI", 8))
        label = f"Caching preview…  {self._cached}/{self._total}"
        p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, label)
        p.end()


class RangeSelector(QWidget):
    """Dual-handle in/out trim slider drawn with QPainter."""
    rangeChanged = pyqtSignal(float, float)   # in_pct, out_pct (0.0–1.0)

    HANDLE_W = 10
    TRACK_H  = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._in  = 0.0
        self._out = 1.0
        self._drag = None   # "in" | "out" | None
        self.setMouseTracking(True)

    # ── public api ────────────────────────────────────────────────────────────
    @property
    def in_pct(self):  return self._in
    @property
    def out_pct(self): return self._out

    def set_in(self, v):
        self._in = max(0.0, min(v, self._out - 0.01))
        self.update(); self.rangeChanged.emit(self._in, self._out)

    def set_out(self, v):
        self._out = max(self._in + 0.01, min(v, 1.0))
        self.update(); self.rangeChanged.emit(self._in, self._out)

    def reset(self):
        self._in, self._out = 0.0, 1.0
        self.update(); self.rangeChanged.emit(0.0, 1.0)

    # ── geometry helpers ──────────────────────────────────────────────────────
    def _track_rect(self):
        hw = self.HANDLE_W
        return (hw, (self.height() - self.TRACK_H) // 2,
                self.width() - hw * 2, self.TRACK_H)

    def _handle_x(self, pct):
        tx, _, tw, _ = self._track_rect()
        return int(tx + pct * tw)

    def _pct_from_x(self, x):
        tx, _, tw, _ = self._track_rect()
        return max(0.0, min(1.0, (x - tx) / tw))

    def _handle_rect(self, pct):
        hw = self.HANDLE_W; hh = self.height()
        cx = self._handle_x(pct)
        return (cx - hw // 2, 4, hw, hh - 8)

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        tx, ty, tw, th = self._track_rect()

        # Full track (dark)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#313244"))
        p.drawRoundedRect(tx, ty, tw, th, 3, 3)

        # Active region (blue)
        x1 = self._handle_x(self._in)
        x2 = self._handle_x(self._out)
        p.setBrush(QColor("#89b4fa"))
        p.drawRect(x1, ty, x2 - x1, th)

        # Handles
        hw = self.HANDLE_W
        for pct, label in ((self._in, "I"), (self._out, "O")):
            hx, hy, hwidth, hheight = self._handle_rect(pct)
            p.setBrush(QColor("#cdd6f4"))
            p.drawRoundedRect(hx, hy, hwidth, hheight, 3, 3)
            p.setPen(QColor("#1e1e2e"))
            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.drawText(hx, hy, hwidth, hheight,
                       Qt.AlignmentFlag.AlignCenter, label)
            p.setPen(Qt.PenStyle.NoPen)

        p.end()

    # ── mouse interaction ────────────────────────────────────────────────────
    def _nearest_handle(self, x):
        xi = self._handle_x(self._in)
        xo = self._handle_x(self._out)
        return "in" if abs(x - xi) <= abs(x - xo) else "out"

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = self._nearest_handle(int(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._drag:
            pct = self._pct_from_x(int(e.position().x()))
            if self._drag == "in":
                self.set_in(pct)
            else:
                self.set_out(pct)
        else:
            # Cursor hint
            x = int(e.position().x())
            xi = self._handle_x(self._in)
            xo = self._handle_x(self._out)
            near = min(abs(x - xi), abs(x - xo))
            self.setCursor(Qt.CursorShape.SizeHorCursor if near < 14
                           else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag = None


class PreviewPanel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        t = _T()
        self.setStyleSheet(
            f"background:{t['bg2']};border:1px solid {t['border']};border-radius:8px;")
        self._pil_img = None          # always keep full-res PIL source
        self._placeholder()

    def _placeholder(self):
        self._pil_img = None
        self._redraw_placeholder()

    def _redraw_placeholder(self):
        w, h = max(self.width(), 640), max(self.height(), 360)
        pix = QPixmap(w, h)
        t = _T()
        pix.fill(QColor(t["bg2"]))
        p = QPainter(pix)
        p.setPen(QPen(QColor(t["surface2"])))
        p.setFont(QFont("Segoe UI", 13))
        p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "🎬  Load a video to see preview")
        p.end()
        super().setPixmap(pix)

    def show_frame(self, img):
        if not PIL_OK:
            return
        self._pil_img = img.convert("RGBA")
        self._repaint()

    def _repaint(self):
        """Render PIL image scaled to fit current widget, maintaining aspect ratio."""
        if self._pil_img is None:
            return
        w = max(self.width(),  320)
        h = max(self.height(), 180)
        # Scale down only (thumbnail won't upscale) — use LANCZOS for quality
        tmp = self._pil_img.copy()
        tmp.thumbnail((w, h), PILImage.LANCZOS)
        data = tmp.tobytes("raw", "RGBA")
        qi   = QImage(data, tmp.width, tmp.height, QImage.Format.Format_RGBA8888)
        # Centre inside the widget — QLabel AlignCenter handles this automatically
        super().setPixmap(QPixmap.fromImage(qi))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pil_img is not None:
            self._repaint()
        else:
            self._redraw_placeholder()


def _sep():
    """Thin horizontal separator line."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{_T()['border']}")
    f.setFixedHeight(1)
    return f


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.srt_data:   Optional[SrtFile] = None
        self.osd_data:   Optional[OsdFile] = None
        self.font_obj:   Optional[OsdFont] = None
        self.video_frame = None
        self.video_fps:  float = 60.0
        self.video_dur:  float = 0.0
        self.cached_frames: dict = {}
        self.worker      = None
        self._font_db:   dict = {}
        self.source_mbps: float = 0.0   # source video bitrate, set after loading
        self._extract_proc = None        # current ffmpeg frame-extract process
        self._prefetch_stop = False      # signal to stop background prefetch
        self._scrub_timer  = QTimer()    # debounce frame-slider scrubbing
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(80)
        self._scrub_timer.timeout.connect(self._do_scrub)
        self._pending_pct  = 0
        self._preview_timer = QTimer()   # debounce position/scale/opacity sliders
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(60)
        self._preview_timer.timeout.connect(self._refresh_preview)
        # Playback state
        self._play_timer   = QTimer()
        self._play_timer.setInterval(100)   # tick every 100ms → ~10fps preview steps
        self._play_timer.timeout.connect(self._play_tick)
        self._playing      = False

        self.setWindowTitle("OnyxFPV OSD Tool")
        # App icon — resolved relative to this script so it works from any CWD
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(APP_STYLE)

        # ── Root splitter: left | centre+bottom | right ───────────────────────
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        cw = QWidget()
        cw.setLayout(root)
        self.setCentralWidget(cw)

        # ── LEFT PANEL (scrollable) ───────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{background:#1e1e2e;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;}"
        )
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(400)

        left_inner = QWidget()
        left_inner.setMinimumWidth(280)
        ll = QVBoxLayout(left_inner)
        ll.setContentsMargins(14, 16, 10, 16)
        ll.setSpacing(10)

        # Header: title + theme toggle button on the same row
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(8)

        h1 = QLabel("OnyxFPV OSD")
        h1.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        h1.setStyleSheet(f"color:{_T()['text']};")
        self._h1 = h1

        self._theme_btn = QPushButton()
        self._theme_btn.setFixedSize(30, 30)
        self._theme_btn.setToolTip("Toggle light / dark theme")
        self._theme_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:15px;}}"
            f"QPushButton:hover{{background:{_T()['surface']};}}"
        )
        self._theme_btn.setIcon(_icon("moon-dark.png", 18))
        self._theme_btn.clicked.connect(self._toggle_theme)

        self._palette_btn = QPushButton()
        self._palette_btn.setFixedSize(30, 30)
        self._palette_btn.setToolTip("Open theme colour editor")
        self._palette_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:15px;}}"
            f"QPushButton:hover{{background:{_T()['surface']};}}"
        )
        self._palette_btn.setText("🎨")
        self._palette_btn.setFont(QFont("Segoe UI", 14))
        self._palette_btn.clicked.connect(self._open_theme_editor)
        self._theme_editor_dlg = None   # lazily created

        hdr_row.addWidget(h1)
        hdr_row.addStretch()
        hdr_row.addWidget(self._palette_btn)
        hdr_row.addWidget(self._theme_btn)
        ll.addLayout(hdr_row)

        h2 = QLabel("MSP-OSD overlay for FPV DVR video")
        h2.setStyleSheet(f"color:{_T()['muted']};font-size:10px;margin-bottom:2px;")
        self._h2 = h2
        ll.addWidget(h2)

        # ── Files group ───────────────────────────────────────────────────────
        fg = QGroupBox("Files")
        fg.setStyleSheet(GROUP_STYLE)
        fgl = QVBoxLayout(fg)
        fgl.setSpacing(4)
        fgl.setContentsMargins(10, 16, 10, 10)

        self.video_row = FileRow("Video",  "Select video…",  "Video (*.mp4 *.mkv *.avi *.mov)",
                                 icon=_icon("video.png", 16))
        self.osd_row   = FileRow("OSD",    "Auto-detected",  "OSD (*.osd)",
                                 icon=_icon("gear.png",  16))
        self.srt_row   = FileRow("SRT",    "Auto-detected",  "SRT (*.srt)",
                                 icon=_icon("wifi.png",  16))
        self.video_row.btn.clicked.disconnect()
        self.video_row.btn.clicked.connect(self._on_video)
        self.osd_row.btn.clicked.disconnect()
        self.osd_row.btn.clicked.connect(self._manual_osd)
        self.srt_row.btn.clicked.disconnect()
        self.srt_row.btn.clicked.connect(self._manual_srt)

        note = QLabel("💡 Select any file — .osd and .srt auto-detected by filename")
        note.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        note.setWordWrap(True)

        fgl.addWidget(self.video_row)
        fgl.addWidget(self.osd_row)
        fgl.addWidget(self.srt_row)
        fgl.addWidget(note)
        ll.addWidget(fg)

        # ── OSD Font group ────────────────────────────────────────────────────
        fontg = QGroupBox("OSD Font")
        fontg.setStyleSheet(GROUP_STYLE)
        fontgl = QVBoxLayout(fontg)
        fontgl.setSpacing(6)
        fontgl.setContentsMargins(10, 16, 10, 10)

        # Firmware row
        fw_row = QHBoxLayout()
        fw_lbl = QLabel("Firmware:")
        fw_lbl.setFixedWidth(68)
        fw_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.fw_combo = QComboBox()
        self.fw_combo.setStyleSheet(COMBO_STYLE)
        self.fw_combo.addItems(list(FIRMWARE_PREFIXES.keys()))
        self.fw_combo.currentTextChanged.connect(self._on_fw_changed)
        fw_row.addWidget(fw_lbl)
        fw_row.addWidget(self.fw_combo, 1)
        fontgl.addLayout(fw_row)

        # Style row
        st_row = QHBoxLayout()
        st_lbl = QLabel("Style:")
        st_lbl.setFixedWidth(68)
        st_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.style_combo = QComboBox()
        self.style_combo.setStyleSheet(COMBO_STYLE)
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        st_row.addWidget(st_lbl)
        st_row.addWidget(self.style_combo, 1)
        fontgl.addLayout(st_row)

        # HD + Custom row
        hd_row = QHBoxLayout()
        self.hd_check = QCheckBox("HD tiles")
        self.hd_check.setChecked(True)
        self.hd_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.hd_check.stateChanged.connect(self._reload_font)
        custom_btn = QPushButton("Custom…")
        custom_btn.setStyleSheet(BTN_SEC)
        custom_btn.setFixedHeight(26)
        custom_btn.clicked.connect(self._custom_font)
        hd_row.addWidget(self.hd_check)
        hd_row.addStretch()
        hd_row.addWidget(custom_btn)
        fontgl.addLayout(hd_row)

        self.font_lbl = QLabel("No font loaded")
        self.font_lbl.setStyleSheet(f"color:{_T()['orange']};font-size:10px;")
        self.font_lbl.setWordWrap(True)
        fontgl.addWidget(self.font_lbl)
        ll.addWidget(fontg)

        # ── Link Status Bar ───────────────────────────────────────────────────
        srtg = QGroupBox("Link Status Bar")
        srtg.setStyleSheet(GROUP_STYLE)
        srtgl = QVBoxLayout(srtg)
        srtgl.setSpacing(4)
        srtgl.setContentsMargins(10, 16, 10, 10)

        self.srt_bar_check = QCheckBox("Show link status bar")
        self.srt_bar_check.setChecked(True)
        self.srt_bar_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.srt_bar_check.stateChanged.connect(self._refresh_preview)

        self.srt_opacity_sl = LabeledSlider("Opacity", 10, 100, 60, "%")
        self.srt_opacity_sl.valueChanged.connect(self._refresh_preview)

        note2 = QLabel("Radio signal, bitrate, GPS, altitude from .srt.\n"
                       "'No MAVLink telemetry' lines are hidden.")
        note2.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        note2.setWordWrap(True)

        srtgl.addWidget(self.srt_bar_check)
        srtgl.addWidget(self.srt_opacity_sl)
        srtgl.addWidget(note2)
        ll.addWidget(srtg)

        ll.addStretch()
        left_scroll.setWidget(left_inner)
        self._left_scroll = left_scroll   # saved for theme reapply

        # ── CENTRE: preview + below-video controls ─────────────────────────────
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(10, 16, 10, 12)
        cl.setSpacing(8)

        prev_lbl = QLabel("Preview")
        prev_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        prev_lbl.setStyleSheet(f"color:{_T()['text']}")
        cl.addWidget(prev_lbl)

        self.preview = PreviewPanel()
        self._preview_panel = self.preview   # saved for theme reapply
        cl.addWidget(self.preview, 1)

        # Frame scrub
        frow = QHBoxLayout()
        fl = QLabel("Frame:")
        fl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        fl.setFixedWidth(46)
        self.frame_sl = QSlider(Qt.Orientation.Horizontal)
        self.frame_sl.setRange(0, 100)
        self.frame_sl.setValue(0)
        self.frame_sl.setStyleSheet(SLIDER_STYLE)
        self.frame_sl.valueChanged.connect(self._on_frame_sl)
        self.frame_lbl = QLabel("0%")
        self.frame_lbl.setFixedWidth(34)
        self.frame_lbl.setStyleSheet(f"color:{_T()['text']};font-size:11px;font-weight:bold;")
        frow.addWidget(fl)
        frow.addWidget(self.frame_sl)
        frow.addWidget(self.frame_lbl)
        cl.addLayout(frow)

        self.frame_info = QLabel("t = 0.0s  |  OSD —")
        self.frame_info.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        cl.addWidget(self.frame_info)

        self.cache_bar = CacheBar()
        cl.addWidget(self.cache_bar)

        # ── Trim range selector ───────────────────────────────────────────────
        trim_hdr = QHBoxLayout()
        trim_lbl = QLabel("Trim")
        trim_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        trim_lbl.setFixedWidth(36)
        self.trim_in_lbl  = QLabel("In: 0:00")
        self.trim_out_lbl = QLabel("Out: —")
        for lb in (self.trim_in_lbl, self.trim_out_lbl):
            lb.setStyleSheet(f"color:{_T()['subtext']};font-size:10px;font-weight:bold;")
        trim_rst = QPushButton("✕")
        trim_rst.setFixedSize(20, 20)
        trim_rst.setStyleSheet(BTN_SEC)
        trim_rst.setToolTip("Reset trim to full video")
        trim_rst.clicked.connect(self._trim_reset)
        trim_hdr.addWidget(trim_lbl)
        trim_hdr.addWidget(self.trim_in_lbl)
        trim_hdr.addStretch()
        trim_hdr.addWidget(self.trim_out_lbl)
        trim_hdr.addWidget(trim_rst)
        cl.addLayout(trim_hdr)

        self.trim_sel = RangeSelector()
        self.trim_sel.rangeChanged.connect(self._on_trim_changed)
        self.trim_sel.rangeChanged.connect(lambda *_: self._update_size_hint())
        cl.addWidget(self.trim_sel)

        # ── Playback controls (icon buttons) ─────────────────────────────────
        play_row = QHBoxLayout()
        play_row.setSpacing(4)

        self.restart_btn = QPushButton()
        self.restart_btn.setIcon(_icon("rewind.png", 20))
        self.restart_btn.setFixedSize(34, 34)
        self.restart_btn.setStyleSheet(BTN_PLAY)
        self.restart_btn.setToolTip("Go to start")
        self.restart_btn.clicked.connect(self._play_restart)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(_icon("play.png", 22))
        self.play_btn.setFixedSize(44, 34)
        self.play_btn.setStyleSheet(BTN_PLAY)
        self.play_btn.setToolTip("Play / Pause")
        self.play_btn.clicked.connect(self._play_toggle)

        ref_btn = QPushButton("Refresh Preview")
        ref_btn.setFixedHeight(34)
        ref_btn.setMinimumWidth(120)
        ref_btn.setStyleSheet(BTN_SEC)
        ref_btn.clicked.connect(self._refresh_preview)


        play_row.addWidget(self.restart_btn)
        play_row.addWidget(self.play_btn)
        play_row.addStretch()
        play_row.addWidget(ref_btn)
        cl.addLayout(play_row)

        # ── Smashicons credit ─────────────────────────────────────────────────
        credit = QLabel(
            'Icons by <a href="https://www.flaticon.com/free-icons/wifi-connection" '
            'style="color:#2a2a3a;text-decoration:none;">Smashicons – Flaticon</a>'
        )
        credit.setStyleSheet(f"color:{_T()['muted']};font-size:8px;")
        credit.setOpenExternalLinks(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignRight)
        cl.addWidget(credit)

        cl.addWidget(_sep())

        # ── Below-video: Fine-tune + Info cards side by side ──────────────────
        below = QHBoxLayout()
        below.setSpacing(10)

        # Fine-tune position
        posg = QGroupBox("Fine-tune Position & Scale")
        posg.setStyleSheet(GROUP_STYLE)
        posgl = QVBoxLayout(posg)
        posgl.setSpacing(4)
        posgl.setContentsMargins(10, 16, 10, 10)

        pos_note = QLabel("OSD auto-fitted to video height, centred.")
        pos_note.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        posgl.addWidget(pos_note)

        self.sl_x     = LabeledSlider("X offset", -400, 400,   0, " px")
        self.sl_y     = LabeledSlider("Y offset", -200, 200,   0, " px")
        self.sl_scale = LabeledSlider("Scale",      50, 150, 100, "%")
        for sl in (self.sl_x, self.sl_y, self.sl_scale):
            sl.valueChanged.connect(self._queue_preview)
            posgl.addWidget(sl)

        rst = QPushButton("↺  Reset")
        rst.setStyleSheet(BTN_SEC)
        rst.setFixedHeight(26)
        rst.clicked.connect(self._reset_pos)
        posgl.addWidget(rst)

        below.addWidget(posg, 2)

        # Info cards
        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(6)
        self.vid_card = InfoCard("Video")
        self.osd_card = InfoCard("OSD")
        self.srt_card = InfoCard("📡 Link")
        cards_layout.addWidget(self.vid_card)
        cards_layout.addWidget(self.osd_card)
        cards_layout.addWidget(self.srt_card)

        below.addWidget(cards_widget, 3)
        cl.addLayout(below)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(360)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 16, 14, 16)
        rl.setSpacing(10)

        out_hdr = QLabel("Output & Encoding")
        out_hdr.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        out_hdr.setStyleSheet(f"color:{_T()['text']}")
        rl.addWidget(out_hdr)

        # Output file
        out_fg = QGroupBox("Output File")
        out_fg.setStyleSheet(GROUP_STYLE)
        out_fgl = QVBoxLayout(out_fg)
        out_fgl.setContentsMargins(10, 16, 10, 10)
        self.out_row = FileRow("Output", "Choose output path…", "", save_mode=True, icon=_icon("save.png", 16))
        out_fgl.addWidget(self.out_row)
        rl.addWidget(out_fg)

        # Encoding settings
        encg = QGroupBox("Encoding")
        encg.setStyleSheet(GROUP_STYLE)
        encgl = QVBoxLayout(encg)
        encgl.setSpacing(7)
        encgl.setContentsMargins(10, 16, 10, 10)

        # Codec row
        codec_row = QHBoxLayout()
        codec_lbl = QLabel("Codec:")
        codec_lbl.setFixedWidth(52)
        codec_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.codec_cb = QComboBox()
        self.codec_cb.addItems(["H.264 (libx264)", "H.265 (libx265)"])
        self.codec_cb.setStyleSheet(COMBO_STYLE)
        self.codec_cb.currentIndexChanged.connect(self._on_codec_changed)
        codec_row.addWidget(codec_lbl)
        codec_row.addWidget(self.codec_cb, 1)
        encgl.addLayout(codec_row)

        # Quality mode: CRF or Bitrate
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Quality:")
        mode_lbl.setFixedWidth(52)
        mode_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.mode_crf_btn = QPushButton("CRF")
        self.mode_mbps_btn = QPushButton("Mbit/s")
        for b in (self.mode_crf_btn, self.mode_mbps_btn):
            b.setFixedHeight(24)
            b.setStyleSheet(BTN_SEC)
            b.setCheckable(True)
        self.mode_crf_btn.setChecked(True)
        self.mode_crf_btn.clicked.connect(lambda: self._set_quality_mode("crf"))
        self.mode_mbps_btn.clicked.connect(lambda: self._set_quality_mode("mbps"))
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.mode_crf_btn)
        mode_row.addWidget(self.mode_mbps_btn)
        mode_row.addStretch()
        encgl.addLayout(mode_row)

        # CRF slider (default 28 — more sane default than 23)
        self.crf_sl = LabeledSlider("CRF", 15, 40, 28)
        self.crf_sl.sl.setToolTip(
            "Lower CRF = better quality but LARGER file.\n"
            "Recommended: 28 (H.265) / 23 (H.264).\n"
            "This is NOT a bitrate — it's a quality factor."
        )
        self.crf_sl.valueChanged.connect(self._update_size_hint)
        encgl.addWidget(self.crf_sl)

        # Bitrate spinbox (hidden by default)
        self.mbps_row = QWidget()
        mbps_lay = QHBoxLayout(self.mbps_row)
        mbps_lay.setContentsMargins(0, 0, 0, 0)
        mbps_lbl = QLabel("Mbit/s:")
        mbps_lbl.setFixedWidth(52)
        mbps_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.mbps_spin = QSpinBox()
        self.mbps_spin.setRange(1, 100)
        self.mbps_spin.setValue(8)
        self.mbps_spin.setSuffix(" Mbit/s")
        self.mbps_spin.setStyleSheet(
            f"QSpinBox{{background:{_T()['surface']};color:{_T()['text']};"
            f"border:1px solid {_T()['border2']};border-radius:4px;padding:3px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;"
            f"background:{_T()['surface2']};border-radius:2px;}}"
        )
        self.mbps_spin.valueChanged.connect(self._update_size_hint)
        mbps_lay.addWidget(mbps_lbl)
        mbps_lay.addWidget(self.mbps_spin, 1)
        self.mbps_row.setVisible(False)
        encgl.addWidget(self.mbps_row)

        # Estimated size hint
        self.size_hint = QLabel("")
        self.size_hint.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        encgl.addWidget(self.size_hint)

        # GPU row — detection runs in background so it never blocks startup
        self.hw_check = QCheckBox("⚡  GPU acceleration")
        self.hw_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.hw_check.setEnabled(False)
        self.hw_check.setChecked(False)
        self.hw_check.stateChanged.connect(self._update_size_hint)
        self.hw_lbl = QLabel("Detecting GPU…")
        self.hw_lbl.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        encgl.addWidget(self.hw_check)
        encgl.addWidget(self.hw_lbl)

        # Kick off GPU detection in background.
        # Result is written to _gpu_result[], polled by a QTimer on the main thread.
        # QTimer.singleShot() from a non-main thread is unreliable on Windows —
        # using a poll timer avoids that entirely.
        _gpu_result = [None]   # None=pending, False=done-no-gpu, dict=done-found
        _gpu_done   = [False]

        def _detect_gpu():
            try:
                _ffp = find_ffmpeg()
                _hw  = detect_hw_encoder(_ffp) if _ffp else None
            except Exception:
                _hw = None
            _gpu_result[0] = _hw if _hw else False
            _gpu_done[0]   = True

        threading.Thread(target=_detect_gpu, daemon=True).start()

        def _poll_gpu():
            if not _gpu_done[0]:
                return   # still running — poll again next tick
            _poll_timer.stop()
            _hw = _gpu_result[0]
            if _hw:
                self.hw_check.setEnabled(True)
                self.hw_check.setChecked(True)
                self.hw_lbl.setText(f"✓ {_hw['name']}")
                self.hw_lbl.setStyleSheet(f"color:{_T()['green']};font-size:10px;")
                self.hw_check.setToolTip(f"✓ {_hw['name']} ({_hw['h264']})")
            else:
                self.hw_lbl.setText("No GPU encoder found")
                self.hw_lbl.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
                self.hw_check.setToolTip("No GPU encoder found (NVENC/AMF/QSV/VAAPI)")
            self._update_size_hint()

        _poll_timer = QTimer(self)
        _poll_timer.setInterval(500)   # check every 500ms
        _poll_timer.timeout.connect(_poll_gpu)
        _poll_timer.start()

        upscale_row = QHBoxLayout()
        upscale_lbl = QLabel("Upscale output:")
        upscale_lbl.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["Off", "1440p  (2560×1440)", "2.7K  (2688×1512)", "4K  (3840×2160)"])
        self.upscale_combo.setStyleSheet(COMBO_STYLE)
        self.upscale_combo.setToolTip(
            "Scale the output video to a higher resolution using Lanczos.\n"
            "Useful when source is 1080p and you want a sharper result on a high-res display."
        )
        upscale_row.addWidget(upscale_lbl)
        upscale_row.addWidget(self.upscale_combo, 1)
        encgl.addLayout(upscale_row)

        rl.addWidget(encg)

        # Progress — custom painted bar (stylesheet chunk is unreliable on Windows)
        self.prog = RenderBar()
        rl.addWidget(self.prog)

        self.status = QLabel("Ready")
        self.status.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)

        # OSD trimmed warning (hidden by default)
        self.osd_warn = QLabel("⚠ No OSD elements in trim window — rendering without OSD overlay")
        self.osd_warn.setStyleSheet(f"color:{_T()['orange']};font-size:10px;")
        self.osd_warn.setWordWrap(True)
        self.osd_warn.setVisible(False)
        rl.addWidget(self.osd_warn)

        # Render + Stop buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.render_btn = QPushButton("  Render Video")
        self.render_btn.setIcon(_icon("render.png", 20))
        self.render_btn.setFixedHeight(42)
        self.render_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.render_btn.setStyleSheet(BTN_PRIMARY)
        self.render_btn.clicked.connect(self._render)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(_icon("stop.png", 20))
        self.stop_btn.setFixedSize(42, 42)
        self.stop_btn.setStyleSheet(BTN_STOP)
        self.stop_btn.setToolTip("Stop render")
        self.stop_btn.clicked.connect(self._stop_render)
        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.render_btn, 1)
        btn_row.addWidget(self.stop_btn)
        rl.addLayout(btn_row)

        # FFmpeg status
        self.ffmpeg_lbl = QLabel()
        self.ffmpeg_lbl.setWordWrap(True)
        self._refresh_ffmpeg_status()
        rl.addWidget(self.ffmpeg_lbl)


        rl.addStretch()

        # ── Assemble root layout ──────────────────────────────────────────────
        root.addWidget(left_scroll)
        root.addWidget(centre, 1)
        root.addWidget(right)

        # Vertical dividers
        for w in (left_scroll, centre):
            div = QFrame()
            div.setFrameShape(QFrame.Shape.VLine)
            self._dividers = getattr(self, '_dividers', [])
            self._dividers.append(div)
            div.setStyleSheet(f"color:{_T()['border']};")
            div.setFixedWidth(1)
            root.insertWidget(root.indexOf(w) + 1, div)

        # Collect buttons and labels for theme reapply
        # (theme uses findChildren — no explicit list needed)

        QTimer.singleShot(200, lambda: self._on_fw_changed(self.fw_combo.currentText()))

    # ── Quality mode ──────────────────────────────────────────────────────────

    def _set_quality_mode(self, mode):
        self.mode_crf_btn.setChecked(mode == "crf")
        self.mode_mbps_btn.setChecked(mode == "mbps")
        self.crf_sl.setVisible(mode == "crf")
        self.mbps_row.setVisible(mode == "mbps")
        self._update_size_hint()

    def _on_codec_changed(self):
        # Adjust default CRF when switching codec
        if "265" in self.codec_cb.currentText():
            if self.crf_sl.value() == 23:
                self.crf_sl.setValue(28)
        else:
            if self.crf_sl.value() == 28:
                self.crf_sl.setValue(23)
        self._update_size_hint()

    def _update_size_hint(self):
        if self.video_dur <= 0:
            self.size_hint.setText("")
            return
        # Use trimmed duration for estimate if trim is set
        in_pct  = self.trim_sel.in_pct  if hasattr(self, 'trim_sel') else 0.0
        out_pct = self.trim_sel.out_pct if hasattr(self, 'trim_sel') else 1.0
        dur     = self.video_dur * (out_pct - in_pct)
        gpu_on = self.hw_check.isChecked() and self.hw_check.isEnabled()
        is_265 = "265" in self.codec_cb.currentText()
        if self.mode_mbps_btn.isChecked():
            mbps   = self.mbps_spin.value()
            est_mb = mbps * dur / 8
            self.size_hint.setText(f"≈ {est_mb:.0f} MB at {mbps} Mbit/s")
            return
        crf = self.crf_sl.value()
        # Anchored to source bitrate for accuracy on FPV high-motion content.
        # After NVENC CQ normalisation (+9 offset applied in video_processor):
        #   CPU H.265 CRF 28 ≈ 0.7× src  CPU H.264 CRF 23 ≈ 1.0× src
        #   GPU NVENC (any)  CRF 23 ≈ 1.2× src  (slightly less efficient than x264)
        # Every 6 CRF steps = 2× bitrate.
        if self.source_mbps > 0.1:
            s = self.source_mbps
            if gpu_on:
                base, ref = s * 1.2, 23
            elif is_265:
                base, ref = s * 0.7, 28
            else:
                base, ref = s * 1.0, 23
        else:
            base, ref = (5.0, 28) if is_265 else (8.0, 23)
        est_mbps = base * (2 ** ((ref - crf) / 6.0))
        est_mb   = est_mbps * dur / 8
        src_note = f"  src {self.source_mbps:.1f} Mbit/s" if self.source_mbps > 0.1 else ""
        gpu_note = " · GPU" if gpu_on else ""
        self.size_hint.setText(f"≈ {est_mb:.0f} MB  (~{est_mbps:.1f} Mbit/s{src_note}{gpu_note})")

    # ── Font ──────────────────────────────────────────────────────────────────

    def _on_fw_changed(self, fw):
        self._font_db = fonts_by_firmware(fw)
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        prefixes = FIRMWARE_PREFIXES.get(fw, [fw])
        def _clean(n):
            for p in prefixes:
                if n.upper().startswith(p.upper()):
                    return n[len(p):]
            return n
        for name in self._font_db:
            self.style_combo.addItem(_clean(name), userData=name)
        self.style_combo.blockSignals(False)
        for i in range(self.style_combo.count()):
            if "Nexus" in self.style_combo.itemText(i):
                self.style_combo.setCurrentIndex(i)
                break
        self._reload_font()

    def _on_style_changed(self):
        self._reload_font()

    def _reload_font(self):
        raw_name = self.style_combo.currentData()
        if not raw_name:
            return
        folder = self._font_db.get(raw_name)
        if not folder:
            return
        self.font_obj = load_font(folder, prefer_hd=self.hd_check.isChecked())
        if self.font_obj:
            v = "HD" if self.hd_check.isChecked() else "SD"
            nc = f", {self.font_obj.n_cols}×256 chars" if self.font_obj.n_cols > 1 else ""
            self.font_lbl.setText(f"✓ {raw_name} ({v})  {self.font_obj.tile_w}×{self.font_obj.tile_h}px{nc}")
            self.font_lbl.setStyleSheet(f"color:{_T()['green']};font-size:10px;")
        else:
            self.font_lbl.setText(f"✗ Could not load {raw_name}")
            self.font_lbl.setStyleSheet(f"color:{_T()['red']};font-size:10px;")
        self._refresh_preview()

    def _custom_font(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Font PNG", "", "PNG (*.png)")
        if p:
            self.font_obj = load_font_from_file(p)
            if self.font_obj:
                self.font_lbl.setText(f"✓ Custom: {os.path.basename(p)}")
                self.font_lbl.setStyleSheet(f"color:{_T()['green']};font-size:10px;")
                self._refresh_preview()

    # ── File selection ────────────────────────────────────────────────────────

    def _auto_detect(self, base_path):
        p = Path(base_path); stem = p.stem; dirp = p.parent
        ext = p.suffix.lower()
        candidates = {
            '.osd': dirp / (stem + ".osd"),
            '.srt': dirp / (stem + ".srt"),
            '.mp4': dirp / (stem + ".mp4"),
        }
        if ext == '.mp4':
            if candidates['.osd'].exists():
                self._load_osd(str(candidates['.osd']))
            else:
                self._try_load_p1_osd(base_path)
            if candidates['.srt'].exists(): self._load_srt(str(candidates['.srt']))
        elif ext == '.osd':
            if candidates['.srt'].exists(): self._load_srt(str(candidates['.srt']))
            if candidates['.mp4'].exists(): self._load_video(str(candidates['.mp4']))
        elif ext == '.srt':
            if candidates['.osd'].exists(): self._load_osd(str(candidates['.osd']))
            if candidates['.mp4'].exists(): self._load_video(str(candidates['.mp4']))

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _toggle_theme(self):
        global _DARK_THEME
        _DARK_THEME = not _DARK_THEME
        _build_styles()
        self._apply_theme()

    def _open_theme_editor(self):
        """Open (or raise) the palette editor dialog."""
        from theme_editor import ThemeEditor
        if self._theme_editor_dlg is None or not self._theme_editor_dlg.isVisible():
            self._theme_editor_dlg = ThemeEditor(self)
            self._theme_editor_dlg.applied.connect(self._on_theme_applied)
            self._theme_editor_dlg.show()
        else:
            self._theme_editor_dlg.raise_()
            self._theme_editor_dlg.activateWindow()

    def _on_theme_applied(self):
        """Called when user clicks Apply in the editor — reload and repaint."""
        _theme_mod.load()          # reload saved JSON into _dark / _light dicts
        _build_styles()            # rebuild all Qt stylesheet strings
        self._apply_theme()        # repaint live UI
        if self._theme_editor_dlg:
            self._theme_editor_dlg.reload_from_theme()   # sync editor panels

    def _apply_theme(self):
        """Reapply all stylesheets after a theme change."""
        t = _T()
        self.setStyleSheet(APP_STYLE)

        # Palette + theme toggle buttons
        _icon_btn_ss = (
            f"QPushButton{{background:transparent;border:none;border-radius:15px;}}"
            f"QPushButton:hover{{background:{t['surface']};}}"
        )
        self._palette_btn.setStyleSheet(_icon_btn_ss)
        self._theme_btn.setIcon(_icon("moon-dark.png" if _DARK_THEME else "moon-light.png", 18))
        self._theme_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:15px;}}"
            f"QPushButton:hover{{background:{t['surface']};}}"
        )
        self._h1.setStyleSheet(f"color:{t['text']};")
        self._h2.setStyleSheet(f"color:{t['muted']};font-size:10px;margin-bottom:2px;")

        self._left_scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:transparent;}}"
            f"QScrollBar:vertical{{background:{t['bg']};width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:{t['surface2']};border-radius:3px;}}"
        )

        # Structural widgets
        for gb in self.findChildren(QGroupBox):
            gb.setStyleSheet(GROUP_STYLE)
        for sl in self.findChildren(QSlider):
            sl.setStyleSheet(SLIDER_STYLE)
        for cb in self.findChildren(QComboBox):
            cb.setStyleSheet(COMBO_STYLE)
        for ck in self.findChildren(QCheckBox):
            ck.setStyleSheet(f"color:{t['text']};font-size:11px;")
        for div in getattr(self, '_dividers', []):
            div.setStyleSheet(f"color:{t['border']};")

        # File rows
        for row in [self.video_row, self.osd_row, self.srt_row, self.out_row]:
            row._name_lbl.setStyleSheet(f"color:{t['subtext']}")
            row.path_lbl.setStyleSheet(PATH_FILLED if row.path else PATH_EMPTY)
            row.btn.setStyleSheet(BTN_SEC)
            row.clr.setStyleSheet(BTN_DANGER)

        # Preview panel
        self._preview_panel.setStyleSheet(
            f"background:{t['bg2']};border:1px solid {t['border']};border-radius:8px;")
        if self._preview_panel._pil_img is None:
            self._preview_panel._redraw_placeholder()

        # Buttons
        self.render_btn.setStyleSheet(BTN_PRIMARY)
        self.stop_btn.setStyleSheet(BTN_STOP)
        self.restart_btn.setStyleSheet(BTN_PLAY)
        self.play_btn.setStyleSheet(BTN_PLAY)

        # SpinBox
        self.mbps_spin.setStyleSheet(
            f"QSpinBox{{background:{t['surface']};color:{t['text']};"
            f"border:1px solid {t['border2']};border-radius:4px;padding:3px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;"
            f"background:{t['surface2']};border-radius:2px;}}"
        )

        # Progress bar — repaint with new theme colours
        self.prog.update()

        # Inline subtext labels (constructed with hardcoded colours at init time)
        for lbl, style in [
            (self.frame_info,  f"color:{t['muted']};font-size:10px;"),
            (self.size_hint,   f"color:{t['muted']};font-size:10px;"),
            (self.status,      f"color:{t['muted']};font-size:10px;"),
            (self.frame_lbl,   f"color:{t['text']};font-size:11px;font-weight:bold;"),
            (self.osd_warn,    f"color:{t['orange']};font-size:10px;"),
        ]:
            lbl.setStyleSheet(style)

        # font_lbl — preserve its success/error state colour if already set
        fl_ss = self.font_lbl.styleSheet()
        if "green" not in fl_ss and t['green'] not in fl_ss:
            # still in initial/error state — use orange (no font) or current red
            if "red" not in fl_ss and t['red'] not in fl_ss:
                self.font_lbl.setStyleSheet(f"color:{t['orange']};font-size:10px;")

        # hw_lbl — preserve detected GPU green, only reset if still pending/absent
        hw_text = self.hw_lbl.text()
        if hw_text.startswith("Detecting") or hw_text.startswith("No GPU") or hw_text == "":
            self.hw_lbl.setStyleSheet(f"color:{t['muted']};font-size:10px;")
        elif hw_text.startswith("✓"):
            self.hw_lbl.setStyleSheet(f"color:{t['green']};font-size:10px;")

        self._refresh_preview()

    def _on_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                            "Video (*.mp4 *.mkv *.avi *.mov)")
        if not p: return
        self.video_row.set_path(p)
        self._load_video(p)
        self._auto_detect(p)

    def _manual_osd(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select OSD File", "", "OSD (*.osd)")
        if p: self.osd_row.set_path(p); self._load_osd(p)

    def _manual_srt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select SRT File", "", "SRT (*.srt)")
        if p: self.srt_row.set_path(p); self._load_srt(p)

    def _load_video(self, path):
        if self._playing: self._play_pause()   # stop any running playback
        if not self.out_row.path:
            self.out_row.set_path(self._make_output_path(path))
        self._prefetch_stop = True           # signal any running prefetch to abort
        self.cached_frames.clear()
        self.cache_bar.finish()
        self._st("Reading video info…")
        self._vi = VideoInfoWorker(path)
        self._vi.result.connect(self._got_vid_info)
        self._vi.start()
        self._extract_at_pct(0)

    def _got_vid_info(self, info):
        self.vid_card.clear()
        if "error" not in info:
            self.video_fps = info.get("fps", 60.0)
            self.video_dur = info.get("duration", 0.0)
            size_mb = info.get("size_mb", 0) or 0
            # Source bitrate in Mbit/s — used to calibrate output size estimate
            self.source_mbps = (size_mb * 8 / max(self.video_dur, 1)) if self.video_dur > 0 else 0
            self.vid_card.add_row("Res",  f"{info.get('width')}×{info.get('height')}")
            self.vid_card.add_row("FPS",  str(info.get("fps", "?")))
            _dm, _ds = divmod(int(self.video_dur), 60)
            self.vid_card.add_row("Dur",  f"{_dm}:{_ds:02d}")
            self.vid_card.add_row("Size", f"{size_mb} MB")
            self._update_size_hint()
            # Kick off background prefetch now that we know the duration
            QTimer.singleShot(400, self._start_prefetch)
        # Trigger preview of frame 0 once we know the duration
        self._refresh_preview()
        self._st("Ready")

    def _start_prefetch(self):
        """Begin background frame extraction across ~20 evenly-spaced positions."""
        if not self.video_row.path or self.video_dur <= 0 or not find_ffmpeg():
            return
        # 20 evenly-spaced positions: 0, 5, 10, … 95, 100 %
        positions = list(range(0, 101, 5))
        # Skip positions already cached
        to_fetch = [p for p in positions if p not in self.cached_frames]
        if not to_fetch:
            return
        self._prefetch_stop = False
        self.cache_bar.start(len(to_fetch))
        threading.Thread(
            target=self._prefetch_frames,
            args=(to_fetch,),
            daemon=True
        ).start()

    def _prefetch_frames(self, positions):
        """Worker thread: extract one frame per position, update CacheBar."""
        ffmpeg = find_ffmpeg()
        if not ffmpeg: return
        done = 0
        for pct in positions:
            if self._prefetch_stop:
                break
            if pct in self.cached_frames:
                done += 1
                QTimer.singleShot(0, lambda d=done: self.cache_bar.update_count(d))
                continue
            t = self.video_dur * pct / 100.0
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            try:
                proc = _hidden_popen(
                    [ffmpeg, "-y", "-ss", str(t), "-i", self.video_row.path,
                     "-vframes", "1", "-q:v", "3", tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                proc.wait(timeout=15)
                if (not self._prefetch_stop and proc.returncode == 0
                        and os.path.exists(tmp.name) and PIL_OK):
                    img = PILImage.open(tmp.name).copy().convert("RGBA")
                    self.cached_frames[pct] = img
                    if self.video_frame is None:
                        self.video_frame = img
                    done += 1
                    _d = done
                    QTimer.singleShot(0, lambda d=_d: self.cache_bar.update_count(d))
            except Exception:
                pass
            finally:
                try: os.unlink(tmp.name)
                except: pass
        QTimer.singleShot(0, self.cache_bar.finish)

    def _load_osd(self, path):
        try:
            self.osd_data = parse_osd(path)
            s = self.osd_data.stats
            self.osd_card.clear()
            self.osd_card.add_row("FC",   s.fc_type or "Unknown")
            if s.total_arm_time: self.osd_card.add_row("Arm",  s.total_arm_time)
            if s.min_battery_v:  self.osd_card.add_row("Batt", f"{s.min_battery_v:.2f}V")
            if s.max_current_a:  self.osd_card.add_row("Curr", f"{s.max_current_a:.1f}A")
            if s.used_mah:       self.osd_card.add_row("mAh",  str(s.used_mah))
            self.osd_card.add_row("Dur",  f"{self.osd_data.duration_ms/1000:.1f}s")
            self.osd_card.add_row("Pkts", str(self.osd_data.frame_count))
            self.osd_row.set_path(path)
            # Auto-select firmware from OSD fc_type
            fc = (self.osd_data.stats.fc_type or "").strip()
            fw_map = {"Betaflight": "Betaflight", "INAV": "INAV",
                      "ArduPilot": "ArduPilot", "ARDU": "ArduPilot"}
            fw_match = fw_map.get(fc)
            if fw_match:
                idx = self.fw_combo.findText(fw_match)
                if idx >= 0:
                    self.fw_combo.setCurrentIndex(idx)   # triggers _on_fw_changed
            self._st(f"✓ OSD: {self.osd_data.frame_count} frames  [{fc or 'Unknown FC'}]")
            self._refresh_preview()
        except Exception as e:
            self._st(f"✗ OSD: {e}")

    def _try_load_p1_osd(self, video_path):
        """Silently try to extract embedded P1 OSD from an MP4. No-op if not a P1 file."""
        try:
            if not detect_p1(video_path):
                return
            self._st("Detected BetaFPV P1 — extracting embedded OSD…")
            p1_data = parse_p1_osd(video_path)
            if not p1_data or not p1_data.frames:
                self._st("P1 OSD: no frames found")
                return
            self.osd_data = p1_to_osd_file(p1_data)
            s = self.osd_data.stats
            self.osd_card.clear()
            self.osd_card.add_row("FC",   s.fc_type or "BetaFPV P1")
            if s.total_arm_time: self.osd_card.add_row("Arm",  s.total_arm_time)
            if s.min_battery_v:  self.osd_card.add_row("Batt", f"{s.min_battery_v:.2f}V")
            if s.max_current_a:  self.osd_card.add_row("Curr", f"{s.max_current_a:.1f}A")
            if s.used_mah:       self.osd_card.add_row("mAh",  str(s.used_mah))
            self.osd_card.add_row("Dur",  f"{self.osd_data.duration_ms/1000:.1f}s")
            self.osd_card.add_row("Pkts", str(self.osd_data.frame_count))
            self.osd_row.set_path("(embedded in video)")
            # P1 runs Betaflight over Walksnail/DJI goggles → always uses BTFL_DJI font
            self._auto_select_font("Betaflight", "BTFL_DJI")
            self._st(f"✓ P1 OSD: {self.osd_data.frame_count} frames embedded")
            self._refresh_preview()
        except Exception as e:
            self._st(f"✗ P1 OSD: {e}")

    def _auto_select_font(self, firmware: str, preferred_folder: str):
        """Select firmware in the fw_combo and pick a specific font folder by name."""
        # Switch firmware tab (Betaflight / INAV / etc.)
        idx = self.fw_combo.findText(firmware)
        if idx >= 0 and self.fw_combo.currentIndex() != idx:
            self.fw_combo.setCurrentIndex(idx)   # triggers _on_fw_changed → rebuilds style_combo
        # Now find the preferred folder in the style combo
        for i in range(self.style_combo.count()):
            if self.style_combo.itemData(i) == preferred_folder:
                self.style_combo.setCurrentIndex(i)
                return
        # Fallback: already set by _on_fw_changed

    def _load_srt(self, path):
        try:
            self.srt_data = parse_srt(path)
            self.srt_card.clear()
            self.srt_card.add_row("Entries", str(len(self.srt_data.entries)))
            self.srt_card.add_row("Dur", f"{self.srt_data.duration_ms/1000:.1f}s")
            if self.srt_data.entries:
                t = self.srt_data.entries[0].telemetry
                if t.radio1_dbm is not None: self.srt_card.add_row("R1", f"{t.radio1_dbm:+d}dBm")
                if t.link_mbps:              self.srt_card.add_row("Mbps", str(t.link_mbps))
            self.srt_row.set_path(path)
            self._st(f"✓ SRT: {len(self.srt_data.entries)} entries")
            self._refresh_preview()
        except Exception as e:
            self._st(f"✗ SRT: {e}")

    # ── Preview ───────────────────────────────────────────────────────────────

    def _video_time_ms(self, pct):
        """Map slider 0-100% → absolute video timestamp in ms (full video duration)."""
        return int(self.video_dur * pct / 100.0 * 1000)

    def _on_frame_sl(self, pct):
        # Update text labels immediately for responsiveness
        self.frame_lbl.setText(f"{pct}%")
        t_ms = self._video_time_ms(pct)
        osd_info = "—"
        if self.osd_data:
            fr = self.osd_data.frame_at_time(t_ms)
            if fr: osd_info = f"pkt {fr.index}"
        t_s = t_ms / 1000
        m, s = divmod(int(t_s), 60)
        self.frame_info.setText(f"t = {m}:{s:02d}  |  OSD {osd_info}")

        if pct in self.cached_frames:
            # Exact frame cached — show it
            self._show_pct(pct)
            return

        if self._playing:
            # During playback: re-composite the nearest cached frame rather than
            # spawning ffmpeg (which is too slow for smooth playback)
            nearest = min(self.cached_frames.keys(),
                          key=lambda k: abs(k - pct)) if self.cached_frames else None
            if nearest is not None:
                self._show_pct(nearest)
            return

        # Stationary scrub: debounce then extract via ffmpeg
        self._pending_pct = pct
        self._scrub_timer.start()

    def _do_scrub(self):
        """Called ~80ms after the slider stops — extract the frame via ffmpeg."""
        pct = self._pending_pct
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def _extract_at_pct(self, pct):
        if not self.video_row.path or not find_ffmpeg(): return
        ffmpeg = find_ffmpeg()
        # Seek to the absolute timestamp (slider maps to full video)
        t = self.video_dur * pct / 100.0 if self.video_dur > 0 else 0.0
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        # Kill any in-flight extraction so we don't pile up ffmpeg processes
        if self._extract_proc and self._extract_proc.poll() is None:
            try: self._extract_proc.kill()
            except Exception: pass
        def _run():
            proc = None
            try:
                proc = _hidden_popen(
                    [ffmpeg, "-y", "-ss", str(t), "-i", self.video_row.path,
                     "-vframes", "1", "-q:v", "2", tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._extract_proc = proc
                proc.wait(timeout=20)
                if proc.returncode == 0 and os.path.exists(tmp.name) and PIL_OK:
                    img = PILImage.open(tmp.name).copy().convert("RGBA")
                    self.cached_frames[pct] = img
                    if self.video_frame is None:
                        self.video_frame = img
                    def _on_frame_ready(p=pct):
                        self._show_pct(p)
                    QTimer.singleShot(0, _on_frame_ready)
            except Exception:
                pass
            finally:
                try: os.unlink(tmp.name)
                except: pass
        threading.Thread(target=_run, daemon=True).start()

    def _show_pct(self, pct):
        img = self.cached_frames.get(pct)
        if img: self.preview.show_frame(self._composite(img, pct))

    def _refresh_preview(self):
        pct = self.frame_sl.value()
        img = self.cached_frames.get(pct) or self.video_frame
        if img: self.preview.show_frame(self._composite(img, pct))

    def _composite(self, img, pct):
        t_ms = self._video_time_ms(pct)
        osd_frame = self.osd_data.frame_at_time(t_ms) if self.osd_data else None
        srt_text = ""
        if self.srt_data and self.srt_bar_check.isChecked():
            td = self.srt_data.get_data_at_time(t_ms)
            if td: srt_text = td.status_line()
        cfg = OsdRenderConfig(
            offset_x     = self.sl_x.value(),
            offset_y     = self.sl_y.value(),
            scale        = self.sl_scale.value() / 100.0,
            show_srt_bar = self.srt_bar_check.isChecked(),
            srt_text     = srt_text,
            srt_opacity  = self.srt_opacity_sl.value() / 100.0,
        )
        if self.font_obj and PIL_OK:
            return render_osd_frame(img, osd_frame, self.font_obj, cfg)
        return render_fallback(img, osd_frame, cfg)

    def _reset_pos(self):
        self.sl_x.setValue(0)
        self.sl_y.setValue(0)
        self.sl_scale.setValue(100)
        self._refresh_preview()

    def _queue_preview(self):
        """Debounced preview refresh — fires 60ms after sliders stop moving."""
        self._preview_timer.start()

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_toggle(self):
        if not self.video_row.path or self.video_dur <= 0:
            return
        if self._playing:
            self._play_pause()
        else:
            self._play_start()

    def _play_start(self):
        self._playing = True
        self.play_btn.setIcon(_icon("pause.png", 22))
        self._play_timer.start()

    def _play_pause(self):
        self._playing = False
        self.play_btn.setIcon(_icon("play.png", 22))
        self._play_timer.stop()

    def _play_restart(self):
        self._play_pause()
        self.frame_sl.setValue(0)
        self._refresh_preview()

    def _play_tick(self):
        """Advance the preview slider by one timer tick (100ms worth of video)."""
        if self.video_dur <= 0:
            self._play_pause()
            return
        current = self.frame_sl.value()
        # Each tick = 100ms of video → slider step = 100ms / duration * 100 (pct)
        step = max(1, round(0.1 / self.video_dur * 100))
        nxt  = current + step
        if nxt >= 100:
            self.frame_sl.setValue(100)
            self._play_pause()   # reached end — stop
        else:
            self.frame_sl.setValue(nxt)
            # _on_frame_sl fires automatically via valueChanged

    # ── Trim ─────────────────────────────────────────────────────────────────

    def _fmt_trim_time(self, pct):
        t = pct * self.video_dur if self.video_dur > 0 else 0
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    def _on_trim_changed(self, in_pct, out_pct):
        self.trim_in_lbl.setText(f"In: {self._fmt_trim_time(in_pct)}")
        self.trim_out_lbl.setText(f"Out: {self._fmt_trim_time(out_pct)}")
        self._refresh_preview()

    @staticmethod
    def _clean_stem(stem: str) -> str:
        """Strip any existing _osd or _osd_<timestamps> suffix from a stem."""
        import re
        # Remove _osd_NNNN-NNNN timestamp suffix variants
        stem = re.sub(r'_osd_\d+[-_]\d+$', '', stem)
        # Remove bare _osd suffix
        stem = re.sub(r'_osd$', '', stem)
        return stem

    def _make_output_path(self, video_path: str, trim_start_s: float = 0.0,
                          trim_end_s: float = 0.0) -> str:
        """
        Build output path: <dir>/<clean_stem>_osd[_MMSS-MMSS].mp4
        Always strips existing _osd/_osd_* from stem first.
        Adds timestamp suffix only when trim is meaningfully set.
        """
        p    = Path(video_path)
        stem = self._clean_stem(p.stem)
        dur  = self.video_dur if self.video_dur > 0 else 0.0

        # Use full video end as default
        t_end = trim_end_s if trim_end_s > 0.01 else dur

        trimmed = (trim_start_s > 0.01) or (t_end < dur - 0.5)
        if trimmed:
            def _fmt(s):
                m, sec = divmod(int(s), 60)
                return f"{m:02d}{sec:02d}"
            ts = f"_{_fmt(trim_start_s)}-{_fmt(t_end)}"
        else:
            ts = ""

        out_name = f"{stem}_osd{ts}.mp4"
        return str(p.parent / out_name)

    def _trim_reset(self):
        self.trim_sel.reset()

    def _set_trim_in(self):
        """Set In point to current frame slider position."""
        pct = self.frame_sl.value() / 100.0
        self.trim_sel.set_in(pct)

    def _set_trim_out(self):
        """Set Out point to current frame slider position."""
        pct = self.frame_sl.value() / 100.0
        self.trim_sel.set_out(pct)

    # ── Render ────────────────────────────────────────────────────────────────

    def _refresh_ffmpeg_status(self):
        ffp = find_ffmpeg()
        if ffp:
            self.ffmpeg_lbl.setText("✓ FFmpeg found")
            self.ffmpeg_lbl.setStyleSheet(f"color:{_T()['green']};font-size:10px;")
            self.ffmpeg_lbl.setToolTip(ffp)
            if hasattr(self, "ffmpeg_install_btn"):
                self.ffmpeg_install_btn.setVisible(False)
        else:
            self.ffmpeg_lbl.setText("⚠ FFmpeg not found")
            self.ffmpeg_lbl.setStyleSheet(f"color:{_T()['red']};font-size:10px;")
            self.ffmpeg_lbl.setToolTip("")

    def _install_ffmpeg(self):
        import platform
        if platform.system() != "Windows":
            QMessageBox.information(self, "Install FFmpeg",
                "Install FFmpeg with your package manager:\n\n"
                "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
                "  Fedora:         sudo dnf install ffmpeg\n"
                "  Arch:           sudo pacman -S ffmpeg\n"
                "  macOS:          brew install ffmpeg\n\n"
                "Then restart the app.")
            return
        reply = QMessageBox.question(
            self, "Install FFmpeg",
            "This will install FFmpeg via winget.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._st("Installing FFmpeg via winget\u2026")
        self.ffmpeg_install_btn.setEnabled(False)
        def _do_install():
            try:
                result = subprocess.run(
                    ["winget", "install", "--id", "Gyan.FFmpeg",
                     "--source", "winget",
                     "--accept-package-agreements",
                     "--accept-source-agreements"],
                    capture_output=True, text=True, timeout=300)
                success = result.returncode == 0
                err_msg = (result.stdout + result.stderr)[-500:]
            except FileNotFoundError:
                success = False
                err_msg = "winget not found. Install FFmpeg manually from https://www.gyan.dev/ffmpeg/builds/"
            except Exception as e:
                success = False
                err_msg = str(e)
            def _done():
                self.ffmpeg_install_btn.setEnabled(True)
                if success:
                    try:
                        import winreg
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
                            path_val, _ = winreg.QueryValueEx(k, "Path")
                        os.environ["PATH"] = path_val + ";" + os.environ.get("PATH", "")
                    except Exception:
                        pass
                    self._refresh_ffmpeg_status()
                    if find_ffmpeg():
                        self._st("✓ FFmpeg installed successfully")
                    else:
                        self._st("FFmpeg installed — restart app to detect it")
                else:
                    self._st("FFmpeg install failed")
                    QMessageBox.critical(self, "Install Failed",
                        "Could not install FFmpeg automatically.\n\n"
                        + err_msg
                        + "\n\nDownload manually:\nhttps://www.gyan.dev/ffmpeg/builds/")
            QTimer.singleShot(0, _done)
        threading.Thread(target=_do_install, daemon=True).start()

    def _render(self):
        if not self.video_row.path:
            QMessageBox.warning(self, "Missing", "Select a video file."); return
        if not self.out_row.path:
            QMessageBox.warning(self, "Missing", "Choose output location."); return
        if not find_ffmpeg():
            QMessageBox.critical(self, "FFmpeg Missing",
                "FFmpeg not found.\n\nRun 'OnyxFPV OSD Tool.bat' to install it automatically,\n"
                "or install manually from https://www.gyan.dev/ffmpeg/builds/")
            return

        codec_map = {"H.264 (libx264)": "libx264", "H.265 (libx265)": "libx265"}
        codec = codec_map.get(self.codec_cb.currentText(), "libx264")

        font_folder = None
        if self.font_obj is not None:
            fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
            candidate = os.path.join(fonts_dir, self.font_obj.name)
            if os.path.isdir(candidate):
                font_folder = candidate

        # Quality: CRF mode or bitrate mode
        if self.mode_mbps_btn.isChecked():
            crf_val  = 23  # not used directly
            # Pass bitrate via CRF field — video_processor will use -b:v
            bitrate_mbps = self.mbps_spin.value()
        else:
            crf_val = self.crf_sl.value()
            bitrate_mbps = None

        # Recompute output filename with final trim timestamps
        trim_s = self.trim_sel.in_pct  * self.video_dur
        trim_e = self.trim_sel.out_pct * self.video_dur
        self.out_row.set_path(
            self._make_output_path(self.video_row.path, trim_s, trim_e)
        )

        # ── Overwrite / rename dialog ─────────────────────────────────────────
        out_path = self.out_row.path
        if out_path and os.path.exists(out_path):
            dlg = QDialog(self)
            dlg.setWindowTitle("File Already Exists")
            dlg.setMinimumWidth(420)
            vl = QVBoxLayout(dlg)
            vl.setSpacing(10)
            vl.setContentsMargins(18, 18, 18, 14)

            warn_lbl = QLabel(f"⚠  <b>{os.path.basename(out_path)}</b> already exists in this folder.")
            warn_lbl.setWordWrap(True)
            warn_lbl.setTextFormat(Qt.TextFormat.RichText)
            warn_lbl.setStyleSheet(f"color:{_T()['text']};font-size:12px;")
            vl.addWidget(warn_lbl)

            name_lbl = QLabel("Save as:")
            name_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
            vl.addWidget(name_lbl)

            name_edit = QLineEdit(os.path.basename(out_path))
            name_edit.setStyleSheet(
                f"background:{_T()['bg2']};color:{_T()['text']};"
                f"border:1px solid {_T()['border2']};border-radius:4px;padding:4px 8px;"
            )
            name_edit.selectAll()
            vl.addWidget(name_edit)

            btn_row2 = QHBoxLayout()
            btn_row2.setSpacing(6)
            overwrite_btn = QPushButton("Overwrite")
            overwrite_btn.setStyleSheet(BTN_DANGER)
            overwrite_btn.setFixedHeight(32)
            save_as_btn = QPushButton("Save with this name")
            save_as_btn.setStyleSheet(BTN_PRIMARY)
            save_as_btn.setFixedHeight(32)
            cancel_btn2 = QPushButton("Cancel")
            cancel_btn2.setStyleSheet(BTN_SEC)
            cancel_btn2.setFixedHeight(32)
            btn_row2.addWidget(cancel_btn2)
            btn_row2.addStretch()
            btn_row2.addWidget(overwrite_btn)
            btn_row2.addWidget(save_as_btn)
            vl.addLayout(btn_row2)

            _result = ["cancel"]
            def _ow():   _result[0] = "overwrite"; dlg.accept()
            def _sa():
                new_name = name_edit.text().strip()
                if not new_name: return
                if not new_name.lower().endswith(".mp4"):
                    new_name += ".mp4"
                new_path = os.path.join(os.path.dirname(out_path), new_name)
                if os.path.exists(new_path) and new_path != out_path:
                    name_edit.setStyleSheet(
                        f"background:{_T()['bg2']};color:{_T()['red']};"
                        f"border:1px solid {_T()['red']};border-radius:4px;padding:4px 8px;"
                    )
                    name_lbl.setText("Save as:  ⚠ that file also exists — pick a different name")
                    return
                _result[0] = new_path
                dlg.accept()
            def _cancel(): dlg.reject()
            overwrite_btn.clicked.connect(_ow)
            save_as_btn.clicked.connect(_sa)
            cancel_btn2.clicked.connect(_cancel)

            dlg.setStyleSheet(f"background:{_T()['bg']};")
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return   # user cancelled
            if _result[0] == "cancel":
                return
            elif _result[0] != "overwrite":
                # user chose a new name
                self.out_row.set_path(_result[0])

        # Upscale target from dropdown
        _upscale_map = {0: "", 1: "1440p", 2: "2.7k", 3: "4k"}
        upscale_target = _upscale_map.get(self.upscale_combo.currentIndex(), "")

        cfg = ProcessingConfig(
            input_video   = self.video_row.path,
            output_video  = self.out_row.path,
            osd_file      = self.osd_row.path or None,
            osd_data      = self.osd_data,        # pass in-memory OSD (covers P1 embedded)
            srt_file      = self.srt_row.path or None,
            codec         = codec,
            crf           = crf_val,
            bitrate_mbps  = bitrate_mbps,
            font_folder   = font_folder,
            prefer_hd     = self.hd_check.isChecked(),
            scale         = self.sl_scale.value() / 100.0,
            offset_x      = self.sl_x.value(),
            offset_y      = self.sl_y.value(),
            show_srt_bar  = self.srt_bar_check.isChecked(),
            srt_opacity   = self.srt_opacity_sl.value() / 100.0,
            use_hw        = self.hw_check.isChecked(),
            trim_start    = self.trim_sel.in_pct  * self.video_dur,
            trim_end      = self.trim_sel.out_pct * self.video_dur,
            upscale_target = upscale_target,
        )

        self.render_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.osd_warn.setVisible(False)   # hide previous warning
        self.prog.setActive(True)
        self.prog.setValue(0)

        # Pre-flight OSD visibility check — warn if no OSD frames in trim window
        if self.osd_data and self.video_dur > 0:
            t_start_ms = int(self.trim_sel.in_pct  * self.video_dur * 1000)
            t_end_ms   = int(self.trim_sel.out_pct * self.video_dur * 1000)
            in_window = [fr for fr in self.osd_data.frames
                         if t_start_ms <= fr.time_ms <= t_end_ms + 500]
            if not in_window:
                self.osd_warn.setVisible(True)

        self.worker = ProcessWorker(cfg)
        self.worker.progress.connect(
            lambda p, m: (self.prog.setValue(p), self._st(m)),
            Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._done)
        self.worker.start()

    def _stop_render(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._st("⏹ Stopped")
            self.prog.setActive(False)
            self.render_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def _done(self, ok, msg):
        self.render_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if ok:
            self.prog.setValue(100)
            self.prog.setActive(False)
            out_path = self.out_row.path
            out_dir  = os.path.dirname(os.path.abspath(out_path))
            self._st(f"✓ Saved: {out_path}")
            # Show warning in label if OSD was trimmed
            if msg:
                self.osd_warn.setText(f"⚠ {msg}")
                self.osd_warn.setVisible(True)

            dlg = QMessageBox(self)
            dlg.setWindowTitle("Done!")
            dlg.setText(f"Saved to:\n{out_path}")
            dlg.setIcon(QMessageBox.Icon.Information)
            open_btn = dlg.addButton("  Open Folder", QMessageBox.ButtonRole.ActionRole)
            dlg.addButton(QMessageBox.StandardButton.Ok)
            dlg.exec()
            if dlg.clickedButton() == open_btn:
                self._open_folder(out_dir)
        else:
            self.prog.setActive(False)
            self._st(f"✗ {msg}")
            QMessageBox.critical(self, "Error", f"Render failed:\n{msg}")

    def _open_folder(self, folder):
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            _hidden_popen(["open", folder])
        else:
            _hidden_popen(["xdg-open", folder])

    def _st(self, msg):
        self.status.setText(msg)


# ─── Styles (Catppuccin Mocha) ────────────────────────────────────────────────

# (styles are generated dynamically by _build_styles() above)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("OnyxFPV OSD Tool")
    app.setOrganizationName("OnyxFPV")
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    splash = SplashScreen()
    splash.show()

    def step(v, msg):
        splash.set_progress(v, msg)
        app.processEvents()

    step(0.15, "Loading OSD parser…")
    step(0.30, "Loading font engine…")
    step(0.48, "Loading video pipeline…")
    step(0.64, "Building interface…")
    win = MainWindow()
    step(0.92, "Checking FFmpeg…")
    app.processEvents()

    splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
VueOSD — Digital FPV OSD Tool
Parse and overlay MSP-OSD data onto FPV DVR video footage.
"""

import sys, os, math, threading, subprocess, json, time

# ── Windows: set AppUserModelID so taskbar shows our icon, not Python's ───────
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VueOSD.OSDTool.1")
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QUrl
from PyQt6.QtGui import (QFont, QPixmap, QImage, QPainter, QColor, QPen, QIcon,
                         QDesktopServices, QStandardItem, QStandardItemModel)


from srt_parser    import parse_srt, SrtFile, SRT_FIELDS
from osd_parser    import parse_osd, OsdFile, GRID_COLS, GRID_ROWS
from p1_osd_parser import detect_p1, parse_p1_osd, p1_to_osd_file
from font_loader   import (fonts_by_firmware, load_font, load_font_from_file,
                           OsdFont, FIRMWARE_PREFIXES)
from osd_renderer  import OsdRenderConfig, render_osd_frame, render_fallback
from video_processor import (ProcessingConfig, ColorTransConfig, process_video,
                             get_video_info, find_ffmpeg, detect_hw_encoder,
                             detect_libplacebo)
from splash_screen   import SplashScreen


try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

from subprocess_utils import _hidden_popen, _hidden_run


# ─── Theme system ─────────────────────────────────────────────────────────────

import theme as _theme_mod   # single source of truth for all colours

_DARK_THEME = True   # module-level flag; toggled by the theme button

# ─── Version & UI scale ───────────────────────────────────────────────────────

VERSION = "1.5"

_UI_SCALE = 1.0
_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

def _fs(n: int) -> int:
    """Scale a font size by the active UI scale factor."""
    return max(6, int(n * _UI_SCALE))

_OSD_OFFSET_MS = 0  # persisted OSD sync offset (ms)

_COLORTRANS_SETTINGS: dict = {}  # loaded from settings.json "colortrans" key

def _load_settings():
    global _UI_SCALE, _OSD_OFFSET_MS, _COLORTRANS_SETTINGS
    try:
        with open(_SETTINGS_FILE) as f:
            data = json.load(f)
        _UI_SCALE      = float(data.get("ui_scale", 1.0))
        _OSD_OFFSET_MS = int(data.get("osd_offset_ms", 0))
        _COLORTRANS_SETTINGS = data.get("colortrans", {})
    except Exception:
        pass

def _save_settings(**extra):
    try:
        data: dict = {}
        try:
            with open(_SETTINGS_FILE) as f:
                data = json.load(f)
        except Exception:
            pass
        data["ui_scale"]      = _UI_SCALE
        data["osd_offset_ms"] = _OSD_OFFSET_MS
        data.update(extra)
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

_load_settings()

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
        f"font-family:'Segoe UI',Arial,sans-serif;font-size:{_fs(12)}px;}}"
        f"QLabel{{color:{t['text']};}}"
        f"QCheckBox{{color:{t['text']};}}"
        f"QScrollArea{{border:none;}}"
    )
    # Light: group titles use subtext (softer), dark: keep accent (blue)
    title_col = t['subtext'] if is_light else t['accent']
    GROUP_STYLE = (
        f"QGroupBox{{border:1px solid {t['border']};border-radius:8px;margin-top:4px;"
        f"padding:4px;font-weight:bold;color:{title_col};font-size:{_fs(11)}px;}}"
        f"QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}"
    )
    PATH_EMPTY  = (f"background:{t['bg2']};color:{t['muted']};border:1px solid {t['border']};"
                   f"border-radius:4px;padding:3px 8px;font-size:{_fs(11)}px;")
    PATH_FILLED = (f"background:{t['bg2']};color:{t['text']};border:1px solid {t['border2']};"
                   f"border-radius:4px;padding:3px 8px;font-size:{_fs(11)}px;")

    # Light theme: buttons use a thin border so they read against the near-white bg
    # without being dark slabs. Dark theme: no border needed (surfaces contrast enough).
    btn_border     = f"1px solid {t['border2']}" if is_light else "none"
    btn_border_hov = f"1px solid {t['border2']}" if is_light else "none"

    BTN_SEC  = (f"QPushButton{{background:{t['surface']};color:{t['text']};"
                f"border:{btn_border};border-radius:6px;"
                f"padding:3px 10px;font-size:{_fs(11)}px;}}"
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
                f"border:{btn_border};border-radius:8px;font-size:{_fs(15)}px;}}"
                f"QPushButton:hover{{background:{t['surface2']};border:{btn_border_hov};}}"
                f"QPushButton:pressed{{background:{t['accent']};color:#ffffff;}}"
                f"QPushButton:disabled{{background:{t['bg']};color:{t['muted']};"
                f"border:1px solid {t['border']};}}")
    BTN_STOP  = (f"QPushButton{{background:{t['red']};color:#ffffff;"
                 f"border:none;border-radius:8px;font-size:{_fs(16)}px;font-weight:bold;}}"
                 f"QPushButton:hover{{background:{t['red']}dd;}}"
                 f"QPushButton:pressed{{background:{t['red']};}}"
                 f"QPushButton:disabled{{background:{t['surface']};color:{t['muted']};"
                 f"border:1px solid {t['border']};}}")
    BTN_DANGER = (f"QPushButton{{background:{t['surface']};color:{t['red']};"
                  f"border:{btn_border};border-radius:6px;font-weight:bold;font-size:{_fs(11)}px;}}"
                  f"QPushButton:hover{{background:{t['red']};color:#ffffff;border:none;}}")
    COMBO_STYLE = (f"QComboBox{{background:{t['surface']};color:{t['text']};"
                   f"border:1px solid {t['border2']};"
                   f"border-radius:4px;padding:3px 8px;font-size:{_fs(11)}px;}}"
                   f"QComboBox::drop-down{{border:none;padding-right:6px;}}"
                   f"QComboBox QAbstractItemView{{background:{t['bg']};color:{t['text']};"
                   f"selection-background-color:{t['surface2']};border:1px solid {t['border2']};}}")
    SLIDER_STYLE = (f"QSlider::groove:horizontal{{background:{t['border']};height:4px;border-radius:2px;}}"
                    f"QSlider::handle:horizontal{{background:{t['accent']};width:14px;height:14px;"
                    f"margin:-5px 0;border-radius:7px;}}"
                    f"QSlider::sub-page:horizontal{{background:{t['accent']};border-radius:2px;}}")
    PROG_STYLE  = (f"QProgressBar{{background:{t['surface']};border-radius:4px;text-align:center;"
                   f"color:{t['text']};font-size:{_fs(11)}px;}}"
                   f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                   f"stop:0 {t['accent']},stop:1 {t['accent2']});border-radius:4px;}}")

# Initialise with dark theme
APP_STYLE = GROUP_STYLE = PATH_EMPTY = PATH_FILLED = ""
BTN_SEC = BTN_PRIMARY = BTN_PLAY = BTN_STOP = BTN_DANGER = ""
COMBO_STYLE = SLIDER_STYLE = PROG_STYLE = ""
_build_styles()

# Frame-slider resolution: 10 000 steps gives ~0.18 s precision on a 30-min clip.
_SL_MAX = 10_000


# ─── Icon helpers ─────────────────────────────────────────────────────────────

def _icons_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")

def _icon(name: str, size: int = 22, color: str = None) -> QIcon:
    """Load an icon tinted to the active theme's icon colour (or an explicit hex colour)."""
    import numpy as np
    path = os.path.join(_icons_dir(), name)
    if not os.path.exists(path):
        return QIcon()
    col = QColor(color if color else _T()["icon"])
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
    def __init__(self, label, placeholder, filter_str, save_mode=False, icon=None,
                 icon_name="", parent=None):
        super().__init__(parent)
        self.filter_str = filter_str
        self.save_mode  = save_mode
        self._path      = ""
        self._icon_name = icon_name   # stored for theme retinting
        self._icon_lbl: Optional[QLabel] = None

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
            self._icon_lbl = icon_lbl
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

    def retint(self):
        """Re-tint the row icon to the current theme's icon colour."""
        if self._icon_lbl and self._icon_name:
            self._icon_lbl.setPixmap(_icon(self._icon_name, 16).pixmap(16, 16))

    @property
    def path(self):
        return self._path


class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(72)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._lbl = QLabel("Drop  .mp4 · .osd · .srt  here")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet(f"color:{_T()['muted']};font-size:11px;")
        lay.addWidget(self._lbl)
        self._update_idle()

    def _update_idle(self):
        self.setStyleSheet(
            f"DropZone{{background:{_T()['bg2']};border:2px dashed {_T()['border2']};"
            f"border-radius:8px;}}"
        )
        self._lbl.setStyleSheet(f"color:{_T()['muted']};font-size:11px;")

    def _update_hover(self):
        self.setStyleSheet(
            f"DropZone{{background:{_T()['surface']};border:2px solid {_T()['accent']};"
            f"border-radius:8px;}}"
        )
        self._lbl.setStyleSheet(f"color:{_T()['accent']};font-size:11px;font-weight:bold;")

    def refresh_theme(self):
        self._update_idle()

    _VALID_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.osd', '.srt'}

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if any(os.path.splitext(p)[1].lower() in self._VALID_EXTS for p in paths):
                event.acceptProposedAction()
                self._update_hover()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._update_idle()

    def dropEvent(self, event):
        self._update_idle()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.splitext(path)[1].lower() in self._VALID_EXTS:
                self.file_dropped.emit(path)
                break
        event.acceptProposedAction()


class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, label, lo, hi, val, suffix="", parent=None):
        super().__init__(parent)
        self._s = suffix
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(58)
        self._lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")

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
        lay.addWidget(self._lbl)
        lay.addWidget(self.sl)
        lay.addWidget(self.vl)

    def value(self): return self.sl.value()
    def setValue(self, v): self.sl.setValue(v)

    def refresh_theme(self):
        self._lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.vl.setStyleSheet(f"color:{_T()['text']};font-size:11px;font-weight:bold;")


class CheckableComboBox(QComboBox):
    """Multi-select combo box with checkable items."""
    selectionChanged = pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.view().viewport().installEventFilter(self)
        self._keys: list[str] = []

    def add_items(self, items: list[tuple[str, str]]):
        """Add (key, label) pairs as checkable items, all checked by default."""
        self._keys.clear()
        self._model.clear()
        for key, label in items:
            item = QStandardItem(label)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
            item.setData(key, Qt.ItemDataRole.UserRole)
            self._model.appendRow(item)
            self._keys.append(key)

    def checked_keys(self) -> set[str]:
        """Return set of keys for currently checked items."""
        result = set()
        for i in range(self._model.rowCount()):
            item = self._model.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.add(item.data(Qt.ItemDataRole.UserRole))
        return result

    def eventFilter(self, obj, event):
        """Keep popup open when clicking items."""
        if obj is self.view().viewport() and event.type() == event.Type.MouseButtonRelease:
            idx = self.view().indexAt(event.pos())
            if idx.isValid():
                item = self._model.itemFromIndex(idx)
                new_state = (Qt.CheckState.Unchecked
                             if item.checkState() == Qt.CheckState.Checked
                             else Qt.CheckState.Checked)
                item.setCheckState(new_state)
                self.selectionChanged.emit(self.checked_keys())
                return True
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        """Show summary text instead of single selected item."""
        from PyQt6.QtWidgets import QStylePainter, QStyleOptionComboBox, QStyle
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        total = self._model.rowCount()
        checked = len(self.checked_keys())
        if checked == 0:
            opt.currentText = "No fields"
        elif checked == total:
            opt.currentText = "All fields"
        else:
            opt.currentText = f"{checked} / {total} fields"
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


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

    def refresh_theme(self):
        for row in range(self._r):
            ki = self._g.itemAtPosition(row, 0)
            vi = self._g.itemAtPosition(row, 1)
            if ki and ki.widget():
                ki.widget().setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
            if vi and vi.widget():
                vi.widget().setStyleSheet(f"color:{_T()['text']};font-size:10px;font-weight:600;")


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


## CacheBar — removed (now in player.py Timeline cached-frame dots)


## RangeSelector — removed (now Timeline in player.py)


## PreviewPanel — removed (now VideoCanvas in player.py)


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
        self.worker      = None
        self._font_db:   dict = {}
        self.source_mbps: float = 0.0   # source video bitrate, set after loading
        self._dividers:    list = []   # legacy, kept for compat

        self.setWindowTitle(f"VueOSD v{VERSION} — Digital FPV OSD Tool")
        # App icon — resolved relative to this script so it works from any CWD
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(APP_STYLE)

        # ── Root splitter: left | centre | right (resizable via drag handles) ──
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(5)
        self.setCentralWidget(self._splitter)

        left_scroll = self._build_left_panel()
        centre = self._build_centre_panel()
        right = self._build_right_panel()

        self._splitter.addWidget(left_scroll)
        self._splitter.addWidget(centre)
        self._splitter.addWidget(right)

        # Initial sizes: left 300, centre stretches, right 300
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([300, 500, 300])

        self._update_splitter_style()

        # Collect buttons and labels for theme reapply
        # (theme uses findChildren — no explicit list needed)

        QTimer.singleShot(200, lambda: self._on_fw_changed("Betaflight"))
        QTimer.singleShot(300, self._cc_restore_from_settings)

    def _build_left_panel(self) -> "QScrollArea":
        """Build and return the left scrollable panel."""
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{background:#1e1e2e;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;}"
        )
        left_scroll.setMinimumWidth(250)
        left_scroll.setMaximumWidth(500)

        left_inner = QWidget()
        left_inner.setMinimumWidth(280)
        ll = QVBoxLayout(left_inner)
        ll.setContentsMargins(14, 16, 10, 16)
        ll.setSpacing(10)

        # Header: title + theme toggle button on the same row
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(8)

        h1 = QLabel("VueOSD")
        h1.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        h1.setStyleSheet(f"color:{_T()['text']};")
        self._h1 = h1

        h2 = QLabel("Digital FPV OSD Tool")
        h2.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        h2.setStyleSheet(f"color:{_T()['text']};")
        h2.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._h2 = h2

        ver = QLabel(f"v{VERSION}")
        ver.setFont(QFont("Segoe UI", _fs(8)))
        ver.setStyleSheet(f"color:{_T()['muted']};")
        ver.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._ver_lbl = ver

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
        hdr_row.addWidget(h2)
        hdr_row.addWidget(ver)
        hdr_row.addStretch()
        hdr_row.addWidget(self._palette_btn)
        hdr_row.addWidget(self._theme_btn)
        ll.addLayout(hdr_row)

        # ── UI Scale selector ─────────────────────────────────────────────────
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 2, 0, 0)
        scale_row.setSpacing(6)
        scale_lbl = QLabel("UI Scale")
        scale_lbl.setStyleSheet(f"color:{_T()['muted']};font-size:{_fs(10)}px;")
        self._scale_lbl = scale_lbl
        self._scale_cb = QComboBox()
        self._scale_cb.addItems(["100%", "125%", "150%", "175%"])
        _scale_vals = [1.0, 1.25, 1.5, 1.75]
        _scale_idx = min(range(len(_scale_vals)), key=lambda i: abs(_scale_vals[i] - _UI_SCALE))
        self._scale_cb.setCurrentIndex(_scale_idx)
        self._scale_cb.setFixedWidth(72)
        self._scale_cb.setStyleSheet(COMBO_STYLE)
        self._scale_cb.currentIndexChanged.connect(self._on_scale_changed)
        scale_row.addWidget(scale_lbl)
        scale_row.addWidget(self._scale_cb)
        scale_row.addStretch()
        ll.addLayout(scale_row)

        # ── Files group ───────────────────────────────────────────────────────
        fg = QGroupBox("Files")
        fg.setStyleSheet(GROUP_STYLE)
        fgl = QVBoxLayout(fg)
        fgl.setSpacing(4)
        fgl.setContentsMargins(10, 16, 10, 10)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        fgl.addWidget(self.drop_zone)

        self.video_row = FileRow("Video",  "Select video…",  "Video (*.mp4 *.mkv *.avi *.mov)",
                                 icon=_icon("video.png", 16), icon_name="video.png")
        self.osd_row   = FileRow("OSD",    "Auto-detected",  "OSD (*.osd)",
                                 icon=_icon("gear.png",  16), icon_name="gear.png")
        self.srt_row   = FileRow("SRT",    "Auto-detected",  "SRT (*.srt)",
                                 icon=_icon("wifi.png",  16), icon_name="wifi.png")
        self.video_row.btn.clicked.disconnect()
        self.video_row.btn.clicked.connect(self._on_video)
        self.osd_row.btn.clicked.disconnect()
        self.osd_row.btn.clicked.connect(self._manual_osd)
        self.srt_row.btn.clicked.disconnect()
        self.srt_row.btn.clicked.connect(self._manual_srt)

        fgl.addWidget(self.video_row)
        fgl.addWidget(self.osd_row)
        fgl.addWidget(self.srt_row)
        ll.addWidget(fg)

        # ── OSD Font group ────────────────────────────────────────────────────
        fontg = QGroupBox("OSD Font")
        fontg.setStyleSheet(GROUP_STYLE)
        fontgl = QVBoxLayout(fontg)
        fontgl.setSpacing(6)
        fontgl.setContentsMargins(10, 16, 10, 10)


        # Style row
        st_row = QHBoxLayout()
        self._st_lbl = QLabel("Style:")
        self._st_lbl.setFixedWidth(68)
        self._st_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.style_combo = QComboBox()
        self.style_combo.setStyleSheet(COMBO_STYLE)
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        st_row.addWidget(self._st_lbl)
        st_row.addWidget(self.style_combo, 1)
        fontgl.addLayout(st_row)

        # HD + Custom row
        hd_row = QHBoxLayout()
        self.hd_check = QCheckBox("HD tiles")
        self.hd_check.setChecked(True)
        self.hd_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.hd_check.stateChanged.connect(self._reload_font)
        self._custom_btn = QPushButton("+  Add Font")
        self._custom_btn.setStyleSheet(BTN_SEC)
        self._custom_btn.setFixedHeight(26)
        self._custom_btn.clicked.connect(self._custom_font)
        hd_row.addWidget(self.hd_check)
        hd_row.addStretch()
        hd_row.addWidget(self._custom_btn)
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

        self.srt_size_sl = LabeledSlider("Size", 75, 200, 100, "%")
        self.srt_size_sl.valueChanged.connect(self._refresh_preview)

        _fields_lbl = QLabel("Visible fields:")
        _fields_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.srt_fields_combo = CheckableComboBox()
        self.srt_fields_combo.add_items(SRT_FIELDS)
        self.srt_fields_combo.setStyleSheet(f"font-size:11px;")
        self.srt_fields_combo.selectionChanged.connect(lambda _: self._refresh_preview())

        note2 = QLabel("Radio signal, bitrate, GPS, altitude from .srt.\n"
                       "'No MAVLink telemetry' lines are hidden.")
        note2.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        note2.setWordWrap(True)

        srtgl.addWidget(self.srt_bar_check)
        srtgl.addWidget(self.srt_opacity_sl)
        srtgl.addWidget(self.srt_size_sl)
        srtgl.addWidget(_fields_lbl)
        srtgl.addWidget(self.srt_fields_combo)
        srtgl.addWidget(note2)
        ll.addWidget(srtg)

        ll.addStretch()
        left_scroll.setWidget(left_inner)
        self._left_scroll = left_scroll   # saved for theme reapply
        return left_scroll

    def _build_centre_panel(self) -> QWidget:
        """Build and return the centre panel (player + controls)."""
        from player import PlayerPanel

        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(10, 16, 10, 12)
        cl.setSpacing(8)

        self._prev_lbl = QLabel("Preview")
        self._prev_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._prev_lbl.setStyleSheet(f"color:{_T()['subtext']}")
        cl.addWidget(self._prev_lbl)

        # ── New player panel (replaces PreviewPanel + slider + trim + transport)
        self._player_panel = PlayerPanel(
            theme_fn=_T,
            icon_fn=_icon,
            btn_play_fn=lambda: BTN_PLAY,
            btn_sec_fn=lambda: BTN_SEC,
            fs_fn=_fs,
            composite_fn=self._composite,
            osd_fn=self._render_osd_overlay,
        )
        cl.addWidget(self._player_panel, 1)

        # Aliases for backward compatibility with render pipeline
        self.trim_sel = self._player_panel.timeline
        self.trim_in_lbl = self._player_panel.trim_in_lbl
        self.trim_out_lbl = self._player_panel.trim_out_lbl

        # Wire trim signals
        self._player_panel.timeline.trimChanged.connect(self._on_trim_changed)
        self._player_panel.timeline.trimChanged.connect(lambda *_: self._update_size_hint())

        # Wire refresh button
        self._player_panel.transport.refreshClicked.connect(self._refresh_preview)

        # Create frame_info alias for theme code
        self.frame_info = self._player_panel.transport.info_lbl

        # ── Credits ───────────────────────────────────────────────────────────
        credit = QLabel(
            'Icons by <a href="https://www.flaticon.com/free-icons/wifi-connection" '
            'style="color:#2a2a3a;text-decoration:none;">Smashicons \u2013 Flaticon</a>'
            '  &middot;  Color correction based on '
            '<a href="https://github.com/sickgreg/VfxEnc" '
            'style="color:#2a2a3a;text-decoration:none;">VfxEnc</a> by sickgreg'
        )
        credit.setStyleSheet(f"color:{_T()['muted']};font-size:8px;")
        credit.setOpenExternalLinks(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignRight)
        credit.setWordWrap(True)
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

        # OSD sync offset row
        sync_row = QHBoxLayout()
        sync_row.setSpacing(4)
        self._sync_lbl = QLabel("OSD offset:")
        self._sync_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:{_fs(11)}px;")
        self._sync_lbl.setFixedWidth(72)
        self.osd_offset_sb = QSpinBox()
        self.osd_offset_sb.setRange(-10000, 10000)
        self.osd_offset_sb.setValue(_OSD_OFFSET_MS)
        self.osd_offset_sb.setSuffix(" ms")
        self.osd_offset_sb.setToolTip(
            "Shift OSD timestamps relative to video.\n"
            "+500 ms → OSD shows data 500 ms later (compensates OSD lagging behind).\n"
            "−500 ms → OSD shows data 500 ms earlier.")
        self.osd_offset_sb.setStyleSheet(
            f"QSpinBox{{background:{_T()['surface']};color:{_T()['text']};"
            f"border:1px solid {_T()['border2']};border-radius:4px;padding:3px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;"
            f"background:{_T()['surface2']};border-radius:2px;}}"
        )
        self.osd_offset_sb.valueChanged.connect(self._on_osd_offset_changed)
        self._rst_offset_btn = QPushButton("↺")
        self._rst_offset_btn.setFixedWidth(28)
        self._rst_offset_btn.setFixedHeight(24)
        self._rst_offset_btn.setStyleSheet(BTN_SEC)
        self._rst_offset_btn.setToolTip("Reset OSD offset to 0")
        self._rst_offset_btn.clicked.connect(lambda: self.osd_offset_sb.setValue(0))
        sync_row.addWidget(self._sync_lbl)
        sync_row.addWidget(self.osd_offset_sb, 1)
        sync_row.addWidget(self._rst_offset_btn)
        posgl.addLayout(sync_row)

        self._rst_pos_btn = QPushButton("↺  Reset")
        self._rst_pos_btn.setStyleSheet(BTN_SEC)
        self._rst_pos_btn.setFixedHeight(26)
        self._rst_pos_btn.clicked.connect(self._reset_pos)
        posgl.addWidget(self._rst_pos_btn)

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
        return centre

    def _build_color_correction_group(self, rl):
        """Build color correction collapsible group and add to layout."""
        t = _T()

        self._cc_group = QGroupBox("Color Correction")
        self._cc_group.setStyleSheet(GROUP_STYLE)
        ccl = QVBoxLayout(self._cc_group)
        ccl.setSpacing(2)
        ccl.setContentsMargins(10, 14, 10, 6)

        self.cc_enable = QCheckBox("Enable color correction")
        self.cc_enable.setStyleSheet(f"color:{t['text']};font-size:11px;")
        self.cc_enable.stateChanged.connect(self._on_color_changed)
        ccl.addWidget(self.cc_enable)

        # Container for all CC controls — hidden until "Enable" is checked
        self._cc_content = QWidget()
        _cc_inner = QVBoxLayout(self._cc_content)
        _cc_inner.setContentsMargins(0, 0, 0, 0)
        _cc_inner.setSpacing(6)
        self._cc_content.setVisible(False)
        self.cc_enable.toggled.connect(self._cc_content.setVisible)
        ccl.addWidget(self._cc_content)
        ccl = _cc_inner  # alias: all remaining widgets go into the inner container

        # ── Helper: create a LabeledSlider, connect it, add to layout ──
        def _slider(attr, label, lo, hi, default, suffix=""):
            s = LabeledSlider(label, lo, hi, default, suffix)
            s.valueChanged.connect(self._on_color_changed)
            ccl.addWidget(s)
            setattr(self, attr, s)

        # ── Helper: create a QCheckBox, connect it, add to layout ──
        def _check(attr, label, checked=False):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setStyleSheet(f"color:{t['text']};font-size:11px;")
            cb.stateChanged.connect(self._on_color_changed)
            ccl.addWidget(cb)
            setattr(self, attr, cb)

        # Reverse ColorTrans sub-section
        _check("cc_reverse", "Reverse ColorTrans", checked=True)
        _slider("cc_yoff",  "Y-offset",  0, 100, 25, "%")
        _slider("cc_blift", "Blk lift",  0, 100, 20)

        ccl.addWidget(_sep())

        # Grading sliders
        _slider("cc_brightness", "Bright",   -100, 100, 0)
        _slider("cc_contrast",   "Contrast",  0, 300, 100, "%")
        _slider("cc_gamma",      "Gamma",     10, 300, 100)
        _slider("cc_lift",       "Lift",      -100, 100, -15)
        _slider("cc_gain",       "Gain",      0, 500, 175)
        _slider("cc_hue",        "Hue",       -180, 180, 0, "°")
        _slider("cc_sat",        "Satur.",    -100, 100, -22)

        # RGB multipliers — button opens popup dialog
        self._cc_rgb_btn = QPushButton("R / G / B:  1.00 / 1.00 / 1.00")
        self._cc_rgb_btn.setStyleSheet(BTN_SEC)
        self._cc_rgb_btn.setFixedHeight(26)
        self._cc_rgb_btn.clicked.connect(self._cc_open_rgb_popup)
        ccl.addWidget(self._cc_rgb_btn)

        # Hidden sliders for RGB values (no UI inline, edited via popup)
        for attr in ("cc_r", "cc_g", "cc_b"):
            s = QSlider(); s.setRange(0, 400); s.setValue(100); s.setVisible(False)
            setattr(self, attr, s)

        ccl.addWidget(_sep())

        # Custom GLSL shader
        self.cc_glsl_row = FileRow("GLSL", "Optional .glsl shader…",
                                    "GLSL Shaders (*.glsl *.hook)")
        self.cc_glsl_row.btn.clicked.disconnect()
        self.cc_glsl_row.btn.clicked.connect(self._cc_browse_glsl)
        ccl.addWidget(self.cc_glsl_row)

        self._cc_glsl_warn = QLabel("")
        self._cc_glsl_warn.setStyleSheet(f"color:{t['orange']};font-size:10px;")
        self._cc_glsl_warn.setWordWrap(True)
        self._cc_glsl_warn.setVisible(False)
        ccl.addWidget(self._cc_glsl_warn)

        # Preset + Reset row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        self._cc_preset_lbl = QLabel("Preset:")
        self._cc_preset_lbl.setStyleSheet(f"color:{t['subtext']};font-size:11px;")
        self._cc_preset_lbl.setFixedWidth(48)
        self.cc_preset_cb = QComboBox()
        self.cc_preset_cb.setStyleSheet(COMBO_STYLE)
        self.cc_preset_cb.currentIndexChanged.connect(self._on_cc_preset_changed)
        self._cc_save_btn = QPushButton("Save…")
        self._cc_save_btn.setStyleSheet(BTN_SEC)
        self._cc_save_btn.setFixedHeight(24)
        self._cc_save_btn.clicked.connect(self._cc_save_preset)
        self._cc_rst_btn = QPushButton("Reset")
        self._cc_rst_btn.setStyleSheet(BTN_SEC)
        self._cc_rst_btn.setFixedHeight(24)
        self._cc_rst_btn.setToolTip("Reset all color correction to defaults")
        self._cc_rst_btn.clicked.connect(self._cc_reset_defaults)
        preset_row.addWidget(self._cc_preset_lbl)
        preset_row.addWidget(self.cc_preset_cb, 1)
        preset_row.addWidget(self._cc_save_btn)
        preset_row.addWidget(self._cc_rst_btn)
        ccl.addLayout(preset_row)

        rl.addWidget(self._cc_group)

        # Load presets into combo
        self._cc_presets = {}  # name -> dict
        self._cc_loading_preset = False
        self._cc_scan_presets()

    def _build_right_panel(self) -> QWidget:
        """Build and return the right panel (encoding + output)."""
        # Outer container: scroll area on top, fixed bottom section
        right_outer = QWidget()
        right_outer.setMinimumWidth(250)
        right_outer.setMaximumWidth(500)
        outer_lay = QVBoxLayout(right_outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        # ── Scrollable area (color correction + encoding settings) ───────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{background:#1e1e2e;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;}"
        )
        self._right_scroll = right_scroll

        right = QWidget()
        right.setMinimumWidth(240)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 14, 4)
        rl.setSpacing(6)

        # ── Color Correction group (collapsible) ─────────────────────────────
        self._build_color_correction_group(rl)

        # ── Output & Encoding (merged) ───────────────────────────────────────
        encg = QGroupBox("Output & Encoding")
        encg.setStyleSheet(GROUP_STYLE)
        encgl = QVBoxLayout(encg)
        encgl.setSpacing(4)
        encgl.setContentsMargins(10, 14, 10, 8)

        # Output file row
        self.out_row = FileRow("Output", "Choose output path…", "", save_mode=True, icon=_icon("save.png", 16), icon_name="save.png")
        encgl.addWidget(self.out_row)

        encgl.addWidget(_sep())

        # Codec row
        codec_row = QHBoxLayout()
        self._codec_lbl = QLabel("Codec:")
        self._codec_lbl.setFixedWidth(52)
        self._codec_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.codec_cb = QComboBox()
        self.codec_cb.addItems(["H.264 (libx264)", "H.265 (libx265)"])
        self.codec_cb.setStyleSheet(COMBO_STYLE)
        self.codec_cb.currentIndexChanged.connect(self._on_codec_changed)
        codec_row.addWidget(self._codec_lbl)
        codec_row.addWidget(self.codec_cb, 1)
        encgl.addLayout(codec_row)

        # Output bitrate: logarithmic slider + spinbox for fine ±1 steps
        self.mbps_row = QWidget()
        mbps_lay = QHBoxLayout(self.mbps_row)
        mbps_lay.setContentsMargins(0, 0, 0, 0)
        mbps_lay.setSpacing(6)
        self._mbps_lbl = QLabel("Mbit/s:")
        self._mbps_lbl.setFixedWidth(52)
        self._mbps_lbl.setStyleSheet(f"color:{_T()['subtext']};font-size:11px;")
        self.mbps_sl = QSlider(Qt.Orientation.Horizontal)
        self.mbps_sl.setRange(0, 1000)
        self.mbps_sl.setValue(self._mbps_to_slider(8))
        self.mbps_sl.setToolTip(
            "Drag to set output bitrate (logarithmic scale, 1–100 Mbit/s).\n"
            "Use the arrows on the right for ±1 Mbit/s fine steps.\n"
            "Auto-set to source average bitrate on video load."
        )
        self.mbps_sl.setStyleSheet(SLIDER_STYLE)
        self.mbps_spin = QSpinBox()
        self.mbps_spin.setRange(1, 100)
        self.mbps_spin.setValue(8)
        self.mbps_spin.setSuffix(" Mbit/s")
        self.mbps_spin.setFixedWidth(90)
        self.mbps_spin.setStyleSheet(
            f"QSpinBox{{background:{_T()['surface']};color:{_T()['text']};"
            f"border:1px solid {_T()['border2']};border-radius:4px;padding:3px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;"
            f"background:{_T()['surface2']};border-radius:2px;}}"
        )
        self.mbps_sl.valueChanged.connect(self._on_mbps_sl_changed)
        self.mbps_spin.valueChanged.connect(self._on_mbps_spin_changed)
        mbps_lay.addWidget(self._mbps_lbl)
        mbps_lay.addWidget(self.mbps_sl)
        mbps_lay.addWidget(self.mbps_spin)
        encgl.addWidget(self.mbps_row)

        # Estimated size hint
        self.size_hint = QLabel("")
        self.size_hint.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        encgl.addWidget(self.size_hint)

        # GPU row — checkbox + status on same line
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(6)
        self.hw_check = QCheckBox("⚡ GPU")
        self.hw_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.hw_check.setEnabled(False)
        self.hw_check.setChecked(False)
        self.hw_check.stateChanged.connect(self._update_size_hint)
        self.hw_lbl = QLabel("Detecting…")
        self.hw_lbl.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        gpu_row.addWidget(self.hw_check)
        gpu_row.addWidget(self.hw_lbl, 1)
        encgl.addLayout(gpu_row)

        # Kick off GPU detection in background.
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
        self._upscale_lbl = QLabel("Upscale output:")
        self._upscale_lbl.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["Off", "1440p  (2560×1440)", "2.7K  (2688×1512)", "4K  (3840×2160)"])
        self.upscale_combo.setStyleSheet(COMBO_STYLE)
        self.upscale_combo.setToolTip(
            "Scale the output video to a higher resolution using Lanczos.\n"
            "Useful when source is 1080p and you want a sharper result on a high-res display."
        )
        upscale_row.addWidget(self._upscale_lbl)
        upscale_row.addWidget(self.upscale_combo, 1)
        encgl.addLayout(upscale_row)

        encgl.addWidget(_sep())

        # Transparent overlay export
        self.transparent_check = QCheckBox("Transparent overlay (ProRes 4444)")
        self.transparent_check.setStyleSheet(f"color:{_T()['text']};font-size:11px;")
        self.transparent_check.setToolTip(
            "Export only the OSD glyphs and SRT bar as a video with\n"
            "a transparent background (ProRes 4444 alpha, .mov).\n\n"
            "Layer it on GoPro/action-cam footage in DaVinci Resolve,\n"
            "Premiere Pro, or Final Cut Pro."
        )
        self.transparent_check.stateChanged.connect(self._on_transparent_changed)
        encgl.addWidget(self.transparent_check)

        rl.addWidget(encg)
        rl.addStretch()
        right_scroll.setWidget(right)
        outer_lay.addWidget(right_scroll, 1)

        # ── Fixed bottom section (render controls — not scrollable) ──────────
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(10, 4, 14, 8)
        bl.setSpacing(4)

        # Progress bar
        self.prog = RenderBar()
        bl.addWidget(self.prog)

        self.status = QLabel("Ready")
        self.status.setStyleSheet(f"color:{_T()['muted']};font-size:10px;")
        self.status.setWordWrap(True)
        bl.addWidget(self.status)

        # OSD trimmed warning (hidden by default)
        self.osd_warn = QLabel("⚠ No OSD elements in trim window — rendering without OSD overlay")
        self.osd_warn.setStyleSheet(f"color:{_T()['orange']};font-size:10px;")
        self.osd_warn.setWordWrap(True)
        self.osd_warn.setVisible(False)
        bl.addWidget(self.osd_warn)

        # Render + Stop buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.render_btn = QPushButton("  Render Video")
        self.render_btn.setIcon(_icon("render.png", 18))
        self.render_btn.setFixedHeight(36)
        self.render_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.render_btn.setStyleSheet(BTN_PRIMARY)
        self.render_btn.clicked.connect(self._render)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(_icon("stop.png", 18))
        self.stop_btn.setFixedSize(36, 36)
        self.stop_btn.setStyleSheet(BTN_STOP)
        self.stop_btn.setToolTip("Stop render")
        self.stop_btn.clicked.connect(self._stop_render)
        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.render_btn, 1)
        btn_row.addWidget(self.stop_btn)
        bl.addLayout(btn_row)

        # FFmpeg status
        self.ffmpeg_lbl = QLabel()
        self.ffmpeg_lbl.setWordWrap(True)
        self._refresh_ffmpeg_status()
        bl.addWidget(self.ffmpeg_lbl)

        outer_lay.addWidget(bottom)
        return right_outer

    def _on_codec_changed(self):
        self._update_size_hint()

    def _on_transparent_changed(self, state):
        """Toggle encoding controls when transparent overlay mode changes."""
        checked = bool(state)
        # Disable controls irrelevant to transparent export
        self.codec_cb.setEnabled(not checked)
        self._codec_lbl.setEnabled(not checked)
        self.mbps_row.setEnabled(not checked)
        self.mbps_sl.setEnabled(not checked)
        self.mbps_spin.setEnabled(not checked)
        self._mbps_lbl.setEnabled(not checked)
        self.size_hint.setVisible(not checked)
        self.hw_check.setEnabled(not checked and self.hw_check.toolTip().startswith("✓"))
        self.upscale_combo.setEnabled(not checked)
        self._upscale_lbl.setEnabled(not checked)
        if hasattr(self, 'cc_enable'):
            self.cc_enable.setEnabled(not checked)
            if checked:
                self._cc_content.setVisible(False)
            else:
                self._cc_content.setVisible(self.cc_enable.isChecked())
        # Update output path extension when toggling
        if self.video_row.path:
            trim_s = self.trim_sel.in_pct  * self.video_dur if self.video_dur > 0 else 0.0
            trim_e = self.trim_sel.out_pct * self.video_dur if self.video_dur > 0 else 0.0
            self.out_row.set_path(
                self._make_output_path(self.video_row.path, trim_s, trim_e))

    @staticmethod
    def _mbps_to_slider(mbps):
        mbps = max(1, min(100, int(mbps)))
        return round(math.log(mbps) / math.log(100) * 1000)

    def _on_mbps_sl_changed(self, pos):
        v = max(1, min(100, round(math.exp(pos / 1000 * math.log(100)))))
        self.mbps_spin.blockSignals(True)
        self.mbps_spin.setValue(v)
        self.mbps_spin.blockSignals(False)
        self._update_size_hint()

    def _on_mbps_spin_changed(self, v):
        self.mbps_sl.blockSignals(True)
        self.mbps_sl.setValue(self._mbps_to_slider(v))
        self.mbps_sl.blockSignals(False)
        self._update_size_hint()

    def _update_size_hint(self):
        if self.video_dur <= 0:
            self.size_hint.setText("")
            return
        # Use trimmed duration for estimate if trim is set
        in_pct  = self.trim_sel.in_pct  if hasattr(self, 'trim_sel') else 0.0
        out_pct = self.trim_sel.out_pct if hasattr(self, 'trim_sel') else 1.0
        dur     = self.video_dur * (out_pct - in_pct)
        mbps    = self.mbps_spin.value()
        est_mb  = mbps * dur / 8
        src_note = f"  src {self.source_mbps:.1f} Mbit/s" if self.source_mbps > 0.1 else ""
        self.size_hint.setText(f"≈ {est_mb:.0f} MB at {mbps} Mbit/s{src_note}")

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
        """Import a font PNG — copies it into fonts/ for permanent use."""
        from font_loader import import_font
        p, _ = QFileDialog.getOpenFileName(self, "Select Font PNG", "", "PNG (*.png)")
        if not p:
            return
        # Derive a clean name from the filename
        stem = os.path.splitext(os.path.basename(p))[0]
        # Detect current firmware context from the selected font's prefix
        fw = "Betaflight"
        raw = self.style_combo.currentData() or ""
        for fw_name, prefixes in FIRMWARE_PREFIXES.items():
            if any(raw.upper().startswith(p.upper()) for p in prefixes):
                fw = fw_name
                break
        folder = import_font(p, fw, stem)
        if folder:
            # Rescan fonts and select the newly imported one
            self._on_fw_changed(fw)
            # Find and select the new font in the combo
            for i in range(self.style_combo.count()):
                if self.style_combo.itemData(i) == folder.name:
                    self.style_combo.setCurrentIndex(i)
                    break
            self.font_lbl.setText(f"\u2713 Imported: {stem}")
            self.font_lbl.setStyleSheet(f"color:{_T()['green']};font-size:10px;")
            self._refresh_preview()

    # ── File selection ────────────────────────────────────────────────────────

    def _on_file_dropped(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.mp4', '.mkv', '.avi', '.mov'):
            self.video_row.set_path(path)
            self._load_video(path)
            self._auto_detect(path)
        elif ext == '.osd':
            self._load_osd(path)
            self._auto_detect(path)
        elif ext == '.srt':
            self._load_srt(path)
            self._auto_detect(path)

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
            if candidates['.mp4'].exists():
                self.video_row.set_path(str(candidates['.mp4']))
                self._load_video(str(candidates['.mp4']))
        elif ext == '.srt':
            if candidates['.osd'].exists(): self._load_osd(str(candidates['.osd']))
            if candidates['.mp4'].exists():
                self.video_row.set_path(str(candidates['.mp4']))
                self._load_video(str(candidates['.mp4']))

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _refresh_theme(self):
        """Rebuild all stylesheet strings and repaint the live UI."""
        _build_styles()
        self._apply_theme()

    def _on_scale_changed(self, idx: int):
        global _UI_SCALE
        _UI_SCALE = [1.0, 1.25, 1.5, 1.75][idx]
        self._refresh_theme()
        _save_settings()

    def _toggle_theme(self):
        global _DARK_THEME
        _DARK_THEME = not _DARK_THEME
        self._refresh_theme()

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
        self._refresh_theme()      # rebuild stylesheets and repaint live UI
        if self._theme_editor_dlg:
            self._theme_editor_dlg.reload_from_theme()   # sync editor panels

    def _update_splitter_style(self):
        """Style the QSplitter drag handles to match the current theme."""
        t = _T()
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {t['border']};
            }}
            QSplitter::handle:hover {{
                background: {t['accent']};
            }}
        """)

    def _apply_theme(self):
        """Reapply all stylesheets after a theme change."""
        t = _T()
        self.setStyleSheet(APP_STYLE)
        self._update_splitter_style()

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
        self._h1.setFont(QFont("Segoe UI", _fs(16), QFont.Weight.Bold))
        self._h1.setStyleSheet(f"color:{t['text']};")
        self._ver_lbl.setFont(QFont("Segoe UI", _fs(8)))
        self._ver_lbl.setStyleSheet(f"color:{t['muted']};")
        self._h2.setFont(QFont("Segoe UI", _fs(16), QFont.Weight.Bold))
        self._h2.setStyleSheet(f"color:{t['text']};")
        self._scale_lbl.setStyleSheet(f"color:{t['muted']};font-size:{_fs(10)}px;")
        self._prev_lbl.setStyleSheet(f"color:{t['subtext']};")

        _scroll_ss = (
            f"QScrollArea{{border:none;background:transparent;}}"
            f"QScrollBar:vertical{{background:{t['bg']};width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:{t['surface2']};border-radius:3px;}}"
        )
        self._left_scroll.setStyleSheet(_scroll_ss)
        self._right_scroll.setStyleSheet(_scroll_ss)

        # Structural widgets
        for gb in self.findChildren(QGroupBox):
            gb.setStyleSheet(GROUP_STYLE)
        for sl in self.findChildren(QSlider):
            sl.setStyleSheet(SLIDER_STYLE)
        for cb in self.findChildren(QComboBox):
            cb.setStyleSheet(COMBO_STYLE)
        for ck in self.findChildren(QCheckBox):
            ck.setStyleSheet(f"color:{t['text']};font-size:{_fs(11)}px;")
        for div in self._dividers:
            div.setStyleSheet(f"color:{t['border']};")
        for w in self.findChildren(LabeledSlider):
            w.refresh_theme()
        for w in self.findChildren(InfoCard):
            w.refresh_theme()

        # File rows
        for row in [self.video_row, self.osd_row, self.srt_row, self.out_row,
                     self.cc_glsl_row]:
            row._name_lbl.setStyleSheet(f"color:{t['subtext']}")
            row.path_lbl.setStyleSheet(PATH_FILLED if row.path else PATH_EMPTY)
            row.btn.setStyleSheet(BTN_SEC)
            row.clr.setStyleSheet(BTN_DANGER)
        self.drop_zone.refresh_theme()

        # Player panel
        self._player_panel.refresh_theme()

        # Buttons + icon retint
        self.render_btn.setStyleSheet(BTN_PRIMARY)
        self.stop_btn.setStyleSheet(BTN_STOP)
        _render_ico_col = "#ffffff" if not _DARK_THEME else t['bg']
        self.render_btn.setIcon(_icon("render.png", 20, _render_ico_col))
        self.stop_btn.setIcon(_icon("stop.png", 20, "#ffffff"))
        # File row icons
        for row in [self.video_row, self.osd_row, self.srt_row, self.out_row]:
            row.retint()
        self._custom_btn.setStyleSheet(BTN_SEC)
        self._rst_pos_btn.setStyleSheet(BTN_SEC)
        self._rst_offset_btn.setStyleSheet(BTN_SEC)
        self._player_panel.trim_rst_btn.setStyleSheet(BTN_SEC)
        self._cc_save_btn.setStyleSheet(BTN_SEC)
        self._cc_rst_btn.setStyleSheet(BTN_SEC)
        self._cc_rgb_btn.setStyleSheet(BTN_SEC)

        # SpinBoxes
        _sb_style = (
            f"QSpinBox{{background:{t['surface']};color:{t['text']};"
            f"border:1px solid {t['border2']};border-radius:4px;padding:3px 6px;}}"
            f"QSpinBox::up-button,QSpinBox::down-button{{width:16px;"
            f"background:{t['surface2']};border-radius:2px;}}"
        )
        self.mbps_spin.setStyleSheet(_sb_style)
        self.osd_offset_sb.setStyleSheet(_sb_style)

        # Progress bar — repaint with new theme colours
        self.prog.update()

        # Inline subtext labels (constructed with hardcoded colours at init time)
        for lbl, style in [
            (self.frame_info,    f"color:{t['muted']};font-size:{_fs(10)}px;"),
            (self.size_hint,     f"color:{t['muted']};font-size:{_fs(10)}px;"),
            (self.status,        f"color:{t['muted']};font-size:{_fs(10)}px;"),
            (self.osd_warn,      f"color:{t['orange']};font-size:{_fs(10)}px;"),
            (self._sync_lbl,     f"color:{t['subtext']};font-size:{_fs(11)}px;"),

            (self._st_lbl,       f"color:{t['subtext']};font-size:{_fs(11)}px;"),
            (self._codec_lbl,    f"color:{t['subtext']};font-size:{_fs(11)}px;"),
            (self._mbps_lbl,     f"color:{t['subtext']};font-size:{_fs(11)}px;"),
            (self._upscale_lbl,  f"color:{t['text']};font-size:{_fs(11)}px;"),
            (self._cc_preset_lbl, f"color:{t['subtext']};font-size:{_fs(11)}px;"),
        ]:
            lbl.setStyleSheet(style)

        # font_lbl — preserve its success/error state colour if already set
        fl_ss = self.font_lbl.styleSheet()
        if "green" not in fl_ss and t['green'] not in fl_ss:
            # still in initial/error state — use orange (no font) or current red
            if "red" not in fl_ss and t['red'] not in fl_ss:
                self.font_lbl.setStyleSheet(f"color:{t['orange']};font-size:{_fs(10)}px;")

        # hw_lbl — preserve detected GPU green, only reset if still pending/absent
        hw_text = self.hw_lbl.text()
        if hw_text.startswith("Detecting") or hw_text.startswith("No GPU") or hw_text == "":
            self.hw_lbl.setStyleSheet(f"color:{t['muted']};font-size:{_fs(10)}px;")
        elif hw_text.startswith("✓"):
            self.hw_lbl.setStyleSheet(f"color:{t['green']};font-size:{_fs(10)}px;")

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
        self._player_panel.controller.stop()
        if not self.out_row.path:
            self.out_row.set_path(self._make_output_path(path))
        self.video_frame = None
        self._st("Reading video info\u2026")
        self._vi = VideoInfoWorker(path)
        self._vi.result.connect(self._got_vid_info)
        self._vi.start()

    def _got_vid_info(self, info):
        self.vid_card.clear()
        if "error" not in info:
            self.video_fps = info.get("fps", 60.0)
            self.video_dur = info.get("duration", 0.0)
            size_mb = info.get("size_mb", 0) or 0
            # Source bitrate in Mbit/s — auto-sets the output bitrate spinbox
            self.source_mbps = (size_mb * 8 / max(self.video_dur, 1)) if self.video_dur > 0 else 0
            if self.source_mbps > 0.5:
                # Clamp to spinbox range; _on_mbps_spin_changed syncs the slider
                self.mbps_spin.setValue(max(1, min(100, round(self.source_mbps))))
            self.vid_card.add_row("Res",  f"{info.get('width')}×{info.get('height')}")
            self.vid_card.add_row("FPS",  str(info.get("fps", "?")))
            _dm, _ds = divmod(int(self.video_dur), 60)
            self.vid_card.add_row("Dur",  f"{_dm}:{_ds:02d}")
            self.vid_card.add_row("Size", f"{size_mb} MB")
            self._update_size_hint()
            # Load video into the player controller (triggers prefetch + frame 0 extract)
            self._player_panel.controller.load_video(
                self.video_row.path, self.video_dur, self.video_fps,
                info.get("width", 0), info.get("height", 0))
        self._st("Ready")

    ## _start_prefetch, _prefetch_frames — removed (now in PlayerController)

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
            if s.efficiency:     self.osd_card.add_row("Eff",  s.efficiency)
            self.osd_card.add_row("Dur",  f"{self.osd_data.duration_ms/1000:.1f}s")
            self.osd_card.add_row("Pkts", str(self.osd_data.frame_count))
            self.osd_row.set_path(path)
            # Auto-select firmware from OSD fc_type
            fc = (self.osd_data.stats.fc_type or "").strip()
            fw_map = {"Betaflight": "Betaflight", "INAV": "INAV",
                      "ArduPilot": "ArduPilot", "ARDU": "ArduPilot"}
            self._on_fw_changed(fw_map.get(fc, "Betaflight"))
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
            if s.efficiency:     self.osd_card.add_row("Eff",  s.efficiency)
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
        """Switch firmware context and pick a specific font folder by name."""
        self._on_fw_changed(firmware)
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

    # ── Keyboard shortcuts ─────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()
        ctrl = self._player_panel.controller

        if key == Qt.Key.Key_Space:
            ctrl.toggle_play()
        elif key == Qt.Key.Key_Left:
            n = 10 if mod & Qt.KeyboardModifier.ShiftModifier else 1
            ctrl.step_backward(n)
        elif key == Qt.Key.Key_Right:
            n = 10 if mod & Qt.KeyboardModifier.ShiftModifier else 1
            ctrl.step_forward(n)
        elif key == Qt.Key.Key_J:
            ctrl.step_backward(1)
        elif key == Qt.Key.Key_K:
            ctrl.shuttle_reset()
        elif key == Qt.Key.Key_L:
            ctrl.shuttle_faster()
        elif key == Qt.Key.Key_Home:
            ctrl.seek_start()
        elif key == Qt.Key.Key_End:
            ctrl.seek_end()
        elif key == Qt.Key.Key_I:
            ctrl.set_trim_in()
        elif key == Qt.Key.Key_O:
            ctrl.set_trim_out()
        else:
            super().keyPressEvent(event)

    # ── Preview (delegated to PlayerController) ─────────────────────────────

    def _video_time_ms(self, pct):
        """Map slider 0-_SL_MAX → absolute video timestamp in ms (full video duration)."""
        return int(self.video_dur * pct / _SL_MAX * 1000)

    def _refresh_preview(self):
        """Refresh current frame with updated OSD composite settings."""
        self._player_panel.controller.refresh_now()

    def _composite(self, img, pct):
        """Composite OSD + SRT onto a video frame at given slider position."""
        t_ms     = self._player_panel.controller.pct_to_video_time_ms(pct) + self.osd_offset_sb.value()
        osd_frame = self.osd_data.frame_at_time(t_ms) if self.osd_data else None
        srt_text = ""
        if self.srt_data and self.srt_bar_check.isChecked():
            td = self.srt_data.get_data_at_time(t_ms)
            if td: srt_text = td.status_line(self.srt_fields_combo.checked_keys())
        cfg = OsdRenderConfig(
            offset_x     = self.sl_x.value(),
            offset_y     = self.sl_y.value(),
            scale        = self.sl_scale.value() / 100.0,
            show_srt_bar = self.srt_bar_check.isChecked(),
            srt_text     = srt_text,
            srt_opacity  = self.srt_opacity_sl.value() / 100.0,
            srt_scale    = self.srt_size_sl.value() / 100.0,
        )
        if self.font_obj and PIL_OK:
            return render_osd_frame(img, osd_frame, self.font_obj, cfg)
        return render_fallback(img, osd_frame, cfg)

    def _render_osd_overlay(self, pct, w, h):
        """Render OSD + SRT as transparent overlay at display size (w, h).

        Used by the two-layer preview system — only re-renders the OSD,
        skipping the expensive video frame copy and resize.
        """
        t_ms = self._player_panel.controller.pct_to_video_time_ms(pct) + self.osd_offset_sb.value()
        osd_frame = self.osd_data.frame_at_time(t_ms) if self.osd_data else None
        srt_text = ""
        if self.srt_data and self.srt_bar_check.isChecked():
            td = self.srt_data.get_data_at_time(t_ms)
            if td: srt_text = td.status_line(self.srt_fields_combo.checked_keys())
        # Scale pixel offsets from video resolution to display resolution
        ctrl = self._player_panel.controller
        ratio = h / ctrl.video_h if ctrl.video_h > 0 else 1.0
        cfg = OsdRenderConfig(
            offset_x     = int(self.sl_x.value() * ratio),
            offset_y     = int(self.sl_y.value() * ratio),
            scale        = self.sl_scale.value() / 100.0,
            show_srt_bar = self.srt_bar_check.isChecked(),
            srt_text     = srt_text,
            srt_opacity  = self.srt_opacity_sl.value() / 100.0,
            srt_scale    = self.srt_size_sl.value() / 100.0,
        )
        canvas = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
        if self.font_obj and PIL_OK:
            return render_osd_frame(canvas, osd_frame, self.font_obj, cfg)
        return render_fallback(canvas, osd_frame, cfg)

    def _on_osd_offset_changed(self, value: int):
        global _OSD_OFFSET_MS
        _OSD_OFFSET_MS = value
        _save_settings()
        self._queue_preview()

    def _reset_pos(self):
        self.sl_x.setValue(0)
        self.sl_y.setValue(0)
        self.sl_scale.setValue(100)
        self._refresh_preview()

    def _queue_preview(self):
        """Debounced preview refresh — fires 60ms after sliders stop moving."""
        self._player_panel.controller.queue_refresh()

    # ── Color Correction ─────────────────────────────────────────────────────

    def _cc_open_rgb_popup(self):
        """Open a small dialog with R/G/B sliders."""
        t = _T()
        dlg = QDialog(self)
        dlg.setWindowTitle("RGB Multipliers")
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(f"background:{t['bg']};")
        vl = QVBoxLayout(dlg)
        vl.setSpacing(8)
        vl.setContentsMargins(16, 16, 16, 12)

        note = QLabel("Adjust per-channel gain (1.0 = neutral)")
        note.setStyleSheet(f"color:{t['muted']};font-size:10px;")
        vl.addWidget(note)

        r_sl = LabeledSlider("Red", 0, 400, self.cc_r.value())
        g_sl = LabeledSlider("Green", 0, 400, self.cc_g.value())
        b_sl = LabeledSlider("Blue", 0, 400, self.cc_b.value())

        # Show values as x.xx instead of raw int
        def _fmt(sl, v):
            sl.vl.setText(f"{v / 100:.2f}")
        for sl in (r_sl, g_sl, b_sl):
            _fmt(sl, sl.value())
            sl.valueChanged.connect(lambda v, s=sl: _fmt(s, v))

        vl.addWidget(r_sl)
        vl.addWidget(g_sl)
        vl.addWidget(b_sl)

        # Live preview: update hidden sliders as popup changes
        def _apply():
            self.cc_r.setValue(r_sl.value())
            self.cc_g.setValue(g_sl.value())
            self.cc_b.setValue(b_sl.value())
            self._on_color_changed()

        r_sl.valueChanged.connect(lambda _: _apply())
        g_sl.valueChanged.connect(lambda _: _apply())
        b_sl.valueChanged.connect(lambda _: _apply())

        btn_row = QHBoxLayout()
        rst_btn = QPushButton("Reset (1.0)")
        rst_btn.setStyleSheet(BTN_SEC)
        rst_btn.clicked.connect(lambda: (r_sl.setValue(100), g_sl.setValue(100), b_sl.setValue(100)))
        ok_btn = QPushButton("Close")
        ok_btn.setStyleSheet(BTN_PRIMARY)
        ok_btn.setFixedHeight(28)
        ok_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(rst_btn)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        vl.addLayout(btn_row)

        dlg.exec()

    def _get_color_config(self) -> ColorTransConfig:
        """Build ColorTransConfig from current UI state."""
        return ColorTransConfig(
            enabled=self.cc_enable.isChecked(),
            reverse_enabled=self.cc_reverse.isChecked(),
            yoff_strength=self.cc_yoff.value() / 100.0,
            black_lift=self.cc_blift.value() / 1000.0,
            brightness=self.cc_brightness.value() / 100.0,
            contrast=self.cc_contrast.value() / 100.0,
            gamma=self.cc_gamma.value() / 100.0,
            lift=self.cc_lift.value() / 100.0,
            gain=self.cc_gain.value() / 100.0,
            hue=float(self.cc_hue.value()),
            saturation=self.cc_sat.value(),
            r_mult=self.cc_r.value() / 100.0,
            g_mult=self.cc_g.value() / 100.0,
            b_mult=self.cc_b.value() / 100.0,
            glsl_shader=self.cc_glsl_row.path or "",
        )

    def _on_color_changed(self, _=None):
        """Called when any color correction slider/checkbox changes."""
        # Update RGB value label
        r, g, b = self.cc_r.value() / 100.0, self.cc_g.value() / 100.0, self.cc_b.value() / 100.0
        self._cc_rgb_btn.setText(f"R / G / B:  {r:.2f} / {g:.2f} / {b:.2f}")

        # Mark preset as "Custom" if user is not loading a preset
        if not self._cc_loading_preset:
            idx = self.cc_preset_cb.findText("Custom")
            if idx < 0:
                self.cc_preset_cb.blockSignals(True)
                self.cc_preset_cb.addItem("Custom")
                self.cc_preset_cb.setCurrentText("Custom")
                self.cc_preset_cb.blockSignals(False)
            else:
                self.cc_preset_cb.blockSignals(True)
                self.cc_preset_cb.setCurrentIndex(idx)
                self.cc_preset_cb.blockSignals(False)

        # Persist color correction settings
        self._cc_save_to_settings()

        # Update player preview color filter
        self._update_player_color_vf()
        self._queue_preview()

    def _update_player_color_vf(self):
        """Build color filter string for preview and pass to PlayerController."""
        cc = self._get_color_config()
        if cc.enabled or (cc.glsl_shader and os.path.isfile(cc.glsl_shader)):
            import tempfile
            from video_processor import _build_color_vf
            ffmpeg = find_ffmpeg()
            if ffmpeg:
                tmp = tempfile.gettempdir()
                vf = _build_color_vf(cc, ffmpeg, tmp)
                self._player_panel.controller.set_color_vf(vf)
                return
        self._player_panel.controller.set_color_vf("")

    def _cc_browse_glsl(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select GLSL Shader", "",
                                            "GLSL Shaders (*.glsl *.hook)")
        if p:
            self.cc_glsl_row.set_path(p)
            # Check libplacebo availability
            ffmpeg = find_ffmpeg()
            if ffmpeg and not detect_libplacebo(ffmpeg):
                self._cc_glsl_warn.setText(
                    "Your FFmpeg lacks libplacebo — GLSL shader will be skipped.\n"
                    "Built-in LUT correction still works.")
                self._cc_glsl_warn.setVisible(True)
            else:
                self._cc_glsl_warn.setVisible(False)
            self._on_color_changed()

    def _cc_scan_presets(self):
        """Scan preset directories and populate combo."""
        self._cc_presets.clear()
        self.cc_preset_cb.blockSignals(True)
        self.cc_preset_cb.clear()

        # Built-in presets
        app_dir = os.path.dirname(os.path.abspath(__file__))
        builtin = os.path.join(app_dir, "presets")

        # User presets (AppData on Windows, ~/.config on Linux)
        if sys.platform == "win32":
            user_dir = os.path.join(os.environ.get("APPDATA", ""), "VueOSD", "presets")
        else:
            user_dir = os.path.join(os.path.expanduser("~"), ".config", "VueOSD", "presets")

        for folder in [builtin, user_dir]:
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(folder, f)) as fh:
                        data = json.load(fh)
                    name = data.get("name", os.path.splitext(f)[0])
                    self._cc_presets[name] = data
                except Exception:
                    pass

        for name in self._cc_presets:
            self.cc_preset_cb.addItem(name)
        self.cc_preset_cb.blockSignals(False)

    def _on_cc_preset_changed(self, idx):
        name = self.cc_preset_cb.currentText()
        if name == "Custom" or name not in self._cc_presets:
            return
        data = self._cc_presets[name]
        self._cc_loading_preset = True
        self.cc_reverse.setChecked(data.get("reverse_enabled", True))
        self.cc_yoff.setValue(int(data.get("yoff_strength", 0.25) * 100))
        self.cc_blift.setValue(int(data.get("black_lift", 0.02) * 1000))
        self.cc_brightness.setValue(int(data.get("brightness", 0.0) * 100))
        self.cc_contrast.setValue(int(data.get("contrast", 1.0) * 100))
        self.cc_gamma.setValue(int(data.get("gamma", 1.0) * 100))
        self.cc_lift.setValue(int(data.get("lift", -0.15) * 100))
        self.cc_gain.setValue(int(data.get("gain", 1.75) * 100))
        self.cc_hue.setValue(int(data.get("hue", 0.0)))
        self.cc_sat.setValue(int(data.get("saturation", -22.5)))
        self.cc_r.setValue(int(data.get("r_mult", 1.0) * 100))
        self.cc_g.setValue(int(data.get("g_mult", 1.0) * 100))
        self.cc_b.setValue(int(data.get("b_mult", 1.0) * 100))
        glsl = data.get("glsl_shader", "")
        if glsl and os.path.isfile(glsl):
            self.cc_glsl_row.set_path(glsl)
        else:
            self.cc_glsl_row.set_path("")
        # Update RGB label and trigger preview (keep flag true so preset name sticks)
        self._on_color_changed()
        self._cc_loading_preset = False

    def _cc_save_preset(self):
        name, ok = QFileDialog.getSaveFileName(
            self, "Save Color Preset", "", "JSON (*.json)")
        if not name or not ok:
            return
        if not name.endswith(".json"):
            name += ".json"
        cc = self._get_color_config()
        data = {
            "name": os.path.splitext(os.path.basename(name))[0],
            "reverse_enabled": cc.reverse_enabled,
            "yoff_strength": cc.yoff_strength,
            "black_lift": cc.black_lift,
            "brightness": cc.brightness,
            "contrast": cc.contrast,
            "gamma": cc.gamma,
            "lift": cc.lift,
            "gain": cc.gain,
            "hue": cc.hue,
            "saturation": cc.saturation,
            "r_mult": cc.r_mult,
            "g_mult": cc.g_mult,
            "b_mult": cc.b_mult,
            "glsl_shader": cc.glsl_shader,
        }
        try:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            with open(name, "w") as f:
                json.dump(data, f, indent=2)
            self._cc_scan_presets()
            self._st(f"Preset saved: {os.path.basename(name)}")
        except Exception as e:
            self._st(f"Failed to save preset: {e}")

    def _cc_reset_defaults(self):
        """Reset all color correction sliders to Reverse ColorTrans defaults."""
        self._cc_loading_preset = True
        self.cc_reverse.setChecked(True)
        self.cc_yoff.setValue(25)
        self.cc_blift.setValue(20)
        self.cc_brightness.setValue(0)
        self.cc_contrast.setValue(100)
        self.cc_gamma.setValue(100)
        self.cc_lift.setValue(-15)
        self.cc_gain.setValue(175)
        self.cc_hue.setValue(0)
        self.cc_sat.setValue(-22)
        self.cc_r.setValue(100)
        self.cc_g.setValue(100)
        self.cc_b.setValue(100)
        self.cc_glsl_row.set_path("")
        self._cc_glsl_warn.setVisible(False)
        # Select Reverse ColorTrans preset if it exists
        idx = self.cc_preset_cb.findText("Reverse ColorTrans")
        if idx >= 0:
            self.cc_preset_cb.blockSignals(True)
            self.cc_preset_cb.setCurrentIndex(idx)
            self.cc_preset_cb.blockSignals(False)
        self._on_color_changed()
        self._cc_loading_preset = False

    def _cc_save_to_settings(self):
        """Persist current color correction state to settings.json."""
        cc = self._get_color_config()
        ct_data = {
            "enabled": cc.enabled,
            "reverse_enabled": cc.reverse_enabled,
            "yoff_strength": cc.yoff_strength,
            "black_lift": cc.black_lift,
            "brightness": cc.brightness,
            "contrast": cc.contrast,
            "gamma": cc.gamma,
            "lift": cc.lift,
            "gain": cc.gain,
            "hue": cc.hue,
            "saturation": cc.saturation,
            "r_mult": cc.r_mult,
            "g_mult": cc.g_mult,
            "b_mult": cc.b_mult,
            "glsl_shader": cc.glsl_shader,
            "preset": self.cc_preset_cb.currentText(),
        }
        _save_settings(colortrans=ct_data)

    def _cc_restore_from_settings(self):
        """Restore color correction state from settings.json on startup."""
        ct = _COLORTRANS_SETTINGS
        if not ct:
            return
        self._cc_loading_preset = True
        try:
            self.cc_enable.setChecked(ct.get("enabled", False))
            self.cc_reverse.setChecked(ct.get("reverse_enabled", True))
            self.cc_yoff.setValue(int(ct.get("yoff_strength", 0.25) * 100))
            self.cc_blift.setValue(int(ct.get("black_lift", 0.02) * 1000))
            self.cc_brightness.setValue(int(ct.get("brightness", 0.0) * 100))
            self.cc_contrast.setValue(int(ct.get("contrast", 1.0) * 100))
            self.cc_gamma.setValue(int(ct.get("gamma", 1.0) * 100))
            self.cc_lift.setValue(int(ct.get("lift", -0.15) * 100))
            self.cc_gain.setValue(int(ct.get("gain", 1.75) * 100))
            self.cc_hue.setValue(int(ct.get("hue", 0.0)))
            self.cc_sat.setValue(int(ct.get("saturation", -22.5)))
            self.cc_r.setValue(int(ct.get("r_mult", 1.0) * 100))
            self.cc_g.setValue(int(ct.get("g_mult", 1.0) * 100))
            self.cc_b.setValue(int(ct.get("b_mult", 1.0) * 100))
            glsl = ct.get("glsl_shader", "")
            if glsl and os.path.isfile(glsl):
                self.cc_glsl_row.set_path(glsl)
            preset = ct.get("preset", "")
            if preset:
                idx = self.cc_preset_cb.findText(preset)
                if idx >= 0:
                    self.cc_preset_cb.setCurrentIndex(idx)
        finally:
            self._cc_loading_preset = False
        # Update RGB label
        r, g, b = self.cc_r.value() / 100.0, self.cc_g.value() / 100.0, self.cc_b.value() / 100.0
        self._cc_rgb_btn.setText(f"R / G / B:  {r:.2f} / {g:.2f} / {b:.2f}")

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
        # Remove _osd_NNNN-NNNN or _overlay_NNNN-NNNN timestamp suffix variants
        stem = re.sub(r'_(?:osd|overlay)_\d+[-_]\d+$', '', stem)
        # Remove bare _osd or _overlay suffix
        stem = re.sub(r'_(?:osd|overlay)$', '', stem)
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

        if hasattr(self, 'transparent_check') and self.transparent_check.isChecked():
            out_name = f"{stem}_overlay{ts}.mov"
        else:
            out_name = f"{stem}_osd{ts}.mp4"
        return str(p.parent / out_name)

    def _trim_reset(self):
        self.trim_sel.reset()

    def _set_trim_in(self):
        """Set In point to current timeline position."""
        self._player_panel.controller.set_trim_in()

    def _set_trim_out(self):
        """Set Out point to current timeline position."""
        self._player_panel.controller.set_trim_out()

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
                "FFmpeg not found.\n\nRun 'VueOSD.bat' to install it automatically,\n"
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

        # Quality: always bitrate mode
        crf_val      = 23   # kept for ProcessingConfig field; unused
        bitrate_mbps = self.mbps_spin.value()

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
                _req_ext = ".mov" if self.transparent_check.isChecked() else ".mp4"
                if not new_name.lower().endswith(_req_ext):
                    new_name += _req_ext
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
            srt_scale     = self.srt_size_sl.value() / 100.0,
            use_hw        = self.hw_check.isChecked(),
            trim_start    = self.trim_sel.in_pct  * self.video_dur,
            trim_end      = self.trim_sel.out_pct * self.video_dur,
            upscale_target = upscale_target,
            osd_offset_ms = self.osd_offset_sb.value(),
            srt_enabled_fields = self.srt_fields_combo.checked_keys(),
            color_config = self._get_color_config() if self.cc_enable.isChecked() and not self.transparent_check.isChecked() else None,
            transparent_export = self.transparent_check.isChecked(),
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
    app.setApplicationName("VueOSD")
    app.setOrganizationName("VueOSD")
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
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

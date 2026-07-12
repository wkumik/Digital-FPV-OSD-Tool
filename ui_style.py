# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
ui_style.py — Theme tokens, UI scale, stylesheets and icon loading.

The stylesheet constants (APP_STYLE, BTN_SEC, ...) are REBOUND by
_build_styles() whenever the theme or UI scale changes. Import this module
(`import ui_style as S`) and read them through it (`S.BTN_SEC`) — a
`from ui_style import BTN_SEC` freezes the value at import time and the
theme system silently stops working.
"""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap

import theme as _theme_mod   # single source of truth for all colours

_DARK_THEME = True   # module-level flag; toggled by the theme button
_UI_SCALE = 1.0


def set_dark_theme(v):
    global _DARK_THEME
    _DARK_THEME = v


def is_dark():
    return _DARK_THEME


def set_ui_scale(v):
    global _UI_SCALE
    _UI_SCALE = v


def ui_scale():
    return _UI_SCALE


def _fs(n: int) -> int:
    """Scale a font size by the active UI scale factor."""
    return max(6, int(n * _UI_SCALE))


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

# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 OnyxFPV — https://github.com/onyxfpv
"""
theme_editor.py — Colour palette editor for OnyxFPV OSD Tool.

Opens as a non-modal dialog.  Each colour token shows:
  • A clickable swatch  (opens native QColorDialog)
  • A hex text field    (type any #RRGGBB value directly)
  • A live preview pill (updates as you type / pick)

On Apply: saves to theme_custom.json and calls back into main.py to
           rebuild styles and repaint the live UI.
On Reset:  restores factory defaults without saving.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme as _theme

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QSizePolicy, QFrame, QApplication,
    QColorDialog, QToolButton, QGridLayout, QTabWidget,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QSize
from PyQt6.QtGui   import QColor, QFont, QPainter, QBrush, QPen, QIcon


# ─── Small colour swatch widget ───────────────────────────────────────────────

class Swatch(QWidget):
    """A clickable rounded rectangle that shows a colour and opens a picker."""
    clicked = pyqtSignal()

    def __init__(self, color: str = "#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color = color

    def color(self) -> str: return self._color

    def set_color(self, hex_str: str):
        self._color = hex_str
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self._color) if QColor(self._color).isValid() else QColor("#888888")
        p.setBrush(QBrush(c))
        p.setPen(QPen(QColor("#00000040"), 1))
        p.drawRoundedRect(2, 2, self.width()-4, self.height()-4, 5, 5)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


# ─── One row: token label + swatch + hex input ────────────────────────────────

class ColourRow(QWidget):
    changed = pyqtSignal(str, str)   # (token_key, new_hex)

    def __init__(self, key: str, label: str, hex_val: str, parent=None):
        super().__init__(parent)
        self._key = key
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(34)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        # Label
        lbl = QLabel(label)
        lbl.setFixedWidth(188)
        lbl.setStyleSheet("color:#c0c0bc;font-size:11px;")
        row.addWidget(lbl)

        # Swatch
        self.swatch = Swatch(hex_val)
        self.swatch.clicked.connect(self._pick)
        row.addWidget(self.swatch)

        # Hex input
        self.edit = QLineEdit(hex_val)
        self.edit.setFixedWidth(82)
        self.edit.setMaxLength(9)
        self.edit.setFont(QFont("Consolas,Courier New", 11))
        self.edit.setStyleSheet(
            "background:#1a1a28;color:#e0e0d8;"
            "border:1px solid #404050;border-radius:4px;padding:2px 6px;"
        )
        self.edit.textChanged.connect(self._on_text)
        row.addWidget(self.edit)

        row.addStretch()

    # ── public api ────────────────────────────────────────────────────────────
    def get_value(self) -> str:
        return self.swatch.color()

    def set_value(self, hex_val: str):
        self.swatch.set_color(hex_val)
        self.edit.blockSignals(True)
        self.edit.setText(hex_val)
        self.edit.blockSignals(False)

    # ── internal ─────────────────────────────────────────────────────────────
    def _pick(self):
        initial = QColor(self.swatch.color())
        c = QColorDialog.getColor(initial, self, "Pick colour",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            hex_val = c.name()  # always #RRGGBB
            self.swatch.set_color(hex_val)
            self.edit.blockSignals(True)
            self.edit.setText(hex_val)
            self.edit.blockSignals(False)
            self.changed.emit(self._key, hex_val)

    def _on_text(self, text: str):
        if not text.startswith("#"):
            text = "#" + text
        c = QColor(text)
        if c.isValid():
            self.edit.setStyleSheet(
                "background:#1a1a28;color:#e0e0d8;"
                "border:1px solid #404050;border-radius:4px;padding:2px 6px;"
            )
            self.swatch.set_color(text)
            self.changed.emit(self._key, text)
        else:
            self.edit.setStyleSheet(
                "background:#1a1a28;color:#f38ba8;"
                "border:1px solid #f38ba8;border-radius:4px;padding:2px 6px;"
            )


# ─── One palette column ───────────────────────────────────────────────────────

class PalettePanel(QWidget):
    """Scrollable list of colour rows grouped by section, for one theme variant."""

    def __init__(self, palette: dict, title: str, parent=None):
        super().__init__(parent)
        self._rows: dict[str, ColourRow] = {}
        self._palette = dict(palette)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Column title
        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color:#e8e8e0;padding:0 0 10px 0;")
        outer.addWidget(title_lbl)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background:transparent;border:none;")
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(content)
        vl.setContentsMargins(0, 0, 8, 0)
        vl.setSpacing(2)
        scroll.setWidget(content)

        for group_name, keys in _theme.TOKEN_GROUPS:
            # Section header
            sep = QLabel(group_name.upper())
            sep.setStyleSheet(
                "color:#606070;font-size:9px;font-weight:bold;letter-spacing:1px;"
                "padding:10px 0 4px 0;"
            )
            vl.addWidget(sep)

            # Thin rule
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color:#303040;max-height:1px;")
            vl.addWidget(line)

            for key in keys:
                label = _theme.TOKEN_LABELS.get(key, key)
                val   = palette.get(key, "#888888")
                row   = ColourRow(key, label, val)
                row.changed.connect(self._on_changed)
                self._rows[key] = row
                vl.addWidget(row)

        vl.addStretch()

    def _on_changed(self, key: str, val: str):
        self._palette[key] = val

    def get_palette(self) -> dict:
        return {k: r.get_value() for k, r in self._rows.items()}

    def reset_to(self, defaults: dict):
        for key, row in self._rows.items():
            if key in defaults:
                row.set_value(defaults[key])
        self._palette = dict(defaults)


# ─── Live preview strip ───────────────────────────────────────────────────────

class PreviewStrip(QWidget):
    """Shows a row of fake UI elements using the currently-edited palette."""

    def __init__(self, get_dark_fn, get_light_fn, parent=None):
        super().__init__(parent)
        self._get_dark  = get_dark_fn
        self._get_light = get_light_fn
        self.setFixedHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def refresh(self):
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        half = w // 2 - 6

        def _draw_side(x_off: int, pal: dict, label: str):
            # bg
            p.setBrush(QBrush(QColor(pal.get("bg", "#1e1e2e"))))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x_off, 4, half, h-8, 6, 6)

            # surface pill (button-like)
            p.setBrush(QBrush(QColor(pal.get("surface", "#313244"))))
            p.drawRoundedRect(x_off+8, 12, 60, 22, 5, 5)
            p.setPen(QPen(QColor(pal.get("text", "#cdd6f4"))))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(x_off+8, 12, 60, 22, Qt.AlignmentFlag.AlignCenter, "Button")

            # accent pill (primary button)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(pal.get("accent", "#89b4fa"))))
            p.drawRoundedRect(x_off+76, 12, 70, 22, 5, 5)
            p.setPen(QPen(QColor("#ffffff")))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.drawText(x_off+76, 12, 70, 22, Qt.AlignmentFlag.AlignCenter, "Render")

            # text
            p.setPen(QPen(QColor(pal.get("text", "#cdd6f4"))))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(x_off+154, 12, 60, 10, Qt.AlignmentFlag.AlignLeft, "OSD Tool")
            p.setPen(QPen(QColor(pal.get("muted", "#6c7086"))))
            p.drawText(x_off+154, 26, 70, 10, Qt.AlignmentFlag.AlignLeft, "Ready")

            # status dots
            for i, col_key in enumerate(["green", "red", "orange"]):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(pal.get(col_key, "#888"))))
                p.drawEllipse(x_off + half - 42 + i*14, 19, 9, 9)

            # label
            p.setPen(QPen(QColor(pal.get("subtext", "#888"))))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(x_off, h-10, half, 10, Qt.AlignmentFlag.AlignCenter, label)

        _draw_side(0,        self._get_dark(),  "Dark theme")
        _draw_side(half+12,  self._get_light(), "Light theme")
        p.end()


# ─── Main editor dialog ───────────────────────────────────────────────────────

class ThemeEditor(QDialog):
    """
    Non-modal colour editor dialog.
    Emits `applied` when the user clicks Apply — main.py connects this
    to rebuild styles and repaint.
    """
    applied = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theme Editor — OnyxFPV OSD Tool")
        self.setMinimumSize(860, 640)
        self.resize(920, 720)
        self.setStyleSheet("""
            QDialog, QWidget { background: #12121e; color: #d0d0c8; font-family: 'Segoe UI', Arial; }
            QScrollBar:vertical { background: #1a1a2a; width: 6px; border-radius: 3px; }
            QScrollBar::handle:vertical { background: #40405a; border-radius: 3px; }
            QTabWidget::pane { border: 1px solid #30304a; border-radius: 6px; }
            QTabBar::tab { background: #1e1e30; color: #808090; padding: 6px 18px;
                           border-radius: 4px 4px 0 0; font-size: 11px; }
            QTabBar::tab:selected { background: #2a2a40; color: #e0e0d8; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Theme Editor")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color:#e8e8e0;")
        sub = QLabel("Click a swatch or type a hex value  •  Changes preview live")
        sub.setStyleSheet("color:#505060;font-size:11px;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(sub)
        root.addLayout(hdr)

        # ── Live preview ──────────────────────────────────────────────────────
        self.preview = PreviewStrip(
            lambda: self.dark_panel.get_palette(),
            lambda: self.light_panel.get_palette(),
        )
        self.preview.setStyleSheet("background:#0e0e18;border-radius:6px;")
        root.addWidget(self.preview)

        # ── Two palette columns ───────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(20)

        self.dark_panel  = PalettePanel(_theme.get_dark(),  "🌙  Dark theme")
        self.light_panel = PalettePanel(_theme.get_light(), "☀️  Light theme")

        # Wire any change → preview refresh
        for row in list(self.dark_panel._rows.values()) + list(self.light_panel._rows.values()):
            row.changed.connect(lambda *_: self.preview.refresh())

        cols.addWidget(self.dark_panel,  1)

        # Vertical divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("color:#303040;")
        cols.addWidget(div)

        cols.addWidget(self.light_panel, 1)
        root.addLayout(cols, 1)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_ss = lambda bg, fg="#e8e8e0": (
            f"QPushButton{{background:{bg};color:{fg};border:none;border-radius:6px;"
            f"padding:8px 22px;font-size:12px;font-weight:bold;}}"
            f"QPushButton:hover{{opacity:0.85;}}"
        )
        foot = QHBoxLayout()
        foot.setSpacing(8)

        reset_btn = QPushButton("↺  Reset to defaults")
        reset_btn.setStyleSheet(
            "QPushButton{background:#1e1e30;color:#808090;border:1px solid #303040;"
            "border-radius:6px;padding:8px 18px;font-size:11px;}"
            "QPushButton:hover{background:#2a2a40;color:#c0c0b8;}"
        )
        reset_btn.clicked.connect(self._reset)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton{background:#1e1e30;color:#a0a0a0;border:1px solid #303040;"
            "border-radius:6px;padding:8px 20px;font-size:12px;}"
            "QPushButton:hover{background:#2a2a40;}"
        )
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("✓  Apply & Save")
        apply_btn.setStyleSheet(
            "QPushButton{background:#4a7cf0;color:#ffffff;border:none;"
            "border-radius:6px;padding:8px 24px;font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#3a6ce0;}"
            "QPushButton:pressed{background:#2a5cd0;}"
        )
        apply_btn.clicked.connect(self._apply)

        foot.addWidget(reset_btn)
        foot.addStretch()
        foot.addWidget(cancel_btn)
        foot.addWidget(apply_btn)
        root.addLayout(foot)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _apply(self):
        dark  = self.dark_panel.get_palette()
        light = self.light_panel.get_palette()
        _theme.save(dark, light)
        self.applied.emit()

    def _reset(self):
        self.dark_panel.reset_to(_theme.DARK_DEFAULT)
        self.light_panel.reset_to(_theme.LIGHT_DEFAULT)
        self.preview.refresh()

    def reload_from_theme(self):
        """Called after external theme reload to sync panel state."""
        self.dark_panel.reset_to(_theme.get_dark())
        self.light_panel.reset_to(_theme.get_light())
        self.preview.refresh()

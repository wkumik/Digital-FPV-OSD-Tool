# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
ui_widgets.py — Generic, app-agnostic Qt widgets used across the VueOSD UI.

Contains: FileRow, DropZone, LabeledSlider, CheckableComboBox,
CollapsibleGroupBox, InfoCard.

Style constants are rebound by ui_style._build_styles() on every theme/scale
change, so they are read through the module at use time (S.BTN_SEC), never
imported by name.
"""

import os
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGroupBox, QSlider, QComboBox, QGridLayout, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel

import ui_style as S
from ui_style import _T, _icon


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
        self.path_lbl.setStyleSheet(S.PATH_EMPTY)
        self.path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.path_lbl.setFixedHeight(28)
        self.path_lbl.setMinimumWidth(60)
        self.path_lbl.setTextFormat(Qt.TextFormat.PlainText)

        self.btn = QPushButton("Save As" if save_mode else "Browse")
        self.btn.setFixedSize(68, 28)
        self.btn.setStyleSheet(S.BTN_SEC)
        self.btn.clicked.connect(self._browse)

        self.clr = QPushButton("✕")
        self.clr.setFixedSize(28, 28)
        self.clr.setStyleSheet(S.BTN_DANGER)
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
            self.path_lbl.setStyleSheet(S.PATH_FILLED)
            self.path_lbl.setToolTip(path)
            self.clr.setVisible(True)
        else:
            self.path_lbl.setText("No file selected")
            self.path_lbl.setStyleSheet(S.PATH_EMPTY)
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
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(72)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._lbl = QLabel("Drop  .mp4 · .ts · .osd · .srt  here")
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

    _VALID_EXTS = {'.mp4', '.ts', '.mkv', '.avi', '.mov', '.osd', '.srt'}

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
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        paths = [p for p in paths
                 if p and os.path.splitext(p)[1].lower() in self._VALID_EXTS]
        if paths:
            self.files_dropped.emit(paths)
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
        self.sl.setStyleSheet(S.SLIDER_STYLE)

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


class CollapsibleGroupBox(QGroupBox):
    """A QGroupBox whose title bar collapses/expands its contents on click.

    Stays a real QGroupBox so S.GROUP_STYLE, the theme-restyle pass, and the
    usual "add widgets to its layout" pattern all keep working unchanged — only
    the class name at the call site differs. Click the title (the ▾/▸ arrow
    marks the state); content is simply hidden so the box shrinks to its title.
    """
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title_text = title
        self._expanded = True
        self._sync_title()

    def _sync_title(self):
        super().setTitle(f"{'▾' if self._expanded else '▸'}  {self._title_text}")

    def setTitle(self, title: str):
        self._title_text = title
        self._sync_title()

    def titleText(self) -> str:
        return self._title_text

    def isExpanded(self) -> bool:
        return self._expanded

    def setExpanded(self, expanded: bool):
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._sync_title()
        for child in self.findChildren(
                QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            child.setVisible(expanded)
        self.updateGeometry()

    def _label_rect(self):
        """Vertical extent of the title label, for hit-testing the header."""
        try:
            from PyQt6.QtWidgets import QStyle, QStyleOptionGroupBox
            opt = QStyleOptionGroupBox()
            self.initStyleOption(opt)
            return self.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox, opt,
                QStyle.SubControl.SC_GroupBoxLabel, self)
        except Exception:
            return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            y = ev.position().toPoint().y()
            lr = self._label_rect()
            in_header = (lr.top() <= y <= lr.bottom()) if lr is not None \
                else (y <= self.fontMetrics().height() + 6)
            if in_header:
                self.setExpanded(not self._expanded)
                ev.accept()
                return
        super().mousePressEvent(ev)


class InfoCard(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(S.GROUP_STYLE)
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

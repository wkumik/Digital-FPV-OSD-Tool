# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_canvas_edit.py — Mouse-driven editors owned by VideoCanvas.

Contains: RegionEditor (hide-region draw/remove) and WidgetEditor (widget
select/drag/resize). Both are plain helper objects held by the canvas, not
QWidgets: they paint through the canvas's QPainter and emit through the
canvas's signals via a back-reference.

Precedence: the canvas offers each event to the widget editor first, then the
region editor — the same order the two used to sit in as if/elif branches.
"""

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen


class RegionEditor:
    """Hide-region overlay: draw-to-add and click-to-remove rectangles.

    Geometry conversion between canvas pixels and OSD grid cells is supplied
    by MainWindow through set_callbacks(), so the canvas stays agnostic of
    font/OSD state.
    """

    def __init__(self, canvas):
        self._canvas = canvas
        # Modes: None = off, "add" = draw-to-add, "remove" = click-to-remove.
        self.mode: str | None = None
        self.regions: list[tuple[int, int, int, int]] = []
        # px_to_cell(x, y)      → (row, col) | None     (canvas px → OSD grid cell)
        # cell_to_px(r0,c0,r1,c1) → (x, y, w, h) | None (grid rect → canvas px rect)
        self.px_to_cell = None
        self.cell_to_px = None
        self._drag_start: tuple[int, int] | None = None  # canvas px
        self._drag_cur: tuple[int, int] | None = None
        self._hover_idx: int = -1

    # ── state ─────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str | None):
        self.mode = mode
        self._drag_start = None
        self._drag_cur = None

    def set_callbacks(self, px_to_cell, cell_to_px):
        self.px_to_cell = px_to_cell
        self.cell_to_px = cell_to_px

    def set_regions(self, regions):
        self.regions = list(regions or [])

    # ── painting ──────────────────────────────────────────────────────────────

    def paint(self, p: QPainter, t: dict):
        """Draw existing hide regions + the in-progress drag rectangle."""
        if not self.regions and self._drag_start is None and self.mode is None:
            return
        cell_to_px = self.cell_to_px
        # Saved regions
        if cell_to_px is not None:
            for i, reg in enumerate(self.regions):
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
                                 220 if i == self._hover_idx else 180)
                pen = QPen(pen_col, 2 if i == self._hover_idx else 1)
                pen.setStyle(Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)
        # In-progress drag rectangle
        if self.mode == "add" and self._drag_start and self._drag_cur:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_cur
            rx = min(x0, x1); ry = min(y0, y1)
            rw = abs(x1 - x0); rh = abs(y1 - y0)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 200, 80, 60))
            p.drawRect(rx, ry, rw, rh)
            pen = QPen(QColor(255, 200, 80, 230), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rx, ry, rw, rh)

    def hit_test(self, x: int, y: int) -> int:
        """Return index of region whose painted rect contains (x, y), else -1."""
        if not self.regions or self.cell_to_px is None:
            return -1
        for i, reg in enumerate(self.regions):
            rect = self.cell_to_px(*reg)
            if rect is None:
                continue
            rx, ry, rw, rh = rect
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return -1

    # ── mouse (each returns True when it handled the event) ───────────────────

    def mouse_press(self, event, pos) -> bool:
        if self.mode == "add" and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (pos.x(), pos.y())
            self._drag_cur   = (pos.x(), pos.y())
            self._canvas.update()
            return True
        if self.mode == "remove" and event.button() == Qt.MouseButton.LeftButton:
            idx = self.hit_test(pos.x(), pos.y())
            if idx >= 0:
                self._canvas.regionRemoveRequested.emit(idx)
            return True
        return False

    def mouse_move(self, event, pos) -> bool:
        if self.mode == "add" and self._drag_start is not None:
            self._drag_cur = (pos.x(), pos.y())
            self._canvas.update()
            return True
        if self.mode == "remove":
            new_idx = self.hit_test(pos.x(), pos.y())
            if new_idx != self._hover_idx:
                self._hover_idx = new_idx
                self._canvas.update()
            return True
        # Not in a region mode: clear any stale hover, then let the canvas
        # carry on with its placeholder handling.
        if self._hover_idx != -1:
            self._hover_idx = -1
            self._canvas.update()
        return False

    def mouse_release(self, event, pos) -> bool:
        if self.mode == "add" and self._drag_start is not None \
                and event.button() == Qt.MouseButton.LeftButton:
            x0, y0 = self._drag_start
            x1, y1 = pos.x(), pos.y()
            self._drag_start = None
            self._drag_cur = None
            if abs(x1 - x0) >= 3 and abs(y1 - y0) >= 3 \
                    and self.px_to_cell is not None:
                c_a = self.px_to_cell(x0, y0)
                c_b = self.px_to_cell(x1, y1)
                if c_a is not None and c_b is not None:
                    r0 = min(c_a[0], c_b[0]); r1 = max(c_a[0], c_b[0])
                    c0 = min(c_a[1], c_b[1]); c1 = max(c_a[1], c_b[1])
                    self._canvas.regionAdded.emit((r0, c0, r1, c1))
            self._canvas.update()
            return True
        return False


class WidgetEditor:
    """Widget select / drag / resize chrome on the canvas.

    ``widgets`` mirrors MainWindow's widget list (rects in normalised source
    coords). ``src_rect_fn`` returns the SOURCE video area in canvas pixels so
    we can convert between normalised (0..1) and pixel space.
    """

    _RESIZE_MARGIN = 7   # px band around a border that grabs a resize handle

    def __init__(self, canvas):
        self._canvas = canvas
        # edit = False -> widgets render via overlay only (no chrome).
        # edit = True  -> selection chrome drawn, mouse drags widgets.
        self.edit: bool = False
        self.widgets: list = []          # list[widgets.Widget]
        self.selected: int = -1
        self.hover: int = -1
        self.src_rect_fn = None          # () -> (sx, sy, sw, sh) or None
        self._drag_offset: tuple[int, int] | None = None  # (dx, dy) px within widget
        self._drag_idx: int = -1
        # Edge/corner resize of the selected widget. _resize_edge is a combo of
        # 't'/'b' and 'l'/'r' (e.g. 'tl', 'r'); _resize_bounds is the widget's
        # [left, top, right, bottom] in normalised source coords, with only the
        # dragged edges moving so the opposite edge stays pinned.
        self._resize_idx: int = -1
        self._resize_edge: str | None = None
        self._resize_bounds: list[float] | None = None

    # ── state ─────────────────────────────────────────────────────────────────

    def set_edit(self, enabled: bool):
        self.edit = bool(enabled)
        self._drag_offset = None
        self._drag_idx = -1

    def set_widgets(self, widgets):
        self.widgets = list(widgets or [])
        # Drop stale selection if list shrank
        if self.selected >= len(self.widgets):
            self.selected = -1

    def set_selected(self, idx: int) -> bool:
        """Set the highlighted widget. Returns True when the canvas should repaint."""
        if idx == self.selected:
            return False
        self.selected = int(idx) if 0 <= idx < len(self.widgets) else -1
        return True

    # ── geometry ──────────────────────────────────────────────────────────────

    def src_rect_px(self):
        """Source video rect in canvas px, or None if unknown."""
        fn = self.src_rect_fn
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

    def rect_px(self, w):
        """Resolve a widget's pixel rect on the canvas, or None."""
        src = self.src_rect_px()
        if src is None:
            return None
        sx, sy, sw, sh = src
        ww = max(1, int(round(w.w * sw)))
        wh = max(1, int(round(w.h * sh)))
        cx = sx + int(round(w.x * sw))
        cy = sy + int(round(w.y * sh))
        return cx - ww // 2, cy - wh // 2, ww, wh

    def hit_test(self, x: int, y: int) -> int:
        """Return index of widget under (x, y) — selected first, then topmost."""
        # Selected widget wins (keeps drag responsive when widgets overlap)
        sel = self.selected
        if 0 <= sel < len(self.widgets):
            r = self.rect_px(self.widgets[sel])
            if r is not None:
                rx, ry, rw, rh = r
                if rx <= x < rx + rw and ry <= y < ry + rh:
                    return sel
        for i in range(len(self.widgets) - 1, -1, -1):
            r = self.rect_px(self.widgets[i])
            if r is None:
                continue
            rx, ry, rw, rh = r
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return -1

    def resize_edge_at(self, w, x: int, y: int) -> "str | None":
        """Return the resize edge under (x, y) for widget ``w`` — a combo of
        't'/'b' and 'l'/'r' (corners), a single side, or None when the cursor
        isn't on a border."""
        r = self.rect_px(w)
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
    def resize_cursor(edge: str):
        if edge in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if edge in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        if edge in ("l", "r"):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    # ── painting ──────────────────────────────────────────────────────────────

    def paint(self, p: QPainter):
        """Selection border + hover outline for widgets in edit mode."""
        if not self.widgets:
            return
        for i, w in enumerate(self.widgets):
            r = self.rect_px(w)
            if r is None:
                continue
            rx, ry, rw, rh = r
            qr = QRect(rx, ry, rw, rh)
            if i == self.selected:
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
            elif i == self.hover:
                pen = QPen(QColor(200, 200, 200, 180), 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)
            else:
                pen = QPen(QColor(180, 180, 180, 120), 1, Qt.PenStyle.DotLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(qr)

    # ── mouse (each returns True when it handled the event) ───────────────────

    def mouse_press(self, event, pos) -> bool:
        if not (self.edit and event.button() == Qt.MouseButton.LeftButton):
            return False
        c = self._canvas
        # Resize takes priority: if the press lands on a border of the
        # already-selected widget, drag that edge instead of moving.
        sel = self.selected
        if 0 <= sel < len(self.widgets):
            edge = self.resize_edge_at(self.widgets[sel], pos.x(), pos.y())
            if edge:
                w = self.widgets[sel]
                self._resize_idx = sel
                self._resize_edge = edge
                self._resize_bounds = [w.x - w.w / 2, w.y - w.h / 2,
                                       w.x + w.w / 2, w.y + w.h / 2]
                c.setCursor(self.resize_cursor(edge))
                c.update()
                return True
        idx = self.hit_test(pos.x(), pos.y())
        if idx != self.selected:
            self.selected = idx
            c.widgetSelected.emit(idx)
        if idx >= 0:
            r = self.rect_px(self.widgets[idx])
            if r is not None:
                rx, ry, _rw, _rh = r
                self._drag_offset = (pos.x() - rx, pos.y() - ry)
                self._drag_idx = idx
                c.setCursor(Qt.CursorShape.ClosedHandCursor)
        c.update()
        return True

    def mouse_move(self, event, pos) -> bool:
        if not self.edit:
            return False
        c = self._canvas
        # Active resize — drag one edge/corner, keep the opposite pinned.
        if self._resize_idx >= 0 and self._resize_bounds is not None:
            src = self.src_rect_px()
            if src is not None and self._resize_idx < len(self.widgets):
                sx, sy, sw, sh = src
                ux = max(0.0, min(1.0, (pos.x() - sx) / sw))
                uy = max(0.0, min(1.0, (pos.y() - sy) / sh))
                l, t, r, b = self._resize_bounds
                e, MIN = self._resize_edge, 0.03
                if "l" in e: l = min(ux, r - MIN)
                if "r" in e: r = max(ux, l + MIN)
                if "t" in e: t = min(uy, b - MIN)
                if "b" in e: b = max(uy, t + MIN)
                self._resize_bounds = [l, t, r, b]
                w = self.widgets[self._resize_idx]
                w.x, w.y, w.w, w.h = (l + r) / 2, (t + b) / 2, r - l, b - t
                c.widgetMoved.emit(self._resize_idx, w.x, w.y, w.w, w.h)
                c.update()
            return True
        # Active drag — move the widget, normalise back to source coords,
        # emit a continuous move signal so the host can refresh the preview.
        if self._drag_idx >= 0 and self._drag_offset is not None:
            src = self.src_rect_px()
            if src is not None and self._drag_idx < len(self.widgets):
                sx, sy, sw, sh = src
                w = self.widgets[self._drag_idx]
                dx, dy = self._drag_offset
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
                c.widgetMoved.emit(self._drag_idx, nx, ny, w.w, w.h)
                c.update()
            return True
        # Hover — resize cursor on the selected widget's border wins, else
        # a light dashed outline on whichever widget is under the cursor.
        sel = self.selected
        if 0 <= sel < len(self.widgets):
            edge = self.resize_edge_at(self.widgets[sel], pos.x(), pos.y())
            if edge:
                c.setCursor(self.resize_cursor(edge))
                return True
        new_idx = self.hit_test(pos.x(), pos.y())
        if new_idx != self.hover:
            self.hover = new_idx
            c.update()
        c.setCursor(Qt.CursorShape.OpenHandCursor if new_idx >= 0
                    else Qt.CursorShape.ArrowCursor)
        return True

    def mouse_release(self, event, pos) -> bool:
        c = self._canvas
        if self.edit and self._resize_idx >= 0 \
                and event.button() == Qt.MouseButton.LeftButton:
            idx = self._resize_idx
            self._resize_idx = -1
            self._resize_edge = None
            self._resize_bounds = None
            if 0 <= idx < len(self.widgets):
                w = self.widgets[idx]
                c.widgetCommit.emit(idx, w.x, w.y, w.w, w.h)
            c.update()
            return True
        if self.edit and self._drag_idx >= 0 \
                and event.button() == Qt.MouseButton.LeftButton:
            idx = self._drag_idx
            self._drag_idx = -1
            self._drag_offset = None
            c.setCursor(Qt.CursorShape.OpenHandCursor)
            if 0 <= idx < len(self.widgets):
                w = self.widgets[idx]
                # widgetCommit lets the host persist + do a full-quality refresh
                c.widgetCommit.emit(idx, w.x, w.y, w.w, w.h)
            c.update()
            return True
        return False

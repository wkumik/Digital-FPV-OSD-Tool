# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_widget_draw.py - Draw functions for non-map widget types.

Each function renders one widget type onto a PIL RGBA image. They share the
same signature:  fn(img, w, value, unit, default_fmt, rx, ry, ww, wh)

Not part of the public API — imported by widgets.py only.
"""

from __future__ import annotations

import math
from typing import Any

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

from _widget_primitives import (
    _parse_color, _font, _format_value, _draw_text_with_shadow, _polar,
)


def _draw_digital(img: Any, w: Any,
                  value: Any, unit: str, default_fmt: str,
                  rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    color    = _parse_color(style.get("color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("opacity", 1.0))))
    bg_alpha = int(255 * float(style.get("bg_opacity", 0.0)))
    bg_color = _parse_color(style.get("bg_color", "#000000"), alpha=bg_alpha)
    label    = str(style.get("label", ""))
    fmt      = str(style.get("fmt", default_fmt))
    show_unit = bool(style.get("show_unit", True))
    align    = str(style.get("align", "center"))
    shadow   = bool(style.get("shadow", True))

    val_str = _format_value(value, fmt, default_fmt)
    parts: list[str] = []
    if label:
        parts.append(label)
    parts.append(val_str)
    if show_unit and unit:
        parts.append(unit)
    text = " ".join(parts)

    font = _font(int(wh * float(style.get("font_scale", 0.85))))
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    if align == "left":
        tx = rx
    elif align == "right":
        tx = rx + ww - tw
    else:
        tx = rx + (ww - tw) // 2
    ty = ry + (wh - th) // 2 - bb[1]

    if bg_color[3] > 0:
        pad = max(2, int(wh * 0.10))
        draw.rounded_rectangle(
            [tx - pad, ty + bb[1] - pad, tx + tw + pad, ty + bb[1] + th + pad],
            radius=max(2, int(wh * 0.12)), fill=bg_color,
        )

    if shadow:
        _draw_text_with_shadow(draw, (tx, ty), text, font, fill=color,
                               shadow_alpha=int(style.get("shadow_alpha", 130)))
    else:
        draw.text((tx, ty), text, font=font, fill=color)


def _draw_bar(img: Any, w: Any,
              value: Any, unit: str, default_fmt: str,
              rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    color    = _parse_color(style.get("color", "#3FA9F5"),
                            alpha=int(255 * float(style.get("opacity", 0.80))))
    bg_color = _parse_color(style.get("bg_color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("bg_opacity", 0.18))))
    orient     = str(style.get("orientation", "horizontal"))
    show_value = bool(style.get("show_value", True))

    draw = ImageDraw.Draw(img)

    if wh < 4 or ww < 4:
        return

    if orient == "vertical":
        cap_h = int(wh * 0.18) if show_value else 0
        gap_above_cap = 2 if cap_h > 0 else 0
        if cap_h + gap_above_cap > wh - 2:
            cap_h = 0
            gap_above_cap = 0
        bar_top = ry
        bar_bot = ry + wh - cap_h - gap_above_cap
        bar_left = rx
        bar_right = rx + ww
        bar_h = max(2, bar_bot - bar_top)
        radius = max(1, int(min(ww, bar_h) * 0.20))
        if bar_bot > bar_top:
            draw.rounded_rectangle([bar_left, bar_top, bar_right, bar_bot],
                                   radius=radius, fill=bg_color)
            fill_h = int(bar_h * frac)
            if fill_h > 0:
                draw.rounded_rectangle(
                    [bar_left, bar_bot - fill_h, bar_right, bar_bot],
                    radius=radius, fill=color,
                )
        cap_y0 = bar_bot + gap_above_cap
        cap_y1 = ry + wh
        text_anchor = (cap_y0, cap_y1)
        draw_value = show_value and cap_h > 0
    else:
        if show_value:
            text_h = max(10, int(wh * 0.45))
            gap    = max(1, int(wh * 0.08))
            if text_h + gap > wh - 2:
                text_h = max(0, wh - 2 - gap)
                if text_h < 6:
                    text_h = 0
                    gap = 0
            bar_top = ry + text_h + gap
        else:
            text_h = 0
            gap    = 0
            bar_top = ry
        bar_bot  = ry + wh
        bar_h    = max(2, bar_bot - bar_top)
        radius   = max(1, int(min(ww, bar_h) * 0.30))
        if bar_bot > bar_top:
            draw.rounded_rectangle([rx, bar_top, rx + ww, bar_bot],
                                   radius=radius, fill=bg_color)
            fill_w = int(ww * frac)
            if fill_w > 0:
                draw.rounded_rectangle([rx, bar_top, rx + fill_w, bar_bot],
                                       radius=radius, fill=color)
        text_anchor = (ry, ry + text_h)
        draw_value = show_value and text_h > 0

    if draw_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        if bool(style.get("show_unit", True)) and unit:
            val_str = f"{val_str} {unit}"
        cap_y0, cap_y1 = text_anchor
        cap_h = max(1, cap_y1 - cap_y0)
        font = _font(int(cap_h * float(style.get("font_scale", 0.85))))
        bb = draw.textbbox((0, 0), val_str, font=font)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        align = str(style.get("align", "left"))
        if align == "right":
            tx = rx + ww - tw
        elif align == "center":
            tx = rx + (ww - tw) // 2
        else:
            tx = rx
        ty = cap_y0 + (cap_h - th) // 2 - bb[1]
        _draw_text_with_shadow(draw, (tx, ty), val_str, font,
                               fill=_parse_color(style.get("text_color", "#FFFFFF")),
                               shadow_alpha=int(style.get("shadow_alpha", 110)))


def _draw_gauge(img: Any, w: Any,
                value: Any, unit: str, default_fmt: str,
                rx: int, ry: int, ww: int, wh: int) -> None:
    style = w.style
    if value is None:
        return

    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    color    = _parse_color(style.get("color", "#3FA9F5"),
                            alpha=int(255 * float(style.get("opacity", 0.90))))
    track    = _parse_color(style.get("bg_color", "#FFFFFF"),
                            alpha=int(255 * float(style.get("bg_opacity", 0.14))))
    start_angle   = float(style.get("start_angle", 135.0))
    span_angle    = float(style.get("span_angle", 270.0))
    thickness_pct = float(style.get("thickness", 0.10))
    show_value    = bool(style.get("show_value", True))
    rounded_caps  = bool(style.get("rounded_caps", True))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    size_s   = min(pw, ph)
    cx_s     = pw // 2
    cy_s     = ph // 2
    half_s   = size_s // 2
    gx0_s    = cx_s - half_s
    gy0_s    = cy_s - half_s
    gx1_s    = gx0_s + size_s
    gy1_s    = gy0_s + size_s
    thick_s  = max(2 * SS, int(size_s * thickness_pct))

    end_angle = start_angle + span_angle
    pdraw.arc([gx0_s, gy0_s, gx1_s, gy1_s], start_angle, end_angle,
              fill=track, width=thick_s)
    v_end = start_angle + span_angle * frac
    if frac > 0.001:
        pdraw.arc([gx0_s, gy0_s, gx1_s, gy1_s], start_angle, v_end,
                  fill=color, width=thick_s)
        if rounded_caps:
            arc_r = ((gx1_s - gx0_s) - thick_s) / 2.0
            cap_r = thick_s / 2.0
            acx   = (gx0_s + gx1_s) / 2.0
            acy   = (gy0_s + gy1_s) / 2.0
            for ang_deg in (start_angle, v_end):
                rad = math.radians(ang_deg)
                px  = acx + arc_r * math.cos(rad)
                py  = acy + arc_r * math.sin(rad)
                pdraw.ellipse([px - cap_r, py - cap_r,
                               px + cap_r, py + cap_r], fill=color)

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        font = _font(int(size_s * float(style.get("font_scale", 0.32))))
        bb = pdraw.textbbox((0, 0), val_str, font=font)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        tx = cx_s - tw // 2
        ty = cy_s - th // 2 - bb[1] - int(size_s * (0.04 if unit else 0.0))
        text_color = _parse_color(style.get("text_color", "#FFFFFF"))
        pdraw.text((tx, ty), val_str, font=font, fill=text_color)
        if bool(style.get("show_unit", True)) and unit:
            ufont = _font(int(size_s * float(style.get("unit_scale", 0.13))))
            ubb = pdraw.textbbox((0, 0), unit, font=ufont)
            uw = ubb[2] - ubb[0]
            utx = cx_s - uw // 2
            uty = ty + th + max(2, int(size_s * 0.015))
            pdraw.text((utx, uty), unit, font=ufont,
                       fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)


def _draw_semicircle(img: Any, w: Any,
                     value: Any, unit: str, default_fmt: str,
                     rx: int, ry: int, ww: int, wh: int) -> None:
    """Top half-arc with a perpendicular needle marker that slides along the arc."""
    style = w.style
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    color        = _parse_color(style.get("color", "#FFFFFF"),
                                alpha=int(255 * float(style.get("opacity", 0.95))))
    track        = _parse_color(style.get("bg_color", "#FFFFFF"),
                                alpha=int(255 * float(style.get("bg_opacity", 0.25))))
    needle_color = _parse_color(style.get("needle_color", "#FF3030"))
    thickness_pct = float(style.get("thickness", 0.07))
    show_value    = bool(style.get("show_value", True))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    margin = max(2 * SS, int(min(pw, ph) * 0.06))
    cx = pw / 2.0
    cy = ph - margin
    r  = min(pw / 2.0 - margin, cy - margin)
    if r < 4 * SS:
        return
    thick = max(2 * SS, int(r * thickness_pct * 2))

    bb = [cx - r, cy - r, cx + r, cy + r]
    pdraw.arc(bb, 180, 360, fill=track, width=thick)
    end_angle = 180.0 + 180.0 * frac
    if frac > 0.001:
        pdraw.arc(bb, 180, end_angle, fill=color, width=thick)

    inner_r = r - thick / 2.0 - SS
    outer_r = r + thick / 2.0 + SS * 1.5
    ix, iy = _polar(cx, cy, inner_r, end_angle)
    ox, oy = _polar(cx, cy, outer_r, end_angle)
    pdraw.line([(ix, iy), (ox, oy)],
               fill=needle_color, width=max(2 * SS, int(thick * 0.55)))

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        has_unit = bool(style.get("show_unit", True)) and unit
        val_center_y = cy - r * (0.50 if has_unit else 0.40)
        font = _font(int(r * float(style.get("font_scale", 0.50))))
        tbb = pdraw.textbbox((0, 0), val_str, font=font)
        tw_ = tbb[2] - tbb[0]
        th_ = tbb[3] - tbb[1]
        tx = cx - tw_ / 2.0 - tbb[0]
        ty = val_center_y - th_ / 2.0 - tbb[1]
        text_color = _parse_color(style.get("text_color", "#FFFFFF"))
        pdraw.text((tx, ty), val_str, font=font, fill=text_color)

        if has_unit:
            ufont = _font(int(r * float(style.get("unit_scale", 0.18))))
            ubb = pdraw.textbbox((0, 0), unit, font=ufont)
            uw_ = ubb[2] - ubb[0]
            val_visible_bottom = ty + tbb[1] + th_
            uty = val_visible_bottom + max(2 * SS, int(r * 0.04)) - ubb[1]
            utx = cx - uw_ / 2.0 - ubb[0]
            pdraw.text((utx, uty), unit, font=ufont,
                       fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)


def _draw_tickdial(img: Any, w: Any,
                   value: Any, unit: str, default_fmt: str,
                   rx: int, ry: int, ww: int, wh: int) -> None:
    """Shallow upper arc with tick marks + numeric labels and a rotating needle."""
    style = w.style
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    tick_color   = _parse_color(style.get("color", "#FFFFFF"),
                                alpha=int(255 * float(style.get("opacity", 0.95))))
    needle_color = _parse_color(style.get("needle_color", "#FF3030"))
    show_value   = bool(style.get("show_value", True))
    n_majors     = int(style.get("major_ticks", 5))
    n_minors_per = int(style.get("minor_per_major", 4))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    margin = max(2 * SS, int(min(pw, ph) * 0.04))
    cx = pw / 2.0
    text_budget = 0
    if show_value:
        text_budget = int(ph * (0.30 if unit else 0.22))
    r = min(pw / 2.0 - margin, ph - text_budget - margin * 2)
    if r < 6 * SS:
        return
    cy = margin + r

    tick_len_major = max(3 * SS, int(r * 0.13))
    tick_len_minor = max(2 * SS, int(r * 0.06))
    tick_w_major   = max(1 * SS, int(r * 0.022))
    tick_w_minor   = max(1 * SS, int(r * 0.011))

    start_angle = 180.0
    span_angle  = 180.0

    for i in range(n_majors - 1):
        for k in range(1, n_minors_per + 1):
            t = (i + k / (n_minors_per + 1)) / (n_majors - 1)
            ang = start_angle + span_angle * t
            x0, y0 = _polar(cx, cy, r,                  ang)
            x1, y1 = _polar(cx, cy, r - tick_len_minor, ang)
            pdraw.line([(x0, y0), (x1, y1)], fill=tick_color, width=tick_w_minor)

    label_font = _font(int(r * float(style.get("label_scale", 0.13))))
    for i in range(n_majors):
        t = i / max(1, n_majors - 1)
        ang = start_angle + span_angle * t
        x0, y0 = _polar(cx, cy, r,                  ang)
        x1, y1 = _polar(cx, cy, r - tick_len_major, ang)
        pdraw.line([(x0, y0), (x1, y1)], fill=tick_color, width=tick_w_major)
        label_val = vmin + (vmax - vmin) * t
        fmt = str(style.get("scale_fmt", "{:.0f}"))
        try:
            lbl = fmt.format(label_val)
        except (ValueError, KeyError, IndexError):
            lbl = f"{label_val:.0f}"
        lbb = pdraw.textbbox((0, 0), lbl, font=label_font)
        lw_ = lbb[2] - lbb[0]
        lh_ = lbb[3] - lbb[1]
        lx_anchor, ly_anchor = _polar(cx, cy, r - tick_len_major - lh_ * 1.1, ang)
        ltx = lx_anchor - lw_ / 2.0 - lbb[0]
        lty = ly_anchor - lh_ / 2.0 - lbb[1]
        pdraw.text((ltx, lty), lbl, font=label_font,
                   fill=_parse_color(style.get("label_color", "#DDDDDD")))

    needle_ang = start_angle + span_angle * frac
    needle_len = r * float(style.get("needle_length", 0.92))
    nx, ny = _polar(cx, cy, needle_len, needle_ang)
    tail_len = r * float(style.get("needle_tail", 0.12))
    tx_, ty_ = _polar(cx, cy, tail_len, needle_ang + 180.0)
    pdraw.line([(tx_, ty_), (nx, ny)],
               fill=needle_color, width=max(2 * SS, int(r * 0.028)))
    cap_r = max(2 * SS, int(r * 0.06))
    pdraw.ellipse([cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r],
                  fill=needle_color)

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        has_unit = bool(style.get("show_unit", True)) and unit
        vfont = _font(int(r * float(style.get("font_scale", 0.28))))
        vbb = pdraw.textbbox((0, 0), val_str, font=vfont)
        vw_ = vbb[2] - vbb[0]
        vh_ = vbb[3] - vbb[1]
        val_top_y = cy + cap_r + max(2 * SS, int(r * 0.05))
        vtx = cx - vw_ / 2.0 - vbb[0]
        vty = val_top_y - vbb[1]
        pdraw.text((vtx, vty), val_str, font=vfont,
                   fill=_parse_color(style.get("text_color", "#FFFFFF")))
        if has_unit:
            ufont = _font(int(r * float(style.get("unit_scale", 0.13))))
            ubb = pdraw.textbbox((0, 0), unit, font=ufont)
            uw_ = ubb[2] - ubb[0]
            val_visible_bottom = vty + vbb[1] + vh_
            utx = cx - uw_ / 2.0 - ubb[0]
            uty = val_visible_bottom + max(1 * SS, int(r * 0.025)) - ubb[1]
            pdraw.text((utx, uty), unit, font=ufont,
                       fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)


def _draw_ring(img: Any, w: Any,
               value: Any, unit: str, default_fmt: str,
               rx: int, ry: int, ww: int, wh: int) -> None:
    """Thin full-circle ring; progress arc starts at 12 o'clock, sweeps clockwise."""
    style = w.style
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    color = _parse_color(style.get("color", "#3FA9F5"),
                         alpha=int(255 * float(style.get("opacity", 0.95))))
    track = _parse_color(style.get("bg_color", "#FFFFFF"),
                         alpha=int(255 * float(style.get("bg_opacity", 0.18))))
    thickness_pct = float(style.get("thickness", 0.06))
    show_value    = bool(style.get("show_value", True))
    rounded_caps  = bool(style.get("rounded_caps", True))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    size  = min(pw, ph)
    cx    = pw / 2.0
    cy    = ph / 2.0
    r     = size / 2.0 - max(2 * SS, int(size * 0.04))
    if r < 4 * SS:
        return
    thick = max(2 * SS, int(size * thickness_pct))

    bb = [cx - r, cy - r, cx + r, cy + r]
    pdraw.arc(bb, 0, 360, fill=track, width=thick)
    start_angle = 270.0
    end_angle   = start_angle + 360.0 * frac
    if frac > 0.001:
        pdraw.arc(bb, start_angle, end_angle, fill=color, width=thick)
        if rounded_caps:
            cap_r = thick / 2.0
            for ang_deg in (start_angle, end_angle):
                px, py = _polar(cx, cy, r, ang_deg)
                pdraw.ellipse([px - cap_r, py - cap_r,
                               px + cap_r, py + cap_r], fill=color)

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        has_unit = bool(style.get("show_unit", True)) and unit
        font = _font(int(size * float(style.get("font_scale", 0.28))))
        tbb = pdraw.textbbox((0, 0), val_str, font=font)
        tw_ = tbb[2] - tbb[0]
        th_ = tbb[3] - tbb[1]
        val_center_y = cy - (size * 0.07 if has_unit else 0.0)
        tx = cx - tw_ / 2.0 - tbb[0]
        ty = val_center_y - th_ / 2.0 - tbb[1]
        pdraw.text((tx, ty), val_str, font=font,
                   fill=_parse_color(style.get("text_color", "#FFFFFF")))
        if has_unit:
            ufont = _font(int(size * float(style.get("unit_scale", 0.12))))
            ubb = pdraw.textbbox((0, 0), unit, font=ufont)
            uw_ = ubb[2] - ubb[0]
            val_visible_bottom = ty + tbb[1] + th_
            utx = cx - uw_ / 2.0 - ubb[0]
            uty = val_visible_bottom + max(2 * SS, int(size * 0.025)) - ubb[1]
            pdraw.text((utx, uty), unit, font=ufont,
                       fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)


def _draw_analog(img: Any, w: Any,
                 value: Any, unit: str, default_fmt: str,
                 rx: int, ry: int, ww: int, wh: int) -> None:
    """Classic round instrument: dark face, tick marks, numeric scale, white needle."""
    style = w.style
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return

    vmin = float(style.get("min", 0.0))
    vmax = float(style.get("max", 100.0))
    rng  = vmax - vmin
    frac = 0.0 if rng <= 0 else max(0.0, min(1.0, (v - vmin) / rng))

    face_color    = _parse_color(style.get("face_color", "#101418"),
                                 alpha=int(255 * float(style.get("face_opacity", 0.92))))
    bezel_color   = _parse_color(style.get("bezel_color", "#404040"))
    tick_color    = _parse_color(style.get("color", "#FFFFFF"),
                                 alpha=int(255 * float(style.get("opacity", 0.95))))
    needle_color  = _parse_color(style.get("needle_color", "#FFFFFF"))
    label_color   = _parse_color(style.get("label_color", "#DDDDDD"))
    show_value    = bool(style.get("show_value", False))
    n_majors      = int(style.get("major_ticks", 6))
    n_minors_per  = int(style.get("minor_per_major", 4))
    start_angle   = float(style.get("start_angle", 135.0))
    span_angle    = float(style.get("span_angle", 270.0))

    SS = max(1, int(style.get("supersample", 3)))
    pw = max(1, ww * SS)
    ph = max(1, wh * SS)
    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)

    size = min(pw, ph)
    cx   = pw / 2.0
    cy   = ph / 2.0
    r    = size / 2.0 - max(2 * SS, int(size * 0.03))
    if r < 8 * SS:
        return

    bb_face = [cx - r, cy - r, cx + r, cy + r]
    bezel_w = max(2 * SS, int(r * 0.05))
    pdraw.ellipse(bb_face, fill=face_color, outline=bezel_color, width=bezel_w)

    tick_len_major = max(3 * SS, int(r * 0.14))
    tick_len_minor = max(2 * SS, int(r * 0.07))
    tick_w_major   = max(1 * SS, int(r * 0.025))
    tick_w_minor   = max(1 * SS, int(r * 0.013))
    tick_outer     = r - bezel_w

    for i in range(n_majors - 1):
        for k in range(1, n_minors_per + 1):
            t = (i + k / (n_minors_per + 1)) / (n_majors - 1)
            ang = start_angle + span_angle * t
            x0, y0 = _polar(cx, cy, tick_outer,                  ang)
            x1, y1 = _polar(cx, cy, tick_outer - tick_len_minor, ang)
            pdraw.line([(x0, y0), (x1, y1)], fill=tick_color, width=tick_w_minor)

    label_font = _font(int(r * float(style.get("label_scale", 0.14))))
    for i in range(n_majors):
        t = i / max(1, n_majors - 1)
        ang = start_angle + span_angle * t
        x0, y0 = _polar(cx, cy, tick_outer,                  ang)
        x1, y1 = _polar(cx, cy, tick_outer - tick_len_major, ang)
        pdraw.line([(x0, y0), (x1, y1)], fill=tick_color, width=tick_w_major)

        label_val = vmin + (vmax - vmin) * t
        fmt = str(style.get("scale_fmt", "{:.0f}"))
        try:
            lbl = fmt.format(label_val)
        except (ValueError, KeyError, IndexError):
            lbl = f"{label_val:.0f}"
        lbb = pdraw.textbbox((0, 0), lbl, font=label_font)
        lw_ = lbb[2] - lbb[0]
        lh_ = lbb[3] - lbb[1]
        lx_anchor, ly_anchor = _polar(
            cx, cy, tick_outer - tick_len_major - max(lh_, lw_) * 0.7, ang)
        ltx = lx_anchor - lw_ / 2.0 - lbb[0]
        lty = ly_anchor - lh_ / 2.0 - lbb[1]
        pdraw.text((ltx, lty), lbl, font=label_font, fill=label_color)

    needle_ang = start_angle + span_angle * frac
    needle_len = (tick_outer - tick_len_major) * float(style.get("needle_length", 0.95))
    nx, ny = _polar(cx, cy, needle_len, needle_ang)
    tail_len = r * float(style.get("needle_tail", 0.12))
    tx_, ty_ = _polar(cx, cy, tail_len, needle_ang + 180.0)
    pdraw.line([(tx_, ty_), (nx, ny)],
               fill=needle_color, width=max(2 * SS, int(r * 0.030)))
    cap_r = max(3 * SS, int(r * 0.07))
    pdraw.ellipse([cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r],
                  fill=needle_color)
    inner_cap_r = max(1 * SS, int(cap_r * 0.45))
    pdraw.ellipse([cx - inner_cap_r, cy - inner_cap_r,
                   cx + inner_cap_r, cy + inner_cap_r], fill=face_color)

    inner_label = str(style.get("label", "")).strip()
    if inner_label:
        ifont = _font(int(r * float(style.get("inner_label_scale", 0.11))))
        ibb = pdraw.textbbox((0, 0), inner_label, font=ifont)
        iw_ = ibb[2] - ibb[0]
        ih_ = ibb[3] - ibb[1]
        itx = cx - iw_ / 2.0 - ibb[0]
        ity = cy - r * 0.28 - ih_ / 2.0 - ibb[1]
        pdraw.text((itx, ity), inner_label, font=ifont, fill=label_color)

    if bool(style.get("show_unit", True)) and unit:
        ufont = _font(int(r * float(style.get("unit_scale", 0.11))))
        ubb = pdraw.textbbox((0, 0), unit, font=ufont)
        uw_ = ubb[2] - ubb[0]
        uh_ = ubb[3] - ubb[1]
        utx = cx - uw_ / 2.0 - ubb[0]
        uty = cy + r * 0.32 - uh_ / 2.0 - ubb[1]
        pdraw.text((utx, uty), unit, font=ufont,
                   fill=_parse_color(style.get("unit_color", "#B8B8B8")))

    if show_value:
        fmt = str(style.get("fmt", default_fmt))
        val_str = _format_value(v, fmt, default_fmt)
        vfont = _font(int(r * float(style.get("font_scale", 0.20))))
        vbb = pdraw.textbbox((0, 0), val_str, font=vfont)
        vw_ = vbb[2] - vbb[0]
        vh_ = vbb[3] - vbb[1]
        vtx = cx - vw_ / 2.0 - vbb[0]
        vty = cy + r * 0.28 - vh_ / 2.0 - vbb[1]
        pdraw.text((vtx, vty), val_str, font=vfont,
                   fill=_parse_color(style.get("text_color", "#FFFFFF")))

    if SS > 1:
        patch = patch.resize((ww, wh), Image.LANCZOS)
    img.paste(patch, (rx, ry), patch)

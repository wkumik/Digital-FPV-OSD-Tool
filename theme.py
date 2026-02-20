# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
theme.py — Colour palette definitions for VueOSD.

Edit this file manually, or use the in-app Theme Editor (palette icon in toolbar).
Changes take effect immediately when you click Apply in the editor.

Colour token guide
──────────────────
  bg          Main window background
  bg2         Inset areas — path labels, spinbox backgrounds
  bg3         Deeper inset — used in group headers
  surface     Button / combo background (resting state)
  surface2    Button / combo background (hover)
  surface3    Button / combo background (pressed / active)
  text        Primary body text
  subtext     Secondary labels (60% importance)
  muted       Disabled / placeholder / hint text (38%)
  accent      Slider fill, active states, primary button fill
  accent2     Primary button hover / gradient end
  green       Success / armed indicator text
  red         Error / stop button / danger action
  orange      Warning text
  border      Subtle dividers and group borders
  border2     Input borders, stronger dividers
  icon        Icon tint colour
"""

import json, os

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme_custom.json")

# ── Built-in defaults ─────────────────────────────────────────────────────────

DARK_DEFAULT = {
    "bg":       "#1e1e2e",
    "bg2":      "#181825",
    "bg3":      "#11111b",
    "surface":  "#313244",
    "surface2": "#45475a",
    "surface3": "#585b70",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "muted":    "#6c7086",
    "accent":   "#89b4fa",
    "accent2":  "#b4befe",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "orange":   "#fab387",
    "border":   "#313244",
    "border2":  "#45475a",
    "icon":     "#cdd6f4",
}

LIGHT_DEFAULT = {
    "bg":       "#fafafa",
    "bg2":      "#f2f2f0",
    "bg3":      "#e8e8e5",
    "surface":  "#f0f0ee",
    "surface2": "#e4e4e1",
    "surface3": "#d8d8d4",
    "text":     "#1c1c1c",
    "subtext":  "#555552",
    "muted":    "#999994",
    "accent":   "#525760",
    "accent2":  "#666f83",
    "green":    "#155a15",
    "red":      "#b01025",
    "orange":   "#8a4200",
    "border":   "#ddddd9",
    "border2":  "#c8c8c4",
    "icon":     "#3a3a38",
}

# Human-readable labels for the editor UI
TOKEN_LABELS = {
    "bg":       "Window background",
    "bg2":      "Inset background",
    "bg3":      "Deep inset",
    "surface":  "Button / combo (rest)",
    "surface2": "Button / combo (hover)",
    "surface3": "Button / combo (pressed)",
    "text":     "Primary text",
    "subtext":  "Secondary text",
    "muted":    "Disabled / hint text",
    "accent":   "Accent (primary button, slider)",
    "accent2":  "Accent hover / gradient end",
    "green":    "Success / armed",
    "red":      "Error / stop / danger",
    "orange":   "Warning",
    "border":   "Subtle border",
    "border2":  "Input border",
    "icon":     "Icon tint",
}

# Group tokens into logical sections for the editor
TOKEN_GROUPS = [
    ("Backgrounds",  ["bg", "bg2", "bg3"]),
    ("Surfaces",     ["surface", "surface2", "surface3"]),
    ("Text",         ["text", "subtext", "muted"]),
    ("Accent",       ["accent", "accent2"]),
    ("Status",       ["green", "red", "orange"]),
    ("Borders",      ["border", "border2"]),
    ("Icons",        ["icon"]),
]


# ── Active palettes (mutated at runtime) ──────────────────────────────────────

_dark  = dict(DARK_DEFAULT)
_light = dict(LIGHT_DEFAULT)


def get_dark()  -> dict: return _dark
def get_light() -> dict: return _light


def load():
    """Load customisations from theme_custom.json (if it exists)."""
    global _dark, _light
    _dark  = dict(DARK_DEFAULT)
    _light = dict(LIGHT_DEFAULT)
    if not os.path.exists(_FILE):
        return
    try:
        data = json.loads(open(_FILE, encoding="utf-8").read())
        if "dark"  in data:
            for k, v in data["dark"].items():
                if k in _dark and _is_hex(v):
                    _dark[k] = v
        if "light" in data:
            for k, v in data["light"].items():
                if k in _light and _is_hex(v):
                    _light[k] = v
    except Exception:
        pass   # silently ignore corrupt file — defaults remain


def save(dark: dict, light: dict):
    """Persist customisations to theme_custom.json."""
    global _dark, _light
    _dark  = {k: v for k, v in dark.items()  if _is_hex(v)}
    _light = {k: v for k, v in light.items() if _is_hex(v)}
    data = json.dumps({"dark": _dark, "light": _light}, indent=2)
    open(_FILE, "w", encoding="utf-8").write(data)


def reset():
    """Discard customisations and restore factory defaults."""
    global _dark, _light
    _dark  = dict(DARK_DEFAULT)
    _light = dict(LIGHT_DEFAULT)
    if os.path.exists(_FILE):
        try: os.remove(_FILE)
        except: pass


def _is_hex(v: str) -> bool:
    """Return True if v looks like a valid #RRGGBB or #RRGGBBAA hex colour."""
    if not isinstance(v, str): return False
    v = v.strip()
    if not v.startswith("#"): return False
    h = v[1:]
    return len(h) in (6, 8) and all(c in "0123456789abcdefABCDEF" for c in h)


# Load any saved customisations immediately on import
load()

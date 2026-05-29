# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VueOSD — Digital FPV OSD Tool** overlays MSP-OSD telemetry data onto FPV DVR video footage using a Python/PyQt6 GUI with an FFmpeg backend. It parses `.osd` binary files or `.srt` subtitle telemetry, renders OSD glyphs via NumPy+Pillow, and pipes the result to FFmpeg for GPU-accelerated encoding (NVENC, AMF, QSV, VAAPI, VideoToolbox).

## Running the Application

```bash
# Windows (auto-installs deps on first run)
"VueOSD.bat"

# Linux/macOS (creates venv, installs deps, launches)
chmod +x run.sh && ./run.sh

# Direct Python launch (after deps installed)
python main.py
```

## Building Distributables

```bash
# Windows → dist/VueOSD.exe
build.bat

# Linux/macOS → dist/VueOSD
./build.sh
```

Both scripts use PyInstaller. FFmpeg must be on PATH.

## Running Tests

```bash
python -m pytest tests/        # or: python -m unittest discover tests
python -m pytest tests/ -v     # verbose
```

## Dependencies

- **Python 3.10+** required
- `pip install -r requirements.txt` — installs PyQt6, Pillow, NumPy
- **FFmpeg** must be on PATH (Windows: auto-installed via `winget` through `bootstrap.py`)

## Architecture

The application is a pipeline: parse → render → pipe → encode.

```
main.py (PyQt6 GUI — MainWindow, FileRow, LabeledSlider, RenderBar)
    │
    ├── player.py              — Video player (VideoCanvas, Timeline, TransportBar,
    │                              PlayerController, PlayerPanel)
    │
    ├── osd_parser.py          — Binary .osd format parser (timestamp-indexed, bisect lookup)
    ├── p1_osd_parser.py       — BetaFPV P1 MP4 embedded OSD extraction
    ├── srt_parser.py          — .srt subtitle telemetry parser (speed, alt, sats, signal, GPS)
    ├── osd_decoder.py         — Reverse-parses numeric values from OSD glyph grid
    │                              (firmware-specific symbol tables + slim-digit auto-detection,
    │                               Plus Code decoding, raw GPS coordinate extraction)
    ├── font_loader.py         — OSD font sheet loader (Betaflight/INAV/ArduPilot HD fonts)
    │
    ├── osd_renderer.py        — Composites OSD glyph grid + SRT bar → PNG (NumPy+PIL)
    │
    ├── widgets.py             — Custom widget overlay framework (public API)
    │   ├── _widget_primitives.py — Shared helpers (color, font, shadow, polar coords)
    │   ├── _widget_draw.py       — 7 widget type draw functions (digital, bar, gauge,
    │   │                            semicircle, tickdial, ring, analog)
    │   └── _widget_map.py        — GPS track map widget (projection, smoothing, Catmull-Rom)
    │       └── _widget_map_tiles.py — Tile underlay (OSM street / Esri satellite via Web Mercator)
    │
    ├── video_processor.py     — Orchestrates FFmpeg subprocess; auto-detects GPU encoders
    │
    ├── updater.py             — GitHub release check + self-update + fonts downloader
    ├── subprocess_utils.py    — Popen/run wrappers that suppress console windows on Windows
    ├── theme.py               — Dark/light palette definitions; persists to theme_custom.json
    ├── theme_editor.py        — In-app theme customization dialog
    └── splash_screen.py       — Animated progress screen (used during encoding)
```

**Critical rendering path:** Python renders OSD-only PNG frames (~1 ms/frame) and pipes them to a concurrently running FFmpeg process that decodes the video, applies the overlay filter, and GPU-encodes the output. Python is never in the per-frame hot loop with FFmpeg.

**Two render entry points** for widgets:
- `render_widgets_pil()` — PIL preview path (single frame, called from the Qt paint loop)
- `blend_widgets_numpy()` — NumPy export hot path (Porter-Duff over into the frame buffer FFmpeg reads)

## Key Implementation Details

**Widget system** (`widgets.py`): 8 types: digital, bar, gauge, semicircle, tickdial, ring, analog, map. Widgets use normalised 0..1 coordinates (centre-anchored so resizing keeps the visual anchor stable). Data sources unify SRT telemetry fields with OSD-decoded values via a `TelemetryFrame` class. Each widget has a `style` dict for visual customization. The map widget is "telemetry-aware" — it receives the full `TelemetryFrame` (not just a scalar) to access the entire GPS track.

**OSD decoder** (`osd_decoder.py`): Reverse-parses numeric values from the MSP-OSD glyph grid by locating firmware-specific unit-icon "anchors" and walking adjacent digit cells. Firmware auto-detection selects the right symbol table (Betaflight, INAV, ArduPilot). Slim digit glyphs (0x80-0xFF range) are auto-detected per-font via IoU template matching against ASCII digit bitmaps. Also decodes Plus Codes (Open Location Code) and raw GPS lat/lon coordinates from the grid.

**Widget smoothing** (`TelemetryFrame._alpha_beta_visual_value`): An alpha-beta filter runs on a pre-decoded per-frame value track at a fixed 16 ms step, with lookahead to compensate filter lag. The geometry value is smoothed; the text readout always stays on the exact current OSD frame value. Smoothness=0% (the default for old widgets) bypasses the filter entirely for backward compatibility. Filtered tracks are cached per-OsdFile with weakref finalizers so the cache self-cleans when files are replaced.

**GPS map widget** (`_widget_map.py`): Three-tier track source priority: SRT GPS (1 Hz, sub-metre) → raw OSD lat/lon → OSD Plus Codes (~14 m grid). Equirectangular projection with cos(lat) correction. Track polyline is smoothed (moving average → Catmull-Rom spline) and cached per (track, patch size) so playback doesn't recompute per frame. Tile underlay ("street"/"satellite" via `_widget_map_tiles.py`) layers a Web-Mercator basemap under the track; tiles cache to disk under `%APPDATA%/VueOSD/tilecache`.

**GPU encoder detection** (`video_processor.py`): Tests NVENC → AMF → QSV → VAAPI → VideoToolbox → libx264. NVIDIA detection has a 20-second timeout for CUDA context initialization. Results are cached at startup.

**OSD parsing** (`osd_parser.py`): Binary format with timestamp-indexed frames; uses `bisect` for O(log n) frame lookup by video timestamp.

**Font system** (`font_loader.py`): Supports multi-column HD font sheets. Fonts live in `fonts/` with prefixes: `BTFL_` (Betaflight), `INAV` (INAV), `ARDU_` (ArduPilot). Quicksilver is not supported.

**Theme system** (`theme.py`): 16-color token system (backgrounds, surfaces, text, accent, status, borders). User overrides stored in `theme_custom.json` at runtime.

**Auto-updater** (`updater.py`): Hits GitHub API for latest release, compares against running VERSION, downloads source zipball, mirrors contents over the install directory (preserving protected files like `settings.json`). Also supports a "Download fonts" button that fetches `fonts.zip` from the latest release that has one attached.

**Bootstrap flow** (`bootstrap.py`): On Windows, re-launches as `pythonw.exe` to hide the console, creates a venv, installs requirements, and auto-installs FFmpeg via `winget`. Only runs when the packaged `.bat` launcher is used.

**Firmware auto-detection** (`main.py` `_load_osd()`): Reads the 4-byte FC type tag from the OSD header (`BTFL` → Betaflight, `INAV` → INAV, `ARDU` → ArduPilot) and calls `_on_fw_changed()` directly. Unknown tags fall back to Betaflight. There is no firmware selector in the UI.

**Video player** (`player.py`): Contains `VideoCanvas` (frame display with Quick Start placeholder), `Timeline` (scrub + trim handles + cache dots), `TransportBar` (play/pause, step, shuttle controls), `PlayerController` (FFmpeg pipe management, frame cache, playback state machine), and `PlayerPanel` (container). Keyboard shortcuts (Space, Left/Right, J/K/L, I/O, Home/End) are handled in `MainWindow.keyPressEvent()` and delegated to `PlayerController`.

**Windows taskbar integration** (`main.py`): Sets `AppUserModelID` via ctypes for proper icon display. Subprocess calls use `subprocess_utils.py` wrappers that hide console windows via `STARTF_USESHOWWINDOW` + `CREATE_NO_WINDOW`.

## Codebase Conventions

- Private submodules (not part of the public API) are prefixed with `_`: `_widget_primitives.py`, `_widget_draw.py`, `_widget_map.py`, `_widget_map_tiles.py`
- All `.py` files include an SPDX license header and copyright line
- UI classes use PyQt6 naming conventions (camelCase methods), while data/logic modules use snake_case
- Caching is aggressive: projection results, visual tracks, basemaps, and slim-digit maps are all cached by identity or content hash, with weakref-based eviction for per-file caches

## UI Style Guidelines (PyQt6)

Match the existing look — read a nearby control and copy its pattern before adding a new one.

**Icons: use PNG assets via the `_icon()` helper, never emoji.** The canonical pattern
is `btn.setIcon(_icon("render.png", 18))` — `_icon(name, size, color=None)` (`main.py:217`)
loads a PNG from `icons/` and can retint it to a theme colour. For a button with both
icon and text, give the label leading spaces so the icon has room: `QPushButton("  Render Video")`
(see `render_btn`, `stop_btn`, and the player transport buttons in `player.py`).
- **Do NOT put emoji in button/label text** (🔒 👁 🙈 ✎ 🔍 🎨 etc.). They render
  inconsistently across platforms and clash with the PNG icon set. Plain ASCII/text
  labels are fine; iconography must come from `icons/`.
- A few legacy unicode glyphs exist (`＋ Add`, `− Remove`, `↺ Reset`, `✕`). These are
  tolerated holdovers, not patterns to copy — prefer a PNG icon for anything new.
- Available icons in `icons/`: `back, error, folder, gear, moon-dark, moon-light,
  next, pause, play, render, rewind, save, settings, stop, video, wifi, wifi2` (`.png`).
  If a control needs an icon that isn't there (lock/unlock, eye/eye-off, …), **add a
  new PNG to `icons/` in the same flat monochrome style** (retintable via `_icon()`'s
  `color` arg) rather than substituting an emoji or unicode glyph.

**Button styling: use the shared constants.** Every button's `setStyleSheet` takes a
module-level style string: `BTN_SEC` (default), `BTN_PRIMARY`, `BTN_PLAY`, `BTN_STOP`,
`BTN_DANGER`. Don't hand-roll per-button CSS. Any `BTN_SEC` button must also be listed
in the theme-restyle loop in `_apply_theme()` (the `for _b in (...)` block) or it keeps
the stale palette when the user switches light/dark; icon buttons get their icon
re-applied there too so retinting follows the theme.

**Sizing & layout.** Group boxes use `CollapsibleGroupBox` + `GROUP_STYLE`, content
margins `(10, 16, 10, 10)`, `setSpacing(4)`. Use the local `_lbl()` helper for field
labels and `_fs()` for all font sizes (never hard-code px) so they scale with the UI
scale setting.

**State sync.** When a `_refresh_*` method programmatically updates a control that has a
connected signal, wrap the update in `blockSignals(True)/(False)` so it doesn't re-fire
handlers — see `_refresh_map_box()`.

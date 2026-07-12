# VueOSD Refactor Plan — player + UI style layer

**Executor:** Claude Sonnet, working in this repository.
**Baseline:** commit `e66c0ff` (branch `main`). Line numbers below were taken at this
commit — treat them as hints. **Always locate code by symbol name / grep, never by raw
line offset.**
**Goal:** structural refactor only. Zero behavior change. Every stage ends with the app
working exactly as before.

---

## 0. Ground rules (read before every stage)

1. **Behavior-preserving.** No feature changes, no perf tuning, no "while I'm here"
   fixes. If you spot a real bug, note it in the final report — do not fix it here.
2. **One stage = one commit.** Run the full verification protocol (§9) before each
   commit. Commit messages are given per stage. Do **not** push unless the user asks.
3. **If a stage fails verification** and you cannot fix it quickly: revert the touched
   files (`git checkout -- <files>`), record the failure in your final report, and move
   on. Stages 6–7 are independent of stages 1–5.
4. **Conventions (from CLAUDE.md — read it):**
   - New private modules are prefixed `_` (like `_widget_draw.py`).
   - Every new `.py` file starts with the same SPDX + copyright header as `player.py`
     (copy the first 2 lines verbatim) followed by a short module docstring.
   - No emoji anywhere. UI code keeps PyQt6 camelCase method names; logic modules
     use snake_case.
   - Match surrounding comment density. Do not add comments that narrate the refactor.
5. **Threading invariants (critical — do not "improve"):**
   - Worker threads never call Qt objects directly; they marshal to the main thread
     via `QTimer.singleShot(0, ...)`. Preserve every such hop exactly.
   - `PlayerController._buf_lock` guards `_buf_frame` between the reader thread and
     the display timer. Keep the pull model (timer pulls latest frame) as is.
   - `cached_frames` is accessed from threads without a lock. This is a known,
     accepted quirk. Do **not** add locking in this refactor.
   - Playback pacing semantics (first-frame clock anchor, drop-if-late, re-anchor past
     `_MAX_PLAYBACK_LAG_S = 0.35`) were just fixed and verified. The refactor moves
     this logic; it must not alter it. Stage 2 gives the exact target code.
6. **Do not touch:** `video_processor.py` export pipelines (`_overlay_pipeline`,
   `_chroma_key_pipeline`, `_srt_only_pipeline`, `_reencode_only`, `process_video`),
   `osd_*.py`, `widgets.py` and `_widget_*.py`, `updater.py`, `bootstrap.py`. See §8.
7. Keep `git status` clean of stray files. Scratch/verification scripts go to the
   session scratchpad directory, never the repo.

---

## 1. Current anatomy (facts you will need)

### player.py (1944 lines) — the primary target

| Symbol | Lines (approx) | Role |
|---|---|---|
| donate constants `_DONATE_*` | 33–61 | strings used by TransportBar |
| `_SL_MAX = 10_000`, `_CACHE_MAX = 64`, `_MAX_PLAYBACK_LAG_S = 0.35` | 64–70 | module constants |
| `class VideoCanvas(QWidget)` | 75–853 | frame display + aspect padding + placeholder + **region-edit overlay** + **widget-edit overlay** + mouse handling for both editors |
| `class Timeline(QWidget)` | 855–1061 | scrub track, trim handles, cache dots |
| `class TransportBar(QWidget)` | 1062–1204 | play/step/shuttle buttons, time label, speed, donate link |
| `class PlayerController` | 1206–1890 | video state, frame cache, scrub/extract, prefetch, playback pipe + pacing, OSD refresh, trim, shuttle |
| `class PlayerPanel(QWidget)` | 1892–1944 | assembles canvas + timeline + transport + controller |

Duplication inside `PlayerController` (the core smell): **three near-identical FFmpeg
rawvideo command constructions** in `_extract_at_pct` (~1462), `_prefetch_frames`
(~1531), and `play()` (~1576) — the first two lack `-hwaccel`/`-an`, `play()` has both.
This divergence already caused a real bug (hwaccel was added to only one site).
Likewise `PILImage.frombytes(...).convert("RGBA")` appears 4×.

### Coupling from main.py (public API — must keep working)

- `from player import PlayerPanel` (local import inside `MainWindow`, ~line 1379).
- `PlayerPanel(theme_fn, icon_fn, btn_play_fn, btn_sec_fn, fs_fn, composite_fn, osd_fn)`
  — keyword construction, see main.py ~1392.
- Attributes read by main.py: `panel.controller`, `panel.timeline`, `panel.canvas`,
  `panel.transport`, `panel.trim_in_lbl`, `panel.trim_out_lbl`,
  `panel.transport.info_lbl`, `panel.refresh_theme()`.
- `Timeline` signals/attrs used: `seekRequested`, `trimChanged`, `_position` (yes,
  main.py reads the private `_position` directly — keep it).
- `VideoCanvas` API used: `set_region_callbacks`, `set_regions`,
  `set_region_edit_mode`, `regionAdded`, `regionRemoveRequested`, `resized`,
  `set_widget_src_rect_fn`, `set_widgets`, `set_widget_edit`, `set_widget_selected`,
  `widgetSelected`, `widgetMoved`, `widgetCommit`, `set_frame`, `set_base_frame`,
  `set_osd_overlay`, `set_placeholder`, `set_aspect_padding`, `has_frame`,
  `display_size`, `display_origin`.
- `TransportBar` signals used: `playClicked`, `restartClicked`, `skipEndClicked`,
  `stepBackClicked`, `stepFwdClicked`, `refreshClicked`, `previewQualityChanged`,
  plus `set_playing/set_time/set_info/set_speed/refresh_theme`.
- `PlayerController` API used by main.py (grep to confirm the full list before you
  start: `rg -n "controller\." main.py`): `load_video`, `stop`, `pause`,
  `toggle_play`, `seek_start`, `seek_end`, `step_forward`, `step_backward`,
  `shuttle_faster`, `shuttle_slower`, `shuttle_reset`, `set_trim_in`, `set_trim_out`,
  `pct_to_video_time_ms`, `queue_refresh`, `refresh_now`, `set_color_vf`,
  `set_full_quality_preview`, `clear`, and attributes `video_fps`, `video_dur`,
  `video_w`, `video_h`, `full_quality_preview`, `pts_list`.
- main.py defines its **own** `_SL_MAX = 10_000` (~line 209). Leave that duplicate
  alone; it is not imported from player.

### main.py (4303 lines) — secondary target

Module-level UI-style layer (to be extracted in stages 6–7):

- `_DARK_THEME` (~62), `_UI_SCALE` (~68), `_fs()` (~71), `_T()` (~110),
  `_build_styles()` (~115–202) which **assigns module globals**:
  `APP_STYLE, GROUP_STYLE, PATH_EMPTY, PATH_FILLED, BTN_SEC, BTN_PRIMARY, BTN_PLAY,
  BTN_STOP, BTN_DANGER` (+ read `_build_styles` yourself for the complete list, e.g.
  `SLIDER_STYLE` — move **everything** it assigns).
- `_icons_dir()` (~214), `_icon()` (~217).
- Generic widgets: `FileRow` (279), `DropZone` (369), `LabeledSlider` (425),
  `CheckableComboBox` (464), `CollapsibleGroupBox` (528), `InfoCard` (590).
  (`RenderBar` (626) and the `*Worker` QThreads are app-specific — they stay.)
- Settings load/save (`_load_settings` ~81, `_save_settings` ~95) read/write
  `_UI_SCALE`; `_toggle_theme`/`_on_scale_changed` mutate `_DARK_THEME`/`_UI_SCALE`
  via `global`.

### Tests

`tests/test_widget_smoothing.py` (20 tests) — covers widgets/decoder/parser only.
**Nothing covers the player**, which is why stages 2–3 add tests around the seams
you create. Tests run with `python -m unittest discover tests` (pytest is not
installed).

---

## 2. Target layout after stages 1–4

```
player.py             — PlayerController (coordinator) + PlayerPanel
                        + re-exports: VideoCanvas, Timeline, TransportBar, _SL_MAX
_player_common.py     — constants, PIL import guard, rawvideo_cmd(), rgba_from_rgb24()
_player_views.py      — VideoCanvas, Timeline, TransportBar (+ donate constants)
_player_extract.py    — FrameCache (dict-like + eviction)
_player_playback.py   — plan_frame() pure pacing fn + PlaybackEngine
tests/test_player_pacing.py
tests/test_frame_cache.py
tests/test_player_common.py
```

Import DAG (must stay acyclic):
`_player_common` ← `_player_views`, `_player_extract`, `_player_playback` ← `player`.
New modules must **never** import `player` or `main`.

---

## Stage 1 — `_player_common.py`: dedupe FFmpeg command + frame decode

**Create `_player_common.py`** (SPDX header + docstring) containing:

```python
_SL_MAX = 10_000
_CACHE_MAX = 64
# (move the _MAX_PLAYBACK_LAG_S constant + its comment block here from player.py)

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False


def rawvideo_cmd(ffmpeg, video_path, seek_s, scale_w, scale_h,
                 color_vf="", vframes=None, hwaccel=False):
    """FFmpeg command producing rgb24 rawvideo on stdout.

    hwaccel: best-effort GPU decode (playback pipe); FFmpeg falls back to
    software on its own if no hwaccel is available.
    """
    cmd = [ffmpeg]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-ss", str(seek_s), "-i", video_path]
    vf_parts = [f for f in [color_vf, f"scale={scale_w}:{scale_h}"] if f]
    cmd += ["-vf", ",".join(vf_parts)]
    if vframes is not None:
        cmd += ["-vframes", str(vframes)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-v", "quiet", "pipe:1"]
    return cmd


def rgba_from_rgb24(data, w, h):
    """bytes -> RGBA PIL image, or None if data is short or PIL is missing."""
    nbytes = w * h * 3
    if not PIL_OK or not data or len(data) < nbytes:
        return None
    return PILImage.frombytes("RGB", (w, h), data[:nbytes]).convert("RGBA")
```

**In `player.py`:**

1. Import the new names; delete the local `_SL_MAX`, `_CACHE_MAX`,
   `_MAX_PLAYBACK_LAG_S` definitions and the local `try: from PIL...` block
   (import `PIL_OK`/`PILImage` from `_player_common` — `VideoCanvas` uses them).
   Keep `_SL_MAX` importable from `player` (i.e. `from _player_common import _SL_MAX`
   at module level is enough).
2. Replace the three inline FFmpeg command constructions with `rawvideo_cmd(...)`:
   - `_extract_at_pct._run`: `vframes=1, hwaccel=False`
   - `_prefetch_frames`: `vframes=1, hwaccel=False`
   - `play()`: `vframes=None, hwaccel=True`
   Note: the two extraction sites currently lack `-an`; the unified builder adds it.
   That is a deliberate no-op (video-only rawvideo pipe) — accept it.
3. Replace the four `frombytes(...).convert("RGBA")` sites (`_extract_at_pct`,
   `_prefetch_frames`, `_reader_thread`) with `rgba_from_rgb24(...)`. In
   `_reader_thread`, a `None` return should `continue` (matches current behavior:
   short reads already `break` before decode; PIL_OK is guaranteed there in practice —
   guard anyway).
   The `PIL_OK` checks folded into the helper replace the explicit `and PIL_OK` tests.
4. Hoist the `import bisect` from inside `_reader_thread` to the module imports.
5. Replace the three `getattr(self, '_color_vf', '')` with `self._color_vf`
   (it is always set in `__init__`). Same for `getattr(self, "full_quality_preview", False)`
   in `_extraction_dims`, and remove the `if hasattr(self, "cached_frames")` guard in
   `set_full_quality_preview` (always true after `__init__`).

**Add `tests/test_player_common.py`:** assert `rawvideo_cmd` output exactly for
(a) extraction (no hwaccel, vframes=1, no color_vf), (b) playback (hwaccel, no
vframes), (c) color_vf present → it precedes `scale=` in the `-vf` argument, joined
with a comma. Assert `rgba_from_rgb24` returns `None` on short data and a `(w,h)`
RGBA image on exact-size data (construct 2×2 bytes).

**Verify (§9), then commit:**
`refactor(player): extract shared FFmpeg command + frame decode helpers`

---

## Stage 2 — pure pacing function + tests

**Create `_player_playback.py`** (header + docstring) with the constant moved in
stage 1 available via `from _player_common import _MAX_PLAYBACK_LAG_S` and this
**exact** function:

```python
def plan_frame(now, t0, rel, frame_dur, speed,
               max_lag=_MAX_PLAYBACK_LAG_S):
    """Decide what to do with a decoded frame on the playback clock.

    now       — time.monotonic() at decode
    t0        — playback clock anchor (None before the first frame)
    rel       — frame's offset from the seek point, seconds, already
                divided by speed
    frame_dur — nominal seconds per source frame
    speed     — playback speed factor

    Returns (action, t0, deadline) where action is "show" or "drop".
    t0 is (re-)anchored here: on the first frame, and whenever the
    pipeline has fallen further behind than max_lag — dropping forever
    would freeze the preview, so we re-anchor and show the frame
    (slower than real time beats a frozen preview).
    """
    if t0 is None:
        t0 = now - rel
    deadline = t0 + rel
    late = now - deadline
    if late > frame_dur / speed:
        if late <= max_lag:
            return "drop", t0, deadline
        t0 = now - rel
        deadline = now
    return "show", t0, deadline
```

**Rewire `PlayerController._reader_thread`** to call it — the block from `now =
time.monotonic()` through the late/re-anchor logic becomes:

```python
now = time.monotonic()
first = t0 is None
action, t0, deadline = plan_frame(now, t0, rel, frame_dur, speed)
if first or self._ffplay_t0 != t0:
    self._ffplay_t0 = t0   # keep _display_tick's position clock in step
if action == "drop":
    continue
```

(Keep the surrounding comment block about bounded dropping — move it onto
`plan_frame` if it reads better there; don't duplicate it.)

**Add `tests/test_player_pacing.py`** covering at minimum:
1. First frame anchors: `t0=None` → `show`, `t0 == now - rel`, `deadline == now`.
2. On-time frame → `show`, t0 unchanged.
3. Late by less than one frame interval → `show` (not drop).
4. Late by 2× frame interval but under max_lag → `drop`, t0 unchanged.
5. Late beyond max_lag → `show` with re-anchored `t0 == now - rel`, `deadline == now`.
6. speed=2.0 halves the effective drop threshold (late of 1.5×frame_dur/2 → drop).
Use plain floats (e.g. now=100.0), no mocking needed — the function is pure.

**Verify, commit:**
`refactor(player): extract pure playback pacing decision + tests`

---

## Stage 3 — `FrameCache`

**In `_player_extract.py`** (new file, header + docstring): a dict-like cache so
existing call sites keep their syntax:

```python
from _player_common import _CACHE_MAX

class FrameCache:
    """pct -> PIL image cache with distance-based eviction.

    Dict-like on purpose: PlayerController used a bare dict before; call
    sites use `in`, indexing, .get, .keys, iteration, len, .clear.
    NOT thread-safe — same accepted quirk as the dict it replaces.
    """
    def __init__(self, max_items=_CACHE_MAX):
        self._d = {}
        self._max = max_items
    def __contains__(self, pct): ...
    def __getitem__(self, pct): ...
    def __setitem__(self, pct, img): ...
    def __delitem__(self, pct): ...
    def __iter__(self): return iter(self._d)
    def __len__(self): ...
    def get(self, pct, default=None): ...
    def keys(self): ...
    def clear(self): ...
    def evict_around(self, current_pct):
        """Evict entries farthest from current_pct until within max size."""
        while len(self._d) > self._max:
            worst = max(self._d, key=lambda k: abs(k - current_pct))
            del self._d[worst]
    def nearest(self, pct):
        """Key closest to pct, or None if empty."""
        return min(self._d, key=lambda k: abs(k - pct)) if self._d else None
```

**In `player.py`:**
- `self.cached_frames = FrameCache()` (drop `self._cache_max`).
- Delete `_evict_cache`; its one call site in `_extract_at_pct` becomes
  `self.cached_frames.evict_around(pct)`.
- Replace the two open-coded `min(self.cached_frames, key=...)` nearest-lookups
  (`_on_seek`, `pause`) with `self.cached_frames.nearest(...)` (in `pause` the
  current code does `min(self.cached_frames.keys(), ...)` — same thing).
- Timeline notification calls (`self.timeline.add_cached/set_cached`) stay in the
  controller exactly where they are — the cache stays view-free.

**Add `tests/test_frame_cache.py`:** eviction keeps the entries nearest to
`current_pct` and caps at `max_items`; `nearest()` on empty → None, otherwise the
closest key; dict-like ops round-trip.

**Verify, commit:**
`refactor(player): extract FrameCache with distance-based eviction`

---

## Stage 4 — split player.py into modules

Pure moves — **no logic edits** in this stage.

### 4a. `_player_views.py`

Move: the `_DONATE_*` constants, `VideoCanvas`, `Timeline`, `TransportBar`, plus
exactly the imports they need (Qt classes, `QUrl`/`QDesktopServices`/`random` for the
donate note, `PIL_OK`/`PILImage` from `_player_common`). `Timeline` uses no player
constants; `VideoCanvas` doesn't either — check with grep before assuming.

### 4b. `PlaybackEngine` into `_player_playback.py`

Move play/pipe/reader/display internals out of `PlayerController` into:

```python
class PlaybackEngine:
    """Owns the playback FFmpeg pipe, reader thread, frame buffer and the
    60 Hz display timer. Reads controller state (video path/dims/fps/dur,
    pts_list, color_vf, composite_fn, speed) via `self._c`; drives the
    canvas/timeline/transport through it. Created once per controller."""
    def __init__(self, ctrl):
        self._c = ctrl
        ...
```

Method distribution:

| Stays on PlayerController (public API / coordination) | Moves to PlaybackEngine |
|---|---|
| `toggle_play` (delegates: `if self.engine.playing: self.pause() else: self.play()`) | `play(seek_s)` — body of current `play()` minus the seek computation |
| `play()` → computes `seek_s` from timeline, calls `self.engine.play(seek_s)` | `_build_play_snap` |
| `pause()` — sets transport, calls `self.engine.stop_pipe()`, keeps the cached-frame/OSD restore block | `_reader_thread` |
| `stop`, seek/step/shuttle/trim methods, `_do_refresh`, `_update_osd` etc. | `_display_tick` + `_display_timer` (created in engine `__init__`) |
| | `stop_pipe()` (current `_stop_pipe`) |
| | `_on_playback_finished` → engine method that calls `self._c.pause()` etc. |

State mapping (rename inside moved code): `self._ffplay_proc/_ffplay_stop/_ffplay_t0/
_ffplay_seek/_ffplay_w/_ffplay_h/_buf_lock/_buf_frame/_buf_pct/_display_timer` →
engine attributes (drop the `_ffplay_` prefix where it reads naturally, or keep the
names — your choice, but be consistent). `self._playing` → `engine.playing`.
Controller state accessed from moved code becomes `self._c.X`: `video_path`,
`video_dur`, `video_fps`, `timeline`, `transport`, `canvas`, `_composite_fn`,
`_speed`, `_color_vf`, `pts_list`, `_pts_ready`, `_extraction_dims`, `pause`.

Compatibility shims on the controller (main.py and intra-file callers use these):
`self._playing` is referenced by `_display_tick` (moves with it) — grep `player.py`
and `main.py` for `_playing` when done; nothing outside the moved code may reference
it, otherwise add a property `_playing` → `self.engine.playing`.

`_stop_pipe` is called from `play()` (old body) and `pause()`; both routes now go
through `engine`. Grep for remaining `_ffplay`/`_buf_`/`_display_timer` references in
`player.py` after the move — there must be none.

### 4c. `player.py` final shape

- Imports: `from _player_views import VideoCanvas, Timeline, TransportBar`,
  `from _player_extract import FrameCache`, `from _player_playback import
  PlaybackEngine, plan_frame`, `from _player_common import _SL_MAX, ...`.
- Module docstring updated to describe the split (mention each `_player_*` module in
  the CLAUDE.md architecture style).
- `PlayerController.__init__` signature unchanged; constructs `self.engine =
  PlaybackEngine(self)` where the playback state block used to be.
- Keep `VideoCanvas`, `Timeline`, `TransportBar`, `_SL_MAX` importable from `player`
  (the re-import above does this).

### 4d. Update CLAUDE.md

Architecture tree: under `player.py`, list the new `_player_*` submodules the same way
`widgets.py` lists `_widget_*`.

**Verify (§9 in full — including the import-cycle check), commit:**
`refactor(player): split views, frame cache, and playback engine into _player_* modules`

---

## Stage 5 (OPTIONAL — attempt only if stages 1–4 verified clean)

### VideoCanvas editor split — `_player_canvas_edit.py`

`VideoCanvas` mixes display with two mouse-driven editors. Extract them as helper
objects owned by the canvas (composition, not mixins):

- `RegionEditor`: state `_region_edit_mode`, regions list, px↔cell callbacks, drag
  rect; methods `paint(p, canvas, theme)` (current `_paint_regions`), `hit_test`,
  and the region branches of `mousePressEvent` / `mouseMoveEvent` /
  `mouseReleaseEvent`.
- `WidgetEditor`: state `_widget_edit`, widgets list, selection idx, src-rect fn,
  drag/resize state; methods `paint` (current `_paint_widget_chrome`),
  `_widget_rect_px`, `_widget_hit_test`, `_widget_resize_edge_at`, `_resize_cursor`,
  and the widget branches of the three mouse handlers.

The canvas keeps every public method/signal listed in §1 and delegates:
`mousePressEvent` asks `region_editor.mouse_press(e)` then
`widget_editor.mouse_press(e)`, each returning True if handled — **preserve the
current if/elif precedence between region mode and widget mode exactly as it is in
the existing handlers** (read them first; region edit mode takes priority while
active). Signals stay on the canvas; editors call `canvas.regionAdded.emit(...)` etc.
through a back-reference.

This stage cannot be fully verified without a human: after the automated protocol,
**stop and ask the user** to manually test region draw/remove and widget
drag/resize/select before committing. If the user is not available, leave this stage
uncommitted on a branch `refactor/canvas-editors` and say so in the report.

Commit (only after user OK):
`refactor(player): split canvas region/widget editors into _player_canvas_edit`

---

## Stage 6 — extract the UI style layer from main.py → `ui_style.py`

Motivation: `main.py` (4303 lines) owns theme tokens, scale, icons, and the
stylesheet builder as mutable module globals; nothing can be extracted from it while
widgets read those globals. This stage creates the leaf module everything else can
import.

**Create `ui_style.py`** and move, verbatim: `_DARK_THEME`, `_UI_SCALE`, `_fs`,
`_T`, `_build_styles` **and every global it assigns** (read the function; the list
includes `APP_STYLE, GROUP_STYLE, PATH_EMPTY, PATH_FILLED, BTN_SEC, BTN_PRIMARY,
BTN_PLAY, BTN_STOP, BTN_DANGER` and any others such as `SLIDER_STYLE`),
`_icons_dir`, `_icon`. Add accessors:

```python
def set_ui_scale(v): global _UI_SCALE; _UI_SCALE = v
def ui_scale(): return _UI_SCALE
def set_dark_theme(v): global _DARK_THEME; _DARK_THEME = v
def is_dark(): return _DARK_THEME
```

**Rewiring main.py — the stale-binding trap (read carefully):**

- Functions (`_T`, `_fs`, `_icon`, `_icons_dir`, `_build_styles`) are never rebound —
  importing them by name is safe: `from ui_style import _T, _fs, _icon, _icons_dir,
  _build_styles`.
- The style **constants are rebound at runtime** by `_build_styles()`. A
  `from ui_style import BTN_SEC` would freeze the value at import time and the theme
  system would silently break. Therefore: `import ui_style as S` and rewrite every
  bare reference in main.py — `BTN_SEC` → `S.BTN_SEC`, `APP_STYLE` → `S.APP_STYLE`,
  etc., for **every** constant `_build_styles` assigns. Do it constant-by-constant
  with word-boundary search (`rg -n "\bBTN_SEC\b" main.py`), and after each one
  verify zero bare references remain outside f-string contents that merely *mention*
  the token name (there shouldn't be any — the styles are opaque CSS strings).
- `_DARK_THEME` / `_UI_SCALE` mutations in main.py (`_toggle_theme`,
  `_on_scale_changed`, `_load_settings`, `_save_settings`, `_apply_theme`'s
  moon-icon choice) become accessor calls (`S.set_dark_theme(...)`, `S.is_dark()`,
  `S.set_ui_scale(...)`, `S.ui_scale()`).
- player.py is **not** rewired — `PlayerPanel` keeps receiving `theme_fn`, `fs_fn`,
  etc. by injection (public API invariant). main.py now passes `S._T`, `S._fs`,
  `lambda: S.BTN_PLAY`, `lambda: S.BTN_SEC` — note the lambdas already exist for
  exactly this reason; keep them lambdas.
- `theme.py` interplay: `_T()` reads from `theme` module (`_theme_mod`). Move that
  import along; `ui_style` may import `theme`, never the reverse.

Verification additions for this stage: launch smoke (§9) **plus** grep-based
assertions: `rg -c "\bBTN_" main.py` must equal the count of `S.BTN_` occurrences
(no bare names), and `python -c "import ui_style"` works headlessly. Ask the user to
manually verify theme toggle + UI scale change at the end; note it in the report if
they haven't yet.

**Commit:** `refactor(ui): extract theme/scale/stylesheet layer into ui_style.py`

---

## Stage 7 — extract generic widgets from main.py → `ui_widgets.py`

Move classes (verbatim, with their imports): `FileRow`, `DropZone`, `LabeledSlider`,
`CheckableComboBox`, `CollapsibleGroupBox`, `InfoCard`. They reference the style
layer — inside the moved code, rewrite those references the same way as stage 6
(`import ui_style as S`; `S.PATH_EMPTY`, `S._T()`, `S._fs(...)`, `S.SLIDER_STYLE`,
`S.GROUP_STYLE`...). Constants at *use time* via `S.`, functions may be imported by
name. `RenderBar`, `ProcessWorker`, `VideoInfoWorker`, updater workers, and
`MainWindow` stay in main.py.

main.py then does `from ui_widgets import FileRow, DropZone, LabeledSlider,
CheckableComboBox, CollapsibleGroupBox, InfoCard` (class objects are never rebound —
safe to import by name).

`ui_widgets` may import `ui_style`; never `main`. Check: `rg -n "import main"`
must stay empty.

**Commit:** `refactor(ui): move generic widgets from main.py into ui_widgets.py`

---

## 8. Explicitly OUT of scope (do not do these)

- **video_processor.py export pipelines.** `_overlay_pipeline` /
  `_chroma_key_pipeline` / `_srt_only_pipeline` share structure, but they are the
  revenue path (rendering) with no automated coverage; deduplicating them without an
  export regression harness is how output gets silently corrupted. Leave them.
- Decomposing `MainWindow` itself beyond stages 6–7 (file intake, widget-props panel,
  render orchestration). Follow-up project.
- Adding type hints, docstring sweeps, import sorting, or reformatting untouched code.
- Any threading model changes (QThread migration, locks, executors).
- Renaming public attributes/signals, "fixing" the `timeline._position` private
  access from main.py, or changing settings/JSON formats.

## 9. Verification protocol (run before EVERY commit)

```powershell
# 1. Byte-compile everything touched (plus main.py always):
python -m py_compile main.py player.py ui_style.py ui_widgets.py _player_common.py _player_views.py _player_extract.py _player_playback.py   # only files that exist
# 2. Import graph is acyclic / modules import cleanly, headless:
python -c "import player"
# 3. Tests:
python -m unittest discover tests
# 4. Launch smoke (GUI appears on the user's desktop for ~10 s — that is expected):
$p = Start-Process python -ArgumentList "main.py" -PassThru
Start-Sleep -Seconds 12
if ($p.HasExited) { throw "app exited on launch: $($p.ExitCode)" }
Stop-Process -Id $p.Id -Force
```

Also stage-specific greps given inline above. If the smoke launch throws, read the
console by running `python main.py` in the foreground briefly and capture the traceback.

## 10. Final report to the user

After the last stage, summarize: stages completed/committed (with hashes), stages
skipped or reverted and why, bugs noticed but not fixed, and this **manual QA
checklist** for the user (things automation can't cover):

- Drop video + `.osd` + `.srt` together (differently-named files) → all three load.
- Play / pause / scrub / step / shuttle (J-K-L), trim in/out — behaves as before.
- Playback of the high-bitrate RubyFPV `rec_*.mp4` stays smooth (~real time).
- Region editor: draw, move, remove a hide-region. Widget editor: select, drag,
  resize a widget; map widget still renders.
- Theme toggle (dark/light), theme editor apply, UI scale 100→150%, all buttons
  restyle (watch for stale colors — that's the stale-binding trap of stage 6).
- Short export render completes and the OSD is positioned as before.

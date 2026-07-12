# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
player.py — Video player for VueOSD.

Contains: PlayerController (coordinator) and PlayerPanel (container widget).
The pieces it coordinates live in private submodules:

    _player_common.py    — constants, PIL guard, FFmpeg command + frame decode
    _player_views.py     — VideoCanvas, Timeline, TransportBar
    _player_extract.py   — FrameCache (distance-based eviction)
    _player_playback.py  — plan_frame pacing decision + PlaybackEngine

VideoCanvas, Timeline, TransportBar and _SL_MAX stay importable from here.
"""

import threading, subprocess

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QTimer

from _player_common import _SL_MAX, rawvideo_cmd, rgba_from_rgb24
from _player_views import VideoCanvas, Timeline, TransportBar
from _player_extract import FrameCache
from _player_playback import PlaybackEngine, plan_frame
from subprocess_utils import _hidden_popen
from video_processor import find_ffmpeg, get_frame_pts
from osd_renderer import OsdRenderConfig, render_osd_frame, render_fallback


# ─── PlayerController ─────────────────────────────────────────────────────────

class PlayerController:
    """Manages playback state, FFmpeg processes, frame cache, and seeking.

    This is NOT a QObject — it is owned by PlayerPanel and called directly.
    It uses QTimer and threading internally.
    """

    def __init__(self, canvas: VideoCanvas, timeline: Timeline,
                 transport: TransportBar, composite_fn, osd_fn=None):
        self.canvas = canvas
        self.timeline = timeline
        self.transport = transport
        self._composite_fn = composite_fn  # callable(pil_img, pct) -> pil_img (playback)
        self._osd_fn = osd_fn              # callable(pct, w, h) -> PIL RGBA (preview overlay)

        # Video state
        self.video_path = ""
        self.video_dur = 0.0
        self.video_fps = 60.0
        self.video_w = 0
        self.video_h = 0
        self.full_quality_preview = False  # True = decode at native res
        self._color_vf = ""   # color correction video filter string

        # Cache
        self.cached_frames = FrameCache()   # pct -> PIL Image

        # Seeking
        self._extract_proc = None
        self._scrub_timer = QTimer()
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(30)
        self._scrub_timer.timeout.connect(self._do_scrub)
        self._pending_pct = 0

        # Drag-aware resolution: 480p while dragging, full after 200ms idle
        self._drag_active = False
        self._drag_idle_timer = QTimer()
        self._drag_idle_timer.setSingleShot(True)
        self._drag_idle_timer.setInterval(200)
        self._drag_idle_timer.timeout.connect(self._on_drag_idle)

        # PTS list for accurate OSD sync on videos with packet loss
        self.pts_list = []       # per-frame PTS in seconds
        self._pts_ready = False

        # Prefetch
        self._prefetch_stop = False

        # Playback
        self._speed = 1.0
        self._speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
        self._speed_idx = 2  # index into _speed_levels (1.0x)
        self.engine = PlaybackEngine(self)

        # Preview debounce (30ms — OSD-only rendering is fast)
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(30)
        self._preview_timer.timeout.connect(self._do_refresh)

        # Wire signals
        self.timeline.seekRequested.connect(self._on_seek)
        self.canvas.resized.connect(self._on_canvas_resized)
        self.transport.playClicked.connect(self.toggle_play)
        self.transport.restartClicked.connect(self.seek_start)
        self.transport.skipEndClicked.connect(self.seek_end)
        self.transport.stepBackClicked.connect(lambda: self.step_backward(1))
        self.transport.stepFwdClicked.connect(lambda: self.step_forward(1))

    # ── Video loading ─────────────────────────────────────────────────────────

    def set_color_vf(self, vf: str):
        """Set color correction video filter for preview extraction.

        When set, the filter is prepended to -vf in all FFmpeg extract/play commands.
        Pass "" to disable.
        """
        changed = self._color_vf != vf
        self._color_vf = vf
        if changed and self.video_path:
            # Invalidate cache and refresh
            self.cached_frames.clear()
            self.timeline.set_cached(set())
            self._do_refresh()
            self._start_prefetch()

    def set_full_quality_preview(self, enabled: bool):
        """Toggle native-resolution preview decoding. Drops the cache so the
        next refresh re-extracts at the new quality."""
        new = bool(enabled)
        if new == self.full_quality_preview:
            return
        self.full_quality_preview = new
        self.cached_frames.clear()
        self.timeline.set_cached(set())
        if self.video_path and self.video_w > 0 and self.video_h > 0:
            pct = int(self.timeline._position * _SL_MAX)
            self._extract_at_pct(pct)

    def load_video(self, path, duration, fps, width=0, height=0):
        """Called after video info is available."""
        self.stop()
        self.video_path = path
        self.video_dur = duration
        self.video_fps = fps
        self.video_w = width or 0
        self.video_h = height or 0
        self.cached_frames.clear()
        self.pts_list = []
        self._pts_ready = False
        self.timeline.set_position(0.0)
        self.timeline.set_cached(set())
        self.transport.set_time(0, duration)
        self._extract_at_pct(0)
        # Start prefetch after a short delay
        QTimer.singleShot(400, self._start_prefetch)
        # Probe PTS in background for accurate OSD sync
        threading.Thread(target=self._probe_pts, args=(path,), daemon=True).start()

    def clear(self):
        self.stop()
        self.video_path = ""
        self.video_dur = 0.0
        self.cached_frames.clear()
        self.pts_list = []
        self._pts_ready = False
        self.canvas.set_placeholder()
        self.timeline.set_position(0.0)
        self.timeline.set_cached(set())

    def _probe_pts(self, path):
        """Background: probe per-frame PTS for accurate OSD sync."""
        pts = get_frame_pts(path)
        if pts and path == self.video_path:
            self.pts_list = pts
            self._pts_ready = True

    def pct_to_video_time_ms(self, pct):
        """Convert slider pct (0–_SL_MAX) to video time in ms.

        Uses PTS list when available (accurate for videos with dropped frames),
        falls back to linear mapping otherwise.
        """
        if self._pts_ready and self.pts_list:
            idx = int(pct / _SL_MAX * (len(self.pts_list) - 1))
            idx = max(0, min(idx, len(self.pts_list) - 1))
            return int(self.pts_list[idx] * 1000)
        return int(self.video_dur * pct / _SL_MAX * 1000) if self.video_dur > 0 else 0

    def _extraction_dims(self, cap=1080):
        """Return (scale_w, scale_h) for frame extraction, preserving the
        video's native aspect ratio.  Falls back to canvas AR if video
        dimensions are unknown.
        When `full_quality_preview` is set, returns native source dims so
        the preview shows pixel-perfect frames at the cost of decode speed."""
        if self.full_quality_preview and self.video_w > 0 and self.video_h > 0:
            sw = self.video_w - (self.video_w % 2)
            sh = self.video_h - (self.video_h % 2)
            return max(2, sw), max(2, sh)
        if self.video_w > 0 and self.video_h > 0:
            ar = self.video_w / self.video_h
        else:
            cw = max(self.canvas.width(), 320)
            ch = max(self.canvas.height(), 180)
            ar = cw / ch
        scale_h = min(max(self.canvas.height(), 180), cap)
        scale_w = int(scale_h * ar)
        # FFmpeg requires even dimensions
        scale_w -= scale_w % 2
        scale_h -= scale_h % 2
        return max(2, scale_w), max(2, scale_h)

    # ── Seeking ───────────────────────────────────────────────────────────────

    def _on_seek(self, pos):
        """Called when user clicks/drags the timeline."""
        if self.engine.playing:
            self.pause()
        pct = int(pos * _SL_MAX)
        self._update_time_labels(pct)

        # Mark drag active; reset idle timer
        self._drag_active = True
        self._drag_idle_timer.start()

        if pct in self.cached_frames:
            self._show_pct(pct)
            return

        # Show nearest cached frame as placeholder while extracting
        if self.cached_frames:
            nearest = self.cached_frames.nearest(pct)
            self._show_pct(nearest)

        self._pending_pct = pct
        self._scrub_timer.start()

    def _do_scrub(self):
        pct = self._pending_pct
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def _on_drag_idle(self):
        """Drag ended — re-extract current position at full resolution."""
        self._drag_active = False
        pct = int(self.timeline._position * _SL_MAX)
        # Re-extract at full quality (evict the low-res cached version)
        if pct in self.cached_frames:
            del self.cached_frames[pct]
        self._extract_at_pct(pct)

    def _extract_at_pct(self, pct):
        if not self.video_path or not find_ffmpeg():
            return
        ffmpeg = find_ffmpeg()
        t = self.video_dur * pct / _SL_MAX if self.video_dur > 0 else 0.0

        # Kill in-flight extraction
        prev = self._extract_proc
        if prev is not None and prev.poll() is None:
            try:
                prev.kill()
            except Exception:
                pass
        self._extract_proc = None

        cap = 480 if self._drag_active else 1080
        scale_w, scale_h = self._extraction_dims(cap)
        nbytes = scale_w * scale_h * 3

        def _run():
            try:
                cmd = rawvideo_cmd(ffmpeg, self.video_path, t, scale_w, scale_h,
                                   color_vf=self._color_vf, vframes=1, hwaccel=False)
                proc = _hidden_popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                self._extract_proc = proc
                data = proc.stdout.read(nbytes)
                proc.wait(timeout=20)
                img = rgba_from_rgb24(data, scale_w, scale_h)
                if img is not None:
                    self.cached_frames[pct] = img
                    self.cached_frames.evict_around(pct)
                    self.timeline.add_cached(pct / _SL_MAX)
                    def _ready(p=pct):
                        self._show_pct(p)
                    QTimer.singleShot(0, _ready)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def _show_pct(self, pct):
        img = self.cached_frames.get(pct)
        if img is None:
            return
        self.canvas.set_base_frame(img)
        self._update_osd(pct)

    def _update_time_labels(self, pct):
        t_s = self.video_dur * pct / _SL_MAX if self.video_dur > 0 else 0.0
        self.transport.set_time(t_s, self.video_dur)

    # ── Prefetch ──────────────────────────────────────────────────────────────

    def _start_prefetch(self):
        if not self.video_path or self.video_dur <= 0 or not find_ffmpeg():
            return
        positions = list(range(0, _SL_MAX + 1, _SL_MAX // 20))
        to_fetch = [p for p in positions if p not in self.cached_frames]
        if not to_fetch:
            return
        self._prefetch_stop = False
        threading.Thread(target=self._prefetch_frames, args=(to_fetch,),
                         daemon=True).start()

    def _prefetch_frames(self, positions):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return
        scale_w, scale_h = self._extraction_dims(1080)
        nbytes = scale_w * scale_h * 3

        for pct in positions:
            if self._prefetch_stop:
                break
            if pct in self.cached_frames:
                continue
            t = self.video_dur * pct / _SL_MAX
            try:
                cmd = rawvideo_cmd(ffmpeg, self.video_path, t, scale_w, scale_h,
                                   color_vf=self._color_vf, vframes=1, hwaccel=False)
                proc = _hidden_popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                data = proc.stdout.read(nbytes)
                proc.wait(timeout=15)
                img = None if self._prefetch_stop else rgba_from_rgb24(data, scale_w, scale_h)
                if img is not None:
                    self.cached_frames[pct] = img
                    self.timeline.add_cached(pct / _SL_MAX)
            except Exception:
                pass

    # ── Playback ──────────────────────────────────────────────────────────────

    def toggle_play(self):
        if not self.video_path or self.video_dur <= 0:
            return
        if self.engine.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        seek_s = self.video_dur * self.timeline._position
        self.engine.play(seek_s)

    def pause(self):
        self.engine.stop_display()
        self.transport.set_playing(False)
        self.engine.stop_pipe()
        # After playback, restore raw base frame for two-layer mode
        # (playback set_frame bakes OSD into the base — must replace it)
        if self._osd_fn and self.cached_frames:
            pct = int(self.timeline._position * _SL_MAX)
            raw = self.cached_frames.get(pct)
            if raw is None:
                raw = self.cached_frames.get(self.cached_frames.nearest(pct))
            if raw is not None:
                self.canvas.set_base_frame(raw)
                self._update_osd(pct)

    def stop(self):
        self.pause()
        self._prefetch_stop = True

    # ── Seek helpers ──────────────────────────────────────────────────────────

    def seek_start(self):
        self.pause()
        self.timeline.set_position(0.0)
        pct = 0
        self._update_time_labels(pct)
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def seek_end(self):
        self.pause()
        self.timeline.set_position(1.0)
        pct = _SL_MAX
        self._update_time_labels(pct)
        if pct in self.cached_frames:
            self._show_pct(pct)
        else:
            self._extract_at_pct(pct)

    def step_forward(self, n=1):
        if self.engine.playing:
            self.pause()
        if self.video_dur <= 0 or self.video_fps <= 0:
            return
        frame_step = n / (self.video_fps * self.video_dur) if self.video_dur > 0 else 0.01
        new_pos = min(1.0, self.timeline._position + frame_step)
        self.timeline.set_position(new_pos)
        pct = int(new_pos * _SL_MAX)
        self._update_time_labels(pct)
        self._pending_pct = pct
        self._scrub_timer.start()

    def step_backward(self, n=1):
        if self.engine.playing:
            self.pause()
        if self.video_dur <= 0 or self.video_fps <= 0:
            return
        frame_step = n / (self.video_fps * self.video_dur) if self.video_dur > 0 else 0.01
        new_pos = max(0.0, self.timeline._position - frame_step)
        self.timeline.set_position(new_pos)
        pct = int(new_pos * _SL_MAX)
        self._update_time_labels(pct)
        self._pending_pct = pct
        self._scrub_timer.start()

    # ── Shuttle (J/K/L) ──────────────────────────────────────────────────────

    def shuttle_faster(self):
        if self._speed_idx < len(self._speed_levels) - 1:
            self._speed_idx += 1
        self._speed = self._speed_levels[self._speed_idx]
        self.transport.set_speed(self._speed)
        if not self.engine.playing:
            self.play()

    def shuttle_slower(self):
        """J key — step backward (FFmpeg can't pipe reverse frames)."""
        self.step_backward(1)

    def shuttle_reset(self):
        """K key — toggle play/pause and reset speed to 1x."""
        self._speed_idx = 2
        self._speed = 1.0
        self.transport.set_speed(1.0)
        self.toggle_play()

    # ── Preview refresh ───────────────────────────────────────────────────────

    def queue_refresh(self):
        """Debounced preview refresh — called when OSD settings change."""
        self._preview_timer.start()

    def _do_refresh(self):
        """Refresh OSD overlay only — base frame stays cached."""
        pct = int(self.timeline._position * _SL_MAX)
        # Ensure we have a base frame displayed
        if not self.canvas.has_frame() and self.cached_frames:
            nearest = self.cached_frames.nearest(pct)
            self.canvas.set_base_frame(self.cached_frames[nearest])
        self._update_osd(pct)

    def _update_osd(self, pct=None):
        """Render OSD overlay at canvas display size and apply it."""
        if pct is None:
            pct = int(self.timeline._position * _SL_MAX)
        if self._osd_fn and self.canvas.has_frame():
            w, h = self.canvas.display_size()
            if w > 0 and h > 0:
                osd_img = self._osd_fn(pct, w, h)
                self.canvas.set_osd_overlay(osd_img)

    def _on_canvas_resized(self):
        """Canvas was resized — re-render OSD at new display size."""
        self._preview_timer.start()

    def refresh_now(self):
        """Immediate preview refresh (no debounce)."""
        self._do_refresh()

    # ── Trim helpers ──────────────────────────────────────────────────────────

    def set_trim_in(self):
        self.timeline.set_in(self.timeline._position)

    def set_trim_out(self):
        self.timeline.set_out(self.timeline._position)


# ─── PlayerPanel ──────────────────────────────────────────────────────────────

class PlayerPanel(QWidget):
    """Container widget: VideoCanvas + Timeline + TransportBar.

    Drop-in replacement for the old centre panel's player area.
    """

    def __init__(self, theme_fn, icon_fn, btn_play_fn, btn_sec_fn,
                 fs_fn, composite_fn, osd_fn=None, parent=None):
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.canvas = VideoCanvas(theme_fn, fs_fn)
        self.timeline = Timeline(theme_fn)
        self.transport = TransportBar(theme_fn, icon_fn, btn_play_fn, btn_sec_fn)
        self.controller = PlayerController(
            self.canvas, self.timeline, self.transport, composite_fn, osd_fn)

        # Trim labels row
        self._trim_row = QWidget()
        trim_lay = QHBoxLayout(self._trim_row)
        trim_lay.setContentsMargins(0, 0, 0, 0)
        trim_lay.setSpacing(4)
        t = theme_fn()
        self.trim_in_lbl = QLabel("In: 0:00")
        self.trim_out_lbl = QLabel("Out: \u2014")
        for lb in (self.trim_in_lbl, self.trim_out_lbl):
            lb.setStyleSheet(f"color:{t['subtext']};font-size:10px;font-weight:bold;")
        self.trim_rst_btn = QPushButton("\u2715")
        self.trim_rst_btn.setFixedSize(20, 20)
        self.trim_rst_btn.setStyleSheet(btn_sec_fn())
        self.trim_rst_btn.setToolTip("Reset trim to full video")
        self.trim_rst_btn.clicked.connect(self.timeline.reset)

        trim_lay.addWidget(self.trim_in_lbl)
        trim_lay.addStretch()
        trim_lay.addWidget(self.trim_out_lbl)
        trim_lay.addWidget(self.trim_rst_btn)

        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.timeline)
        lay.addWidget(self._trim_row)
        lay.addWidget(self.transport)

    def refresh_theme(self):
        t = self.canvas._T()
        self.canvas.update()
        self.timeline.update()
        self.transport.refresh_theme()
        for lb in (self.trim_in_lbl, self.trim_out_lbl):
            lb.setStyleSheet(f"color:{t['subtext']};font-size:10px;font-weight:bold;")

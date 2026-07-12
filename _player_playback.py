# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_playback.py — Playback pipe, reader thread and display pump.

Contains: plan_frame (pure pacing decision) and PlaybackEngine.
"""

import time, bisect, threading, subprocess

from PyQt6.QtCore import QTimer

from _player_common import (
    _SL_MAX, _MAX_PLAYBACK_LAG_S, rawvideo_cmd, rgba_from_rgb24,
)
from subprocess_utils import _hidden_popen
from video_processor import find_ffmpeg


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


class PlaybackEngine:
    """Owns the playback FFmpeg pipe, reader thread, frame buffer and the
    60 Hz display timer. Reads controller state (video path/dims/fps/dur,
    pts_list, color_vf, composite_fn, speed) via `self._c`; drives the
    canvas/timeline/transport through it. Created once per controller."""

    def __init__(self, ctrl):
        self._c = ctrl

        self.playing = False
        self.proc = None
        self.stop_ev = threading.Event()
        self.t0 = 0.0
        self.seek = 0.0
        self.w = 0
        self.h = 0

        # Display timer (pull model) — 60Hz
        self._display_timer = QTimer()
        self._display_timer.setInterval(16)
        self._display_timer.timeout.connect(self._display_tick)

        # Frame buffer (written by reader thread, read by display timer)
        self._buf_lock = threading.Lock()
        self._buf_frame = None   # PIL image
        self._buf_pct = 0.0

    def play(self, seek_s):
        self.stop_pipe()
        ffmpeg = find_ffmpeg()
        c = self._c
        if not ffmpeg or not c.video_path:
            return

        # Playback runs at 720p to keep pipe traffic + per-frame composite
        # inside the real-time budget (high-bitrate 1080p90 sources overrun
        # it at 1080p). Paused/scrubbed frames still extract at 1080, and
        # full_quality_preview overrides this to native dims.
        scale_w, scale_h = c._extraction_dims(720)

        self.stop_ev.clear()
        self.seek = seek_s
        self.t0 = time.monotonic()
        self.w = scale_w
        self.h = scale_h
        self.playing = True
        c.transport.set_playing(True)

        # Best-effort GPU decode (~3x faster on HEVC); FFmpeg falls back
        # to software decoding on its own if no hwaccel is available.
        cmd = rawvideo_cmd(ffmpeg, c.video_path, seek_s, scale_w, scale_h,
                           color_vf=c._color_vf, vframes=None, hwaccel=True)

        try:
            self.proc = _hidden_popen(
                cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
                bufsize=scale_w * scale_h * 3 * 4,
            )
        except Exception:
            self.playing = False
            c.transport.set_playing(False)
            return

        # Snapshot composite settings
        snap = self._build_play_snap(seek_s, scale_w, scale_h)

        self._display_timer.start()
        threading.Thread(target=self._reader_thread, args=(snap,),
                         daemon=True).start()

    def _build_play_snap(self, seek_s, w, h):
        c = self._c
        return dict(
            seek_s=seek_s,
            w=w, h=h,
            fps=max(1.0, c.video_fps),
            speed=c._speed,
        )

    def _reader_thread(self, snap):
        """Background: read raw frames, composite OSD, deposit in buffer."""
        c = self._c
        proc = self.proc
        stop = self.stop_ev
        w, h = snap["w"], snap["h"]
        nbytes = w * h * 3
        seek = snap["seek_s"]
        fps = snap["fps"]
        speed = snap["speed"]
        frame_dur = 1.0 / fps
        t0 = None   # anchored when the first decoded frame arrives

        # Snapshot PTS list for this playback session
        pts = c.pts_list if c._pts_ready else []
        # Find starting frame index in PTS list
        pts_base_idx = 0
        if pts:
            pts_base_idx = bisect.bisect_left(pts, seek)

        frame_idx = 0
        while not stop.is_set():
            try:
                data = proc.stdout.read(nbytes)
            except Exception:
                break
            if not data or len(data) < nbytes:
                break

            # Video timestamp for this frame and its offset from the seek
            # point (its slot on the playback clock). When per-frame PTS are
            # known, pace by them — VFR sources (Ruby onboard recordings)
            # deliver frames slower than the nominal container fps, and a
            # fixed-fps schedule would mark every frame late and drop it.
            if pts and (pts_base_idx + frame_idx) < len(pts):
                video_time_s = pts[pts_base_idx + frame_idx]
            else:
                video_time_s = seek + frame_idx * frame_dur
            rel = (video_time_s - seek) / speed
            frame_idx += 1

            # Anchor the playback clock to the first DECODED frame, not to
            # play() — FFmpeg spawn + seek + decoder init can take a second,
            # and a clock started back then would put every frame behind
            # schedule from the outset, with no way to recover.
            #
            # Frame dropping: when compositing (OSD grid + widgets + map) can't
            # keep up, skip this frame's composite + display so playback stays
            # locked to real time instead of drifting into slow motion. The raw
            # pipe read above already happened, so FFmpeg keeps streaming — we
            # just discard frames we're too late to show. Threshold is one
            # displayed-frame interval (frame_dur / speed). Dropping is
            # bounded: past _MAX_PLAYBACK_LAG_S the machine simply can't keep
            # up, and dropping forever would freeze the preview — re-anchor
            # the clock to this frame and show it instead.
            now = time.monotonic()
            first = t0 is None
            action, t0, deadline = plan_frame(now, t0, rel, frame_dur, speed)
            if first or self.t0 != t0:
                self.t0 = t0   # keep _display_tick's position clock in step
            if action == "drop":
                continue

            img = rgba_from_rgb24(data, w, h)
            if img is None:
                continue

            try:
                pct_norm = video_time_s / c.video_dur if c.video_dur > 0 else 0.0
                pct_sl = int(pct_norm * _SL_MAX)

                # Composite OSD
                if c._composite_fn:
                    composited = c._composite_fn(img, pct_sl)
                else:
                    composited = img

                with self._buf_lock:
                    self._buf_frame = composited
                    self._buf_pct = pct_norm
            except Exception:
                pass

            # Pace to real-time (adjusted by speed) — only sleep when ahead.
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0.002:
                time.sleep(sleep_s)

        if not stop.is_set():
            QTimer.singleShot(0, self._on_playback_finished)

    def _display_tick(self):
        """Main thread: pull latest frame from buffer, display it."""
        if not self.playing:
            return
        c = self._c

        # Update timeline position from elapsed time
        elapsed = time.monotonic() - self.t0
        video_pos = self.seek + elapsed * c._speed
        if video_pos >= c.video_dur:
            c.pause()
            c.timeline.set_position(1.0)
            c.transport.set_time(c.video_dur, c.video_dur)
            return
        pos_norm = video_pos / c.video_dur if c.video_dur > 0 else 0.0
        c.timeline.set_position(pos_norm)
        c.transport.set_time(video_pos, c.video_dur)

        # Pull frame from buffer
        with self._buf_lock:
            frame = self._buf_frame
            self._buf_frame = None

        if frame is not None:
            c.canvas.set_frame(frame)

    def _on_playback_finished(self):
        c = self._c
        if self.playing:
            c.pause()
            c.timeline.set_position(1.0)
            c.transport.set_time(c.video_dur, c.video_dur)

    def stop_display(self):
        """Stop the 60 Hz display pump — the playback clock stops advancing."""
        self.playing = False
        self._display_timer.stop()

    def stop_pipe(self):
        self.stop_ev.set()
        proc = self.proc
        if proc:
            try:
                proc.stdout.close()
                proc.kill()
            except Exception:
                pass
            self.proc = None

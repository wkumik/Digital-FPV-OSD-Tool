# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_playback.py — Pure playback pacing decision for the player module.
"""

from _player_common import _MAX_PLAYBACK_LAG_S


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

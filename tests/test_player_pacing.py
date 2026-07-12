import unittest

from _player_playback import plan_frame


class PlanFrameTest(unittest.TestCase):
    def test_first_frame_anchors(self):
        action, t0, deadline = plan_frame(now=100.0, t0=None, rel=0.5,
                                          frame_dur=1 / 60, speed=1.0)
        self.assertEqual(action, "show")
        self.assertEqual(t0, 100.0 - 0.5)
        self.assertEqual(deadline, 100.0)

    def test_on_time_frame_shows_and_keeps_t0(self):
        t0_in = 90.0
        now = t0_in + 0.5  # exactly on schedule
        action, t0, deadline = plan_frame(now=now, t0=t0_in, rel=0.5,
                                          frame_dur=1 / 60, speed=1.0)
        self.assertEqual(action, "show")
        self.assertEqual(t0, t0_in)

    def test_late_by_less_than_one_frame_interval_shows(self):
        frame_dur = 1 / 60
        t0_in = 90.0
        deadline = t0_in + 0.5
        now = deadline + frame_dur * 0.5  # late, but under one frame interval
        action, t0, _ = plan_frame(now=now, t0=t0_in, rel=0.5,
                                   frame_dur=frame_dur, speed=1.0)
        self.assertEqual(action, "show")
        self.assertEqual(t0, t0_in)

    def test_late_by_two_frame_intervals_under_max_lag_drops(self):
        frame_dur = 1 / 60
        t0_in = 90.0
        deadline = t0_in + 0.5
        now = deadline + frame_dur * 2  # late by 2x frame interval
        action, t0, _ = plan_frame(now=now, t0=t0_in, rel=0.5,
                                   frame_dur=frame_dur, speed=1.0, max_lag=0.35)
        self.assertEqual(action, "drop")
        self.assertEqual(t0, t0_in)

    def test_late_beyond_max_lag_reanchors_and_shows(self):
        frame_dur = 1 / 60
        t0_in = 90.0
        rel = 0.5
        deadline = t0_in + rel
        now = deadline + 0.5  # beyond max_lag of 0.35
        action, t0, new_deadline = plan_frame(now=now, t0=t0_in, rel=rel,
                                              frame_dur=frame_dur, speed=1.0,
                                              max_lag=0.35)
        self.assertEqual(action, "show")
        self.assertEqual(t0, now - rel)
        self.assertEqual(new_deadline, now)

    def test_speed_halves_effective_drop_threshold(self):
        frame_dur = 1 / 60
        speed = 2.0
        t0_in = 90.0
        rel = 0.5
        deadline = t0_in + rel
        # late of 1.5x frame_dur/speed exceeds the frame_dur/speed threshold
        now = deadline + 1.5 * (frame_dur / speed)
        action, t0, _ = plan_frame(now=now, t0=t0_in, rel=rel,
                                   frame_dur=frame_dur, speed=speed, max_lag=0.35)
        self.assertEqual(action, "drop")
        self.assertEqual(t0, t0_in)


if __name__ == "__main__":
    unittest.main()

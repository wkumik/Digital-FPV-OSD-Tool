import unittest

from _player_extract import FrameCache


class FrameCacheEvictionTest(unittest.TestCase):
    def test_evict_around_keeps_entries_nearest_current_pct(self):
        cache = FrameCache(max_items=3)
        for pct in (0, 100, 200, 5000, 9999):
            cache[pct] = f"img{pct}"
        cache.evict_around(current_pct=100)
        self.assertEqual(len(cache), 3)
        self.assertIn(0, cache)
        self.assertIn(100, cache)
        self.assertIn(200, cache)
        self.assertNotIn(5000, cache)
        self.assertNotIn(9999, cache)

    def test_evict_around_no_op_when_under_max(self):
        cache = FrameCache(max_items=10)
        cache[0] = "a"
        cache[1] = "b"
        cache.evict_around(current_pct=0)
        self.assertEqual(len(cache), 2)


class FrameCacheNearestTest(unittest.TestCase):
    def test_nearest_on_empty_returns_none(self):
        cache = FrameCache()
        self.assertIsNone(cache.nearest(50))

    def test_nearest_returns_closest_key(self):
        cache = FrameCache()
        cache[10] = "a"
        cache[50] = "b"
        cache[90] = "c"
        self.assertEqual(cache.nearest(48), 50)
        self.assertEqual(cache.nearest(11), 10)
        self.assertEqual(cache.nearest(89), 90)


class FrameCacheDictOpsTest(unittest.TestCase):
    def test_round_trip_dict_like_ops(self):
        cache = FrameCache()
        self.assertEqual(len(cache), 0)
        cache[5] = "img"
        self.assertIn(5, cache)
        self.assertEqual(cache[5], "img")
        self.assertEqual(cache.get(5), "img")
        self.assertIsNone(cache.get(6))
        self.assertEqual(list(cache), [5])
        self.assertEqual(list(cache.keys()), [5])
        del cache[5]
        self.assertNotIn(5, cache)
        cache[1] = "x"
        cache[2] = "y"
        cache.clear()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
_player_extract.py — Frame extraction cache for the player module.
"""

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

    def __contains__(self, pct):
        return pct in self._d

    def __getitem__(self, pct):
        return self._d[pct]

    def __setitem__(self, pct, img):
        self._d[pct] = img

    def __delitem__(self, pct):
        del self._d[pct]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def get(self, pct, default=None):
        return self._d.get(pct, default)

    def keys(self):
        return self._d.keys()

    def clear(self):
        self._d.clear()

    def evict_around(self, current_pct):
        """Evict entries farthest from current_pct until within max size."""
        while len(self._d) > self._max:
            worst = max(self._d, key=lambda k: abs(k - current_pct))
            del self._d[worst]

    def nearest(self, pct):
        """Key closest to pct, or None if empty."""
        return min(self._d, key=lambda k: abs(k - pct)) if self._d else None

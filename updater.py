# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD - https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
updater.py - GitHub release check + in-place self-update.

VueOSD ships as source (no PyInstaller exe on GitHub releases), so updating
means: hit the GitHub API for the latest release tag, compare against the
running VERSION, and on user consent download the source zipball and copy
its contents over the install directory.

Python source files aren't locked on Windows the way running .exe files are,
so we can overwrite main.py / widgets.py / etc. from inside the live
process - we just need to re-exec to pick up the new code.

Pure stdlib (urllib + zipfile + shutil) so no new wheel goes into
requirements.txt.

UI bits live in main.py - this module is GUI-agnostic, so the worker can run
on any thread and `download_and_apply` can be progress-reported into any
Qt / tk / console wrapper.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

GITHUB_REPO = "wkumik/Digital-FPV-OSD-Tool"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_URL       = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=10"
USER_AGENT = "VueOSD-Updater"

# Asset filename to look for inside GitHub Releases when the user clicks
# "Download fonts". Releases ship slim source via .gitattributes export-ignore,
# so fonts live as a separate `fonts.zip` asset that the maintainer attaches
# to any release.
FONTS_ASSET_NAME = "fonts.zip"

# Files we never overwrite during an update - they hold user state.
_PROTECTED_FILES = {
    "settings.json",
    "theme_custom.json",
    "vueosd_crash.log",
    "VueOSD.lnk",
}

# Directories we skip while mirroring - they're either developer state
# (.git) or environment / build artefacts that the zipball wouldn't contain
# anyway but might match by name accidentally.
_PROTECTED_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".claude",
}


@dataclass
class ReleaseInfo:
    """Subset of the GitHub release JSON that the updater actually uses."""
    tag:         str   # "v1.7"
    version:     str   # "1.7" (tag with leading v/V stripped)
    name:        str   # "VueOSD v1.7"
    body:        str   # release notes (markdown)
    zipball_url: str   # GitHub's auto-generated source archive URL
    html_url:    str   # release page on github.com


# ----- Version comparison ---------------------------------------------------

def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse '1.6', 'v1.6.1', or '1.6-rc2' into a tuple of ints.

    Anything past the first non-numeric segment is dropped, so a pre-release
    'v1.7-beta' compares equal to 'v1.7'. Good enough for VueOSD's tagging
    convention; revisit if we ever publish '1.7.1'-style patch releases.
    """
    v = v.lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts: list[int] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts) or (0,)


def is_newer(remote_version: str, current_version: str) -> bool:
    return _version_tuple(remote_version) > _version_tuple(current_version)


# ----- GitHub API -----------------------------------------------------------

def check_latest_release(timeout: float = 5.0) -> Optional[ReleaseInfo]:
    """Fetch the latest release once. Returns None on ANY failure - network
    errors, rate limits, parse errors - so the caller never has to handle
    exceptions just to silently skip the check.
    """
    req = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"User-Agent": USER_AGENT,
                 "Accept":     "application/vnd.github+json"},
    )
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

    tag = str(data.get("tag_name") or "")
    zipball = str(data.get("zipball_url") or "")
    if not tag or not zipball:
        return None
    return ReleaseInfo(
        tag         = tag,
        version     = tag.lstrip("vV"),
        name        = str(data.get("name") or tag),
        body        = str(data.get("body") or ""),
        zipball_url = zipball,
        html_url    = str(data.get("html_url") or ""),
    )


def find_update(current_version: str,
                timeout: float = 5.0) -> Optional[ReleaseInfo]:
    """Return ReleaseInfo if a newer release than ``current_version`` exists,
    else None. Convenience wrapper around check_latest_release + version
    comparison."""
    info = check_latest_release(timeout=timeout)
    if info is None:
        return None
    if not is_newer(info.version, current_version):
        return None
    return info


# ----- Download + install ---------------------------------------------------

def download_and_apply(release: ReleaseInfo,
                       install_dir: str,
                       progress_cb: Optional[Callable[[int, str], None]] = None,
                       ) -> None:
    """Download the release zipball, extract it, and mirror its contents over
    ``install_dir``. Files listed in ``_PROTECTED_FILES`` and any directory in
    ``_PROTECTED_DIRS`` are preserved.

    ``progress_cb(percent, message)`` is called periodically when supplied.
    Percent is 0..100. Raises on any failure - the caller surfaces the error.
    """
    def _p(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    _p(0, "Starting download...")
    req = urllib.request.Request(
        release.zipball_url,
        headers={"User-Agent": USER_AGENT,
                 "Accept":     "application/octet-stream"},
    )

    with tempfile.TemporaryDirectory(prefix="vueosd_update_") as tmp:
        zip_path = os.path.join(tmp, "release.zip")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(zip_path, "wb") as f:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        # Reserve 80% of the bar for the download; the rest
                        # is unzip + copy which are usually instant.
                        _p(int(done * 80 / total), "Downloading update...")
                    else:
                        _p(40, "Downloading update...")

        _p(82, "Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # GitHub's source zipball wraps everything in a single top-level dir
        # like 'wkumik-Digital-FPV-OSD-Tool-<sha>'. Find it.
        candidates = [n for n in os.listdir(tmp)
                      if os.path.isdir(os.path.join(tmp, n))]
        if not candidates:
            raise RuntimeError("update archive contained no directory")
        src_root = os.path.join(tmp, candidates[0])

        _p(90, "Installing...")
        _mirror_tree(src_root, install_dir)
        _p(100, "Update applied.")


def _mirror_tree(src: str, dst: str) -> None:
    """Copy every file in ``src`` into the matching path under ``dst``.

    Behaviour:
      - Creates missing target directories.
      - Overwrites existing files (.py + assets).
      - Skips files in _PROTECTED_FILES and any path component in
        _PROTECTED_DIRS.
      - Does NOT delete files that exist in dst but not in src - keeps the
        user's loose files (sample .osd / .srt sidecars, custom fonts they
        dropped in) safe.
    """
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in _PROTECTED_DIRS]
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        for fname in files:
            if fname in _PROTECTED_FILES:
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target, fname)
            try:
                shutil.copy2(src_path, dst_path)
            except PermissionError:
                # Re-raise with a more helpful message - usually means the
                # user installed under a path that needs admin rights.
                raise PermissionError(
                    f"Can't overwrite {dst_path} - the install directory "
                    f"is read-only or open in another program. Close any "
                    f"editors / shells in the install folder and try again."
                ) from None


# ----- Restart --------------------------------------------------------------

def restart_app() -> None:
    """Re-launch this Python process with the same argv, then exit.

    The new process inherits whatever interpreter started us (`sys.executable`)
    - usually the venv's pythonw.exe on Windows, system python elsewhere -
    so we don't need to know how the user originally launched the app.
    """
    py = sys.executable
    args = [py] + sys.argv
    install_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # Detach: close stdio file handles in the parent so the child isn't
    # terminated when we exit.
    kwargs: dict = {"cwd": install_dir, "close_fds": True}
    if sys.platform == "win32":
        # DETACHED_PROCESS so the new process doesn't keep our console alive.
        kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS
    subprocess.Popen(args, **kwargs)
    sys.exit(0)


def install_dir_is_git_checkout(install_dir: str) -> bool:
    """True when the install dir is a git working copy. Used to add a warning
    to the update dialog - overwriting tracked files will show up as a noisy
    set of modifications in `git status`."""
    return os.path.isdir(os.path.join(install_dir, ".git"))


# ----- Fonts asset (manual "Download fonts" button) -------------------------

@dataclass
class FontsAsset:
    """A `fonts.zip` asset attached to a GitHub release."""
    name:         str
    download_url: str
    size:         int   # bytes, 0 when GitHub didn't report it
    release_tag:  str
    release_name: str


def find_fonts_asset(timeout: float = 5.0) -> Optional[FontsAsset]:
    """Scan the 10 most recent GitHub releases for the first attached
    fonts.zip asset and return its metadata.

    Returns None on any network / parse error - the caller surfaces it.
    Picking the newest matching release means the maintainer can refresh the
    fonts pack by attaching a new fonts.zip to ANY release; no special tag
    convention required.
    """
    req = urllib.request.Request(
        RELEASES_URL,
        headers={"User-Agent": USER_AGENT,
                 "Accept":     "application/vnd.github+json"},
    )
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            releases = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(releases, list):
        return None
    target = FONTS_ASSET_NAME.lower()
    for rel in releases:
        for asset in (rel.get("assets") or []):
            if str(asset.get("name", "")).lower() == target:
                url = str(asset.get("browser_download_url") or "")
                if not url:
                    continue
                return FontsAsset(
                    name         = str(asset.get("name") or FONTS_ASSET_NAME),
                    download_url = url,
                    size         = int(asset.get("size") or 0),
                    release_tag  = str(rel.get("tag_name") or ""),
                    release_name = str(rel.get("name") or ""),
                )
    return None


def download_fonts(asset: FontsAsset, install_dir: str,
                   progress_cb: Optional[Callable[[int, str], None]] = None,
                   ) -> int:
    """Download ``asset`` into <install_dir>/fonts/. Returns the count of
    font subdirectories present after extraction.

    Existing font folders with the same name are replaced (the zip wins);
    user-added fonts outside the zip are left in place.
    """
    def _p(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    fonts_dir = os.path.join(install_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)

    _p(0, "Connecting to GitHub...")
    req = urllib.request.Request(
        asset.download_url,
        headers={"User-Agent": USER_AGENT,
                 "Accept":     "application/octet-stream"},
    )

    with tempfile.TemporaryDirectory(prefix="vueosd_fonts_") as tmp:
        zip_path = os.path.join(tmp, "fonts.zip")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            total = int(r.headers.get("Content-Length") or asset.size or 0)
            done = 0
            with open(zip_path, "wb") as f:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        _p(int(done * 80 / total), "Downloading fonts...")
                    else:
                        _p(40, "Downloading fonts...")

        _p(85, "Extracting...")
        extract_root = os.path.join(tmp, "_extracted")
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)

        # The zip may wrap everything in a single 'fonts/' top-level dir or
        # it may dump font folders at the root. Detect both and copy from
        # the directory whose direct children are the per-font subdirs.
        src_root = _locate_fonts_root(extract_root)
        if src_root is None:
            raise RuntimeError(
                "fonts.zip didn't contain any font directories - the asset "
                "may be malformed.")

        _p(92, "Installing fonts...")
        for name in os.listdir(src_root):
            src = os.path.join(src_root, name)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(fonts_dir, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        _p(100, "Done.")

    n_fonts = sum(1 for x in os.listdir(fonts_dir)
                  if os.path.isdir(os.path.join(fonts_dir, x)))
    return n_fonts


def _locate_fonts_root(extract_root: str) -> Optional[str]:
    """Find the directory inside ``extract_root`` that DIRECTLY contains the
    per-font subdirectories.

    Handles two layouts:
      1. fonts.zip/fonts/BTFL_Sphere/...  (wrapped)
      2. fonts.zip/BTFL_Sphere/...        (flat)
    """
    # Flat layout: extract_root itself has font dirs as direct children.
    flat_dirs = [d for d in os.listdir(extract_root)
                 if os.path.isdir(os.path.join(extract_root, d))]
    if not flat_dirs:
        return None
    if any(_looks_like_font_dir(os.path.join(extract_root, d))
           for d in flat_dirs):
        return extract_root
    # Wrapped layout: try one level deeper.
    for d in flat_dirs:
        nested = os.path.join(extract_root, d)
        nested_dirs = [n for n in os.listdir(nested)
                       if os.path.isdir(os.path.join(nested, n))]
        if any(_looks_like_font_dir(os.path.join(nested, n))
               for n in nested_dirs):
            return nested
    return None


def _looks_like_font_dir(path: str) -> bool:
    """Heuristic: a font dir contains at least one .png file."""
    try:
        return any(f.lower().endswith(".png") for f in os.listdir(path))
    except OSError:
        return False

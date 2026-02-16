# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 OnyxFPV — https://github.com/onyxfpv
import sys
"""
video_processor.py  –  Render OSD + SRT overlay onto video.

Architecture (fast path):
──────────────────────────────────────────────────────────────────
Python renders ONLY the OSD overlay frames (~10 fps) → pipe to FFmpeg.
FFmpeg reads the OSD pipe + source video file simultaneously, composites
them natively in C using the 'overlay' filter, and encodes with NVENC.
Python NEVER touches a single raw video frame.

  Python:  1671 OSD frames × ~1ms = ~1.7s  (runs in background thread)
  FFmpeg:  decode + overlay + NVENC encode  (runs concurrently)
  Total:   max(1.7s, ffmpeg_time)           (overlapped)

vs old approach (Python in video frame loop):
  Python + encode: 7.5s per 5s of 1080p60
  New approach:    ~1.2s per 5s with NVENC, ~4s CPU-only

GPU encoder priority (auto-detected at startup):
  NVIDIA:  h264_nvenc / hevc_nvenc
  AMD:     h264_amf   / hevc_amf
  Intel:   h264_qsv   / hevc_qsv
  Linux:   h264_vaapi / hevc_vaapi
  macOS:   h264_videotoolbox
"""

import subprocess
import shutil
import os
import json
import threading
import tempfile
from dataclasses import dataclass
from typing import Optional, Callable

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# Suppress console window on Windows for ALL subprocess calls.
# Use STARTUPINFO (more reliable than creationflags alone).
def _hidden_popen(*args, **kwargs):
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", si)
        kwargs.setdefault("creationflags", 0x08000000)
    return subprocess.Popen(*args, **kwargs)

def _hidden_run(*args, **kwargs):
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", si)
        kwargs.setdefault("creationflags", 0x08000000)
    return subprocess.run(*args, **kwargs)


@dataclass
class ProcessingConfig:
    input_video:  str
    output_video: str
    osd_file:     Optional[str]  = None
    srt_file:     Optional[str]  = None
    codec:        str   = "libx264"
    crf:          int   = 23
    preset:       str   = "medium"
    font_folder:  Optional[str]  = None
    prefer_hd:    bool  = True
    scale:        float = 1.0
    offset_x:     int   = 0
    offset_y:     int   = 0
    show_srt_bar:  bool  = True
    srt_opacity:   float = 0.6   # SRT bar background opacity
    use_hw:        bool  = False
    bitrate_mbps:  float = None   # if set, use -b:v instead of -crf
    trim_start:    float = 0.0    # seconds, 0 = beginning
    trim_end:      float = 0.0    # seconds, 0 = end of file
    upscale_target: str  = ""     # "" = no upscale | "1440p" | "2.7k" | "4k"
    osd_data:      object = None  # pre-parsed OsdFile (e.g. P1 embedded OSD)


# ── GPU encoder detection ─────────────────────────────────────────────────────

_HW_CANDIDATES = [
    ("h264_nvenc",         "hevc_nvenc",        "NVIDIA NVENC"),
    ("h264_amf",           "hevc_amf",          "AMD AMF"),
    ("h264_qsv",           "hevc_qsv",          "Intel QSV"),
    ("h264_vaapi",         "hevc_vaapi",         "VAAPI"),
    ("h264_videotoolbox",  "hevc_videotoolbox",  "Apple VideoToolbox"),
    ("h264_v4l2m2m",       "hevc_v4l2m2m",       "V4L2 M2M"),
]

_hw_probe_cache: Optional[dict] = None   # None = not yet probed; {} = probed, no GPU


def _run_with_hard_timeout(cmd, timeout_s=5):
    """Run a subprocess with a hard kill after timeout_s seconds.
    Returns (returncode, stdout_bytes, stderr_bytes) or raises TimeoutError."""
    import subprocess as _sp
    kwargs = {}
    if sys.platform == "win32":
        si = _sp.STARTUPINFO()
        si.dwFlags |= _sp.STARTF_USESHOWWINDOW
        si.wShowWindow = _sp.SW_HIDE
        kwargs["startupinfo"] = si
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return proc.returncode, out, err
    except _sp.TimeoutExpired:
        proc.kill()
        proc.communicate()   # drain pipes after kill
        raise TimeoutError(f"process timed out after {timeout_s}s")


# Phrases in ffmpeg stderr that definitively mean "no GPU device present".
# Any other non-zero exit might be a driver/init hiccup — we treat it as
# "available" so we don't wrongly hide the GPU option from the user.
_NO_DEVICE_PHRASES = [
    "no nvenc capable device",
    "no capable device",
    "no encode device",
    "nvenc_err_no_encode_device",
    "nvenc_err_unsupported_device",
    "mfx_err",
    "mfx session",
    "no opencl",
    "no vaapi",
    "device type cuda not found",
    "cannot load nvcuda.dll",
    "cannot load libnvcuvid",
    "cuda_error_no_device",
]


def detect_hw_encoder(ffmpeg: str) -> Optional[dict]:
    """
    Detect a working hardware encoder via a short test encode.

    Key design decisions:
      • 20 s timeout — NVENC must initialise the CUDA context on first call,
        which on Windows with a cold driver can take 10-15 s.
      • Fixed pixel-format flags — NVENC wants -pix_fmt as an encoder output
        flag, not a -vf filter.  Using -vf format= before the encoder can
        cause spurious failures on some driver versions.
      • Stderr inspection — if the encode fails we check stderr for definitive
        "no device" phrases.  Any other failure (driver hiccup, wrong pix fmt
        in a specific build) is treated as "available" so we don't silently
        hide a working GPU from the user.

    Returns {"name", "h264", "h265", "vaapi"} or None.
    Caches result — detection only runs once per session.
    """
    global _hw_probe_cache
    if _hw_probe_cache is not None:
        return _hw_probe_cache if _hw_probe_cache else None

    # Step 1: which encoders are compiled into this ffmpeg build (fast, no GPU)
    try:
        _, out, _ = _run_with_hard_timeout(
            [ffmpeg, "-encoders", "-hide_banner"], timeout_s=8)
        compiled = set(out.decode("utf-8", errors="replace").split())
    except Exception:
        compiled = set()

    for h264, h265, name in _HW_CANDIDATES:
        if h264 not in compiled:
            continue  # encoder not built in — skip without any subprocess

        # Step 2: test encode — 1 frame, null output, correct pix_fmt per encoder
        # NVENC minimum frame size is 145x145. Use 256x256 @ 30fps to satisfy
        # all encoder constraints (NVENC, AMF, QSV all accept this).
        if "vaapi" in h264:
            cmd = [ffmpeg, "-y",
                   "-vaapi_device", "/dev/dri/renderD128",
                   "-f", "lavfi", "-i", "color=black:size=256x256:rate=30",
                   "-frames:v", "1",
                   "-vf", "format=nv12,hwupload",
                   "-c:v", h264, "-f", "null", "-"]
        elif "amf" in h264:
            # AMF requires nv12 input; yuv420p causes an init failure on some drivers
            cmd = [ffmpeg, "-y",
                   "-f", "lavfi", "-i", "color=black:size=256x256:rate=30",
                   "-frames:v", "1",
                   "-c:v", h264, "-pix_fmt", "nv12",
                   "-f", "null", "-"]
        else:
            cmd = [ffmpeg, "-y",
                   "-f", "lavfi", "-i", "color=black:size=256x256:rate=30",
                   "-frames:v", "1",
                   "-c:v", h264, "-pix_fmt", "yuv420p",
                   "-f", "null", "-"]

        try:
            rc, _, err = _run_with_hard_timeout(cmd, timeout_s=20)
        except TimeoutError:
            # Timed out — CUDA/driver failed to initialise even in 20 s.
            # This is a genuine "no working GPU" signal.
            continue
        except Exception:
            continue

        if rc == 0:
            _hw_probe_cache = {"name": name, "h264": h264,
                               "h265": h265, "vaapi": "vaapi" in h264}
            return _hw_probe_cache

        # Non-zero exit — inspect stderr
        err_lo = err.decode("utf-8", errors="replace").lower()

        if any(phrase in err_lo for phrase in _NO_DEVICE_PHRASES):
            # Definitively no device of this type — try the next candidate
            continue

        # Unknown failure (wrong pix-fmt, minor driver issue, etc.).
        # The encoder IS compiled in and didn't say "no device", so report it
        # as available — the user's actual encode will likely work fine.
        _hw_probe_cache = {"name": name, "h264": h264,
                           "h265": h265, "vaapi": "vaapi" in h264}
        return _hw_probe_cache

    _hw_probe_cache = {}  # definitive miss — cache so we never probe again
    return None


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ["ffmpeg.exe", "ffmpeg"]:
        candidate = os.path.join(script_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def get_video_info(video_path: str) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ff = find_ffmpeg()
        if ff:
            candidate = ff.replace("ffmpeg", "ffprobe")
            if os.path.exists(candidate):
                ffprobe = candidate
    if not ffprobe:
        return {"error": "ffprobe not found"}
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", video_path]
    try:
        result = _hidden_run(cmd, capture_output=True, text=True, timeout=30)
        data   = json.loads(result.stdout)
        info   = {}
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["width"]    = stream.get("width", 0)
                info["height"]   = stream.get("height", 0)
                info["codec"]    = stream.get("codec_name", "unknown")
                rfr              = stream.get("r_frame_rate", "30/1")
                num, den         = rfr.split("/")
                info["fps"]      = round(int(num) / max(int(den), 1), 3)
                info["duration"] = float(data.get("format", {}).get("duration", 0))
                info["size_mb"]  = round(
                    int(data.get("format", {}).get("size", 0)) / 1_048_576, 1)
        return info
    except Exception as e:
        return {"error": str(e)}


_STDERR_CAP = 32 * 1024   # keep last 32 KB of ffmpeg stderr

def _drain(pipe, store: list):
    """Drain a pipe to a list[0] string, capped to avoid unbounded RAM use."""
    chunks = []
    total  = 0
    try:
        for chunk in iter(lambda: pipe.read(4096), b""):
            chunks.append(chunk)
            total += len(chunk)
            # Drop oldest chunks once we exceed the cap
            while total > _STDERR_CAP and chunks:
                dropped = chunks.pop(0)
                total  -= len(dropped)
    except Exception:
        pass
    store[0] = b"".join(chunks).decode("utf-8", errors="replace")


def _read_exactly(pipe, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


# ── Main entry point ──────────────────────────────────────────────────────────

def process_video(
    config: ProcessingConfig,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg not found!\nGet from https://www.gyan.dev/ffmpeg/builds/")
    if not PIL_OK:
        raise RuntimeError("Pillow is required: pip install Pillow")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from osd_parser   import parse_osd
    from srt_parser   import parse_srt
    from font_loader  import load_font
    from osd_renderer import OsdRenderer, OsdRenderConfig
    from pathlib      import Path

    # ── Load data ─────────────────────────────────────────────────────────────
    osd_data = srt_data = font = None

    # Accept pre-parsed OSD data directly (e.g. P1 embedded OSD — no .osd file)
    if config.osd_data is not None:
        osd_data = config.osd_data
    elif config.osd_file and os.path.isfile(config.osd_file):
        try:    osd_data = parse_osd(config.osd_file)
        except Exception as e:
            if progress_callback: progress_callback(0, f"⚠ OSD: {e}")

    if config.srt_file and os.path.isfile(config.srt_file):
        try:    srt_data = parse_srt(config.srt_file)
        except Exception as e:
            if progress_callback: progress_callback(0, f"⚠ SRT: {e}")

    if config.font_folder and os.path.isdir(config.font_folder):
        try:    font = load_font(Path(config.font_folder), prefer_hd=config.prefer_hd)
        except Exception as e:
            if progress_callback: progress_callback(0, f"⚠ Font: {e}")

    if osd_data is None and srt_data is None:
        return _reencode_only(ffmpeg, config, progress_callback)

    # ── Video info ─────────────────────────────────────────────────────────────
    info = get_video_info(config.input_video)
    if "error" in info or not info.get("width"):
        raise RuntimeError(f"Cannot read video: {info.get('error','unknown')}")

    width    = info["width"]
    height   = info["height"]
    fps      = info["fps"]
    duration = info["duration"]

    # ── Resolve encoder ────────────────────────────────────────────────────────
    hw_info  = detect_hw_encoder(ffmpeg) if config.use_hw else None
    use_vaapi = hw_info and hw_info.get("vaapi", False)

    if hw_info and config.use_hw:
        codec_map   = {"libx264": hw_info["h264"], "libx265": hw_info["h265"]}
        encoder     = codec_map.get(config.codec, hw_info["h264"])
        enc_label   = f"{hw_info['name']} ({encoder})"
        # GPU rate control — each encoder has a different "CRF equivalent":
        #   NVENC: -rc:v vbr -cq X  (VBR + constant quality, X same scale as CRF)
        #   VAAPI: -rc_mode VBR -qp X
        #   AMF:   -rc cqp -qp_i X -qp_p X
        #   QSV:   -global_quality X
        #   -qp alone forces I-frame-only quality and produces huge files!
        if "nvenc" in encoder:
            # NVENC CQ scale is shifted vs x264/x265 CRF.
            # CQ 23 on NVENC ≈ CRF 14 in x264 (near-lossless, huge files).
            # Apply +9 offset so the user's CRF slider maps to similar file sizes:
            #   slider 23 → CQ 32, slider 28 → CQ 37, slider 18 → CQ 27
            nvenc_cq = min(51, config.crf + 9)
            quality_args = ["-rc:v", "vbr", "-cq", str(nvenc_cq),
                            "-maxrate", "50M",
                            "-bufsize", "100M"]
        elif "vaapi" in encoder:
            quality_args = ["-rc_mode", "VBR", "-qp", str(config.crf)]
        elif "amf" in encoder:
            quality_args = ["-rc", "cqp",
                            "-qp_i", str(config.crf),
                            "-qp_p", str(config.crf)]
        elif "qsv" in encoder:
            quality_args = ["-global_quality", str(config.crf)]
        else:
            quality_args = ["-cq", str(config.crf)]
        preset_args = ["-preset", "p6"] if "nvenc" in encoder else \
                      (["-preset", "medium"] if "qsv" in encoder else [])
        # pix_fmt_args: for hardware encoders the filter_complex format= node
        # already delivers the correct pixel format to the encoder — passing
        # -pix_fmt after -c:v confuses NVENC on Linux ("Operation not permitted")
        # and AMF on Windows.  Only VAAPI needs special handling (hwupload path).
        # CPU encoders still need an explicit -pix_fmt yuv420p.
        use_nvenc = "nvenc" in encoder
        use_amf   = "amf"   in encoder
        if use_vaapi:
            pix_fmt_args = ["-pix_fmt", "nv12"]
        elif use_nvenc or use_amf:
            # NVENC and AMF: format is already set by filter_complex; no extra flag.
            # Passing -pix_fmt after -c:v h264_nvenc causes "Operation not permitted"
            # on Linux and AMF init failure on Windows.
            pix_fmt_args = []
        else:
            # QSV and any other HW encoder: explicit yuv420p is safe and expected
            pix_fmt_args = ["-pix_fmt", "yuv420p"]
    else:
        encoder      = config.codec
        enc_label    = f"CPU ({encoder})"
        if config.bitrate_mbps:
            quality_args = ["-b:v", f"{config.bitrate_mbps}M",
                            "-maxrate", f"{config.bitrate_mbps * 1.5:.1f}M",
                            "-bufsize", f"{config.bitrate_mbps * 2:.1f}M"]
        else:
            quality_args = ["-crf", str(config.crf)]
        preset_args  = ["-preset", config.preset]
        pix_fmt_args = ["-pix_fmt", "yuv420p"]

    # ── Choose pipeline ────────────────────────────────────────────────────────
    # Fast path: OSD overlay pipe — Python handles ONLY the OSD frames,
    # FFmpeg handles all video frame I/O natively in C.
    if font and osd_data:
        return _overlay_pipeline(
            ffmpeg, config, osd_data, srt_data, font,
            width, height, fps, duration,
            encoder, enc_label, quality_args, preset_args, pix_fmt_args,
            use_vaapi, hw_info, progress_callback)

    # Fallback: SRT-only (no OSD font) — just burn SRT bar via Python
    return _srt_only_pipeline(
        ffmpeg, config, srt_data,
        width, height, fps, duration,
        encoder, enc_label, quality_args, preset_args, pix_fmt_args,
        progress_callback)


# ── Upscale target → filter string ───────────────────────────────────────────
_UPSCALE_HEIGHTS = {"1440p": 1440, "2.7k": 1512, "4k": 2160}

def _upscale_filter(target: str, fc_fmt, use_vaapi: bool) -> str:
    """Build the filter_complex string for overlay + optional upscale."""
    h = _UPSCALE_HEIGHTS.get((target or "").lower())
    if use_vaapi:
        # VAAPI: no format= node, hwupload instead
        if h:
            return f"[0:v][1:v]overlay=shortest=1,scale=-2:{h}:flags=lanczos,hwupload[v]"
        return "[0:v][1:v]overlay=shortest=1,hwupload[v]"
    if h:
        return f"[0:v][1:v]overlay=shortest=1,scale=-2:{h}:flags=lanczos,format={fc_fmt}[v]"
    return f"[0:v][1:v]overlay=shortest=1,format={fc_fmt}[v]"


# ── Fast pipeline: OSD overlay ─────────────────────────────────────────────────

def _overlay_pipeline(
    ffmpeg, config, osd_data, srt_data, font,
    width, height, fps, duration,
    encoder, enc_label, quality_args, preset_args, pix_fmt_args,
    use_vaapi, hw_info, progress_callback,
):
    """
    Python renders OSD frames (~10fps) → pipe to FFmpeg as a second input.
    FFmpeg overlays the OSD stream onto the source video and encodes.
    Python never reads or writes a single raw video frame.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from osd_renderer import OsdRenderer, OsdRenderConfig

    total_frames = max(1, int(duration * fps))
    render_cfg = OsdRenderConfig(
        offset_x    = config.offset_x,
        offset_y    = config.offset_y,
        scale       = config.scale,
        show_srt_bar= config.show_srt_bar,
        srt_opacity = config.srt_opacity,
    )
    renderer = OsdRenderer(width, height, font, render_cfg)

    # Trim window — default to full video
    _t_start = config.trim_start if config.trim_start > 0.01 else 0.0
    _t_end   = config.trim_end   if config.trim_end   > 0.01 else duration
    _t_dur   = max(0.001, _t_end - _t_start)
    _trim_ss = ["-ss", f"{_t_start:.3f}"] if _t_start > 0.01 else []
    _trim_t  = ["-t",  f"{_t_dur:.3f}"]   if _t_dur  < duration - 0.01 else []

    # OSD availability check — warn if no OSD frames overlap the trim window
    t_start_ms = int(_t_start * 1000)
    t_end_ms   = int(_t_end   * 1000)
    osd_in_window = [fr for fr in osd_data.frames
                     if t_start_ms <= fr.time_ms <= t_end_ms + 500]
    osd_trimmed_warning = ""
    if not osd_in_window:
        osd_trimmed_warning = "No OSD elements in trim window — rendered without OSD overlay"
        if progress_callback:
            progress_callback(0, f"⚠ {osd_trimmed_warning}")

    # How many output frames we will write to the pipe (= trimmed video frame count)
    n_out_frames = max(1, int(_t_dur * fps))

    if progress_callback:
        progress_callback(5, f"{width}x{height} @ {fps}fps · {n_out_frames} frames · {enc_label}")

    # OSD pipe runs at the SAME fps as the video.
    # Each pipe frame is looked up by its absolute timestamp so OSD timing is exact
    # regardless of the OSD file's variable internal frame rate.
    # faststart moves moov atom to front for instant web playback
    movflags = ["-movflags", "+faststart"] if not use_vaapi else []

    # Pixel format for the filter_complex output — must match what the encoder accepts:
    #   NVENC (NVIDIA) → nv12  (its native format; yuv420p causes "Operation not permitted" on Linux)
    #   AMF   (AMD)    → nv12  (yuv420p causes init failure on Windows)
    #   VAAPI          → handled separately with hwupload (no format= node)
    #   CPU / QSV      → yuv420p
    use_nvenc = "nvenc" in encoder
    use_amf   = "amf"   in encoder
    if use_vaapi:
        _fc_fmt = None   # VAAPI path uses hwupload, no format= node needed
    elif use_nvenc or use_amf:
        _fc_fmt = "nv12"
    else:
        _fc_fmt = "yuv420p"

    ffmpeg_cmd = (
        [ffmpeg, "-y"]
        + (["-vaapi_device", "/dev/dri/renderD128"] if use_vaapi else [])
        # Input 0: source video (with optional fast seek)
        + _trim_ss + ["-i", config.input_video] + _trim_t
        # Input 1: OSD overlay pipe — rgba frames at video fps
        + ["-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{width}x{height}",
           "-r", str(fps),
           "-i", "pipe:0"]
        # High-quality RGBA→YUV colour conversion
        + ["-sws_flags", "lanczos+accurate_rnd+full_chroma_int"]
        # Overlay filter: OSD on top of source, optional upscale, then encode
        + ["-filter_complex",
           (_upscale_filter(config.upscale_target, _fc_fmt, use_vaapi))]
        + ["-map", "[v]",
           "-map", "0:a:0?",
           "-c:v", encoder]
        + quality_args + preset_args + pix_fmt_args
        + movflags
        + ["-c:a", "copy", config.output_video]
    )

    ffmpeg_proc = _hidden_popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    ffmpeg_stderr = [""]

    # Drain FFmpeg stderr in background (prevents pipe deadlock, capped)
    _overlay_stderr_store = [""]
    threading.Thread(target=_drain, args=(ffmpeg_proc.stderr, _overlay_stderr_store),
                     daemon=True).start()
    ffmpeg_stderr = _overlay_stderr_store

    # ── OSD render loop ───────────────────────────────────────────────────────
    # For each output video frame i, compute its absolute timestamp, look up
    # the correct OSD frame by time, composite once and write to pipe.
    # Cache composited frames by (osd_frame_index, srt_text) so that repeated
    # OSD frames (which are the vast majority) cost only a dict lookup + write.
    # This guarantees frame-perfect sync: pipe frame i always matches video frame i.

    # Frame cache: (osd_index, srt_text) → composited bytes
    # Capped at 128 entries — OSD updates at ~10fps but video runs at 60fps,
    # so consecutive repeats are caught with a small window. Prevents GB of RAM
    # on long videos with many unique OSD/SRT combinations.
    _CACHE_MAX   = 32
    _frame_cache: dict = {}
    report_every = max(1, n_out_frames // 50)

    try:
        if not osd_in_window:
            # No OSD data in trim window — single blank frame, FFmpeg pads
            ffmpeg_proc.stdin.write(bytes(renderer.composite(None, "")))
            if progress_callback:
                progress_callback(50, f"No OSD in trim window — blank overlay  [{enc_label}]")
        else:
            for i in range(n_out_frames):
                # Absolute timestamp of this video frame in the OSD file's timebase
                abs_t_ms = int((_t_start + i / fps) * 1000)

                osd_frame = osd_data.frame_at_time(abs_t_ms)

                srt_text = ""
                if srt_data and config.show_srt_bar:
                    td = srt_data.get_data_at_time(abs_t_ms)
                    if td:
                        srt_text = td.status_line()

                cache_key = (osd_frame.index if osd_frame else -1, srt_text)
                if cache_key not in _frame_cache:
                    if len(_frame_cache) >= _CACHE_MAX:
                        # Drop the oldest entry (insertion-order dict, Python 3.7+)
                        del _frame_cache[next(iter(_frame_cache))]
                    _frame_cache[cache_key] = bytes(
                        renderer.composite(osd_frame, srt_text))

                ffmpeg_proc.stdin.write(_frame_cache[cache_key])

                if progress_callback and (i % report_every == 0 or i == n_out_frames - 1):
                    pct = min(5 + int(i / n_out_frames * 85), 90)
                    progress_callback(pct,
                        f"Frame {i+1}/{n_out_frames}  ({abs_t_ms/1000:.1f}s)  [{enc_label}]")

    except (BrokenPipeError, OSError) as exc:
            # BrokenPipeError  — POSIX broken pipe (errno 32)
            # OSError(errno=22) — Windows EINVAL raised when writing to a pipe
            #                     whose read end (FFmpeg) has already closed.
            # Both mean FFmpeg exited early — fall through to returncode check.
            import errno as _errno
            if not isinstance(exc, BrokenPipeError) and getattr(exc, 'errno', None) != _errno.EINVAL:
                raise   # genuine unexpected OS error — re-raise
    finally:
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass

    # Wait for FFmpeg to finish encoding (it may still be processing video)
    if progress_callback:
        progress_callback(92, f"Encoding…  [{enc_label}]")

    ffmpeg_proc.wait()

    if ffmpeg_proc.returncode not in (0, None):
        err = ffmpeg_stderr[0]
        if hw_info and config.use_hw:
            # Widen the GPU-failure check to include AMF and generic HW terms
            hw_phrases = ["nvenc", "amf", "vaapi", "qsv", "cuda",
                          "no capable", "no device", "hwaccel", "d3d11"]
            if any(x in err.lower() for x in hw_phrases):
                raise RuntimeError(
                    f"GPU encode failed ({hw_info['name']}):\n{err[-800:]}\n\n"
                    "Try disabling GPU acceleration in Settings.")
        raise RuntimeError(f"Encode failed (exit {ffmpeg_proc.returncode}):\n{err[-2000:]}")

    if progress_callback:
        progress_callback(100, f"✓ Done  [{enc_label}]")

    return osd_trimmed_warning or True


# ── Fallback pipeline: SRT-only (no font loaded) ──────────────────────────────

def _srt_only_pipeline(
    ffmpeg, config, srt_data,
    width, height, fps, duration,
    encoder, enc_label, quality_args, preset_args, pix_fmt_args,
    progress_callback,
):
    """SRT text bar rendered in Python, piped through the old frame-by-frame path."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from osd_renderer import _draw_srt_bar
    from PIL import Image

    total_frames = max(1, int(duration * fps))
    frame_bytes  = width * height * 4

    _t_start = config.trim_start if config.trim_start > 0.01 else 0.0
    _t_end   = config.trim_end   if config.trim_end   > 0.01 else duration
    _t_dur   = _t_end - _t_start
    _trim_ss = ["-ss", f"{_t_start:.3f}"] if _t_start > 0.01 else []
    _trim_t  = ["-t",  f"{_t_dur:.3f}"]   if _t_dur  < duration - 0.01 else []

    if progress_callback:
        progress_callback(5, f"{width}×{height} @ {fps}fps · SRT only · {enc_label}")

    decode_cmd = ([ffmpeg, "-y"]
                  + _trim_ss + ["-i", config.input_video] + _trim_t
                  + ["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
    srt_movflags = ["-movflags", "+faststart"]
    encode_cmd = ([ffmpeg, "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
                  + ["-sws_flags", "lanczos+accurate_rnd+full_chroma_int"]
                  + _trim_ss + ["-i", config.input_video] + _trim_t
                  + ["-map", "0:v:0", "-map", "1:a:0?",
                     "-c:v", encoder]
                  + quality_args + preset_args + pix_fmt_args
                  + srt_movflags
                  + ["-c:a", "copy", config.output_video])

    dec_proc = _hidden_popen(decode_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    enc_proc = _hidden_popen(encode_cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    enc_stderr = [""]
    threading.Thread(target=_drain, args=(dec_proc.stderr, [""]), daemon=True).start()
    threading.Thread(target=_drain, args=(enc_proc.stderr, enc_stderr), daemon=True).start()

    frame_idx    = 0
    t_offset_ms  = int(_t_start * 1000)   # SRT timestamps are absolute
    total_trimmed = max(1, int(_t_dur * fps))
    report_every = max(1, total_trimmed // 200)
    try:
        while True:
            raw = _read_exactly(dec_proc.stdout, frame_bytes)
            if raw is None: break
            # Offset frame time by trim_start so SRT lookup uses absolute timestamp
            t_ms = t_offset_ms + int(frame_idx / fps * 1000)
            srt_text = ""
            if srt_data and config.show_srt_bar:
                td = srt_data.get_data_at_time(t_ms)
                if td: srt_text = td.status_line()
            if srt_text:
                img = Image.frombuffer("RGBA", (width, height), raw, "raw", "RGBA", 0, 1)
                out = img.copy()
                _draw_srt_bar(out, srt_text)
                enc_proc.stdin.write(out.tobytes())
            else:
                enc_proc.stdin.write(raw)
            frame_idx += 1
            if progress_callback and frame_idx % report_every == 0:
                pct = min(5 + int(frame_idx / total_trimmed * 93), 98)
                progress_callback(pct, f"Frame {frame_idx}/{total_trimmed}  [{enc_label}]")
    except BrokenPipeError:
        pass
    finally:
        try: enc_proc.stdin.close()
        except Exception: pass

    dec_proc.wait(); enc_proc.wait()

    if enc_proc.returncode not in (0, None):
        err = enc_stderr[0]
        if isinstance(err, bytes): err = err.decode("utf-8", errors="replace")
        raise RuntimeError(f"Encode failed:\n{err[-2000:]}")

    if progress_callback:
        progress_callback(100, f"✓ Done  [{enc_label}]")
    return True


def _reencode_only(ffmpeg, config, progress_callback):
    if progress_callback:
        progress_callback(5, "Re-encoding…")
    _t_start = config.trim_start if config.trim_start > 0.01 else 0.0
    _t_end   = config.trim_end   if config.trim_end   > 0.01 else 0.0
    _trim_ss = ["-ss", f"{_t_start:.3f}"] if _t_start > 0.01 else []
    _trim_to = ["-to", f"{_t_end:.3f}"]   if _t_end   > 0.01 else []
    cmd = ([ffmpeg, "-y"] + _trim_ss + ["-i", config.input_video] + _trim_to
           + ["-sws_flags", "lanczos+accurate_rnd+full_chroma_int"]
           + ["-c:v", config.codec, "-crf", str(config.crf),
              "-preset", config.preset,
              "-movflags", "+faststart",
              "-c:a", "copy", config.output_video])
    stderr_store = [""]
    proc = _hidden_popen(cmd, stderr=subprocess.PIPE, bufsize=0)
    threading.Thread(target=_drain, args=(proc.stderr, stderr_store), daemon=True).start()
    proc.wait()
    if proc.returncode != 0:
        err = stderr_store[0]
        if isinstance(err, bytes): err = err.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed:\n{err[-1000:]}")
    if progress_callback:
        progress_callback(100, "Done!")
    return True

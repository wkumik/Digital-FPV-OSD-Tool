# VueOSD — Digital FPV OSD Tool

Overlay MSP-OSD data onto FPV DVR video footage. Supports Betaflight, INAV, ArduPilot, BetaFPV P1, Caddx Ascent, [Ruby FPV](https://rubyfpv.com/) and [Onyx FPV](https://onyxfpv.com/).

Reads `.osd` and `.srt` files recorded alongside your DVR video and renders the
HUD elements directly onto the footage — frame-accurate, GPU-accelerated.
![VueOSD](https://github.com/user-attachments/assets/eab2d3aa-94e0-401b-9568-39ad38457fd3)


---

## Features

- Frame-accurate OSD overlay (no sync drift)
- Betaflight, INAV & ArduPilot font support (SneakyFPV HD fonts included)
- Firmware auto-detected from OSD file header
- BetaFPV P1 embedded OSD support
- Caddx Ascent OSD support
- [Ruby FPV](https://rubyfpv.com/) support
- [Onyx FPV](https://onyxfpv.com/) support
- SRT telemetry bar (speed, altitude, satellites, signal)
- GPU-accelerated encoding (NVIDIA NVENC, AMD AMF, Intel QSV)
- Trim, scale, offset, opacity controls
- Live preview with scrubbing
- Upscale to 1440p, 2.7K or 4K
- CRF or target bitrate mode
- Light and dark theme with custom colour editor
- UI scaling (100% – 175%)

---

## Quick Start — Windows

Double-click **`VueOSD.bat`**

On first run it will:
1. Install Python dependencies automatically
2. Install FFmpeg automatically (via winget)
3. Launch the app

No manual setup required.

---

## Quick Start — Linux / macOS

```bash
chmod +x run.sh
./run.sh
```

Install FFmpeg separately:
```bash
sudo apt install ffmpeg        # Ubuntu/Debian
sudo dnf install ffmpeg        # Fedora
sudo pacman -S ffmpeg          # Arch
brew install ffmpeg            # macOS
```

---

## Build Standalone Executable

**Windows** — run `build.bat` after first launch (produces `dist\VueOSD.exe`)
**Linux/macOS** — run `./build.sh` (produces `dist/VueOSD`)

The resulting binary needs no Python installation on the target machine.
Bundle it with `ffmpeg.exe` (Windows) or ensure `ffmpeg` is on PATH.

---

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## File Structure

```
main.py              Main application (PyQt6 GUI)
video_processor.py   FFmpeg pipeline, GPU detection, encoding
osd_renderer.py      Numpy-based OSD compositor (~1 ms/frame)
osd_parser.py        .osd binary format parser
p1_osd_parser.py     BetaFPV P1 embedded OSD extractor
srt_parser.py        .srt telemetry parser
font_loader.py       OSD font loader (multi-column HD fonts)
fonts/               SneakyFPV OSD font packs
```

---

## Credits

See [CREDITS.md](docs/CREDITS.md) for full attribution — including SneakyFPV (fonts)
and Walksnal (original OSD tool concept).

## Licence

MIT — see [docs/LICENSE.md](docs/LICENSE.md)

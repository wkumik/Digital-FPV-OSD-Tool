# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025 VueOSD — https://github.com/wkumik/Digital-FPV-OSD-Tool
"""
srt_parser.py  –  Parse VueOSD SRT telemetry subtitle files.

SRT files are standard subtitle files containing custom telemetry lines.
From the Ruby firmware (rx_video_recording_data.cpp), the SRT content
can include any combination of:

  Line 1 (if MAVLink enabled and data available):
    D: <dist>m/ft  H: <alt>m/ft  <lat>, <lon>  <voltage> V
    (or "No MAVLink telemetry" if FC has no MAVLink)

  Line 2 (second text line, optional per settings):
    <MM:SS>  Radio 1: <dBm> dBm  <SNR> SNR  Radio 2: ...   <Mbps> Mbps

The format is flexible and varies per system configuration.
We parse generically: extract any recognisable values, skip
"No MAVLink telemetry" lines, and show everything else in the
status bar.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Selectable SRT field registry ─────────────────────────────────────────────
SRT_FIELDS: list[tuple[str, str]] = [
    ("flight_time", "Flight Time"),
    ("signal",      "Signal"),
    ("channel",     "Channel"),
    ("freq_mhz",    "Frequency"),
    ("radio1",      "Radio 1"),
    ("radio2",      "Radio 2"),
    ("sbat_v",      "Sky Battery"),
    ("gbat_v",      "Ground Battery"),
    ("voltage_v",   "Voltage"),
    ("delay_ms",    "Latency"),
    ("link_mbps",   "Bitrate"),
    ("altitude_m",  "Altitude"),
    ("distance_m",  "Distance"),
    ("sty_mode",    "STY Mode"),
    ("gsnr",        "Ground SNR"),
    ("ssnr",        "Sky SNR"),
    ("gtemp_c",     "Ground Temp"),
    ("stemp_c",     "Sky Temp"),
    ("frame_fps",   "Frame Rate"),
    ("gerr",        "Ground Errors"),
    ("serr",        "Sky Errors"),
]
ALL_SRT_FIELD_KEYS: set[str] = {k for k, _ in SRT_FIELDS}


@dataclass
class TelemetryData:
    raw_lines:     list[str]       = field(default_factory=list)
    flight_time:   str             = ""
    radio1_dbm:    Optional[int]   = None
    radio1_snr:    Optional[int]   = None
    radio2_dbm:    Optional[int]   = None
    radio2_snr:    Optional[int]   = None
    link_mbps:     Optional[float] = None
    distance_m:    Optional[float] = None
    altitude_m:    Optional[float] = None
    voltage_v:     Optional[float] = None
    gps_lat:       Optional[float] = None
    gps_lon:       Optional[float] = None
    # Walksnail/Avatar-style fields
    signal:        Optional[int]   = None
    channel:       Optional[int]   = None
    freq_mhz:      Optional[float] = None
    sbat_v:        Optional[float] = None
    gbat_v:        Optional[float] = None
    delay_ms:      Optional[int]   = None
    sty_mode:      Optional[int]   = None
    # CaddxFPV/Walksnail fw 39.44.15+ fields
    gsnr:          Optional[float] = None
    ssnr:          Optional[float] = None
    gtemp_c:       Optional[int]   = None
    stemp_c:       Optional[int]   = None
    frame_fps:     Optional[int]   = None
    gerr:          Optional[int]   = None
    serr:          Optional[int]   = None

    def status_line(self, enabled: "set[str] | None" = None) -> str:
        """Build a compact one-line status string for the bottom overlay bar.

        *enabled* — set of field keys to include (see ``SRT_FIELDS``).
        ``None`` means show all fields (backward compatible).
        """
        def _on(key: str) -> bool:
            return enabled is None or key in enabled

        parts = []
        if self.flight_time and _on("flight_time"):
            parts.append(self.flight_time)
        if self.signal is not None and _on("signal"):
            parts.append(f"Sig:{self.signal}")
        if self.channel is not None and _on("channel"):
            parts.append(f"CH:{self.channel}")
        if self.freq_mhz is not None and _on("freq_mhz"):
            parts.append(f"{self.freq_mhz:.0f}MHz")
        if self.radio1_dbm is not None and _on("radio1"):
            s = f"R1:{self.radio1_dbm:+d}dBm"
            if self.radio1_snr is not None:
                s += f" {self.radio1_snr}SNR"
            parts.append(s)
        if self.radio2_dbm is not None and _on("radio2"):
            s = f"R2:{self.radio2_dbm:+d}dBm"
            if self.radio2_snr is not None:
                s += f" {self.radio2_snr}SNR"
            parts.append(s)
        if self.sbat_v is not None and _on("sbat_v"):
            parts.append(f"SBat:{self.sbat_v:.1f}V")
        if self.gbat_v is not None and _on("gbat_v"):
            parts.append(f"GBat:{self.gbat_v:.1f}V")
        if self.voltage_v is not None and _on("voltage_v"):
            parts.append(f"{self.voltage_v:.1f}V")
        if self.delay_ms is not None and _on("delay_ms"):
            parts.append(f"Latency:{self.delay_ms}ms")
        if self.link_mbps is not None and _on("link_mbps"):
            parts.append(f"{self.link_mbps:.1f}Mbps")
        if self.altitude_m is not None and _on("altitude_m"):
            parts.append(f"H:{self.altitude_m:.0f}m")
        if self.distance_m is not None and _on("distance_m"):
            parts.append(f"Dist:{self.distance_m:.0f}m")
        if self.sty_mode is not None and _on("sty_mode"):
            parts.append(f"STY:{self.sty_mode}")
        if self.gsnr is not None and _on("gsnr"):
            parts.append(f"GSNR:{self.gsnr:.1f}")
        if self.ssnr is not None and _on("ssnr"):
            parts.append(f"SSNR:{self.ssnr:.1f}")
        if self.gtemp_c is not None and _on("gtemp_c"):
            parts.append(f"Gtemp:{self.gtemp_c}C")
        if self.stemp_c is not None and _on("stemp_c"):
            parts.append(f"Stemp:{self.stemp_c}C")
        if self.frame_fps is not None and _on("frame_fps"):
            parts.append(f"{self.frame_fps}fps")
        if self.gerr is not None and _on("gerr"):
            parts.append(f"Gerr:{self.gerr}")
        if self.serr is not None and _on("serr"):
            parts.append(f"SErr:{self.serr}")
        return "  ".join(parts)


@dataclass
class SrtEntry:
    index:     int
    start_ms:  int
    end_ms:    int
    telemetry: TelemetryData


@dataclass
class SrtFile:
    entries:     list[SrtEntry] = field(default_factory=list)
    duration_ms: int = 0

    def get_data_at_time(self, timestamp_ms: int) -> Optional[TelemetryData]:
        """Return telemetry for the SRT entry active at timestamp_ms."""
        # Binary-search-friendly: entries are in order
        for e in self.entries:
            if e.start_ms <= timestamp_ms < e.end_ms:
                return e.telemetry
        return None


# ── Regexes ────────────────────────────────────────────────────────────────────

_TS_RE      = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
_SKIP_RE    = re.compile(r"No MAVLink telemetry", re.IGNORECASE)

# Radio: "Radio 1: -65 dBm  12 SNR" or "Radio 1: -65 dBm  12 SNR  "
_RADIO_RE   = re.compile(
    r"Radio\s+(\d+):\s*(-?\d+)\s*dBm(?:\s+(-?\d+)\s*SNR)?", re.IGNORECASE
)
_MBPS_RE        = re.compile(r"(?:Bitrate:)?([\d.]+)\s*Mbps", re.IGNORECASE)
_TIME_RE        = re.compile(r"^\s*(\d{2}):(\d{2})\b")
_DIST_RE        = re.compile(r"\b(?:Distance|D):\s*([\d.]+)\s*(m|ft)", re.IGNORECASE)
_ALT_RE         = re.compile(r"\bH:\s*([\d.]+)\s*(m|ft)", re.IGNORECASE)
_VOLT_RE        = re.compile(r"([\d.]+)\s*V\b")
_GPS_RE         = re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")
# Walksnail/Avatar fields
_SIGNAL_RE      = re.compile(r"Signal:\s*(\d+)", re.IGNORECASE)
_CHANNEL_RE     = re.compile(r"\bCH:\s*(\d+)", re.IGNORECASE)
_HZ_RE          = re.compile(r"\bHz:\s*(\d+)", re.IGNORECASE)
_FLIGHTTIME_RE  = re.compile(r"FlightTime:\s*(\d+)", re.IGNORECASE)
_SBAT_RE        = re.compile(r"SBat:\s*([\d.]+)\s*V?", re.IGNORECASE)
_GBAT_RE        = re.compile(r"GBat:\s*([\d.]+)\s*V?", re.IGNORECASE)
_DELAY_RE       = re.compile(r"Delay:\s*(\d+)\s*ms", re.IGNORECASE)
_STYMODE_RE     = re.compile(r"STYMode:\s*(\d+)", re.IGNORECASE)
# CaddxFPV/Walksnail fw 39.44.15+ fields
_GSNR_RE        = re.compile(r"\bGSNR:\s*([\d.]+)", re.IGNORECASE)
_SSNR_RE        = re.compile(r"\bSSNR:\s*([\d.]+)", re.IGNORECASE)
_GTEMP_RE       = re.compile(r"\bGtemp:\s*(-?\d+)", re.IGNORECASE)
_STEMP_RE       = re.compile(r"\bStemp:\s*(-?\d+)", re.IGNORECASE)
_FRAME_RE       = re.compile(r"\bFrame:\s*(\d+)", re.IGNORECASE)
_GERR_RE        = re.compile(r"\bGerr:\s*(\d+)", re.IGNORECASE)
_SERR_RE        = re.compile(r"\bSErr:\s*(\d+)", re.IGNORECASE)


def _ft_to_m(ft: float) -> float:
    return ft / 3.28084


def _parse_lines(lines: list[str]) -> TelemetryData:
    t = TelemetryData(raw_lines=list(lines))

    for line in lines:
        if _SKIP_RE.search(line):
            continue

        # Flight time MM:SS at start of line
        tm = _TIME_RE.match(line)
        if tm and not t.flight_time:
            t.flight_time = f"{tm.group(1)}:{tm.group(2)}"

        # FlightTime:N  (seconds as integer, e.g. Walksnail)
        ft = _FLIGHTTIME_RE.search(line)
        if ft and not t.flight_time:
            secs = int(ft.group(1))
            t.flight_time = f"{secs // 60}:{secs % 60:02d}"

        # Signal strength
        sig = _SIGNAL_RE.search(line)
        if sig:
            t.signal = int(sig.group(1))

        # RF channel
        ch = _CHANNEL_RE.search(line)
        if ch:
            t.channel = int(ch.group(1))

        # Frequency in kHz (field is labelled Hz but value is kHz) → store as MHz
        hz = _HZ_RE.search(line)
        if hz:
            t.freq_mhz = int(hz.group(1)) / 1_000  # kHz → MHz

        # Per-battery voltages (Walksnail SBat / GBat)
        sbat = _SBAT_RE.search(line)
        if sbat:
            t.sbat_v = float(sbat.group(1))
        gbat = _GBAT_RE.search(line)
        if gbat:
            t.gbat_v = float(gbat.group(1))

        # Link delay
        dly = _DELAY_RE.search(line)
        if dly:
            t.delay_ms = int(dly.group(1))

        # STY mode (BetaFPV P1)
        sty = _STYMODE_RE.search(line)
        if sty:
            t.sty_mode = int(sty.group(1))

        # Ground/Sky SNR, temperature, frame rate, error counters (CaddxFPV fw 39.44.15+)
        gsnr = _GSNR_RE.search(line)
        if gsnr:
            t.gsnr = float(gsnr.group(1))
        ssnr = _SSNR_RE.search(line)
        if ssnr:
            t.ssnr = float(ssnr.group(1))
        gtemp = _GTEMP_RE.search(line)
        if gtemp:
            t.gtemp_c = int(gtemp.group(1))
        stemp = _STEMP_RE.search(line)
        if stemp:
            t.stemp_c = int(stemp.group(1))
        frame = _FRAME_RE.search(line)
        if frame:
            t.frame_fps = int(frame.group(1))
        gerr = _GERR_RE.search(line)
        if gerr:
            t.gerr = int(gerr.group(1))
        serr = _SERR_RE.search(line)
        if serr:
            t.serr = int(serr.group(1))

        # Radio interfaces
        for m in _RADIO_RE.finditer(line):
            idx  = int(m.group(1))
            dbm  = int(m.group(2))
            snr  = int(m.group(3)) if m.group(3) is not None else None
            if idx == 1:
                t.radio1_dbm = dbm
                if snr is not None:
                    t.radio1_snr = snr
            elif idx == 2:
                t.radio2_dbm = dbm
                if snr is not None:
                    t.radio2_snr = snr

        # Bitrate
        mb = _MBPS_RE.search(line)
        if mb:
            t.link_mbps = float(mb.group(1))

        # Distance
        d = _DIST_RE.search(line)
        if d:
            v = float(d.group(1))
            t.distance_m = _ft_to_m(v) if d.group(2).lower() == 'ft' else v

        # Altitude
        h = _ALT_RE.search(line)
        if h:
            v = float(h.group(1))
            t.altitude_m = _ft_to_m(v) if h.group(2).lower() == 'ft' else v

        # GPS
        gps = _GPS_RE.search(line)
        if gps:
            t.gps_lat = float(gps.group(1))
            t.gps_lon = float(gps.group(2))

        # Generic voltage — only if no SBat/GBat matched on the same line
        # (GPS coords never contain a literal "V", so they can't false-match _VOLT_RE)
        if not sbat and not gbat:
            volt = _VOLT_RE.search(line)
            if volt:
                t.voltage_v = float(volt.group(1))

    return t


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h)*3_600_000 + int(m)*60_000 + int(s)*1_000 + int(ms)


def parse_srt(path: str) -> SrtFile:
    srt = SrtFile()
    idx: Optional[int] = None
    start_ms = end_ms = 0
    data_lines: list[str] = []

    def _flush():
        if idx is not None:
            srt.entries.append(SrtEntry(
                index=idx, start_ms=start_ms, end_ms=end_ms,
                telemetry=_parse_lines(data_lines),
            ))

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n').strip()

            if not line:
                _flush()
                idx = None
                data_lines = []
                continue

            # Sequence number
            if idx is None and line.isdigit():
                idx = int(line)
                continue

            # Timestamp line
            ts = _TS_RE.match(line)
            if ts and idx is not None and not data_lines:
                g = ts.groups()
                start_ms = _to_ms(*g[:4])
                end_ms   = _to_ms(*g[4:])
                continue

            data_lines.append(line)

    _flush()  # handle file without trailing blank line

    if srt.entries:
        srt.duration_ms = srt.entries[-1].end_ms

    return srt

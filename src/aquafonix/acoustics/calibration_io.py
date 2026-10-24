"""Find hydrophone calibration in a recording's metadata.

Looks in the WAV file's text chunks (LIST/INFO, iXML, GUANO, bext, ...),
libsndfile's string metadata, and sidecar files next to the recording
(``<name>.xml``, ``<name>.log.xml``, ``<name>.json``, ``<name>.txt``,
``calibration.*``).  Recognises

* end-to-end calibration ("full scale = 172.5 dB re 1 uPa"),
* hydrophone sensitivity ("-165 dB re 1 V/uPa"),
* pre-amplifier / recorder gain ("gain 20 dB"),
* full-scale voltage ("full scale 2.5 Vpp", "ADC range 1 V peak").

Vendors write these differently, so this is pattern matching: the result
says where each value came from and the user can always override it.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

_NUM = r"([-+]?\d{1,3}(?:\.\d+)?)"
END_TO_END = [
    re.compile(r"(?:end[\s_-]*to[\s_-]*end|system|full[\s_-]*scale|clip(?:ping)?\s*level)"
               r"[^0-9\n]{0,40}" + r"(\d{2,3}(?:\.\d+)?)" + r"\s*dB\s*re\.?\s*1?\s*[µu]\s*Pa(?!\s*/)",
               re.I),
]
SENSITIVITY = [
    re.compile(_NUM + r"\s*dB\s*re\.?\s*1?\s*V\s*/\s*[µu]\s*Pa", re.I),
    re.compile(r"sensitivity[^0-9\-+\n]{0,40}" + _NUM, re.I),
]
GAIN = [re.compile(r"\bgain[^0-9\-+\n]{0,25}" + _NUM + r"\s*dB", re.I)]
FULL_SCALE = [
    re.compile(r"(?:full[\s_-]*scale|adc[\s_-]*range|input[\s_-]*range|v[\s_-]*fs|vref|"
               r"peak[\s_-]*voltage)[^0-9\n]{0,30}(\d+(?:\.\d+)?)\s*(mV|V)\s*(pp|p-p|peak|pk)?",
               re.I),
]


@dataclass
class CalibrationHint:
    sensitivity_db: float | None = None
    gain_db: float | None = None
    full_scale_v: float | None = None
    end_to_end_db: float | None = None  # dB re 1 uPa at digital full scale
    source: str = ""

    @property
    def found(self) -> bool:
        return self.end_to_end_db is not None or self.sensitivity_db is not None

    def values(self) -> tuple[float, float, float]:
        """``(sensitivity dB re 1 V/uPa, gain dB, full-scale V)`` for the panel."""
        if self.end_to_end_db is not None:
            return -self.end_to_end_db, 0.0, 1.0
        return (self.sensitivity_db if self.sensitivity_db is not None else -165.0,
                self.gain_db or 0.0, self.full_scale_v or 1.0)

    def describe(self) -> str:
        if self.end_to_end_db is not None:
            return f"full scale = {self.end_to_end_db:g} dB re 1 µPa ({self.source})"
        parts = [f"sensitivity {self.sensitivity_db:g} dB re 1 V/µPa"]
        if self.gain_db is not None:
            parts.append(f"gain {self.gain_db:g} dB")
        if self.full_scale_v is not None:
            parts.append(f"full scale {self.full_scale_v:g} V")
        return ", ".join(parts) + f" ({self.source})"


def wav_text_chunks(path: str, max_chunk: int = 1 << 20) -> list[str]:
    """Text found in a RIFF/RF64 WAV file's non-audio chunks."""
    out = []
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64", b"BW64") or head[8:12] != b"WAVE":
                return out
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                if cid == b"data" or size > max_chunk or size == 0xFFFFFFFF:
                    if cid == b"data" and size == 0xFFFFFFFF:
                        break  # RF64 data size lives in ds64; nothing useful after it
                    f.seek(size + (size & 1), 1)
                    continue
                body = f.read(size + (size & 1))
                if cid in (b"fmt ", b"fact", b"ds64", b"PEAK", b"cue "):
                    continue
                text = body.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
                if text.strip() and printable > 0.8 * len(text):
                    out.append(text)
    except OSError:
        pass
    return out


def _flatten_json(obj, prefix="") -> list[str]:
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            lines += _flatten_json(v, f"{prefix}{k} ")
        return lines
    if isinstance(obj, list):
        return [ln for v in obj for ln in _flatten_json(v, prefix)]
    return [f"{prefix}: {obj}"]


def sidecar_texts(path: str) -> list[tuple[str, str]]:
    p = Path(path)
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    candidates = [p.with_name(stem + ext) for ext in
                  (".xml", ".log.xml", ".json", ".txt", ".cal", ".ini", ".meta")]
    candidates += sorted(p.parent.glob("calibration.*")) + sorted(p.parent.glob("*.cal"))
    out = []
    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size < (1 << 20) and c != p:
                text = c.read_text(encoding="utf-8", errors="ignore")
                if c.suffix.lower() == ".json":
                    try:
                        text = "\n".join(_flatten_json(json.loads(text)))
                    except ValueError:
                        pass
                out.append((c.name, re.sub(r"<[^>]+>", " ", text) if c.suffix == ".xml"
                            else text))
        except OSError:
            continue
    return out


def parse_text(text: str) -> CalibrationHint:
    hint = CalibrationHint()
    for rx in END_TO_END:
        m = rx.search(text)
        if m and 100 <= float(m.group(1)) <= 260:
            hint.end_to_end_db = float(m.group(1))
            break
    for rx in SENSITIVITY:
        m = rx.search(text)
        if m:
            v = float(m.group(1))
            if 100 <= abs(v) <= 260:
                hint.sensitivity_db = -abs(v)
                break
    for rx in GAIN:
        m = rx.search(text)
        if m and -40 <= float(m.group(1)) <= 100:
            hint.gain_db = float(m.group(1))
            break
    for rx in FULL_SCALE:
        m = rx.search(text)
        if m:
            v = float(m.group(1)) / (1000 if m.group(2).lower() == "mv" else 1)
            if (m.group(3) or "").lower() in ("pp", "p-p"):
                v /= 2
            if 0.001 <= v <= 100:
                hint.full_scale_v = v
                break
    return hint


def find_calibration(path: str) -> CalibrationHint | None:
    """Calibration from the file's metadata or sidecar files, or None."""
    sources: list[tuple[str, str]] = [("WAV metadata", t) for t in wav_text_chunks(path)]
    try:
        import soundfile as sf

        with sf.SoundFile(path) as f:
            meta = " \n".join(f"{k}: {v}" for k, v in f.copy_metadata().items() if v)
        if meta:
            sources.append(("file tags", meta))
    except Exception:
        pass
    sources += sidecar_texts(path)
    best = None
    for name, text in sources:
        hint = parse_text(text)
        if hint.found:
            hint.source = name
            if hint.end_to_end_db is not None:
                return hint
            best = best or hint
    return best

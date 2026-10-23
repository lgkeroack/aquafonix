"""Audio file input/output (WAV, FLAC, AIFF, ... via libsndfile)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPEN_FILTER = ("Audio files (*.wav *.flac *.aif *.aiff *.ogg *.w64 *.rf64 *.caf);;"
               "All files (*)")


@dataclass
class AudioInfo:
    sample_rate: float
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def info(path: str) -> AudioInfo:
    import soundfile as sf

    i = sf.info(path)
    return AudioInfo(float(i.samplerate), int(i.channels), int(i.frames))


def read(path: str, start: float = 0.0, duration: float | None = None) -> tuple[np.ndarray, float]:
    """``(data[n, channels] float64, sample_rate)``."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        f.seek(int(start * f.samplerate))
        n = -1 if duration is None else int(duration * f.samplerate)
        data = f.read(n, dtype="float64", always_2d=True)
        return data, float(f.samplerate)


def write(path: str, data: np.ndarray, sample_rate: float) -> None:
    """WAV is written as 32-bit float (lossless for any scale); FLAC as
    24-bit PCM (clipped to +-1)."""
    import soundfile as sf

    data = np.nan_to_num(np.asarray(data, dtype=np.float64))
    if data.ndim == 1:
        data = data[:, None]
    lower = path.lower()
    if lower.endswith(".flac"):
        sf.write(path, np.clip(data, -1, 1), int(round(sample_rate)), subtype="PCM_24")
    else:
        sf.write(path, data.astype(np.float32), int(round(sample_rate)), subtype="FLOAT")

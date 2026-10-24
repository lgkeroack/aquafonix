"""User-built signature catalog.

Snip a sound, give it a name, and later occurrences are flagged as
"Suspected: <name>".  Every confirmation adds the occurrence as another
example; a correction moves it to the right label (and teaches the wrong
label not to match it); a rejection adds a *negative* example.  So the
catalog grows and sharpens with use.

Matching:
* ``spectral`` - normalised cross-correlation of the floor-relative
  spectrogram inside the signature's frequency band (sounds and calls);
* ``shape`` - normalised cross-correlation of the waveform itself (slow
  sensor data or graphs, where the shape is what matters).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from aquafonix import audiofile
from aquafonix.acoustics.filters import resample
from aquafonix.acoustics.spectral import (auto_nfft, blur_freq, power_frames, snr_frames,
                                          to_db)

log = logging.getLogger(__name__)

METHODS = {
    "auto": "Automatic",
    "spectral": "Spectrogram (sounds, calls, clicks)",
    "shape": "Waveform shape (slow sensor data, graphs)",
}
CATEGORIES = ["", "Marine mammal", "Fish", "Invertebrate", "Vessel / machinery",
              "Sonar / echosounder", "Seismic / impulsive", "Weather / sea state",
              "Equipment / self-noise", "Other"]
MAX_EXAMPLES = 30  # per kind, per signature
MAX_CLIP_S = 10.0
SHAPE_MAX_SAMPLES = 4096
DEFAULT_THRESHOLD = 0.6


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class Example:
    id: str
    file: str
    sample_rate: float
    kind: str = "positive"  # "positive" | "negative"
    f_lo: float = 0.0
    f_hi: float = 0.0  # 0 = up to Nyquist
    created: str = field(default_factory=_now)
    source: str = ""
    floor_hz: list[float] = field(default_factory=list)
    floor_db: list[float] = field(default_factory=list)
    data: str = "audio"  # "audio" clip (WAV) or spectrogram "patch" (NPZ)


@dataclass
class Signature:
    id: str
    label: str
    category: str = ""
    notes: str = ""
    method: str = "spectral"
    threshold: float | None = None
    enabled: bool = True
    created: str = field(default_factory=_now)
    examples: list[Example] = field(default_factory=list)
    confirmed: int = 0
    rejected: int = 0

    @property
    def positives(self) -> list[Example]:
        return [e for e in self.examples if e.kind == "positive"]

    @property
    def negatives(self) -> list[Example]:
        return [e for e in self.examples if e.kind == "negative"]


def pick_method(method: str, sample_rate: float) -> str:
    if method in ("spectral", "shape"):
        return method
    return "spectral" if sample_rate >= 2000 else "shape"


def estimate_band(clip: np.ndarray, sample_rate: float, floor_hz=None, floor_db=None,
                  nfft: int | None = None) -> tuple[float, float]:
    """Frequency band where a snipped sound rises above the noise floor."""
    clip = np.asarray(clip, dtype=float)
    nfft = nfft or auto_nfft(sample_rate)
    if len(clip) < nfft:
        return 0.0, sample_rate / 2
    p = power_frames(clip, nfft, nfft // 2)
    d = to_db(p)
    freqs = np.fft.rfftfreq(nfft, 1 / sample_rate)
    if floor_hz is not None and len(floor_hz):
        floor = np.interp(freqs, floor_hz, floor_db)
    else:
        floor = np.percentile(d, 20, axis=0)
    snr = snr_frames(d, floor).max(axis=0)
    active = snr > 6
    if active.sum() < 2:
        return 0.0, sample_rate / 2
    weight = np.where(active, 10 ** (snr / 10) - 1, 0)
    cdf = np.cumsum(weight) / weight.sum()
    lo = float(freqs[np.searchsorted(cdf, 0.03)])
    hi = float(freqs[min(np.searchsorted(cdf, 0.97), len(freqs) - 1)])
    return max(0.0, lo * 0.85), min(sample_rate / 2, hi * 1.15)


class SignatureLibrary:
    """The catalog on disk: ``signatures.json`` plus one WAV per example."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.path = self.root / "signatures.json"
        self.signatures: list[Signature] = []
        self.version = 0
        self._lock = threading.RLock()
        self._clip_cache: dict[str, tuple[np.ndarray, float]] = {}
        self.load()

    # -- persistence ---------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self.signatures = []
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    for s in raw.get("signatures", []):
                        exs = [Example(**e) for e in s.pop("examples", [])]
                        self.signatures.append(Signature(**s, examples=exs))
                except Exception:
                    log.exception("could not read %s", self.path)
            self.version += 1

    def save(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            data = {"format": 1, "signatures": [asdict(s) for s in self.signatures]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self.path)
            self.version += 1

    # -- queries ---------------------------------------------------------------
    def labels(self) -> list[str]:
        with self._lock:
            return sorted((s.label for s in self.signatures), key=str.lower)

    def get(self, sig_id: str) -> Signature | None:
        with self._lock:
            return next((s for s in self.signatures if s.id == sig_id), None)

    def find(self, label: str) -> Signature | None:
        key = label.strip().lower()
        with self._lock:
            return next((s for s in self.signatures if s.label.lower() == key), None)

    def load_clip(self, example: Example) -> tuple[np.ndarray, float]:
        cached = self._clip_cache.get(example.id)
        if cached is None:
            data, rate = audiofile.read(str(self.clips_dir / example.file))
            cached = (data.mean(axis=1), rate)
            self._clip_cache[example.id] = cached
        return cached

    # -- edits -----------------------------------------------------------------
    def add_example(self, label: str, clip: np.ndarray, sample_rate: float,
                    f_lo: float = 0.0, f_hi: float = 0.0, kind: str = "positive",
                    method: str = "auto", category: str = "", notes: str = "",
                    source: str = "", floor: tuple | None = None) -> tuple[Signature, Example]:
        """Add a snipped clip (mono) as an example; creates the signature if
        the label is new."""
        label = label.strip()
        if not label:
            raise ValueError("a name is required")
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        clip = clip[: int(MAX_CLIP_S * sample_rate)]
        if len(clip) < 8:
            raise ValueError("the selection is too short")
        with self._lock:
            sig = self.find(label)
            if sig is None:
                sig = Signature(uuid.uuid4().hex[:12], label, category, notes,
                                pick_method(method, sample_rate))
                self.signatures.append(sig)
            else:
                if category:
                    sig.category = category
                if notes and notes not in sig.notes:
                    sig.notes = (sig.notes + "\n" + notes).strip()
            ex_id = uuid.uuid4().hex[:12]
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{sig.id}_{ex_id}.wav"
            audiofile.write(str(self.clips_dir / fname), clip, sample_rate)
            ex = Example(ex_id, fname, float(sample_rate), kind, float(f_lo), float(f_hi),
                         source=source)
            if floor is not None:
                hz, fdb = floor
                step = max(1, len(hz) // 256)
                ex.floor_hz = [float(v) for v in hz[::step]]
                ex.floor_db = [float(v) for v in fdb[::step]]
            sig.examples.append(ex)
            same = [e for e in sig.examples if e.kind == kind]
            while len(same) > MAX_EXAMPLES:  # keep the first (defining) example
                victim = same[1]
                self._delete_example_file(victim)
                sig.examples.remove(victim)
                same.remove(victim)
            self.save()
            return sig, ex

    def add_patch_example(self, label: str, freqs: np.ndarray, frame_period: float,
                          db: np.ndarray, floor: np.ndarray | None, f_lo: float = 0.0,
                          f_hi: float = 0.0, kind: str = "positive", category: str = "",
                          notes: str = "", source: str = "") -> tuple[Signature, Example]:
        """Add a spectrogram patch (frames x freqs, dB) as an example: used
        when the sound was snipped from a spectrogram read off the screen."""
        label = label.strip()
        if not label:
            raise ValueError("a name is required")
        db = np.asarray(db, dtype=np.float32)
        if db.ndim != 2 or db.shape[0] < 1 or db.shape[1] < 3:
            raise ValueError("the selection is too small")
        with self._lock:
            sig = self.find(label)
            if sig is None:
                sig = Signature(uuid.uuid4().hex[:12], label, category, notes, "spectral")
                self.signatures.append(sig)
            ex_id = uuid.uuid4().hex[:12]
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{sig.id}_{ex_id}.npz"
            floor = np.asarray(floor if floor is not None else np.nanpercentile(db, 20, axis=0),
                               dtype=np.float32)
            np.savez_compressed(self.clips_dir / fname, freqs=np.asarray(freqs, np.float32),
                                frame_period=np.float32(frame_period), db=db, floor=floor)
            ex = Example(ex_id, fname, 1.0 / max(frame_period, 1e-9), kind, float(f_lo),
                         float(f_hi), source=source, data="patch")
            sig.examples.append(ex)
            same = [e for e in sig.examples if e.kind == kind]
            while len(same) > MAX_EXAMPLES:
                victim = same[1]
                self._delete_example_file(victim)
                sig.examples.remove(victim)
                same.remove(victim)
            self.save()
            return sig, ex

    def load_patch(self, example: Example) -> dict:
        cached = self._clip_cache.get(example.id)
        if cached is None:
            with np.load(self.clips_dir / example.file) as z:
                cached = {k: z[k] for k in z.files}
            self._clip_cache[example.id] = cached
        return cached

    def _delete_example_file(self, ex: Example) -> None:
        self._clip_cache.pop(ex.id, None)
        try:
            (self.clips_dir / ex.file).unlink()
        except OSError:
            pass

    def update(self, sig: Signature) -> None:
        self.save()

    def rename(self, sig: Signature, label: str) -> None:
        label = label.strip()
        other = self.find(label)
        if other is not None and other is not sig:
            raise ValueError(f"'{label}' already exists")
        sig.label = label
        self.save()

    def remove_signature(self, sig: Signature) -> None:
        with self._lock:
            for ex in sig.examples:
                self._delete_example_file(ex)
            self.signatures = [s for s in self.signatures if s.id != sig.id]
            self.save()

    def remove_example(self, sig: Signature, ex: Example) -> None:
        with self._lock:
            self._delete_example_file(ex)
            sig.examples = [e for e in sig.examples if e.id != ex.id]
            if not sig.positives:
                self.remove_signature(sig)
            else:
                self.save()


# =============================================================================
# matching
# =============================================================================
@dataclass
class SpectralTemplate:
    sig_id: str
    example_id: str
    kind: str
    r0: int
    r1: int
    patch: np.ndarray  # (rows, frames), zero mean, unit norm
    duration: float

    @property
    def frames(self) -> int:
        return self.patch.shape[1]


@dataclass
class ShapeTemplate:
    sig_id: str
    example_id: str
    kind: str
    wave: np.ndarray  # zero mean, unit norm
    duration: float


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(np.ceil(np.log2(max(n, 2)))))


def ncc_spectral(block: np.ndarray, tpl: SpectralTemplate, fft_block: np.ndarray | None = None,
                 min_mean: float = 1.0) -> np.ndarray:
    """Scores for every alignment of ``tpl`` in ``block`` (bins, frames)
    blurred floor-relative SNR. Output index k = window ending at frame
    ``k + tpl.frames - 1``."""
    t = tpl.frames
    nf = block.shape[1]
    k = nf - t + 1
    if k <= 0 or tpl.r1 > block.shape[0]:
        return np.zeros(0)
    w = block[tpl.r0:tpl.r1]
    n_fft = _next_pow2(nf)
    fw = fft_block[tpl.r0:tpl.r1] if fft_block is not None else np.fft.rfft(w, n_fft, axis=1)
    fp = np.fft.rfft(tpl.patch, n_fft, axis=1)
    num = np.fft.irfft((fw * np.conj(fp)).sum(axis=0), n_fft)[:k]
    c1 = np.concatenate([[0.0], np.cumsum(w.sum(axis=0))])
    c2 = np.concatenate([[0.0], np.cumsum((w * w).sum(axis=0))])
    s1, s2 = c1[t:] - c1[:-t], c2[t:] - c2[:-t]
    n = w.shape[0] * t
    var = s2 - s1 * s1 / n
    ok = (var > 1e-9) & (s1 / n >= min_mean)
    out = np.zeros(k)
    out[ok] = num[ok] / np.sqrt(var[ok])
    return np.clip(out, -1.0, 1.0)


def ncc_shape(x: np.ndarray, tpl: ShapeTemplate) -> np.ndarray:
    """Normalised cross-correlation of a waveform template along ``x``."""
    L = len(tpl.wave)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    k = len(x) - L + 1
    if k <= 0:
        return np.zeros(0)
    n_fft = _next_pow2(len(x))
    num = np.fft.irfft(np.fft.rfft(x, n_fft) * np.conj(np.fft.rfft(tpl.wave, n_fft)), n_fft)[:k]
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s1, s2 = c1[L:] - c1[:-L], c2[L:] - c2[:-L]
    var = s2 - s1 * s1 / L
    out = np.zeros(k)
    ok = var > 1e-9 * np.maximum(s2, 1e-300)  # skip flat (constant) stretches
    out[ok] = num[ok] / np.sqrt(var[ok])
    return np.clip(out, -1.0, 1.0)


class Matcher:
    """Turns the library into templates for the current analysis grid and
    scores data against them."""

    def __init__(self, library: SignatureLibrary) -> None:
        self.library = library
        self.spectral: list[SpectralTemplate] = []
        self.shape: list[ShapeTemplate] = []
        self._key = None
        self.max_frames = 0
        self.max_shape = 0

    @property
    def empty(self) -> bool:
        return not self.spectral and not self.shape

    def threshold(self, sig_id: str, default: float) -> float:
        sig = self.library.get(sig_id)
        return sig.threshold if sig is not None and sig.threshold else default

    def label(self, sig_id: str) -> str:
        sig = self.library.get(sig_id)
        return sig.label if sig is not None else "?"

    def prepare(self, sample_rate: float, nfft: int, hop: int) -> None:
        self._prepare((self.library.version, round(sample_rate, 3), nfft, hop),
                      np.fft.rfftfreq(nfft, 1 / sample_rate), hop / sample_rate,
                      (sample_rate, nfft, hop))

    def prepare_grid(self, freqs: np.ndarray, frame_period: float) -> None:
        """Templates for spectrogram frames that come from the screen (only
        patch examples apply: there is no audio to transform)."""
        self._prepare((self.library.version, "grid", len(freqs), float(freqs[0]),
                       float(freqs[-1]), round(frame_period, 6)), np.asarray(freqs, float),
                      frame_period, None)

    def _prepare(self, key, freqs, frame_period, audio) -> None:
        if key == self._key:
            return
        self._key = key
        spectral, shape = [], []
        with self.library._lock:
            sigs = [s for s in self.library.signatures if s.enabled]
        for sig in sigs:
            for ex in sig.examples:
                if ex.data == "patch":
                    try:
                        tpl = self._patch_template(sig, ex, freqs, frame_period)
                    except Exception:
                        log.warning("unusable patch for %s/%s", sig.label, ex.id, exc_info=True)
                        tpl = None
                    if tpl is not None:
                        spectral.append(tpl)
                    continue
                if audio is None:
                    continue
                sample_rate, nfft, hop = audio
                try:
                    clip, rate = self.library.load_clip(ex)
                except Exception:
                    log.warning("missing clip for %s/%s", sig.label, ex.id)
                    continue
                if sig.method == "shape":
                    tpl = self._shape_template(sig, ex, clip, rate, sample_rate)
                    if tpl is not None:
                        shape.append(tpl)
                else:
                    tpl = self._spectral_template(sig, ex, clip, rate, sample_rate, nfft, hop,
                                                  freqs)
                    if tpl is not None:
                        spectral.append(tpl)
        self.spectral, self.shape = spectral, shape
        self.max_frames = max((t.frames for t in spectral), default=0)
        self.max_shape = max((len(t.wave) for t in shape), default=0)

    def _spectral_template(self, sig, ex, clip, rate, sample_rate, nfft, hop, freqs):
        if abs(rate - sample_rate) > 1e-6:
            if ex.f_lo >= sample_rate / 2:
                return None
            clip = resample(clip, rate, sample_rate)
        d = to_db(power_frames(clip, nfft, hop))
        if len(d) < 1:
            return None
        if ex.floor_hz:
            floor = np.interp(freqs, ex.floor_hz, ex.floor_db)
        else:
            floor = np.percentile(d, 20, axis=0) if len(d) >= 5 else d.min(axis=0)
        snr = blur_freq(snr_frames(d, floor), axis=1)  # (frames, bins)
        hi_hz = ex.f_hi if ex.f_hi > 0 else sample_rate / 2
        r0 = int(np.searchsorted(freqs, ex.f_lo, side="left"))
        r1 = int(np.searchsorted(freqs, hi_hz, side="right"))
        r0, r1 = max(0, min(r0, len(freqs) - 3)), min(len(freqs), max(r1, r0 + 3))
        band = snr[:, r0:r1]
        active = np.flatnonzero(band.mean(axis=1) > 1.5)
        if len(active):
            band = band[max(0, active[0] - 1):active[-1] + 2]
        patch = band.T - band.mean()
        norm = np.linalg.norm(patch)
        if norm < 1e-6:
            return None
        return SpectralTemplate(sig.id, ex.id, ex.kind, r0, r1, patch / norm,
                                band.shape[0] * hop / sample_rate)

    def _patch_template(self, sig, ex, freqs, frame_period):
        """Resample a stored spectrogram patch onto the current grid."""
        z = self.library.load_patch(ex)
        pf, pdb, floor = z["freqs"].astype(float), z["db"].astype(float), z["floor"].astype(float)
        dt_src = float(z["frame_period"])
        snr = np.clip(np.nan_to_num(pdb - floor[None, :]), 0.0, 40.0)  # (frames, f)
        on_grid = np.stack([np.interp(freqs, pf, row, left=0.0, right=0.0) for row in snr])
        n_new = max(1, int(round(len(on_grid) * dt_src / frame_period)))
        if n_new != len(on_grid):
            src_t = np.arange(len(on_grid)) * dt_src
            new_t = np.arange(n_new) * frame_period
            on_grid = np.stack([np.interp(new_t, src_t, on_grid[:, k])
                                for k in range(on_grid.shape[1])], axis=1)
        snr = blur_freq(on_grid, axis=1)
        hi_hz = ex.f_hi if ex.f_hi > 0 else float(freqs[-1])
        r0 = int(np.searchsorted(freqs, ex.f_lo, side="left"))
        r1 = int(np.searchsorted(freqs, hi_hz, side="right"))
        r0, r1 = max(0, min(r0, len(freqs) - 3)), min(len(freqs), max(r1, r0 + 3))
        band = snr[:, r0:r1]
        active = np.flatnonzero(band.mean(axis=1) > 1.5)
        if len(active):
            band = band[max(0, active[0] - 1):active[-1] + 2]
        patch = band.T - band.mean()
        norm = np.linalg.norm(patch)
        if norm < 1e-6:
            return None
        return SpectralTemplate(sig.id, ex.id, ex.kind, r0, r1, patch / norm,
                                band.shape[0] * frame_period)

    def _shape_template(self, sig, ex, clip, rate, sample_rate):
        n = int(round(len(clip) * sample_rate / rate))
        if n < 4:
            return None
        if n != len(clip):
            clip = np.interp(np.linspace(0, len(clip) - 1, n), np.arange(len(clip)), clip)
        clip = clip[:SHAPE_MAX_SAMPLES]
        w = clip - clip.mean()
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            return None
        return ShapeTemplate(sig.id, ex.id, ex.kind, w / norm, len(w) / sample_rate)

    # -- scoring ---------------------------------------------------------------
    def spectral_scores(self, block: np.ndarray, kind: str = "positive"):
        """``[(template, scores)]`` over a blurred SNR block (bins, frames)."""
        out = []
        if block.shape[1] == 0:
            return out
        n_fft = _next_pow2(block.shape[1])
        fft_block = np.fft.rfft(block, n_fft, axis=1)
        for tpl in self.spectral:
            if tpl.kind == kind:
                out.append((tpl, ncc_spectral(block, tpl, fft_block)))
        return out

    def shape_scores(self, x: np.ndarray, kind: str = "positive"):
        return [(tpl, ncc_shape(x, tpl)) for tpl in self.shape if tpl.kind == kind]

    def rank(self, pos_scores, neg_scores) -> list[tuple[str, float, bool]]:
        """Best score per signature over a region: ``[(sig_id, score,
        vetoed)]`` best first; *vetoed* if a negative example of that
        signature matched at least as well."""
        best: dict[str, float] = {}
        for tpl, s in pos_scores:
            if len(s):
                best[tpl.sig_id] = max(best.get(tpl.sig_id, -1.0), float(s.max()))
        neg: dict[str, float] = {}
        for tpl, s in neg_scores:
            if len(s):
                neg[tpl.sig_id] = max(neg.get(tpl.sig_id, -1.0), float(s.max()))
        ranked = [(sid, sc, neg.get(sid, -1.0) >= sc) for sid, sc in best.items()]
        return sorted(ranked, key=lambda r: -r[1])

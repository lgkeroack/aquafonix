"""Saved events: retroactive captures ("the last 30 s"), selections and
detections, each stored as a clip plus a JSON sidecar with tags and notes.

Layout::

    <data>/events/20261023_140211_harbour-seal/
        clip.wav      (or clip.csv for non-uniform streams)
        event.json
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from aquafonix import audiofile

log = logging.getLogger(__name__)


@dataclass
class EventRecord:
    id: str
    created: str
    kind: str = "capture"  # capture | selection | detection
    title: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    verified: bool = False
    source: str = ""
    sample_rate: float | None = None
    channels: list[str] = field(default_factory=list)
    t0: float = 0.0  # stream time of the first sample
    t1: float = 0.0
    wall_start: str | None = None
    clip: str = ""
    detections: list[dict] = field(default_factory=list)
    folder: str = ""  # filled in when loaded

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:40] or "event"


class EventStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def add(self, x: np.ndarray, y: np.ndarray, names: list[str], sample_rate: float | None,
            kind: str = "capture", title: str = "", tags: list[str] | None = None,
            notes: str = "", source: str = "", wall_start: float | None = None,
            detections: list[dict] | None = None) -> EventRecord:
        if not len(x):
            raise ValueError("nothing to save: the history is empty for that time span")
        now = dt.datetime.now()
        base = f"{now:%Y%m%d_%H%M%S}_{_slug(title or (tags[0] if tags else kind))}"
        folder = self.root / base
        n = 1
        while folder.exists():
            n += 1
            folder = self.root / f"{base}-{n}"
        folder.mkdir(parents=True)
        if sample_rate:
            clip = "clip.wav"
            audiofile.write(str(folder / clip), y, sample_rate)
        else:
            clip = "clip.csv"
            with open(folder / clip, "w", newline="", encoding="utf-8") as f:
                f.write(",".join(["time_s", *names]) + "\n")
                np.savetxt(f, np.column_stack([x, y]), delimiter=",", fmt="%.10g")
        rec = EventRecord(
            id=folder.name, created=now.isoformat(timespec="seconds"), kind=kind, title=title,
            tags=list(tags or []), notes=notes, source=source,
            sample_rate=float(sample_rate) if sample_rate else None, channels=list(names),
            t0=float(x[0]), t1=float(x[-1]),
            wall_start=(dt.datetime.fromtimestamp(wall_start).isoformat(timespec="milliseconds")
                        if wall_start else None),
            clip=clip, detections=list(detections or []), folder=str(folder))
        self.save(rec)
        return rec

    def save(self, rec: EventRecord) -> None:
        folder = Path(rec.folder)
        data = asdict(rec)
        data.pop("folder")
        tmp = folder / "event.json.tmp"
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(folder / "event.json")

    def list(self) -> list[EventRecord]:
        out = []
        if not self.root.exists():
            return out
        for meta in self.root.glob("*/event.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                data["folder"] = str(meta.parent)
                known = EventRecord.__dataclass_fields__
                out.append(EventRecord(**{k: v for k, v in data.items() if k in known}))
            except Exception:
                log.warning("skipping unreadable event %s", meta, exc_info=True)
        return sorted(out, key=lambda r: r.created, reverse=True)

    def delete(self, rec: EventRecord) -> None:
        shutil.rmtree(rec.folder, ignore_errors=True)

    def clip_path(self, rec: EventRecord) -> Path:
        return Path(rec.folder) / rec.clip

    def load_clip(self, rec: EventRecord) -> tuple[np.ndarray, np.ndarray, float | None]:
        """``(x, y[n, ch], sample_rate)``."""
        path = self.clip_path(rec)
        if rec.sample_rate:
            y, rate = audiofile.read(str(path))
            x = rec.t0 + np.arange(len(y)) / rate
            return x, y, rate
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return data[:, 0], data[:, 1:], None

    def all_tags(self) -> list[str]:
        tags = {t for rec in self.list() for t in rec.tags}
        return sorted(tags, key=str.lower)

    def export_csv(self, path: str) -> int:
        recs = self.list()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "created", "wall_start", "kind", "title", "tags", "verified",
                        "duration_s", "source", "notes", "clip"])
            for r in recs:
                w.writerow([r.id, r.created, r.wall_start or "", r.kind, r.title,
                            "; ".join(r.tags), r.verified, f"{r.duration:.3f}", r.source,
                            r.notes, str(self.clip_path(r))])
        return len(recs)

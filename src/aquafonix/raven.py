"""Raven selection tables (tab-separated text, as used by Raven Pro/Lite and
many bioacoustics tools): export detections and saved events, and import
annotated selections to teach the signature catalogue.

Raven times are seconds from the start of the sound file (or file
sequence); ``Begin File`` and ``File Offset (s)`` locate a selection inside
one file of a sequence.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STANDARD = ("Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
            "Low Freq (Hz)", "High Freq (Hz)")
#: columns that commonly hold the label of a selection, best first
LABEL_COLUMNS = ("Annotation", "Species", "Label", "Call Type", "Class", "Sound Type", "Type",
                 "Tag", "Tags", "Category", "Code", "ID")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v).replace("\t", " ").replace("\n", " ")


def write_table(path: str | Path, rows: list[dict], extra: tuple[str, ...] = ()) -> None:
    """``rows`` hold the standard columns (``begin``, ``end``, ``low``,
    ``high``, ``channel``) plus any ``extra`` columns by name."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(STANDARD + tuple(extra)) + "\n")
        for i, r in enumerate(rows, 1):
            vals = [i, "Spectrogram 1", r.get("channel", 1), r["begin"], r["end"],
                    r.get("low", 0.0), r.get("high", 0.0)] + [r.get(c) for c in extra]
            f.write("\t".join(_fmt(v) for v in vals) + "\n")


def _wall(ts: float | None) -> str:
    return "" if ts is None else dt.datetime.fromtimestamp(ts).isoformat(
        sep=" ", timespec="milliseconds")


def detection_rows(detections, channel: int, wall_time=None, locate=None) -> list[dict]:
    """Rows for detections.  ``wall_time(t)`` gives the clock time of stream
    time ``t``; ``locate(t)`` gives ``(file name, offset s)`` for recordings."""
    rows = []
    for d in detections:
        r = {"begin": d.t0, "end": d.t1, "low": d.f_lo, "high": d.f_hi, "channel": channel,
             "Annotation": d.label or "", "Status": d.status, "Detection": d.text,
             "Score": round(d.score, 3) if d.score else None, "SNR (dB)": d.snr_db,
             "Bearing (deg)": d.bearing, "Begin Date Time": _wall(
                 wall_time(d.t0) if wall_time else None)}
        if locate is not None:
            got = locate(d.t0)
            if got:
                r["Begin File"], r["File Offset (s)"] = got
        rows.append(r)
    return rows


DETECTION_EXTRA = ("Annotation", "Status", "Detection", "Score", "SNR (dB)", "Bearing (deg)",
                   "Begin Date Time", "Begin File", "File Offset (s)")


def event_rows(records, store) -> list[dict]:
    """One row per saved event; each event's clip is its own sound file, so
    a selection covers the whole clip (``File Offset`` 0)."""
    rows = []
    for rec in records:
        hz = [(d.get("f_lo") or 0.0, d.get("f_hi") or 0.0) for d in rec.detections]
        lo = min((a for a, b in hz if b > a), default=0.0)
        hi = max((b for a, b in hz if b > a), default=(rec.sample_rate or 0) / 2)
        rows.append({"begin": rec.t0, "end": rec.t1, "low": lo, "high": hi, "channel": 1,
                     "Annotation": ", ".join(rec.tags) or rec.title, "Title": rec.title,
                     "Notes": rec.notes, "Kind": rec.kind, "Verified": rec.verified,
                     "Begin Date Time": rec.wall_start or "",
                     "Begin File": str(store.clip_path(rec)), "File Offset (s)": 0.0,
                     "Event ID": rec.id})
    return rows


EVENT_EXTRA = ("Annotation", "Title", "Notes", "Kind", "Verified", "Begin Date Time",
               "Begin File", "File Offset (s)", "Event ID")


# -- import ----------------------------------------------------------------------------
@dataclass
class Selection:
    number: str
    begin: float
    end: float
    low: float = 0.0
    high: float = 0.0
    channel: int = 1
    label: str = ""
    file: str = ""
    file_offset: float | None = None
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class SelectionTable:
    path: Path
    columns: list[str]
    selections: list[Selection]

    def label_columns(self) -> list[str]:
        """Non-standard columns, likely label columns first."""
        std = {c.casefold() for c in STANDARD} | {"begin file", "file offset (s)",
                                                  "begin path", "delta time (s)"}
        rest = [c for c in self.columns if c.casefold() not in std]
        rank = {c.casefold(): i for i, c in enumerate(LABEL_COLUMNS)}
        return sorted(rest, key=lambda c: rank.get(c.casefold(), len(rank)))

    def set_label_column(self, column: str | None) -> None:
        for s in self.selections:
            s.label = (s.fields.get(column, "") if column else "").strip()

    def labels(self) -> list[str]:
        return sorted({s.label for s in self.selections if s.label})


def _float(text: str, default: float = 0.0) -> float:
    try:
        return float(str(text).strip().replace(",", "."))
    except ValueError:
        return default


def read_table(path: str | Path) -> SelectionTable:
    path = Path(path)
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = [h.strip() for h in next(reader, [])]
        if not header:
            raise ValueError("the file is empty")
        rows = [r for r in reader if any(c.strip() for c in r)]
    lower = {h.casefold(): i for i, h in enumerate(header)}

    def col(*names):
        return next((lower[n.casefold()] for n in names if n.casefold() in lower), None)

    c_sel, c_view = col("Selection"), col("View")
    c_begin, c_end = col("Begin Time (s)", "Begin Time"), col("End Time (s)", "End Time")
    if c_begin is None or c_end is None:
        raise ValueError("not a Raven selection table (no 'Begin Time (s)' / 'End Time (s)' "
                         "columns)")
    c_lo, c_hi = col("Low Freq (Hz)", "Low Frequency (Hz)"), col("High Freq (Hz)",
                                                               "High Frequency (Hz)")
    c_ch, c_file = col("Channel"), col("Begin File", "Begin Path")
    c_off = col("File Offset (s)", "Beg File Samp (s)")
    seen: dict[str, Selection] = {}
    out = []
    for i, r in enumerate(rows):
        r = r + [""] * (len(header) - len(r))
        number = r[c_sel].strip() if c_sel is not None else str(i + 1)
        if number in seen:  # Raven writes a Waveform and a Spectrogram row per selection
            if c_view is not None and r[c_view].lower().startswith("spectrogram"):
                prev = seen[number]
                prev.low = _float(r[c_lo]) if c_lo is not None else prev.low
                prev.high = _float(r[c_hi]) if c_hi is not None else prev.high
            continue
        sel = Selection(number, _float(r[c_begin]), _float(r[c_end]),
                        _float(r[c_lo]) if c_lo is not None else 0.0,
                        _float(r[c_hi]) if c_hi is not None else 0.0,
                        int(_float(r[c_ch], 1)) if c_ch is not None else 1,
                        file=r[c_file].strip() if c_file is not None else "",
                        file_offset=_float(r[c_off]) if c_off is not None and r[c_off].strip()
                        else None,
                        fields={h: r[j].strip() for j, h in enumerate(header)})
        if sel.end > sel.begin:
            seen[number] = sel
            out.append(sel)
    table = SelectionTable(path, header, out)
    cols = table.label_columns()
    table.set_label_column(cols[0] if cols else None)
    return table


def resolve_audio(sel: Selection, table_dir: Path, default_audio: Path | None) -> tuple[
        Path | None, float]:
    """``(audio file, start offset s)`` of a selection: its ``Begin File``
    (looked for next to the table) at ``File Offset``, or the default audio
    file at ``Begin Time``."""
    if sel.file:
        p = Path(sel.file)
        for cand in (p, table_dir / p.name, (default_audio.parent / p.name)
                     if default_audio else None):
            if cand is not None and cand.is_file():
                return cand, sel.file_offset if sel.file_offset is not None else sel.begin
    if default_audio is not None and default_audio.is_file():
        return default_audio, sel.begin
    return None, 0.0


@dataclass
class ImportResult:
    added: int = 0
    skipped: int = 0
    labels: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def import_selections(table: SelectionTable, library, default_audio: Path | None = None,
                      max_s: float = 30.0, progress=None) -> ImportResult:
    """Add every labelled selection as an example of the signature of that
    name (created if new), so the matcher flags later occurrences."""
    from aquafonix import audiofile

    res = ImportResult()
    todo = [s for s in table.selections if s.label]
    res.skipped = len(table.selections) - len(todo)
    for i, sel in enumerate(todo):
        if progress is not None and progress(i, len(todo)) is False:
            break
        path, start = resolve_audio(sel, table.path.parent, default_audio)
        if path is None:
            res.skipped += 1
            res.problems.append(f"selection {sel.number}: sound file not found"
                                + (f" ({sel.file})" if sel.file else ""))
            continue
        try:
            data, rate = audiofile.read(str(path), start, min(sel.end - sel.begin, max_s))
        except Exception as exc:
            res.skipped += 1
            res.problems.append(f"selection {sel.number}: {exc}")
            continue
        ch = min(max(sel.channel - 1, 0), data.shape[1] - 1) if data.ndim == 2 else 0
        clip = data[:, ch] if data.ndim == 2 else data
        hi = sel.high if sel.high > sel.low else 0.0
        try:
            library.add_example(sel.label, clip, rate, sel.low if hi else 0.0, hi,
                                source=f"Raven: {table.path.name} #{sel.number}")
        except ValueError as exc:
            res.skipped += 1
            res.problems.append(f"selection {sel.number}: {exc}")
            continue
        res.added += 1
        res.labels[sel.label] = res.labels.get(sel.label, 0) + 1
    return res

"""Alert rules: which sounds or conditions should get the operator's
attention, and how (sound, pop-up, saving the moment as an event).

Detection rules fire on each matching detection; condition rules (level,
noise floor, propeller) fire when the condition starts, with a few dB of
hysteresis so a level hovering at the threshold doesn't chatter.  Every rule
has a cooldown, counted in stream time so replayed recordings behave the
same at any speed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np

KINDS = {
    "signature": "A sound matching my signatures",
    "spike": "Any spike above the noise floor",
    "level": "Broadband level above a threshold",
    "floor": "Noise floor above a threshold",
    "propeller": "Propeller modulation (DEMON)",
}
DETECTION_KINDS = ("signature", "spike")
HYSTERESIS_DB = 3.0


@dataclass
class AlertRule:
    name: str = "New alert"
    enabled: bool = True
    kind: str = "signature"
    labels: tuple[str, ...] = ()  # signature rules: these names only (empty = any)
    min_score: float = 0.0  # signature rules
    min_snr_db: float = 0.0  # detection rules
    level_db: float = 120.0  # level / floor rules
    sector: tuple[float, float] | None = None  # only sounds from these bearings
    cooldown_s: float = 10.0
    sound: bool = True
    popup: bool = True
    save_event: bool = False

    def describe(self) -> str:
        if self.kind == "signature":
            what = ("suspected " + ", ".join(self.labels)) if self.labels else "any signature"
            if self.min_score:
                what += f" (score ≥ {self.min_score:.2f})"
        elif self.kind == "spike":
            what = "any spike" + (f" ≥ {self.min_snr_db:g} dB SNR" if self.min_snr_db else "")
        elif self.kind == "level":
            what = f"level ≥ {self.level_db:g} dB"
        elif self.kind == "floor":
            what = f"noise floor ≥ {self.level_db:g} dB"
        else:
            what = "propeller modulation"
        if self.sector and self.kind in DETECTION_KINDS:
            what += f" from {self.sector[0]:g}–{self.sector[1]:g}°"
        return what

    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = list(self.labels)
        d["sector"] = list(self.sector) if self.sector else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AlertRule":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["labels"] = tuple(d.get("labels") or ())
        sector = d.get("sector")
        d["sector"] = (float(sector[0]), float(sector[1])) if sector else None
        if d.get("kind") not in KINDS:
            d["kind"] = "signature"
        return cls(**d)


@dataclass
class Alert:
    rule: AlertRule
    message: str
    stream_t: float  # stream time of the sound / condition
    detection_id: int | None = None
    wall_time: float = field(default_factory=time.time)  # when it was raised

    def to_dict(self) -> dict:
        return {"rule": self.rule.name, "kind": self.rule.kind, "message": self.message,
                "t": self.stream_t, "detection_id": self.detection_id,
                "raised": self.wall_time}


def detection_message(det) -> str:
    parts = [det.text]  # includes the match score
    if np.isfinite(det.snr_db):
        parts.append(f"SNR {det.snr_db:.0f} dB")
    if det.bearing is not None and np.isfinite(det.bearing):
        parts.append(f"bearing {det.bearing:.0f}°")
    return parts[0] + (" (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else "")


class AlertEngine:
    """Evaluates the rules against new detections and the noise report."""

    def __init__(self, rules: list[AlertRule] | None = None) -> None:
        self.rules: list[AlertRule] = list(rules or [])
        self._last: dict[int, float] = {}  # id(rule) -> stream time of last alert
        self._active: dict[int, bool] = {}  # condition rules currently "on"

    def set_rules(self, rules: list[AlertRule]) -> None:
        self.rules = list(rules)
        self.reset()

    def reset(self) -> None:
        self._last.clear()
        self._active.clear()

    def _cooled(self, rule: AlertRule, t: float) -> bool:
        last = self._last.get(id(rule))
        # a jump backwards in stream time (seek, restart) re-arms the rule
        return last is None or t < last or t - last >= rule.cooldown_s

    def check_detection(self, det) -> list[Alert]:
        out = []
        for rule in self.rules:
            if not rule.enabled or rule.kind not in DETECTION_KINDS:
                continue
            if rule.kind == "signature":
                if det.status not in ("suspected", "confirmed", "corrected") or not det.label:
                    continue
                if rule.labels and det.label.casefold() not in {
                        lb.casefold() for lb in rule.labels}:
                    continue
                if det.score < rule.min_score:
                    continue
            if np.isfinite(det.snr_db) and det.snr_db < rule.min_snr_db:
                continue
            if rule.sector is not None:
                b = det.bearing
                lo, hi = sorted(rule.sector)
                if b is None or not np.isfinite(b) or not lo <= b <= hi:
                    continue
            if not self._cooled(rule, det.t0):
                continue
            self._last[id(rule)] = det.t0
            out.append(Alert(rule, detection_message(det), det.t0, det.id))
        return out

    def check_report(self, report, stream_t: float) -> list[Alert]:
        out = []
        for rule in self.rules:
            if not rule.enabled or rule.kind in DETECTION_KINDS:
                continue
            if rule.kind == "propeller":
                d = getattr(report, "demon", None)
                value, on = None, d is not None and d.blades is not None
                off = not on
            else:
                value = report.level_db if rule.kind == "level" else report.floor_db
                if value is None or not np.isfinite(value):
                    continue
                on = value >= rule.level_db
                off = value < rule.level_db - HYSTERESIS_DB
            key = id(rule)
            if on and not self._active.get(key):
                self._active[key] = True
                if self._cooled(rule, stream_t):
                    self._last[key] = stream_t
                    if rule.kind == "propeller":
                        msg = f"Propeller: {report.demon.describe()}"
                    else:
                        what = "Level" if rule.kind == "level" else "Noise floor"
                        msg = f"{what} {value:.1f} dB ≥ {rule.level_db:g} dB"
                    out.append(Alert(rule, msg, stream_t))
            elif off:
                self._active[key] = False
        return out

"""Open data outputs: detections, alerts and levels as JSON messages, so
other software (a plotter, a logger, a vessel-tracking system, a script)
can follow the monitor live.

Each message is one JSON object with a ``type`` field (``detection``,
``alert`` or ``levels``).  Sinks:

* UDP: one message per datagram to ``host:port``;
* TCP: a server any number of clients can connect to; newline-delimited;
* webhook: an HTTP POST per message (``application/json``);
* file: appended to a JSON-lines file.

Every sink runs on its own thread with a bounded queue: a slow or absent
receiver drops messages rather than stalling the monitor.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import select
import socket
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aquafonix import __version__

log = logging.getLogger(__name__)
SCHEMA = 1


@dataclass(frozen=True)
class OutputSettings:
    udp: bool = False
    udp_host: str = "127.0.0.1"
    udp_port: int = 50555
    tcp: bool = False
    tcp_port: int = 50556
    webhook: bool = False
    webhook_url: str = ""
    file: bool = False
    file_path: str = ""
    detections: bool = True
    alerts: bool = True
    levels: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSettings":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    @property
    def any(self) -> bool:
        return self.udp or self.tcp or (self.webhook and bool(self.webhook_url)) or (
            self.file and bool(self.file_path))


def _clean(v):
    """JSON-safe values: NaN/inf -> null, numpy -> Python."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return round(v, 6) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(timespec="milliseconds")


def encode(msg: dict) -> bytes:
    body = {"schema": SCHEMA, "app": "aquafonix", "version": __version__, **msg}
    return json.dumps(_clean(body), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def detection_message(det, wall_time: float | None, event: str = "new",
                      source: str = "") -> dict:
    return {"type": "detection", "event": event, "id": det.id, "kind": det.kind,
            "status": det.status, "label": det.label, "text": det.text,
            "score": det.score, "t0": det.t0, "t1": det.t1, "duration": det.t1 - det.t0,
            "f_lo": det.f_lo, "f_hi": det.f_hi, "snr_db": det.snr_db,
            "bearing_deg": det.bearing, "bearing_quality": det.bearing_q,
            "time": iso(wall_time), "source": source}


def alert_message(alert, wall_time: float | None, source: str = "") -> dict:
    return {"type": "alert", **alert.to_dict(), "time": iso(wall_time),
            "raised": iso(alert.wall_time), "source": source}


def levels_message(t: float, wall_time: float | None, row, units: str, band, tol: dict,
                   demon=None, source: str = "") -> dict:
    spl, lpk, sel, floor, band_db = (float(v) for v in row)
    msg = {"type": "levels", "t": t, "time": iso(wall_time), "units": units,
           "spl_db": spl, "peak_db": lpk, "sel_db": sel, "floor_db": floor,
           "band_db": band_db if band is not None else None,
           "band_hz": list(band) if band is not None else None,
           "third_octaves_db": tol, "source": source}
    if demon is not None and demon.shaft_hz is not None:
        msg["demon"] = {"shaft_hz": demon.shaft_hz, "blade_hz": demon.blade_hz,
                        "blades": demon.blades}
    return msg


# -- sinks ---------------------------------------------------------------------------
class _Sink:
    name = "sink"

    def __init__(self) -> None:
        self.q: queue.Queue[bytes | None] = queue.Queue(maxsize=2000)
        self.sent = 0
        self.dropped = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._loop, name=f"out-{self.name}",
                                        daemon=True)
        self._stop = threading.Event()

    def start(self) -> "_Sink":
        self._thread.start()
        return self

    def put(self, data: bytes) -> None:
        try:
            self.q.put_nowait(data)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        try:
            self._open()
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            log.warning("%s output: %s", self.name, self.error)
            return
        try:
            while not self._stop.is_set():
                try:
                    data = self.q.get(timeout=self._poll_s())
                except queue.Empty:
                    self._idle()
                    continue
                if data is None:
                    break
                try:
                    self._send(data)
                    self.sent += 1
                    self.error = None
                except Exception as exc:
                    self.dropped += 1
                    self.error = str(exc) or type(exc).__name__
        finally:
            self._close()

    def _poll_s(self) -> float:
        return 0.5

    def _open(self) -> None: ...
    def _send(self, data: bytes) -> None: ...
    def _idle(self) -> None: ...
    def _close(self) -> None: ...

    def status(self) -> str:
        s = f"{self.name}: {self.sent} sent"
        if self.dropped:
            s += f", {self.dropped} dropped"
        if self.error:
            s += f" ({self.error})"
        return s


class UdpSink(_Sink):
    name = "UDP"

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.addr = (host, int(port))

    def _open(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def _send(self, data: bytes) -> None:
        self.sock.sendto(data, self.addr)

    def _close(self) -> None:
        self.sock.close()


class TcpServerSink(_Sink):
    name = "TCP"

    def __init__(self, port: int, host: str = "0.0.0.0") -> None:
        super().__init__()
        self.addr = (host, int(port))
        self.clients: list[socket.socket] = []
        self.port = int(port)

    def _open(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(self.addr)
        self.server.listen(8)
        self.server.setblocking(False)
        self.port = self.server.getsockname()[1]

    def _poll_s(self) -> float:
        return 0.2

    def _accept(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self.server], [], [], 0)
            except (OSError, ValueError):
                return
            if not ready:
                return
            try:
                conn, _addr = self.server.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(1.0)
            self.clients.append(conn)

    def _idle(self) -> None:
        self._accept()

    def _send(self, data: bytes) -> None:
        self._accept()
        line = data + b"\n"
        for conn in list(self.clients):
            try:
                conn.sendall(line)
            except OSError:  # client went away (or is hopelessly slow)
                self.clients.remove(conn)
                conn.close()

    def _close(self) -> None:
        for conn in self.clients:
            conn.close()
        self.clients = []
        self.server.close()

    def status(self) -> str:
        return super().status() + f", {len(self.clients)} client(s) on port {self.port}"


class WebhookSink(_Sink):
    name = "Webhook"

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def _send(self, data: bytes) -> None:
        req = urllib.request.Request(self.url, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": f"aquafonix/{__version__}"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read(256)


class FileSink(_Sink):
    name = "File"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = Path(path)

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "ab")

    def _send(self, data: bytes) -> None:
        self.fh.write(data + b"\n")
        self.fh.flush()

    def _close(self) -> None:
        self.fh.close()


class OutputHub:
    """Fans messages out to the configured sinks."""

    def __init__(self) -> None:
        self.settings = OutputSettings()
        self.sinks: list[_Sink] = []

    def configure(self, settings: OutputSettings) -> None:
        if settings == self.settings and self.sinks:
            return
        self.close()
        self.settings = settings
        s = settings
        if s.udp:
            self.sinks.append(UdpSink(s.udp_host, s.udp_port).start())
        if s.tcp:
            self.sinks.append(TcpServerSink(s.tcp_port).start())
        if s.webhook and s.webhook_url:
            self.sinks.append(WebhookSink(s.webhook_url).start())
        if s.file and s.file_path:
            self.sinks.append(FileSink(s.file_path).start())

    def wants(self, kind: str) -> bool:
        s = self.settings
        return bool(self.sinks) and {"detection": s.detections, "alert": s.alerts,
                                     "levels": s.levels}.get(kind, True)

    def publish(self, msg: dict) -> None:
        if not self.wants(msg.get("type", "")):
            return
        data = encode(msg)
        for sink in self.sinks:
            sink.put(data)

    def status(self) -> str:
        return " · ".join(s.status() for s in self.sinks) or "No outputs enabled"

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
        self.sinks = []

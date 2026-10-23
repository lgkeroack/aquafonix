"""Byte-stream sources: serial port, UDP, TCP client/server and file tail."""

from __future__ import annotations

import os
import socket

from aquafonix.parsing import ParserConfig, TextLineParser
from aquafonix.sources.base import ByteStreamSource


def list_serial_ports() -> list[tuple[str, str]]:
    """``[(device, description)]`` for the serial ports on this machine."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    ports = []
    for p in sorted(list_ports.comports(), key=lambda p: p.device):
        desc = p.description if p.description and p.description != "n/a" else ""
        ports.append((p.device, desc))
    return ports


class SerialSource(ByteStreamSource):
    def __init__(self, port: str, baudrate: int = 115200,
                 parser_config: ParserConfig | None = None) -> None:
        super().__init__(parser_config)
        self.port = port
        self.baudrate = int(baudrate)
        self._ser = None

    def _begin(self) -> None:
        import serial

        if not self.port:
            raise ValueError("choose a serial port")
        self._ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        self.status = f"Reading {self.port} @ {self.baudrate} baud"

    def _step(self) -> None:
        data = self._ser.read(max(1, self._ser.in_waiting))
        if data:
            self.handle_bytes(data)

    def _end(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


class UdpSource(ByteStreamSource):
    """Listens for datagrams; in text mode each datagram is one or more lines."""

    def __init__(self, port: int, host: str = "0.0.0.0",
                 parser_config: ParserConfig | None = None) -> None:
        super().__init__(parser_config)
        self.host = host or "0.0.0.0"
        self.port = int(port)
        self._sock: socket.socket | None = None

    def _begin(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self.status = f"Listening on UDP {self.host}:{self.port}"

    def _step(self) -> None:
        try:
            data, _addr = self._sock.recvfrom(65536)
        except (socket.timeout, BlockingIOError):
            return
        except OSError:
            if self._stop.is_set():
                return
            raise
        if isinstance(self.parser, TextLineParser) and not data.endswith(b"\n"):
            data += b"\n"
        self.handle_bytes(data)

    def _end(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class TcpClientSource(ByteStreamSource):
    """Connects to a TCP server and reconnects automatically if dropped."""

    def __init__(self, host: str, port: int, parser_config: ParserConfig | None = None) -> None:
        super().__init__(parser_config)
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self._sock: socket.socket | None = None

    def _begin(self) -> None:
        self.status = f"Connecting to {self.host}:{self.port}…"

    def _step(self) -> None:
        if self._sock is None:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=2)
                self._sock.settimeout(0.1)
                self.reset_parser()
                self.status = f"Connected to {self.host}:{self.port}"
            except OSError as exc:
                self.status = f"Connecting to {self.host}:{self.port}… ({exc})"
                self._stop.wait(1.0)
            return
        try:
            data = self._sock.recv(65536)
        except socket.timeout:
            return
        except OSError as exc:
            data, why = b"", str(exc)
        else:
            why = "closed by server"
        if data:
            self.handle_bytes(data)
            return
        self._sock.close()
        self._sock = None
        self.status = f"Disconnected ({why}); retrying…"
        self._stop.wait(0.5)

    def _end(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class TcpServerSource(ByteStreamSource):
    """Listens on a TCP port and reads from one client at a time."""

    def __init__(self, port: int, host: str = "0.0.0.0",
                 parser_config: ParserConfig | None = None) -> None:
        super().__init__(parser_config)
        self.host = host or "0.0.0.0"
        self.port = int(port)
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None

    def _begin(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._server.settimeout(0.1)
        self.port = self._server.getsockname()[1]
        self.status = f"Waiting for a client on TCP port {self.port}"

    def _step(self) -> None:
        if self._client is None:
            try:
                self._client, addr = self._server.accept()
            except socket.timeout:
                return
            self._client.settimeout(0.1)
            self.reset_parser()
            self.status = f"Client {addr[0]}:{addr[1]} connected"
            return
        try:
            data = self._client.recv(65536)
        except socket.timeout:
            return
        except OSError:
            data = b""
        if data:
            self.handle_bytes(data)
            return
        self._client.close()
        self._client = None
        self.status = f"Client left; waiting on TCP port {self.port}"

    def _end(self) -> None:
        for s in (self._client, self._server):
            if s is not None:
                s.close()
        self._client = self._server = None


class FileTailSource(ByteStreamSource):
    """Follows a growing file (like ``tail -f``); copes with truncation."""

    def __init__(self, path: str, from_start: bool = False,
                 parser_config: ParserConfig | None = None) -> None:
        super().__init__(parser_config)
        self.path = path
        self.from_start = from_start
        self._f = None

    def _begin(self) -> None:
        if not self.path:
            raise ValueError("choose a file to follow")
        self._f = open(self.path, "rb")
        if not self.from_start:
            self._f.seek(0, os.SEEK_END)
        self.status = f"Following {os.path.basename(self.path)}"

    def _step(self) -> None:
        data = self._f.read(1 << 16)
        if data:
            self.handle_bytes(data)
            return
        try:
            if os.path.getsize(self.path) < self._f.tell():  # truncated / rotated
                self._f.close()
                self._f = open(self.path, "rb")
                self.reset_parser()
        except OSError:
            pass
        self._stop.wait(0.05)

    def _end(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

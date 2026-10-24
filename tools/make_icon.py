"""Render the application icon (PNG for the window, ICO for the exe).

    python tools/make_icon.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QColor, QGuiApplication, QImage, QLinearGradient, QPainter,  # noqa: E402
                           QPainterPath, QPen)

ROOT = Path(__file__).resolve().parents[1]


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256
    bg = QLinearGradient(0, 0, 0, size)
    bg.setColorAt(0, QColor(18, 52, 86))
    bg.setColorAt(1, QColor(6, 18, 34))
    p.setBrush(bg)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 48 * s, 48 * s)
    # spectrogram-like bars rising from the bottom
    for i in range(9):
        h = (30 + 60 * abs(math.sin(i * 0.9 + 0.4))) * s
        x = (40 + i * 20) * s
        p.setBrush(QColor(76, 201, 240, 70))
        p.drawRoundedRect(QRectF(x, 214 * s - h, 12 * s, h), 3 * s, 3 * s)
    # the waveform: a call burst
    path = QPainterPath()
    for i in range(201):
        u = i / 200
        env = math.exp(-((u - 0.5) / 0.2) ** 2)
        y = 118 - 70 * env * math.sin(2 * math.pi * (3 * u + 9 * u * u))
        pt = QPointF((28 + 200 * u) * s, y * s)
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    pen = QPen(QColor(76, 201, 240), max(2.0, 11 * s), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.setPen(QPen(QColor(255, 255, 255, 230), max(1.0, 4 * s), Qt.SolidLine, Qt.RoundCap,
                  Qt.RoundJoin))
    p.drawPath(path)
    p.end()
    return img


def main() -> int:
    QGuiApplication(sys.argv[:1])
    png = ROOT / "src" / "aquafonix" / "assets" / "icon.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    render(256).save(str(png))
    ico = ROOT / "packaging" / "aquafonix.ico"
    ico.parent.mkdir(parents=True, exist_ok=True)
    render(256).save(str(ico), "ICO")
    print("wrote", png, "and", ico)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared fixtures.  GUI tests need a display: on Linux CI run them under
``xvfb-run``; without one, Qt falls back to the offscreen platform and the
tests that capture the real screen are skipped."""

from __future__ import annotations

import os
import sys
import time
import traceback

import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
        and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _screen_capture_works() -> bool:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    try:
        from aquafonix.screen.capture import MssGrabber, Region

        g = MssGrabber()
        try:
            g.grab(Region(0, 0, 4, 4))
        finally:
            g.close()
        return True
    except Exception:
        return False


needs_screen = pytest.mark.skipif(not _screen_capture_works(),
                                  reason="no real display to capture")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    from aquafonix.app import apply_theme

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


class Gui:
    """Helpers for driving the UI: pumping events, waiting for conditions,
    and catching exceptions raised inside Qt slots."""

    def __init__(self, app) -> None:
        self.app = app
        self.errors: list[str] = []
        self.infos: list[str] = []

    def pump(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents()
            time.sleep(0.01)

    def wait_until(self, cond, timeout: float, step: float = 0.05) -> bool:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(step)
        self.app.processEvents()
        return bool(cond())


@pytest.fixture
def gui(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    g = Gui(qapp)

    def hook(t, v, tb):
        g.errors.append("".join(traceback.format_exception(t, v, tb)))

    monkeypatch.setattr(sys, "excepthook", hook)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(
        lambda *a, **k: g.infos.append(str(a[2] if len(a) > 2 else a)) or 0))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: g.errors.append(f"warning dialog: {a[2] if len(a) > 2 else a}") or 0))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    yield g
    for w in qapp.topLevelWidgets():
        if w.isVisible() and type(w).__name__ not in ("MainWindow",):
            w.close()
    qapp.processEvents()
    assert not g.errors, "\n".join(g.errors[:3])


@pytest.fixture
def window(gui, tmp_path):
    """A main window with its own settings file and data folder."""
    from PySide6.QtCore import QSettings

    from aquafonix.ui.main_window import MainWindow

    win = MainWindow(QSettings(str(tmp_path / "Aquafonix.ini"), QSettings.IniFormat),
                     tmp_path / "data")
    win.resize(1400, 900)
    win.show()
    gui.pump(0.3)
    yield win
    win.stop()
    win.persist = False
    win.close()
    gui.pump(0.1)

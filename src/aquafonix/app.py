"""Application entry point: command line, settings location, theme, logging."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

from aquafonix import APP_NAME, __version__

log = logging.getLogger("aquafonix")

EPILOG = """\
examples:
  Aquafonix.exe --source serial --set serial.port=COM3 --set serial.baud=115200 --start
  Aquafonix.exe --source network --set network.protocol=udp --set network.port=9000 --start
  Aquafonix.exe --source screen --region 100,200,640,320 --start
  Aquafonix.exe --source audio --start

Any control can be set with --set NAME=VALUE; names match the keys under
[controls] in Aquafonix.ini.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=APP_NAME, description=(
        "Real-time monitor for audio, serial/network/file data streams and graphs on screen."),
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["audio", "serial", "network", "file", "screen",
                                        "generator"], help="source to select at startup")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="set a control, e.g. serial.port=COM3 (repeatable)")
    p.add_argument("--region", metavar="X,Y,W,H",
                   help="screen region to watch, in physical pixels")
    p.add_argument("--start", action="store_true", help="start streaming immediately")
    p.add_argument("--settings", metavar="PATH", help="settings file to use")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return p.parse_args(argv)


def _writable_dir(path: Path) -> bool:
    try:
        probe = path / f".{APP_NAME}.write-test"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def settings_path() -> Path:
    """Portable: ``Aquafonix.ini`` next to the exe when that folder is
    writable, otherwise in the user's config folder."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if _writable_dir(exe_dir):
            return exe_dir / f"{APP_NAME}.ini"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    folder = base / APP_NAME
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{APP_NAME}.ini"


def setup_logging(log_file: Path) -> None:
    # A windowed exe has no console: give stray writes somewhere to go.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        handlers.append(logging.handlers.RotatingFileHandler(
            log_file, maxBytes=512_000, backupCount=1, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _install_excepthook() -> None:
    def hook(exc_type, exc, tb):
        log.error("unhandled exception", exc_info=(exc_type, exc, tb))
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            text = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{text}\n\n"
                                 "Details are in the log file next to the settings file.")
    sys.excepthook = hook


def apply_theme(app) -> None:
    import pyqtgraph as pg
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    window, base, text = QColor(36, 40, 46), QColor(26, 29, 34), QColor(222, 226, 232)
    accent = QColor(76, 201, 240)
    for role, color in ((QPalette.Window, window), (QPalette.WindowText, text),
                        (QPalette.Base, base), (QPalette.AlternateBase, window),
                        (QPalette.ToolTipBase, base), (QPalette.ToolTipText, text),
                        (QPalette.Text, text), (QPalette.Button, QColor(48, 53, 61)),
                        (QPalette.ButtonText, text), (QPalette.BrightText, QColor(255, 80, 80)),
                        (QPalette.Highlight, accent), (QPalette.HighlightedText, QColor(0, 0, 0)),
                        (QPalette.Link, accent), (QPalette.PlaceholderText, QColor(140, 146, 156))):
        pal.setColor(role, color)
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, QColor(110, 115, 122))
    app.setPalette(pal)
    pg.setConfigOptions(background=(18, 20, 24), foreground=(200, 205, 212), antialias=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ini = Path(args.settings) if args.settings else settings_path()
    setup_logging(ini.with_suffix(".log"))
    _install_excepthook()
    log.info("%s %s starting; settings in %s", APP_NAME, __version__, ini)

    if sys.platform == "win32":  # own taskbar group/icon instead of python's
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_NAME)
        except Exception:
            pass

    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    icon = Path(__file__).parent / "assets" / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    apply_theme(app)

    from aquafonix.screen.capture import Region
    from aquafonix.ui.main_window import MainWindow

    win = MainWindow(QSettings(str(ini), QSettings.IniFormat), ini.parent / f"{APP_NAME}Data")
    try:
        if args.source:
            win.select_source(args.source)
        for item in args.set:
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"--set expects NAME=VALUE, got {item!r}")
            win.set_control(name.strip(), value.strip())
        if args.region:
            win.screen_panel.set_region(Region.parse(args.region))
    except (KeyError, ValueError) as exc:
        QMessageBox.warning(win, APP_NAME, f"Command line: {exc}")
    win.show()
    if args.start:
        QTimer.singleShot(0, win.start)
    return app.exec()

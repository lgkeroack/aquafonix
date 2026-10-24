"""The real main window driven through its main flows (needs a display or
Qt's offscreen platform; screen capture tests need a real display)."""

import json
import socket
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QFileDialog, QInputDialog, QWidget

from aquafonix.alerts import AlertRule
from aquafonix.outputs import OutputSettings
from aquafonix.screen.capture import Region
from conftest import needs_screen
from signals import calls_recording, pair_from_bearing, propeller_noise


def npts(item):
    x = item.getData()[0]
    return 0 if x is None else len(x)


@pytest.fixture
def no_dialogs(monkeypatch, tmp_path):
    """Answer file dialogs and naming prompts without showing them."""
    from aquafonix.ui import dialogs
    import aquafonix.ui.main_window as mw

    answers = {"save": str(tmp_path / "out.csv"), "open": [], "label": "upsweep call"}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (answers["save"], "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (answers["open"].pop(0), "")))
    monkeypatch.setattr(dialogs.IdentifyDialog, "exec",
                        lambda self: (self.name.setCurrentText(answers["label"]), True)[1])
    monkeypatch.setattr(mw, "ask_label", lambda *a, **k: "boat noise")
    monkeypatch.setattr(mw, "beep", lambda: answers.setdefault("beeps", []).append(1))
    return answers


def write_wav(path: Path, x: np.ndarray, fs: int) -> Path:
    sf.write(path, x.astype(np.float32), fs, subtype="FLOAT")
    return path


def replay(win, gui, path, speed="4", until=None, timeout=20.0):
    win.select_source("audiofile")
    win.file_panel.set_path(str(path))
    win.set_control("audiofile.speed", speed)
    assert win.start()
    until = until or (win.source.duration - 0.5)
    gui.wait_until(lambda: win.source.position >= until or win.source.finished, timeout)
    gui.pump(1.0)


def test_live_generator_draws_everything(window, gui):
    win = window
    win.select_source("generator")
    win.set_control("gen.rate", "8000")
    win.set_control("gen.freq", "440")
    assert win.start()
    assert gui.wait_until(lambda: npts(next(iter(win.time_plot.curves.values()))) > 100, 5)
    assert gui.wait_until(lambda: win.spectro.image.image is not None, 5)
    assert gui.wait_until(lambda: npts(win.levels_plot.fixed[0]) > 0, 5)
    assert "samples/s" in win.status_label.text()
    for w in (win.spectrum_on, win.tol_on, win.spec_log, win.spec_snr, win.peak_on):
        w.setChecked(True)
    assert gui.wait_until(lambda: npts(win.spec_plot.current) > 10 and len(win.tol_plot.labels) > 10,
                          5)
    win.zoom_x(0.5)
    win.zoom_y(0.5)
    win.fit_view()
    win.act_pause.setChecked(True)
    gui.pump(0.2)
    win.time_plot.setXRange(0.5, 1.0)
    gui.pump(0.3)
    vx = win._view[0]
    assert 0.45 <= vx[0] <= 0.5 and 0.95 <= vx[-1] <= 1.05  # explores the history
    win.act_pause.setChecked(False)
    win.capture_event()
    gui.pump(0.3)
    assert len(win.store.list()) == 1


def test_replay_identify_and_review(window, gui, no_dialogs, tmp_path):
    win = window
    fs = 16000
    path = write_wav(tmp_path / "20240301_060000.wav", calls_recording(fs), fs)
    win.select_source("audiofile")
    win.file_panel.set_path(str(path))
    win.set_control("audiofile.speed", "4")
    assert win.start()
    assert win.transport.isVisible() and win.rec_view.isVisible()
    assert gui.wait_until(lambda: win.analyzer.processed_index > 1.6 * fs, 10)
    win.source.set_paused(True)
    gui.pump(0.3)
    win._spectro_select(0.95, 1.45, win.spectro.fy(1300), win.spectro.fy(3400), True)
    win.identify_selection()
    gui.pump(0.2)
    assert [s.label for s in win.library.signatures] == ["upsweep call"]
    win.analyse_selection()
    gui.pump(0.5)
    detail = [w for w in gui.app.topLevelWidgets() if type(w).__name__ == "SelectionDetail"][-1]
    assert len(detail.rows) > 8
    detail.export_csv()
    assert all((tmp_path / n).exists() for n in ("out.csv", "out_psd.csv",
                                                  "out_third_octaves.csv"))
    detail.close()
    win.act_pause.setChecked(False)
    win.source.set_paused(False)
    gui.wait_until(lambda: win.source.position >= 27, 15)
    gui.pump(1.0)
    dets = win.analyzer.all_detections()
    sus = [d for d in dets if d.status == "suspected"]
    assert len(sus) >= 3 and all(d.wall_time for d in sus)
    assert win.det_panel.table.rowCount() == len(dets)
    win.confirm_detection(sus[0].id)
    assert len(win.library.find("upsweep call").positives) == 2
    win.reject_detection(sus[1].id)
    assert len(win.library.find("upsweep call").negatives) == 1
    win.correct_detection(sus[2].id)
    assert win.library.find("boat noise") is not None and sus[2].label == "boat noise"
    unid = [d for d in dets if d.status == "unidentified"]
    assert any(abs(d.t0 - 10) < 0.3 for d in unid)  # the click
    win._seek_to(5.0)
    gui.pump(0.5)
    assert 4.5 < win.source.position < 8
    win.stop()
    assert len(win.store.list()) >= 3  # reviewed detections were kept as events


def test_two_hydrophones_bearing_and_demon(window, gui, no_dialogs, tmp_path):
    win = window
    fs = 48000
    rng = np.random.default_rng(3)
    src = propeller_noise(fs, 24, 3.0, 4, rng)
    for c in (4.0, 9.0, 14.0, 19.0):
        i = int(c * fs)
        src[i:i + 200] += 0.8 * np.hanning(200) * np.sin(2 * np.pi * 5000 * np.arange(200) / fs)
    a, b = pair_from_bearing(src, fs, 60.0)
    path = write_wav(tmp_path / "20240301_060000.wav", np.column_stack([a, b]), fs)
    for name, value in (("ac.array", "true"), ("ac.cha", "1"), ("ac.chb", "2"),
                        ("ac.spacing", "1.0"), ("display.btr", "true")):
        win.set_control(name, value)
    replay(win, gui, path, "2", until=21)
    an = win.analyzer
    assert win.bearing_view.isVisible() and an.btr.count > 20
    _t, rows = an.btr.latest(an.btr.count)
    assert abs(an.btr_bearings[np.argmax(np.median(rows, axis=0))] - 60) < 3
    clicks = [d for d in an.all_detections() if any(abs(d.t0 - c) < 0.3 for c in (4, 9, 14, 19))]
    assert len(clicks) >= 3 and all(abs(d.bearing - 60) < 3 for d in clicks)
    an._demon_t = 0.0
    an._make_report()
    d = an.report.demon
    assert d is not None and d.blades == 4 and abs(d.shaft_hz - 3.0) < 0.1
    assert any("propeller" in s[0].lower() for s in an.report.sources)
    win.acoustics.show_report(an.report, an.alarm)
    assert "DEMON" in win.acoustics.report.text()
    win.act_pause.setChecked(True)
    win.source.set_paused(True)
    win._spectro_select(5.0, 17.0, win.spectro.fy(1000), win.spectro.fy(10000), True)
    win.analyse_selection()
    gui.pump(0.5)
    detail = [w for w in gui.app.topLevelWidgets() if type(w).__name__ == "SelectionDetail"][-1]
    rows = dict((r[0], r[1]) for r in detail.rows)
    assert abs(float(rows["Bearing (from A→B axis)"]) - 60) < 3
    detail.tabs.setCurrentIndex(1)
    gui.pump(0.5)
    assert detail.demon_result is not None and detail.demon_result.blades == 4
    detail.export_csv()
    assert (tmp_path / "out_demon.csv").exists()
    detail.close()


def test_alerts_outputs_and_raven(window, gui, no_dialogs, tmp_path, monkeypatch):
    from aquafonix.ui import alerts_ui

    win = window
    fs = 16000
    x = calls_recording(fs)
    path = write_wav(tmp_path / "20240301_060000.wav", x, fs)
    win.library.add_example("upsweep call", x[fs:fs + int(0.4 * fs)], fs, 1300, 3400)
    editor = alerts_ui.RuleEditor(win, AlertRule("Upsweeps", labels=("upsweep call",),
                                                 cooldown_s=2.0, save_event=True), [])
    rules = [editor.rule(), AlertRule("Clicks", kind="spike", min_snr_db=25, cooldown_s=0,
                                      popup=False)]
    dlg = alerts_ui.AlertsDialog(win, rules, ["upsweep call"])
    assert dlg.table.rowCount() == 2
    win.alert_engine.set_rules(dlg.rules)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.05)
    outs = OutputSettings(udp=True, udp_port=udp.getsockname()[1], file=True,
                          file_path=str(tmp_path / "stream.jsonl"))
    assert alerts_ui.OutputsDialog(win, outs).settings() == outs
    win.outputs.configure(outs)
    toast = []
    win.toast.notify = lambda title, text, seconds=6.0: toast.append(text)
    replay(win, gui, path, "4", until=29.5)

    def drain():
        msgs = []
        while True:
            try:
                msgs.append(json.loads(udp.recv(65536)))
            except socket.timeout:
                return msgs

    msgs = drain()
    kinds = [m["type"] for m in msgs]
    fired = [a.rule.name for a in win.alerts_panel.alerts]
    assert fired.count("Upsweeps") >= 3 and toast and no_dialogs.get("beeps")
    assert kinds.count("detection") >= 5 and kinds.count("alert") >= 3
    assert kinds.count("levels") >= 20
    lv = next(m for m in msgs if m["type"] == "levels")
    assert lv["time"].startswith("2024-03-01") and len(lv["third_octaves_db"]) > 10
    assert len((tmp_path / "stream.jsonl").read_text().splitlines()) == len(msgs)
    saved = [r for r in win.store.list() if r.kind == "alert"]
    assert len(saved) >= 3 and "Upsweeps" in saved[0].tags and saved[0].duration > 8
    sus = [d for d in win.analyzer.all_detections() if d.status == "suspected"]
    win.confirm_detection(sus[0].id)
    gui.pump(0.3)
    assert any(m["event"] == "update" and m["status"] == "confirmed" for m in drain()
               if m["type"] == "detection")
    no_dialogs["save"] = str(tmp_path / "dets.txt")
    win.export_detections_raven()
    lines = (tmp_path / "dets.txt").read_text().splitlines()
    assert len(lines) == 1 + len(win.analyzer.all_detections())
    assert "20240301_060000.wav" in lines[1]
    win.stop()
    before = len(win.library.find("upsweep call").positives)
    no_dialogs["open"] = [str(tmp_path / "dets.txt")]
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("Annotation",
                                                                               True)))
    win.import_raven()
    assert len(win.library.find("upsweep call").positives) > before


def test_dialogs_open(window):
    from aquafonix.ui.batch_dialog import BatchDialog
    from aquafonix.ui.dialogs import SignatureManager

    window.library.add_example("x", np.random.default_rng(0).standard_normal(4000), 8000)
    assert SignatureManager(window, window.library).table.rowCount() == 1
    assert BatchDialog(window, window.acoustics).integration.count() > 3


def test_settings_restored(gui, tmp_path):
    from aquafonix.ui.main_window import MainWindow

    ini = str(tmp_path / "a.ini")
    win = MainWindow(QSettings(ini, QSettings.IniFormat), tmp_path / "data")
    win.select_source("screen")
    win.spec_log.setChecked(True)
    win.close()
    win2 = MainWindow(QSettings(ini, QSettings.IniFormat), tmp_path / "data")
    try:
        assert win2.source_combo.currentData() == "screen" and win2.spec_log.isChecked()
    finally:
        win2.persist = False
        win2.close()


class _Chart(QWidget):
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setPen(QPen(QColor(200, 200, 200), 1))
        for yy in range(0, 200, 40):
            p.drawLine(0, yy, 400, yy)
        for col, ph in ((QColor(220, 40, 40), 0.0), (QColor(30, 90, 220), 1.5)):
            p.setPen(QPen(col, 2))
            pts = [(xx, 100 + 60 * np.sin(xx / 30 + ph)) for xx in range(0, 400, 2)]
            for (a, b), (c, d) in zip(pts, pts[1:]):
                p.drawLine(int(a), int(b), int(c), int(d))


@needs_screen
def test_screen_region_with_two_traces(window, gui):
    win = window
    win.resize(850, 900)
    win.move(0, 0)
    chart = _Chart()
    chart.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    chart.setGeometry(1000, 40, 400, 200)
    chart.show()
    gui.pump(0.8)
    try:
        win.select_source("screen")
        win.screen_panel.set_region(Region(1000, 40, 400, 200))
        assert win.start()
        gui.pump(1.0)
        win._autodetect_traces(2)
        assert gui.wait_until(lambda: win._last_frame is not None
                              and len(win._last_frame.result.traces) == 2, 5)
        assert [d.name for d in win.screen_panel.trace_defs()] == ["trace 1", "trace 2"]
        assert min(t.coverage for t in win._last_frame.result.traces) > 0.9
    finally:
        chart.close()


@needs_screen
def test_region_selector_and_screen_spectrogram(window, gui, no_dialogs):
    from PySide6.QtTest import QTest

    from aquafonix.screen.specread import ColourScale

    win = window
    win.resize(850, 900)
    win.move(0, 0)
    win.select_source("screen")
    got = []
    win.selector.finished.connect(got.append)
    win.selector.start()
    gui.pump(0.3)
    ov = win.selector.overlays[0]
    QTest.mousePress(ov, Qt.LeftButton, Qt.NoModifier, QPoint(1000, 100))
    QTest.mouseMove(ov, QPoint(1200, 200))
    QTest.mouseRelease(ov, Qt.LeftButton, Qt.NoModifier, QPoint(1200, 200))
    gui.pump(0.3)
    assert got and got[0] == Region(1000, 100, 200, 100)

    lut, _ = ColourScale("viridis", low_db=40, high_db=120).lut()
    rows, view, total = 160, 360, 4000
    truth = 50 + np.random.default_rng(0).normal(0, 2, (rows, total))
    for c0 in range(150, total, 400):
        for k in range(30):
            r = int(rows - 1 - (40 + k * 2))
            truth[r - 2:r + 3, c0 + k] = 105

    def render(lv):
        idx = np.clip(((lv - 40) / 80 * (len(lut) - 1)).round().astype(int), 0, len(lut) - 1)
        return np.ascontiguousarray(lut[idx].round().astype(np.uint8))

    bar = render(np.repeat(np.linspace(120, 40, rows)[:, None], 16, axis=1))
    state = {"pos": view}

    class Spec(QWidget):
        def paintEvent(self, _e):
            img = np.ascontiguousarray(np.concatenate(
                [render(truth[:, state["pos"] - view:state["pos"]]),
                 np.zeros((rows, 10, 3), np.uint8), bar], axis=1))
            q = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                       QImage.Format_RGB888)
            QPainter(self).drawImage(0, 0, q)

    spec = Spec()
    spec.setWindowFlags(Qt.FramelessWindowHint)
    spec.setGeometry(900, 300, view + 26, rows)
    spec.show()
    timer = QTimer()
    timer.timeout.connect(lambda: (state.update(pos=min(total, state["pos"] + 3)), spec.update()))
    timer.start(40)
    try:
        gui.pump(0.5)
        sp = win.screen_panel
        sp.set_region(Region(900, 300, view + 26, rows))
        sp.output.setCurrentIndex(sp.output.findData("spectrogram"))
        for key, px, val in (("ya", rows, 0.0), ("yb", 0, 8000.0)):
            sp.set_cal_px(key, px, emit_line=True)
            sp.cal_val[key].setValue(val)
        assert win.start()
        gui.pump(1.0)
        sp.pick_bar.setChecked(True)
        win._bar_picked(view + 12, 0, view + 24, rows)
        gui.pump(0.2)
        assert sp.scale.currentData() == "bar" and len(sp._bar_colors()) > 50
        sp.set_region(Region(900, 300, view, rows), reset_calibration=False)
        an = win.analyzer
        assert gui.wait_until(lambda: an.spectro is not None and an.spectro.count > 200, 10)
        assert an.external and win.spectro.image.image is not None

        def first_call():
            return next((d for d in an.all_detections() if d.status == "unidentified"
                         and d.f_lo < 3000 < d.f_hi and d.t1 - d.t0 < 1.0 and d.t0 > 0), None)

        assert gui.wait_until(lambda: first_call() is not None, 10)
        first = first_call()
        win.act_pause.setChecked(True)
        win._spectro_select(first.t0, first.t1, win.spectro.fy(first.f_lo),
                            win.spectro.fy(first.f_hi), True)
        no_dialogs["label"] = "screen whistle"
        win.identify_selection()
        sig = win.library.find("screen whistle")
        assert sig is not None and sig.positives[0].data == "patch"
        win.act_pause.setChecked(False)
        assert gui.wait_until(lambda: any(d.status == "suspected"
                                          for d in an.all_detections()), 15)
        win.capture_event()
        gui.pump(0.3)
        rec = win.store.list()[0]
        assert (Path(rec.folder) / "spectrogram.npz").exists()
    finally:
        timer.stop()
        spec.close()

"""Alert rules, JSON outputs (UDP, TCP, webhook, file) and Raven tables."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import soundfile as sf

from aquafonix import outputs, raven
from aquafonix.acoustics.analyzer import Detection
from aquafonix.acoustics.demon import DemonResult
from aquafonix.acoustics.noise import NoiseReport
from aquafonix.acoustics.signatures import SignatureLibrary
from aquafonix.alerts import AlertEngine, AlertRule
from signals import add_chirp


def det(i, t0, status="suspected", label="seal", score=0.8, snr=15.0, bearing=None):
    d = Detection(i, t0, t0 + 0.5, 500, 2000, snr, "match" if label else "event", None, label,
                  score)
    d.status = status if label else "unidentified"
    d.bearing = bearing
    return d


def names(alerts):
    return [a.rule.name for a in alerts]


@pytest.fixture
def engine():
    return AlertEngine([
        AlertRule("seals", kind="signature", labels=("Seal",), cooldown_s=5),
        AlertRule("loud", kind="spike", min_snr_db=20, cooldown_s=0),
        AlertRule("sector", kind="spike", sector=(40, 80), cooldown_s=0),
        AlertRule("level", kind="level", level_db=100, cooldown_s=0),
        AlertRule("prop", kind="propeller", cooldown_s=0),
        AlertRule("off", enabled=False, kind="spike")])


def test_signature_rule_and_cooldown(engine):
    assert names(engine.check_detection(det(1, 10))) == ["seals"]  # label case ignored
    assert names(engine.check_detection(det(2, 12))) == []  # cooling down
    assert names(engine.check_detection(det(3, 16))) == ["seals"]
    assert names(engine.check_detection(det(4, 30, label="dolphin"))) == []
    assert names(engine.check_detection(det(5, 40, status="rejected"))) == []
    assert names(engine.check_detection(det(6, 1))) == ["seals"]  # seek back re-arms


def test_spike_and_sector_rules(engine):
    assert names(engine.check_detection(det(1, 50, label=None, snr=25))) == ["loud"]
    assert names(engine.check_detection(det(2, 60, label=None, snr=12, bearing=60))) == ["sector"]
    assert names(engine.check_detection(det(3, 61, label=None, snr=12, bearing=120))) == []
    assert names(engine.check_detection(det(4, 62, label=None, snr=12))) == []  # no bearing


def test_condition_rules_fire_on_rising_edge_with_hysteresis(engine):
    seq = [engine.check_report(NoiseReport(level_db=v), t)
           for t, v in enumerate((105, 99, 96, 101))]
    assert [names(a) for a in seq] == [["level"], [], [], ["level"]]
    rep = NoiseReport()
    rep.demon = DemonResult(np.zeros(1), np.zeros(1), (1, 2), [], 3.0, 12.0, 4)
    alerts = engine.check_report(rep, 200)
    assert names(alerts) == ["prop"] and "4 blades" in alerts[0].message


def test_rules_round_trip_json(engine):
    data = json.loads(json.dumps([r.to_dict() for r in engine.rules]))
    assert [AlertRule.from_dict(d) for d in data] == engine.rules


def test_outputs_reach_every_sink(tmp_path):
    got_http = []

    class Hook(BaseHTTPRequestHandler):
        def do_POST(self):
            got_http.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Hook)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(3)
    hub = outputs.OutputHub()
    hub.configure(outputs.OutputSettings(
        udp=True, udp_port=udp.getsockname()[1], tcp=True, tcp_port=0, webhook=True,
        webhook_url=f"http://127.0.0.1:{httpd.server_port}/hook", file=True,
        file_path=str(tmp_path / "out.jsonl"), levels=False))
    try:
        tcp = next(s for s in hub.sinks if isinstance(s, outputs.TcpServerSink))
        deadline = time.time() + 3
        while tcp.port == 0 and time.time() < deadline:
            time.sleep(0.05)
        client = socket.create_connection(("127.0.0.1", tcp.port), timeout=3)
        time.sleep(0.5)  # let the server accept
        hub.publish(outputs.detection_message(det(11, 3.5, bearing=float("nan")), 1.7e9,
                                              source="test"))
        alert = AlertEngine([AlertRule("seals", labels=("seal",))]).check_detection(det(12, 9))[0]
        hub.publish(outputs.alert_message(alert, 1.7e9))
        hub.publish(outputs.levels_message(1.0, None, [1, 2, 3, 4, 5], "dBFS", None, {}))
        m1, m2 = json.loads(udp.recv(65536)), json.loads(udp.recv(65536))
        assert m1["type"] == "detection" and m1["label"] == "seal" and m1["bearing_deg"] is None
        assert m1["schema"] == 1 and m1["time"].startswith("2023-11-14T22:13:20")
        assert m2["type"] == "alert" and m2["rule"] == "seals"
        buf = b""
        while buf.count(b"\n") < 2:
            buf += client.recv(65536)
        assert [json.loads(x)["type"] for x in buf.split(b"\n") if x] == ["detection", "alert"]
        deadline = time.time() + 3
        while len(got_http) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert [m["type"] for m in got_http] == ["detection", "alert"]
        def file_lines():
            return (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()

        deadline = time.time() + 3
        while time.time() < deadline and len(file_lines()) < 2:
            time.sleep(0.05)
        assert len(file_lines()) == 2  # levels were off
        client.close()
    finally:
        hub.close()
        httpd.shutdown()


def test_unreachable_webhook_never_blocks():
    hub = outputs.OutputHub()
    hub.configure(outputs.OutputSettings(webhook=True, webhook_url="http://127.0.0.1:9/x"))
    try:
        t0 = time.perf_counter()
        for _ in range(50):
            hub.publish({"type": "alert"})
        assert time.perf_counter() - t0 < 0.05
    finally:
        hub.close()


@pytest.fixture
def recording(tmp_path):
    fs = 16000
    x = 0.01 * np.random.default_rng(0).standard_normal(fs * 20)
    for c in (2, 8, 14):
        add_chirp(x, fs, c, 1500, 3375)
    sf.write(tmp_path / "rec.wav", x, fs)
    return tmp_path / "rec.wav"


def test_raven_round_trip_teaches_signatures(recording, tmp_path):
    dets = [det(1, 2.0, label="upsweep"), det(2, 8.0, label="upsweep"), det(3, 11.0, label=None)]
    for d in dets:
        d.t1, d.f_lo, d.f_hi = d.t0 + 0.4, 1300, 3400
    rows = raven.detection_rows(dets, 1, wall_time=lambda t: 1.7e9 + t,
                                locate=lambda t: ("rec.wav", t))
    raven.write_table(tmp_path / "sel.txt", rows, raven.DETECTION_EXTRA)
    header = (tmp_path / "sel.txt").read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header[:8] == list(raven.STANDARD) + ["Annotation"]
    table = raven.read_table(tmp_path / "sel.txt")
    assert len(table.selections) == 3 and table.label_columns()[0] == "Annotation"
    assert table.labels() == ["upsweep"] and table.selections[0].file == "rec.wav"
    lib = SignatureLibrary(tmp_path / "sigs")
    res = raven.import_selections(table, lib)
    assert res.added == 2 and res.skipped == 1
    assert len(lib.find("upsweep").positives) == 2


def test_raven_pro_table_with_view_rows(recording, tmp_path):
    (tmp_path / "pro.txt").write_text(
        "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
        "High Freq (Hz)\tSpecies\n"
        "1\tWaveform 1\t1\t14.0\t14.4\t0\t8000\tUpsweep\n"
        "1\tSpectrogram 1\t1\t14.0\t14.4\t1300\t3400\tUpsweep\n"
        "2\tSpectrogram 1\t1\t5.0\t5.3\t100\t500\t\n", encoding="utf-8")
    table = raven.read_table(tmp_path / "pro.txt")
    assert len(table.selections) == 2 and table.selections[0].low == 1300
    assert table.label_columns()[0] == "Species"
    lib = SignatureLibrary(tmp_path / "sigs")
    res = raven.import_selections(table, lib, default_audio=recording)
    assert res.added == 1 and lib.find("upsweep") is not None  # names match ignoring case


def test_not_a_raven_table(tmp_path):
    (tmp_path / "x.txt").write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        raven.read_table(tmp_path / "x.txt")

"""Settings panels: one per kind of source. Each builds a Source on demand."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                               QSlider, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from aquafonix.parsing import BINARY_TYPES, ParserConfig
from aquafonix.screen.capture import OUTPUTS, Region, ScreenSource
from aquafonix.screen.extract import (ESTIMATORS, MAX_TRACES, MODES, PROBES, Calibration,
                                      TraceDef, TraceSettings)
from aquafonix.sources import (FileTailSource, GeneratorSource, SerialSource, Source,
                               TcpClientSource, TcpServerSource, UdpSource, list_serial_ports)
from aquafonix.sources.audio import AudioSource, list_audio_devices
from aquafonix.sources.generator import WAVEFORMS
from aquafonix.ui.widgets import ColorButton, combo, named, spin


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class SourcePanel(QWidget):
    key = ""
    title = ""
    #: emitted when a setting that can be applied to a running source changes
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)

    def form(self, title: str | None = None) -> QFormLayout:
        box = QGroupBox(title) if title else QWidget()
        lay = QFormLayout(box)
        lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.layout_.addWidget(box)
        return lay

    def create_source(self) -> Source:
        raise NotImplementedError

    def apply_to(self, source: Source) -> None:
        """Push live-editable settings into a running source."""

    def refresh(self) -> None:
        """Rescan devices/ports."""


class ParserBox(QGroupBox):
    """Text/binary format settings shared by serial, network and file."""

    def __init__(self, prefix: str, parent=None) -> None:
        super().__init__("Data format", parent)
        lay = QFormLayout(self)
        self.mode = named(combo({"text": "Text lines (CSV, numbers, label:value)",
                                 "binary": "Binary frames"}), f"{prefix}.format")
        lay.addRow("Format", self.mode)
        self.x_first = named(QCheckBox("First column is time / X"), f"{prefix}.xfirst")
        self.names = named(QLineEdit(), f"{prefix}.names")
        self.names.setPlaceholderText("optional, e.g. temp, humidity")
        self.dtype = named(combo({k: k for k in BINARY_TYPES}, "int16"), f"{prefix}.dtype")
        self.channels = named(spin(1, 64, 1), f"{prefix}.channels")
        self.big_endian = named(QCheckBox("Big-endian"), f"{prefix}.bigendian")
        self.sync = named(QLineEdit(), f"{prefix}.sync")
        self.sync.setPlaceholderText("optional hex, e.g. AA 55")
        self.rate = named(spin(0, 10_000_000, 0, decimals=3, suffix=" Hz"), f"{prefix}.rate")
        self.rate.setSpecialValueText("arrival time")
        self._text_rows = [(None, self.x_first)]
        self._bin_rows = [("Value type", self.dtype), ("Channels", self.channels),
                          (None, self.big_endian), ("Frame sync", self.sync),
                          ("Sample rate", self.rate)]
        lay.addRow(self.x_first)
        lay.addRow("Channel names", self.names)
        for label, w in self._bin_rows:
            if label:
                lay.addRow(label, w)
            else:
                lay.addRow(w)
        self._layout = lay
        self.mode.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self) -> None:
        binary = self.mode.currentData() == "binary"
        for _, w in self._text_rows:
            self._layout.setRowVisible(w, not binary)
        for _, w in self._bin_rows:
            self._layout.setRowVisible(w, binary)

    def config(self) -> ParserConfig:
        names = [n.strip() for n in self.names.text().split(",") if n.strip()]
        return ParserConfig(mode=self.mode.currentData(), x_from_first=self.x_first.isChecked(),
                            names=names, dtype=self.dtype.currentData(),
                            channels=self.channels.value(),
                            big_endian=self.big_endian.isChecked(),
                            sync_hex=self.sync.text().strip(), sample_rate=self.rate.value())


class AudioPanel(SourcePanel):
    key, title = "audio", "Audio input"
    RATES = ["device default", "8000", "16000", "22050", "44100", "48000", "96000", "192000"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        row = QHBoxLayout()
        self.device = named(combo([]), "audio.device")
        self.device.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device.setMinimumContentsLength(18)
        rescan = QToolButton(text="⟳", toolTip="Rescan audio devices")
        rescan.clicked.connect(self.refresh)
        row.addWidget(self.device, 1)
        row.addWidget(rescan)
        f.addRow("Device", row)
        self.rate = named(combo(self.RATES, "device default"), "audio.rate")
        self.rate.setEditable(True)
        f.addRow("Sample rate", self.rate)
        self.channels = named(spin(1, 32, 2), "audio.channels")
        f.addRow("Channels", self.channels)
        self.info = _hint("")
        self.layout_.addWidget(self.info)
        self.layout_.addWidget(_hint(
            "Pick a microphone / line-in, or <b>System audio (loopback)</b> to watch whatever "
            "the PC is playing (Windows)."))
        self._devices = []
        self.refresh()

    def refresh(self) -> None:
        current = self.device.currentText()
        self._devices, problem = list_audio_devices()
        self.device.blockSignals(True)
        self.device.clear()
        for dev in self._devices:
            self.device.addItem(dev.label)
            self.device.setItemData(self.device.count() - 1,
                                    f"{dev.channels} ch · {dev.default_rate:g} Hz",
                                    Qt.ToolTipRole)
        if current:
            idx = self.device.findText(current)
            if idx >= 0:
                self.device.setCurrentIndex(idx)
        self.device.blockSignals(False)
        msg = problem or ("" if self._devices else "No audio capture devices found.")
        self.info.setText(msg)
        self.info.setVisible(bool(msg))

    def create_source(self) -> Source:
        idx = self.device.currentIndex()
        if idx < 0 or idx >= len(self._devices):
            raise ValueError("no audio device selected")
        text = self.rate.currentText().strip()
        try:
            rate = float(text) if text and text[0].isdigit() else None
        except ValueError:
            raise ValueError(f"invalid sample rate {text!r}") from None
        return AudioSource(self._devices[idx], rate, self.channels.value())


class AudioFilePanel(SourcePanel):
    key, title = "audiofile", "Audio file (replay a recording)"
    SPEEDS = {"0": "Scan (as fast as possible)", "0.1": "0.1×", "0.25": "0.25×", "0.5": "0.5×",
              "1": "1× (real time)", "2": "2×", "4": "4×", "8": "8×", "16": "16×"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        row = QHBoxLayout()
        self.path = named(QLineEdit(), "audiofile.path")
        self.path.setPlaceholderText("WAV / FLAC / AIFF hydrophone recording")
        self.path.editingFinished.connect(self._update_info)
        browse = QToolButton(text="…", toolTip="Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        f.addRow("File", row)
        self.info = QLabel("")
        f.addRow("", self.info)
        self.speed = named(combo(self.SPEEDS, "1"), "audiofile.speed")
        f.addRow("Speed", self.speed)
        self.loop = named(QCheckBox("Loop"), "audiofile.loop")
        f.addRow(self.loop)
        self.layout_.addWidget(_hint(
            "Replays the file through everything else (spectrogram, detection, events). "
            "Use the transport bar above the plots to pause, seek or change speed. "
            "<b>Scan</b> runs detection over the whole file as fast as possible."))

    def _browse(self) -> None:
        from aquafonix.audiofile import OPEN_FILTER

        path, _ = QFileDialog.getOpenFileName(self, "Open recording", self.path.text(),
                                              OPEN_FILTER)
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        self.path.setText(path)
        self._update_info()

    def _update_info(self) -> None:
        from aquafonix import audiofile

        path = self.path.text().strip()
        if not path:
            self.info.setText("")
            return
        try:
            i = audiofile.info(path)
            self.info.setText(f"{i.duration:.1f} s · {i.sample_rate:g} Hz · {i.channels} ch")
        except Exception as exc:
            self.info.setText(f"can't read: {exc}")

    def create_source(self) -> Source:
        from aquafonix.sources.audiofile import AudioFileSource

        return AudioFileSource(self.path.text().strip(), float(self.speed.currentData()),
                               self.loop.isChecked())


class SerialPanel(SourcePanel):
    key, title = "serial", "Serial port (COM)"
    BAUDS = ["9600", "19200", "38400", "57600", "115200", "230400", "250000", "460800",
             "500000", "921600", "1000000", "2000000"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        row = QHBoxLayout()
        self.port = named(combo([]), "serial.port")
        self.port.setEditable(True)
        rescan = QToolButton(text="⟳", toolTip="Rescan serial ports")
        rescan.clicked.connect(self.refresh)
        row.addWidget(self.port, 1)
        row.addWidget(rescan)
        f.addRow("Port", row)
        self.baud = named(combo(self.BAUDS, "115200"), "serial.baud")
        self.baud.setEditable(True)
        f.addRow("Baud rate", self.baud)
        self.parser = ParserBox("serial")
        self.layout_.addWidget(self.parser)
        self.layout_.addWidget(_hint(
            "Works with Arduino-style output: <code>1.2,3.4</code> or "
            "<code>temp:21.5 hum:40</code>, one sample per line."))
        self.refresh()

    def refresh(self) -> None:
        current = self.port.currentText()
        self.port.clear()
        for dev, desc in list_serial_ports():
            self.port.addItem(f"{dev} — {desc}" if desc else dev)
        if current:
            self.port.setEditText(current)

    def create_source(self) -> Source:
        port = self.port.currentText().split(" — ")[0].strip()
        try:
            baud = int(self.baud.currentText())
        except ValueError:
            raise ValueError("baud rate must be a whole number") from None
        return SerialSource(port, baud, self.parser.config())


class NetworkPanel(SourcePanel):
    key, title = "network", "Network (UDP / TCP)"
    PROTOCOLS = {"udp": "UDP — listen for datagrams", "tcp-server": "TCP — listen (server)",
                 "tcp-client": "TCP — connect to a server"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        self.protocol = named(combo(self.PROTOCOLS, "udp"), "network.protocol")
        f.addRow("Protocol", self.protocol)
        self.host = named(QLineEdit("0.0.0.0"), "network.host")
        self.host_label = QLabel("Listen on")
        f.addRow(self.host_label, self.host)
        self.port = named(spin(1, 65535, 9000), "network.port")
        f.addRow("Port", self.port)
        self.parser = ParserBox("network")
        self.layout_.addWidget(self.parser)
        self.layout_.addWidget(_hint(
            "Try it: <code>python -c \"import socket,math,time;s=socket.socket(2,2);"
            "[s.sendto(f'{math.sin(i/9)}'.encode(),('127.0.0.1',9000)) or time.sleep(.01) "
            "for i in range(9999)]\"</code>"))
        self.protocol.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self) -> None:
        client = self.protocol.currentData() == "tcp-client"
        self.host_label.setText("Server" if client else "Listen on")
        if client and self.host.text() in ("", "0.0.0.0"):
            self.host.setText("127.0.0.1")
        elif not client and self.host.text() in ("", "127.0.0.1"):
            self.host.setText("0.0.0.0")

    def create_source(self) -> Source:
        proto, host, port = self.protocol.currentData(), self.host.text().strip(), self.port.value()
        cfg = self.parser.config()
        if proto == "udp":
            return UdpSource(port, host, cfg)
        if proto == "tcp-server":
            return TcpServerSource(port, host, cfg)
        return TcpClientSource(host, port, cfg)


class FilePanel(SourcePanel):
    key, title = "file", "File (follow / tail)"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        row = QHBoxLayout()
        self.path = named(QLineEdit(), "file.path")
        self.path.setPlaceholderText("log / CSV file being written to")
        browse = QToolButton(text="…", toolTip="Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        f.addRow("File", row)
        self.from_start = named(QCheckBox("Read existing contents first"), "file.fromstart")
        f.addRow(self.from_start)
        self.parser = ParserBox("file")
        self.layout_.addWidget(self.parser)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Follow file", self.path.text(),
                                              "Data files (*.csv *.txt *.log *.dat *.bin);;"
                                              "All files (*)")
        if path:
            self.path.setText(path)

    def create_source(self) -> Source:
        return FileTailSource(self.path.text().strip(), self.from_start.isChecked(),
                              self.parser.config())


class GeneratorPanel(SourcePanel):
    key, title = "generator", "Test signal"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        f = self.form()
        self.kind = named(combo({w: w.capitalize() for w in WAVEFORMS}, "sine"), "gen.waveform")
        f.addRow("Waveform", self.kind)
        self.freq = named(spin(0.01, 100_000, 5, decimals=2, suffix=" Hz"), "gen.freq")
        f.addRow("Frequency", self.freq)
        self.amp = named(spin(0, 1e6, 1, decimals=3), "gen.amplitude")
        f.addRow("Amplitude", self.amp)
        self.noise = named(spin(0, 1e6, 0.05, decimals=3), "gen.noise")
        f.addRow("Noise", self.noise)
        self.rate = named(spin(1, 384_000, 1000, suffix=" S/s"), "gen.rate")
        f.addRow("Sample rate", self.rate)
        self.channels = named(spin(1, 16, 2), "gen.channels")
        f.addRow("Channels", self.channels)
        self.layout_.addWidget(_hint("A synthetic signal for trying out the display."))

    def create_source(self) -> Source:
        return GeneratorSource(self.kind.currentData(), self.freq.value(), self.amp.value(),
                               self.rate.value(), self.channels.value(), self.noise.value())


class ScreenPanel(SourcePanel):
    key, title = "screen", "Screen region (graph on screen)"
    select_requested = Signal()
    save_trace_requested = Signal()
    autodetect_requested = Signal(int)
    region_changed = Signal(object)
    #: (key, pixel position) for a calibration line edited in the panel
    cal_line_changed = Signal(str, float)
    CAL_KEYS = ("ya", "yb", "xa", "xb")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._building = True
        # region
        f = self.form("Region")
        sel = QPushButton("Select region on screen…")
        sel.setDefault(True)
        sel.clicked.connect(self.select_requested)
        f.addRow(sel)
        self.rx = named(spin(-100_000, 100_000, 0), "screen.x")
        self.ry = named(spin(-100_000, 100_000, 0), "screen.y")
        self.rw = named(spin(0, 20_000, 0), "screen.w")
        self.rh = named(spin(0, 20_000, 0), "screen.h")
        for box in (self.rx, self.ry, self.rw, self.rh):
            box.valueChanged.connect(self._region_edited)
        pos, size = QHBoxLayout(), QHBoxLayout()
        pos.addWidget(self.rx)
        pos.addWidget(self.ry)
        size.addWidget(self.rw)
        size.addWidget(QLabel("×"))
        size.addWidget(self.rh)
        f.addRow("Left, top", pos)
        f.addRow("Size (px)", size)
        self.outline = named(QCheckBox("Show outline on screen"), "screen.outline")
        self.outline.setChecked(True)
        f.addRow(self.outline)
        self.fps = named(spin(1, 120, 20, suffix=" fps"), "screen.fps")
        f.addRow("Capture rate", self.fps)

        # detection
        f = self.form("Line detection")
        self.mode = named(combo(MODES, "auto"), "screen.mode")
        f.addRow("Find lines by", self.mode)
        self.traces = QTableWidget(0, 3)
        self.traces.setHorizontalHeaderLabels(["Trace name", "Colour", ""])
        self.traces.verticalHeader().setVisible(False)
        hh = self.traces.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.traces.setMinimumHeight(110)
        self.traces.itemChanged.connect(self._traces_edited)
        f.addRow(self.traces)
        crow = QHBoxLayout()
        self.pick = QPushButton("+ Pick from preview")
        self.pick.setCheckable(True)
        self.pick.setToolTip("Then click a line in the preview image to add it as a trace")
        self.auto_n = named(spin(1, MAX_TRACES, 2), "screen.autocount")
        self.auto_n.setToolTip("How many differently coloured lines to look for")
        auto = QPushButton("Auto-detect")
        auto.setToolTip("Find the most prominent line colours in the current frame")
        auto.clicked.connect(lambda: self.autodetect_requested.emit(self.auto_n.value()))
        crow.addWidget(self.pick, 1)
        crow.addWidget(auto)
        crow.addWidget(self.auto_n)
        f.addRow(crow)
        self._traces_state = named(QLineEdit(), "screen.traces")  # persisted, not shown
        self._traces_state.setVisible(False)
        self._traces_state.textChanged.connect(self._load_traces_state)
        self.layout_.addWidget(self._traces_state)
        self.detected = QLabel("–")
        self.detected.setWordWrap(True)
        self.detected.setTextFormat(Qt.RichText)
        f.addRow("Found", self.detected)
        self._det_form = f
        trow = QHBoxLayout()
        self.tolerance = named(QSlider(Qt.Horizontal), "screen.tolerance")
        self.tolerance.setRange(5, 250)
        self.tolerance.setValue(60)
        self.tol_label = QLabel("60")
        self.tol_label.setMinimumWidth(28)
        self.tolerance.valueChanged.connect(lambda v: self.tol_label.setText(str(v)))
        trow.addWidget(self.tolerance, 1)
        trow.addWidget(self.tol_label)
        f.addRow("Tolerance", trow)
        self.ignore_grid = named(QCheckBox("Ignore grid lines and axes"), "screen.ignoregrid")
        self.ignore_grid.setChecked(True)
        f.addRow(self.ignore_grid)
        self.estimator = named(combo(ESTIMATORS, "center"), "screen.estimator")
        f.addRow("Line position", self.estimator)
        self.bridge = named(QCheckBox("Bridge small gaps"), "screen.bridge")
        self.bridge.setChecked(True)
        f.addRow(self.bridge)
        self.show_mask = named(QCheckBox("Highlight detected pixels"), "screen.showmask")
        f.addRow(self.show_mask)

        # calibration
        f = self.form("Calibration")
        self.cal_val, self.cal_px = {}, {}
        labels = {"ya": "Y line A", "yb": "Y line B", "xa": "X line A", "xb": "X line B"}
        defaults = {"ya": 0.0, "yb": 1.0, "xa": 0.0, "xb": 1.0}
        for key in self.CAL_KEYS:
            val = named(spin(-1e9, 1e9, defaults[key], decimals=4), f"screen.cal.{key}")
            px = named(spin(-1e5, 1e5, 0, decimals=1, suffix=" px"), f"screen.cal.{key}.px")
            px.setToolTip("Pixel position inside the region (drag the line in the preview)")
            px.valueChanged.connect(lambda v, k=key: self._cal_px_edited(k, v))
            row = QHBoxLayout()
            row.addWidget(QLabel("value"))
            row.addWidget(val, 1)
            row.addWidget(QLabel("at"))
            row.addWidget(px, 1)
            f.addRow(labels[key], row)
            self.cal_val[key], self.cal_px[key] = val, px
        self.log_y = named(QCheckBox("Logarithmic Y axis"), "screen.logy")
        f.addRow(self.log_y)
        f.addRow(_hint("Drag the dashed A/B lines in the preview onto two known grid lines "
                       "and type their values here."))

        # output
        f = self.form("Output")
        self.output = named(combo(OUTPUTS, "trace"), "screen.output")
        f.addRow("Show", self.output)
        self.probe = named(combo(PROBES, "right"), "screen.probe")
        f.addRow("Value per frame", self.probe)
        self.probe_col = named(spin(0, 100, 50, decimals=1, suffix=" %"), "screen.probecol")
        f.addRow("Column", self.probe_col)
        self._out_form = f
        save = QPushButton("Save current trace as CSV…")
        save.clicked.connect(self.save_trace_requested)
        f.addRow(save)

        for w in (self.mode, self.estimator, self.output, self.probe):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.ignore_grid, self.bridge, self.log_y, self.outline, self.show_mask):
            w.toggled.connect(self._changed)
        for w in (self.fps, self.probe_col, *self.cal_val.values()):
            w.valueChanged.connect(self._changed)
        self.tolerance.valueChanged.connect(self._changed)
        self.layout_.addStretch(1)
        self._building = False
        self._update_visibility()

    # -- state ----------------------------------------------------------------
    def region(self) -> Region | None:
        if self.rw.value() < 1 or self.rh.value() < 1:
            return None
        return Region(self.rx.value(), self.ry.value(), self.rw.value(), self.rh.value())

    def set_region(self, region: Region, reset_calibration: bool = True) -> None:
        for box, v in zip((self.rx, self.ry, self.rw, self.rh),
                          (region.left, region.top, region.width, region.height)):
            box.blockSignals(True)
            box.setValue(v)
            box.blockSignals(False)
        if reset_calibration:
            self.reset_calibration_lines(region.width, region.height)
        self.region_changed.emit(region)
        self._changed()

    def reset_calibration_lines(self, width: int, height: int) -> None:
        for key, px in (("ya", height), ("yb", 0), ("xa", 0), ("xb", width)):
            self.set_cal_px(key, px, emit_line=True)

    def set_cal_px(self, key: str, px: float, emit_line: bool = False) -> None:
        box = self.cal_px[key]
        box.blockSignals(True)
        box.setValue(px)
        box.blockSignals(False)
        if emit_line:
            self.cal_line_changed.emit(key, px)
        self._changed()

    def calibration(self) -> Calibration:
        v, p = self.cal_val, self.cal_px
        return Calibration(p["ya"].value(), v["ya"].value(), p["yb"].value(), v["yb"].value(),
                           p["xa"].value(), v["xa"].value(), p["xb"].value(), v["xb"].value(),
                           self.log_y.isChecked())

    def trace_defs(self) -> list[TraceDef]:
        defs = []
        for row in range(self.traces.rowCount()):
            item = self.traces.item(row, 0)
            button = self.traces.cellWidget(row, 1)
            name = (item.text().strip() if item else "") or f"trace {row + 1}"
            defs.append(TraceDef(name, button.color()))
        return defs

    def set_trace_defs(self, defs: list[TraceDef]) -> None:
        self.traces.blockSignals(True)
        self.traces.setRowCount(0)
        names = set()
        for d in defs[:MAX_TRACES]:
            name = d.name
            while name in names:  # channel names must be unique
                name += "'"
            names.add(name)
            row = self.traces.rowCount()
            self.traces.insertRow(row)
            self.traces.setItem(row, 0, QTableWidgetItem(name))
            button = ColorButton(d.color)
            button.colorChanged.connect(self._traces_edited)
            self.traces.setCellWidget(row, 1, button)
            remove = QToolButton(text="✕", toolTip="Remove this trace")
            remove.clicked.connect(lambda _=False, b=button: self._remove_trace(b))
            self.traces.setCellWidget(row, 2, remove)
        self.traces.blockSignals(False)
        self._traces_edited()

    def _remove_trace(self, button) -> None:
        for row in range(self.traces.rowCount()):
            if self.traces.cellWidget(row, 1) is button:
                defs = self.trace_defs()
                del defs[row]
                self.set_trace_defs(defs)
                return

    def _traces_edited(self, *_args) -> None:
        state = ";".join(f"{d.name.replace(';', ',').replace('=', '-')}="
                         f"#{d.color[0]:02x}{d.color[1]:02x}{d.color[2]:02x}"
                         for d in self.trace_defs())
        self._traces_state.blockSignals(True)
        self._traces_state.setText(state)
        self._traces_state.blockSignals(False)
        self._changed()

    def _load_traces_state(self, text: str) -> None:
        defs = []
        for part in text.split(";"):
            name, _, color = part.rpartition("=")
            c = QColor(color)
            if name and c.isValid():
                defs.append(TraceDef(name, (c.red(), c.green(), c.blue())))
        self.set_trace_defs(defs)

    def trace_settings(self) -> TraceSettings:
        return TraceSettings(mode=self.mode.currentData(), traces=tuple(self.trace_defs()),
                             tolerance=float(self.tolerance.value()),
                             ignore_grid=self.ignore_grid.isChecked(),
                             estimator=self.estimator.currentData(),
                             bridge_gaps=self.bridge.isChecked(),
                             calibration=self.calibration())

    def add_picked_trace(self, rgb) -> None:
        """A colour clicked in the preview becomes a new trace."""
        self.pick.setChecked(False)
        defs = self.trace_defs()
        if len(defs) >= MAX_TRACES:
            return
        defs.append(TraceDef(f"trace {len(defs) + 1}", tuple(int(v) for v in rgb)))
        self.set_trace_defs(defs)
        self.mode.setCurrentIndex(self.mode.findData("colors"))

    def set_detected_traces(self, colors) -> None:
        """Replace the trace list with auto-detected colours."""
        self.set_trace_defs([TraceDef(f"trace {i + 1}", c) for i, c in enumerate(colors)])
        self.mode.setCurrentIndex(self.mode.findData("colors"))

    def show_detected(self, result) -> None:
        parts = []
        for i, tr in enumerate(result.traces):
            if tr.color is None:
                parts.append(f"{tr.name}: {tr.coverage:.0%}")
                continue
            r, g, b = tr.color
            parts.append(f'<span style="color:rgb({r},{g},{b})">■</span> {tr.name}: '
                         f"{tr.coverage:.0%}")
        self.detected.setText("<br>".join(parts) if parts else "no line found")

    # -- source ---------------------------------------------------------------
    def create_source(self) -> Source:
        region = self.region()
        if region is None:
            raise ValueError("Select a screen region first.")
        return ScreenSource(region, self.fps.value(), self.trace_settings(),
                            self.output.currentData(), self.probe.currentData(),
                            self.probe_col.value() / 100.0)

    def apply_to(self, source: Source) -> None:
        if not isinstance(source, ScreenSource):
            return
        region = self.region()
        if region is not None:
            source.region = region
        source.fps = float(self.fps.value())
        source.settings = self.trace_settings()
        source.output = self.output.currentData()
        source.probe = self.probe.currentData()
        source.probe_column = self.probe_col.value() / 100.0

    # -- internals ------------------------------------------------------------
    def _region_edited(self) -> None:
        region = self.region()
        if region is not None:
            self.region_changed.emit(region)
        self._changed()

    def _cal_px_edited(self, key: str, value: float) -> None:
        self.cal_line_changed.emit(key, value)
        self._changed()

    def _changed(self, *_args) -> None:
        if self._building:
            return
        self._update_visibility()
        self.changed.emit()

    def _update_visibility(self) -> None:
        self._out_form.setRowVisible(self.probe_col, self.probe.currentData() == "column")
        self._out_form.setRowVisible(self.probe, self.output.currentData() != "stitch")
        self.traces.setEnabled(self.mode.currentData() == "colors")


PANELS = (AudioPanel, AudioFilePanel, SerialPanel, NetworkPanel, FilePanel, ScreenPanel,
          GeneratorPanel)

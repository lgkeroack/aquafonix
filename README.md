# Aquafonix

A portable desktop monitor for hydrophones and other signals. It watches, in real time:

- **Audio**: microphones, line-in and multi-channel interfaces, plus Windows "what you hear" (WASAPI loopback).
- **Recordings**: WAV, FLAC or AIFF files, or whole folders played back as one continuous recording, with clock times taken from the file names.
- **Data streams**: serial/COM, UDP, TCP or a growing file, as text or binary.
- **The screen**: point it at a graph drawn by another program. It reads up to 8 coloured line traces, or a scrolling spectrogram whose colours it turns back into levels.

On top of that it has a full underwater-acoustics toolkit: calibrated levels, a signature catalogue that you teach and that flags "Suspected: …" sounds, bearings from a pair of hydrophones, propeller (DEMON) analysis, alerts, and live data outputs for other software.

## Getting it

**Windows:** download `Aquafonix.exe` from the latest [release](https://github.com/lgkeroack/aquafonix/releases), or from the *Aquafonix-windows* artifact of a CI run. It is a single file and needs no installation.

The first time you run it, it creates `Aquafonix.ini` (settings) and `AquafonixData\` (signatures and saved events) next to the exe. If that folder is read-only, they go to `%APPDATA%\Aquafonix` instead. Starting takes a few seconds while the exe unpacks.

**From source** (Python 3.12 or newer, on Windows, Linux or macOS):

```sh
pip install -r requirements.txt -e .
aquafonix            # or: python -m aquafonix
```

On Linux, sounddevice needs PortAudio (`libportaudio2`).

## Quick start

| To… | Do this |
|---|---|
| Monitor a hydrophone | **Source → Audio input**, choose the device, **▶ Start** (F5). Enter its sensitivity and gain under **Acoustics → Calibration** for levels in dB re 1 µPa. |
| Replay recordings | **File → Open audio file…**, or drop files or a folder onto the exe (`Aquafonix.exe D:\deployment1\`). |
| Watch a graph on screen | **Source → Screen region → Select region on screen…**, then drag a box around the graph. For a spectrogram, set *Output* to *Spectrogram image* and use *Pick the colour bar in the preview*. |
| Name a sound | Shift-drag around it on the spectrogram, then **🏷 Identify** (I). Later occurrences are flagged *Suspected: name*. **✔ Confirm**, **✎ Correct** or **✖ Reject** them and the catalogue learns from each answer. |
| Keep the last 30 s | **⏺ Capture** (F9) saves the recent history as an event you can tag and annotate. |
| Measure a sound | Select it, then **📊 Analyse** (A): SPL, Lpk, SEL, PSD, third-octave levels, bandwidths, bearing and DEMON, with CSV export. |

## Features

### Display and navigation
- Waveform, spectrogram (linear or log frequency, noise-floor-relative view, adjustable colour bar), PSD with 5/50/95 % percentiles, third-octave bar chart, levels over time, overview strip, and a navigator for the whole recording.
- Zoom (wheel, box or keys), pan, and pause (F6) to explore the in-memory history while data keeps arriving. A cursor readout shows values under the mouse.
- Playback with 0.05–16× time expansion, band filtering, denoising and live listening.

### Acoustic metrics (ISO 18405, IEC 61260-1)
- SPL, Lpk, peak-to-peak and SEL; broadband, filter-band and any third-octave band, integrated over 0.125 s to 10 min.
- Welch PSD in dB re 1 µPa²/Hz with percentiles; all IEC third-octave bands with 5–95 % ranges.
- Calibration from sensitivity, gain and full-scale voltage. It can be read from WAV metadata (INFO, iXML, GUANO, bext) or from sidecar text files, kept as named profiles, and remembered per folder.
- Batch analysis turns a folder of recordings into per-interval and summary CSV reports (L5/L50/L95/Leq).

### Detection and the signature catalogue
- Spikes above the tracked noise floor are highlighted with their time and frequency extent.
- There is no fixed library. You snip a sound and name it, and later occurrences are flagged *Suspected: name* with a score. This works on spectrogram templates, on waveform shape, and on spectrogram patches read from the screen.
- Confirming adds an example. Correcting or rejecting adds a counter-example that stops that false match, so the catalogue grows as you work.
- Annotated **Raven selection tables** can be imported straight into the catalogue.

### Bearings (two hydrophones)
- GCC-PHAT time difference of arrival gives a bearing for every detection, plus a scrolling **bearing-time display**.
- Bearings are measured from the A→B axis: 0° points towards B, 90° is broadside and 180° points towards A. A pair of hydrophones cannot tell left from right.

### DEMON (propeller analysis)
- Demodulates cavitation noise to find the **shaft rate (rpm)** and **blade rate**, and from them the blade count.
- Runs continuously in the noise report and in the analysis window. A detected propeller is listed as a likely noise source.

### Noise analysis
- L5/L50/L95/Leq, crest factor, kurtosis, spectral flatness, tonal lines, likely-source hints and a noise-floor alarm.
- Linear-phase band isolation with no time shift, including presets such as the MSFD 63/125 Hz bands.

### Alerts
Rules can fire on:
- chosen signatures, optionally above a minimum score;
- any spike above a set SNR, optionally only from a bearing sector;
- broadband level above a threshold;
- the noise floor above a threshold;
- propeller modulation.

Each rule can play a sound, show a pop-up and save the moment (5 s before to 5 s after) as a tagged event. Every alert also goes to the Alerts log and to the data outputs. A cooldown, counted in stream time, stops repeats.

### Data outputs (open interface)
Detections (new and reviewed), alerts and per-interval levels are sent as JSON over UDP, a TCP server (newline-delimited), an HTTP webhook, and/or a JSON-lines file. Each sink runs on its own thread, so a slow receiver never stalls the monitor. Example message:

```json
{"schema":1,"app":"aquafonix","version":"0.1.0","type":"detection","event":"new","id":12,
 "kind":"match","status":"suspected","label":"seal call","score":0.84,"t0":125.31,"t1":125.92,
 "f_lo":310.0,"f_hi":2450.0,"snr_db":17.2,"bearing_deg":62.5,"time":"2024-03-01T06:02:05.310+00:00"}
```

`levels` messages carry `spl_db`, `peak_db`, `sel_db`, `floor_db`, `band_db`, `third_octaves_db` and, when a propeller is detected, `demon`.

### Events and exports
- F9 keeps the last *N* seconds. Selections, reviewed detections and alerts can also be saved. Each event gets tags, notes and a verified flag, and can be exported to CSV.
- Detections and saved events export as **Raven selection tables**.

## Compared with other tools

**DAVIS (Abyssens).** The DAVIS column is based on its [product page](https://www.abyssens.fr/davis_underwater-acoustic_data_visualisation_software/).

| | DAVIS | Aquafonix |
|---|---|---|
| Input | DORI `.wav` replay | any WAV/FLAC/AIFF, folders as one timed recording, live audio, data streams, screen capture |
| Spectrogram, SPL over time, PSD | ✓ | ✓, plus percentiles, log frequency, noise-floor view, detection boxes |
| Third-octave levels | 3 bands | all IEC 61260-1 bands, with ranges, and any of them over time |
| Calibration | auto for DORI | file metadata, sidecars, profiles, per folder |
| Area-of-interest analysis | ✓ | ✓, plus Lpk, SEL, bandwidths, kurtosis, bearing, DEMON and export |
| Detection, identification, review | – | ✓, with a self-built catalogue that learns from confirm / correct / reject |
| Events, alerts, batch reports, live data outputs | – | ✓ |

**DSIT BlackFish.** No acoustic product called "Blackfish Technologies" could be found. The closest match is DSIT's BlackFish naval sonar system, which is an integrated sonar with its own hardware. Aquafonix is software for your own hydrophones and does no active sonar, so this compares only the publicly described *passive monitoring* functions:

| | BlackFish (public description) | Aquafonix |
|---|---|---|
| Automatic detection | ✓ | ✓ spikes above the noise floor |
| Classification | ✓ | ✓ user-taught catalogue that improves with review |
| Bearing / tracking display | ✓ | ✓ per-detection bearing and bearing-time display (two hydrophones) |
| Propeller / vessel analysis | – (not described) | ✓ DEMON shaft and blade rate |
| Single-operator alerts | ✓ | ✓ rule-based sound, pop-up and auto-saved clips |
| Open architecture | ✓ | ✓ JSON over UDP/TCP/webhook/file, Raven import/export, CLI |

## Command line

```text
Aquafonix.exe [RECORDING ...] [--source audio|audiofile|serial|network|file|screen|generator]
              [--set NAME=VALUE ...] [--region X,Y,W,H] [--start] [--settings PATH]
              [--smoke-test [REPORT]]
```

Any control can be set with `--set` using its key under `[controls]` in `Aquafonix.ini`, for example `--set serial.port=COM3`. `--smoke-test` checks that the libraries load and the analysis runs, then exits with status 0 if all is well.

## Keyboard

```text
F5  start / stop          F6  pause (explore history) / live
F9  capture the last N seconds as an event
+ / -  zoom time in / out     Ctrl+Up / Ctrl+Down  zoom Y     0  fit
S  selection tool (or Shift-drag on the waveform / spectrogram)
Z  zoom to selection     I  identify selection     P  play selection     A  analyse
L  listen live     Esc  stop playback / clear selection
C / X  confirm / reject the selected detection
```

## Development

```sh
pip install -r requirements-dev.txt -e .
python -m pytest                              # GUI tests use Qt's offscreen platform without a display
xvfb-run -a python -m pytest                  # Linux: also runs the real screen-capture tests
pyinstaller --noconfirm packaging/aquafonix.spec   # -> dist/Aquafonix(.exe)
python tools/make_icon.py                     # regenerate the icon
```

CI runs the tests on Ubuntu (Python 3.12 and 3.13) and Windows. It then builds the Windows exe, smoke-tests it and uploads it as an artifact. A `v*` tag publishes the exe as a release.

Python 3.11 is not supported: PySide6 6.12 over-releases `None` on every `QPainter.setPen()` call there, which crashes within seconds. Python 3.12 and newer are unaffected.

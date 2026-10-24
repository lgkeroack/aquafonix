# PyInstaller spec: one portable, windowed executable.
#
#   pyinstaller packaging/aquafonix.spec
#
# produces dist/Aquafonix.exe (dist/Aquafonix elsewhere).

import re
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (SRC / "aquafonix" / "__init__.py").read_text()).group(1)

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(SRC)],
    datas=[(str(SRC / "aquafonix" / "assets"), "aquafonix/assets")],
    # imported lazily inside functions, so list them explicitly
    hiddenimports=["sounddevice", "soundfile", "serial", "serial.tools.list_ports", "mss"]
    + (["pyaudiowpatch"] if sys.platform == "win32" else []),
    excludes=["tkinter", "matplotlib", "scipy", "IPython", "PyQt5", "PyQt6", "PySide2",
              "pyqtgraph.opengl", "OpenGL", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

version_info = None
if sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo,
                                                     StringStruct, StringTable, VarFileInfo,
                                                     VarStruct, VSVersionInfo)

    nums = tuple(int(n) for n in (VERSION.split(".") + ["0"] * 4)[:4])
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[StringFileInfo([StringTable("040904B0", [
            StringStruct("ProductName", "Aquafonix"),
            StringStruct("FileDescription", "Aquafonix hydrophone and signal monitor"),
            StringStruct("FileVersion", VERSION),
            StringStruct("ProductVersion", VERSION),
            StringStruct("OriginalFilename", "Aquafonix.exe"),
            StringStruct("InternalName", "Aquafonix")])]),
            VarFileInfo([VarStruct("Translation", [1033, 1200])])])

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Aquafonix",
    console=False,
    icon=str(ROOT / "packaging" / "aquafonix.ico"),
    version=version_info,
    upx=False,
    runtime_tmpdir=None,
)

# PyInstaller spec for the LCTIMSpy desktop app (one folder, windowed).
#
#   pyinstaller packaging/lctimspy.spec      (or: packaging/build.ps1)
#
# Output: dist/LCTIMSpy/LCTIMSpy.exe. Needs the Microsoft Edge WebView2 runtime
# on the target machine (present on Windows 11 and current Windows 10); Python
# and Node are not needed.

import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# The legacy notebook API pulls in matplotlib, seaborn and statannotations; the
# app never imports it, so it stays out of the bundle (Phase 7 retires it).
LEGACY = ["LCTIMSpy.lctims", "LCTIMSpy.mobility_analysis", "LCTIMSpy.plot_chromatograms_mobilograms"]

hiddenimports = [m for m in collect_submodules("LCTIMSpy") if m not in LEGACY]
# uvicorn picks its loop/protocol implementations by import string at runtime.
hiddenimports += collect_submodules("uvicorn")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "LCTIMSpy", "server", "static"), "LCTIMSpy/server/static")],
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPECPATH, "hooks")],
    excludes=LEGACY + ["matplotlib", "seaborn", "statannotations", "tkinter", "IPython",
                       "pytest", "PIL", "tests",
                       # pyarrow parts the cache never uses (~17 MB)
                       "pyarrow.flight", "pyarrow._flight", "pyarrow.substrait", "pyarrow._substrait"],
    noarchive=False,
)

# Drop files that are dead weight at runtime: import libraries (*.lib, only for
# compiling against pyarrow) and the native libraries behind the excluded
# pyarrow modules. `--self-test` on the built exe proves nothing needed went.
DROP = ("arrow_flight.dll", "arrow_substrait.dll", "arrow_python_flight.dll")
SOURCE = (".lib", ".h", ".hpp", ".pyx", ".pxd", ".pxi")      # build-time files only
a.binaries = [b for b in a.binaries if not b[0].lower().endswith(SOURCE)
              and os.path.basename(b[0]).lower() not in DROP]
a.datas = [d for d in a.datas if not d[0].lower().endswith(SOURCE)
           and not d[0].replace("\\", "/").startswith(("pyarrow/include/", "pyarrow/src/"))]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LCTIMSpy",
    console=False,
    icon=os.path.join(SPECPATH, "lctimspy.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LCTIMSpy", upx=False)

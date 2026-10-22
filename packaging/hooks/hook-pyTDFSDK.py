"""Bundle pyTDFSDK and Bruker's native timsdata library.

pyTDFSDK installs the library *beside* the package, at
``site-packages/TDF-SDK/timsdata.dll``, and both pyTDFSDK and
``LCTIMSpy.core.sdk`` look for it one level above the package directory. In
the bundle that is ``_internal/TDF-SDK/``, so it is placed there.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_submodules, get_package_paths

hiddenimports = collect_submodules("pyTDFSDK")

site, _pkg = get_package_paths("pyTDFSDK")
# Only this platform's library (the Linux .so alone is 13 MB).
_lib = os.path.join(site, "TDF-SDK", "timsdata.dll" if sys.platform == "win32" else "libtimsdata.so")
binaries = [(_lib, "TDF-SDK")] if os.path.isfile(_lib) else []

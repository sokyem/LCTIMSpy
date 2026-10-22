"""PyInstaller entry script for LCTIMSpy.exe (see lctimspy.spec)."""

import sys

from LCTIMSpy.desktop.main import main

if __name__ == "__main__":
    sys.exit(main())

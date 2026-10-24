"""Entry point for the frozen app (PyInstaller)."""

import sys

from aquafonix.app import main

if __name__ == "__main__":
    sys.exit(main())

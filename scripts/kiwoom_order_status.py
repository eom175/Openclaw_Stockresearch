#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path


def main():
    diagnostics_dir = Path(__file__).resolve().parent / "diagnostics"
    sys.path.insert(0, str(diagnostics_dir))
    runpy.run_path(str(diagnostics_dir / "kiwoom_order_status.py"), run_name="__main__")


if __name__ == "__main__":
    main()

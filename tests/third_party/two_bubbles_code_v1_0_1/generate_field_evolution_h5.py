from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate field-evolution HDF5 from two_bubbles_code."
    )
    parser.add_argument("config", type=Path, help="YAML configuration file.")
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "patched" / "hyper_bubbles.py"
    subprocess.run([sys.executable, str(script), str(args.config)], check=True)


if __name__ == "__main__":
    main()

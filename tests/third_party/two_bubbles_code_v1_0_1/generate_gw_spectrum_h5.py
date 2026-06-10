from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GW-spectrum HDF5 from field-evolution HDF5."
    )
    parser.add_argument("config", type=Path, help="YAML configuration file.")
    parser.add_argument(
        "--field-evolution-h5",
        type=Path,
        help="Input field-evolution HDF5. Defaults to gw.field_evolution_h5 or the run's field HDF5 path.",
    )
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "patched" / "gw_main_run.py"
    cmd = [sys.executable, str(script), str(args.config)]
    if args.field_evolution_h5 is not None:
        cmd.extend(["--field-evolution-h5", str(args.field_evolution_h5)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

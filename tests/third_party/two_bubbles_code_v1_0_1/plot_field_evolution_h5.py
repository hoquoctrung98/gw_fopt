from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gw_fopt_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import yaml
from matplotlib.colors import SymLogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("tests/data/two_bubbles_code_v1_0_1")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    config.setdefault("run", {})
    return config


def run_id(config: dict[str, Any]) -> str:
    return str(config["run"].get("id", "two_bubbles_reference"))


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_path(
        config["run"].get("output_dir", DEFAULT_OUTPUT_DIR), repo_root()
    )


def field_evolution_h5_path(config: dict[str, Any]) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("field_evolution_h5")
    if configured is None:
        return directory / f"{run_id(config)}_field_evolution.h5"
    return resolve_path(configured, directory)


def field_evolution_pdf_path(config: dict[str, Any]) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("field_evolution_pdf")
    if configured is None:
        return directory / f"{run_id(config)}_field_evolution.pdf"
    return resolve_path(configured, directory)


def input_path(config: dict[str, Any], cli_path: Path | None) -> Path:
    if cli_path is not None:
        return resolve_path(cli_path, Path.cwd())
    return field_evolution_h5_path(config)


def plot(config_path: Path, field_evolution_h5: Path | None = None) -> Path:
    config = load_config(config_path.resolve())
    input_h5 = input_path(config, field_evolution_h5)
    output_pdf = field_evolution_pdf_path(config)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as h5:
        sim_group = h5["simulation"]
        z = sim_group["z"][:] - float(sim_group.attrs["d"]) / 2.0
        s = sim_group["s"][:]
        phi = sim_group["phi_region1"][:]

    max_abs = float(np.nanmax(np.abs(phi)))
    linthresh = max(0.02, max_abs * 1.0e-3)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    image = ax.imshow(
        phi,
        origin="lower",
        aspect="auto",
        extent=[float(z[0]), float(z[-1]), float(s[0]), float(s[-1])],
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=linthresh, vmin=-max_abs, vmax=max_abs),
    )
    ax.set_xlabel(r"$z-d/2$")
    ax.set_ylabel(r"$s$")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\phi(s,z)$")
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot field evolution heatmap from field-evolution HDF5."
    )
    parser.add_argument("config", type=Path, help="YAML configuration file.")
    parser.add_argument(
        "--field-evolution-h5", type=Path, help="Input field-evolution HDF5."
    )
    args = parser.parse_args()
    output_pdf = plot(args.config, args.field_evolution_h5)
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()

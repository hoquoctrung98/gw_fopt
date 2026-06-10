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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mpl  # noqa: E402


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


def gw_spectrum_h5_path(config: dict[str, Any]) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("gw_spectrum_h5")
    if configured is None:
        return directory / f"{run_id(config)}_gw_spectrum.h5"
    return resolve_path(configured, directory)


def gw_spectrum_pdf_path(config: dict[str, Any]) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("gw_spectrum_pdf")
    if configured is None:
        return directory / f"{run_id(config)}_gw_spectrum.pdf"
    return resolve_path(configured, directory)


def input_path(config: dict[str, Any], cli_path: Path | None) -> Path:
    if cli_path is not None:
        return resolve_path(cli_path, Path.cwd())
    return gw_spectrum_h5_path(config)


def compute_gw_fopt_spectrum(
    h5: h5py.File,
    *,
    method: str,
    tol: float,
    max_iter: int,
    num_threads: int | None,
    ratio_t_cut: float | None,
    ratio_t_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    from gw_fopt.bubble_gw import two_bubbles

    omega = h5["gw_spectrum/omega"][:]
    cos_thetak = h5["gw_spectrum/k_grid"][:]
    gw_input = h5["gw_input"]
    s = gw_input["s"][:]

    if ratio_t_cut is None:
        ratio_t_cut = float(gw_input.attrs.get("cutoff_ratio", 0.9))

    gw_calc = two_bubbles.GravitationalWaveCalculator(
        initial_field_status="two_bubbles",
        phi1=gw_input["phi_region1"][:][np.newaxis, :, :],
        phi2=gw_input["phi_region2"][:][np.newaxis, :, :],
        z_grid=gw_input["z"][:],
        ds=float(s[1] - s[0]),
        ratio_t_cut=ratio_t_cut,
        ratio_t_0=ratio_t_0,
    )
    if num_threads is not None and num_threads > 0:
        gw_calc.set_num_threads(num_threads)
    gw_calc.set_integration_params(method=method, tol=tol, max_iter=max_iter)
    angular_spectrum = gw_calc.compute_angular_gw_spectrum(
        w_arr=omega,
        cos_thetak_arr=cos_thetak,
    )
    # The Rust calculator is evaluated only for cos(theta_k) >= 0.
    integrated_spectrum = 2.0 * np.trapezoid(angular_spectrum, axis=0, x=cos_thetak)
    return angular_spectrum, integrated_spectrum


def plot(
    config_path: Path,
    gw_spectrum_h5: Path | None = None,
    *,
    skip_gw_fopt: bool = False,
    method: str = "g7k15",
    tol: float = 1e-5,
    max_iter: int = 10,
    num_threads: int | None = None,
    ratio_t_cut: float | None = None,
    ratio_t_0: float = 0.25,
    gw_fopt_plot_scale: float = 1.0,
    # Upstream two_bubbles_code's GW integrator hard-codes a factor of 2 in
    # the z integral for half-domain data. For the full-domain comparison data,
    # this doubles the tensor amplitude, so the plotted spectrum is divided by
    # 2^2.
    upstream_plot_scale: float = 0.25,
) -> Path:
    config = load_config(config_path.resolve())
    input_h5 = input_path(config, gw_spectrum_h5)
    output_pdf = gw_spectrum_pdf_path(config)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as h5:
        omega = h5["gw_spectrum/omega"][:]
        upstream_spectrum = h5["gw_spectrum/dE_dlogomega"][:]
        if skip_gw_fopt:
            gw_fopt_spectrum = None
        else:
            _, gw_fopt_spectrum = compute_gw_fopt_spectrum(
                h5,
                method=method,
                tol=tol,
                max_iter=max_iter,
                num_threads=num_threads,
                ratio_t_cut=ratio_t_cut,
                ratio_t_0=ratio_t_0,
            )

    upstream_plotted = upstream_plot_scale * upstream_spectrum
    if gw_fopt_spectrum is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.loglog(
            omega,
            upstream_plotted,
            marker="o",
            linewidth=1.5,
            label="two_bubbles_code",
        )
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$dE/d\log\omega$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_pdf)
        plt.close(fig)
        return output_pdf

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )
    gw_fopt_plotted = gw_fopt_plot_scale * gw_fopt_spectrum
    axs[0].loglog(
        omega,
        gw_fopt_plotted,
        marker="o",
        ms=4,
        label="gw_fopt",
    )
    axs[0].loglog(
        omega,
        upstream_plotted,
        marker="o",
        color="red",
        ls="--",
        label="two_bubbles_code",
    )

    denominator = gw_fopt_plotted
    ratio = np.divide(
        upstream_plotted,
        denominator,
        out=np.full_like(upstream_plotted, np.nan, dtype=np.float64),
        where=denominator != 0,
    )
    axs[1].plot(omega, ratio, marker="o", color="red", ls="--")

    axs[0].set_ylabel(r"$dE_\mathrm{GW}/d\log\omega$")
    axs[1].set_xlabel(r"$\omega$")
    axs[1].set_ylabel("ratio")
    axs[1].set_ylim(0.0, 2.0)
    axs[0].legend()
    axs[0].set_title("GW spectrum comparison: two_bubbles_code vs gw_fopt")

    axs[0].yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
    axs[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
    for ax in axs:
        ax.grid(True, which="both", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GW spectrum from spectrum HDF5.")
    parser.add_argument("config", type=Path, help="YAML configuration file.")
    parser.add_argument("--gw-spectrum-h5", type=Path, help="Input GW-spectrum HDF5.")
    parser.add_argument(
        "--skip-gw-fopt",
        action="store_true",
        help="Only plot the two_bubbles_code spectrum; do not recompute the gw_fopt comparison.",
    )
    parser.add_argument(
        "--method",
        default="g7k15",
        help="gw_fopt integration method for GravitationalWaveCalculator.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, help="gw_fopt quadrature tolerance."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="gw_fopt quadrature maximum iterations.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of gw_fopt worker threads. Defaults to the calculator default; use a positive value to override.",
    )
    parser.add_argument(
        "--ratio-t-cut",
        type=float,
        default=None,
        help="gw_fopt time-cutoff ratio. Defaults to gw_input.cutoff_ratio from the HDF5.",
    )
    parser.add_argument(
        "--ratio-t-0",
        type=float,
        default=0.25,
        help="gw_fopt exponential cutoff width ratio.",
    )
    parser.add_argument(
        "--gw-fopt-plot-scale",
        type=float,
        default=1.0,
        help="Scale applied to the gw_fopt spectrum in the upper panel.",
    )
    parser.add_argument(
        "--upstream-plot-scale",
        type=float,
        default=0.25,
        help="Scale applied to the two_bubbles_code spectrum before plotting and ratio calculation.",
    )
    args = parser.parse_args()
    output_pdf = plot(
        args.config,
        args.gw_spectrum_h5,
        skip_gw_fopt=args.skip_gw_fopt,
        method=args.method,
        tol=args.tol,
        max_iter=args.max_iter,
        num_threads=args.num_threads,
        ratio_t_cut=args.ratio_t_cut,
        ratio_t_0=args.ratio_t_0,
        gw_fopt_plot_scale=args.gw_fopt_plot_scale,
        upstream_plot_scale=args.upstream_plot_scale,
    )
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()

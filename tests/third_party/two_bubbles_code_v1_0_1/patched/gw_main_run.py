from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy.interpolate import interp1d


DEFAULT_OUTPUT_DIR = Path("tests/data/two_bubbles_code_v1_0_1")
DEFAULT_UPSTREAM_ROOT = (
    Path(__file__).resolve().parents[1] / "upstream" / "two_bubbles_code-v1.0.1"
)

_GW_INTEGRATOR: Any | None = None
_GW_Z: np.ndarray | None = None
_GW_PHI1: np.ndarray | None = None
_GW_PHI2: np.ndarray | None = None
_GW_DS: float | None = None
_GW_WORKDIR: str | None = None
_GW_N_K: int | None = None
_GW_CUTOFF_RATIO: float | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    config.setdefault("run", {})
    config.setdefault("simulation", {})
    config.setdefault("gw", {})
    config.setdefault("upstream", {})
    return config, raw


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


def gw_spectrum_h5_path(config: dict[str, Any]) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("gw_spectrum_h5")
    if configured is None:
        return directory / f"{run_id(config)}_gw_spectrum.h5"
    return resolve_path(configured, directory)


def copied_config_path(config: dict[str, Any], config_path: Path) -> Path:
    directory = output_dir(config)
    configured = config["run"].get("copied_config")
    if configured is None:
        return directory / config_path.name
    return resolve_path(configured, directory)


def upstream_root(config: dict[str, Any]) -> Path:
    root = resolve_path(
        config["upstream"].get("root", DEFAULT_UPSTREAM_ROOT), repo_root()
    )
    if not root.exists():
        raise FileNotFoundError(f"Upstream root does not exist: {root}")
    return root


def build_cython_if_needed(root: Path) -> None:
    gw_spec_dir = root / "gw_spec"
    extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if any(
        (gw_spec_dir / f"u_integrand{suffix}").exists() for suffix in extension_suffixes
    ):
        return
    build_code = (
        "from setuptools import setup, Extension; "
        "from Cython.Build import cythonize; "
        "setup("
        "ext_modules=cythonize(Extension('u_integrand', sources=['u_integrand.pyx'], include_dirs=['./']), annotate=True), "
        "script_args=['build_ext', '--inplace']"
        ")"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [sys.executable, "-c", build_code], cwd=gw_spec_dir, env=env, check=True
    )


def load_field_evolution_hdf5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        sim_group = h5["simulation"]
        return {
            "lambda_bar": float(sim_group.attrs["lambda_bar"]),
            "gamma": sim_group.attrs["gamma"],
            "bubble_type": str(sim_group.attrs.get("bubble_type", "half")),
            "ratio_smax_d": float(sim_group.attrs.get("ratio_smax_d", 1.2)),
            "saved_ratio_smax_d": float(sim_group.attrs.get("saved_ratio_smax_d", 0.0)),
            "d": float(sim_group.attrs["d"]),
            "ds_original": float(sim_group.attrs["ds_original"]),
            "ds_saved": float(sim_group.attrs["ds_saved"]),
            "dz": float(sim_group.attrs["dz"]),
            "n_z_info": int(sim_group.attrs["n_z_info"]),
            "how_often_ds": int(sim_group.attrs["how_often_ds"]),
            "z": sim_group["z"][:],
            "s": sim_group["s"][:],
            "phi_region1": sim_group["phi_region1"][:],
            "phi_mid": sim_group["phi_mid"][:],
            "energy_density": sim_group["energy_density"][:],
            "r_info": sim_group["r_info"][:],
        }


def phi_region2(
    phi0: np.ndarray, d: float, s: np.ndarray, z: np.ndarray, bubble_type: str
) -> np.ndarray:
    n_s = len(s)
    n_z = len(z)
    ds = float(s[1] - s[0])
    dz = float(z[1] - z[0])

    if bubble_type == "one":
        phi0_interp = interp1d(z, phi0, kind="linear", fill_value="extrapolate")
        return np.fromfunction(
            lambda i_s, i_z: phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz) ** 2)),
            (n_s, n_z),
        )

    phimid = int(np.argmax(phi0))
    z_right = z[phimid:] - z[phimid]
    phi_right = phi0[phimid:]
    phi0_interp = interp1d(z_right, phi_right, kind="linear", fill_value="extrapolate")
    return np.fromfunction(
        lambda i_s, i_z: (
            phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz - d / 2) ** 2))
            + phi0_interp(np.sqrt((i_s * ds) ** 2 + (i_z * dz + d / 2) ** 2))
        ),
        (n_s, n_z),
    )


def default_w_bounds(
    lambda_bar: float, phi_region1: np.ndarray, dz: float
) -> tuple[float, float]:
    mass_true_vacuum = np.sqrt(
        1.0 / 18.0 * (9.0 - 8.0 * lambda_bar + 3.0 * np.sqrt(9.0 - 8.0 * lambda_bar))
    )
    wmin = np.pi / ((len(phi_region1[0]) - 1) * dz) / 2.0
    wmax = min(10.0 * mass_true_vacuum, np.pi / dz)
    return float(wmin), float(wmax)


def init_worker(
    gw_spec_dir: str,
    z: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    ds: float,
    workdir: str,
    n_k: int,
    cutoff_ratio: float,
) -> None:
    global \
        _GW_INTEGRATOR, \
        _GW_Z, \
        _GW_PHI1, \
        _GW_PHI2, \
        _GW_DS, \
        _GW_WORKDIR, \
        _GW_N_K, \
        _GW_CUTOFF_RATIO

    sys.path.insert(0, gw_spec_dir)
    try:
        _GW_INTEGRATOR = importlib.import_module("gw_integrator")
    finally:
        sys.path.remove(gw_spec_dir)

    _GW_Z = z
    _GW_PHI1 = phi1
    _GW_PHI2 = phi2
    _GW_DS = float(ds)
    _GW_WORKDIR = str(workdir) + os.sep
    _GW_N_K = int(n_k)
    _GW_CUTOFF_RATIO = float(cutoff_ratio)


def compute_single_omega(
    task: tuple[int, float],
) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    if (
        _GW_INTEGRATOR is None
        or _GW_Z is None
        or _GW_PHI1 is None
        or _GW_PHI2 is None
        or _GW_DS is None
        or _GW_WORKDIR is None
        or _GW_N_K is None
        or _GW_CUTOFF_RATIO is None
    ):
        raise RuntimeError("GW worker has not been initialized.")

    idx, w = task
    result_w, int_k, k_grid, intk = _GW_INTEGRATOR.gw_integral(
        idx,
        _GW_N_K,
        float(w),
        _GW_Z,
        _GW_PHI1,
        _GW_PHI2,
        _GW_DS,
        _GW_WORKDIR,
        _GW_CUTOFF_RATIO,
        False,
    )
    return (
        idx,
        float(result_w),
        float(int_k),
        np.asarray(k_grid, dtype=np.float64),
        np.asarray(intk, dtype=np.float64),
    )


def compute_gw_spectrum(
    root: Path,
    workdir: Path,
    simulation: dict[str, Any],
    wmin: float | None,
    wmax: float | None,
    n_w: int,
    n_k: int,
    cutoff_ratio: float,
    n_workers: int,
) -> dict[str, np.ndarray]:
    default_wmin, default_wmax = default_w_bounds(
        simulation["lambda_bar"], simulation["phi_region1"], simulation["dz"]
    )
    wmin = default_wmin if wmin is None else float(wmin)
    wmax = default_wmax if wmax is None else float(wmax)
    omega = np.geomspace(wmin, wmax, int(n_w))

    region2 = phi_region2(
        simulation["phi_region1"][0],
        simulation["d"],
        simulation["s"],
        simulation["z"],
        simulation["bubble_type"],
    )

    worker_args = (
        str(root / "gw_spec"),
        simulation["z"],
        simulation["phi_region1"],
        region2,
        simulation["ds_saved"],
        str(workdir),
        int(n_k),
        float(cutoff_ratio),
    )
    tasks = [(idx, float(w)) for idx, w in enumerate(omega)]
    if n_workers == 1:
        init_worker(*worker_args)
        results = [compute_single_omega(task) for task in tasks]
    else:
        pool_kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            pool_kwargs["mp_context"] = get_context("fork")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=worker_args,
            **pool_kwargs,
        ) as executor:
            results = list(executor.map(compute_single_omega, tasks))

    d_e_dlogomega = np.empty(len(omega), dtype=np.float64)
    k_grid = None
    k_integrand = np.empty((len(omega), int(n_k)), dtype=np.float64)
    for idx, result_w, int_k, this_k_grid, intk in sorted(
        results, key=lambda item: item[0]
    ):
        omega[idx] = result_w
        d_e_dlogomega[idx] = int_k
        if k_grid is None:
            k_grid = this_k_grid
        k_integrand[idx] = intk

    if k_grid is None:
        k_grid = np.linspace(0.0, 1.0, int(n_k))

    return {
        "omega": omega,
        "dE_dlogomega": d_e_dlogomega,
        "k_grid": k_grid,
        "k_integrand": k_integrand,
        "phi_region2": region2,
        "wmin": np.asarray(wmin),
        "wmax": np.asarray(wmax),
        "default_wmin": np.asarray(default_wmin),
        "default_wmax": np.asarray(default_wmax),
    }


def write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    data = np.asarray(data)
    if data.ndim > 0 and data.size > 16:
        group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
    else:
        group.create_dataset(name, data=data)


def write_metadata(h5: h5py.File, config: dict[str, Any], raw_yaml: str) -> None:
    config_group = h5.create_group("config")
    config_group.attrs["raw_yaml"] = raw_yaml
    config_group.attrs["config_json"] = json.dumps(config, sort_keys=True)

    metadata = h5.create_group("metadata")
    metadata.attrs["source_record"] = "https://zenodo.org/records/14446671"
    metadata.attrs["doi"] = "https://doi.org/10.5281/zenodo.14446671"
    metadata.attrs["upstream_version"] = "1.0.1"
    metadata.attrs["license"] = "Creative Commons Attribution 4.0 International"
    metadata.attrs["generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata.attrs["generator"] = str(Path(__file__).resolve().relative_to(repo_root()))


def write_gw_spectrum_hdf5(
    path: Path,
    config: dict[str, Any],
    raw_yaml: str,
    simulation: dict[str, Any],
    gw: dict[str, np.ndarray],
    field_evolution_h5: Path,
    n_workers: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        write_metadata(h5, config, raw_yaml)

        input_group = h5.create_group("field_evolution_input")
        input_group.attrs["path"] = str(field_evolution_h5)
        for key in ["lambda_bar", "gamma", "bubble_type", "d", "ds_saved", "dz"]:
            input_group.attrs[key] = simulation[key]

        gw_input = h5.create_group("gw_input")
        gw_input.attrs["cutoff_ratio"] = float(config["gw"].get("cutoff_ratio", 0.9))
        gw_input.attrs["n_k"] = int(config["gw"].get("n_k", 51))
        gw_input.attrs["n_w"] = int(config["gw"].get("n_w", 10))
        gw_input.attrs["n_workers"] = int(n_workers)
        gw_input.attrs["wmin"] = float(gw["wmin"])
        gw_input.attrs["wmax"] = float(gw["wmax"])
        gw_input.attrs["default_wmin"] = float(gw["default_wmin"])
        gw_input.attrs["default_wmax"] = float(gw["default_wmax"])
        write_dataset(gw_input, "phi_region1", simulation["phi_region1"])
        write_dataset(gw_input, "phi_region2", gw["phi_region2"])
        write_dataset(gw_input, "z", simulation["z"])
        write_dataset(gw_input, "s", simulation["s"])

        spectrum = h5.create_group("gw_spectrum")
        write_dataset(spectrum, "omega", gw["omega"])
        write_dataset(spectrum, "dE_dlogomega", gw["dE_dlogomega"])
        write_dataset(spectrum, "k_grid", gw["k_grid"])
        write_dataset(spectrum, "k_integrand", gw["k_integrand"])


def field_input_path(config: dict[str, Any], cli_path: Path | None) -> Path:
    if cli_path is not None:
        return resolve_path(cli_path, Path.cwd())
    configured = config["gw"].get("field_evolution_h5")
    if configured is not None:
        return resolve_path(configured, output_dir(config))
    return field_evolution_h5_path(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patched two_bubbles_code GW spectrum to HDF5."
    )
    parser.add_argument("config", type=Path, help="YAML configuration file.")
    parser.add_argument(
        "--field-evolution-h5", type=Path, help="Input field-evolution HDF5."
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config, raw_yaml = load_config(config_path)
    run_config = config["run"]
    gw_config = config["gw"]

    input_h5 = field_input_path(config, args.field_evolution_h5)
    output_h5 = gw_spectrum_h5_path(config)
    if not input_h5.exists():
        raise FileNotFoundError(f"Field-evolution HDF5 does not exist: {input_h5}")
    if output_h5.exists() and not bool(run_config.get("overwrite", False)):
        raise FileExistsError(
            f"{output_h5} exists; set run.overwrite=true to replace it."
        )

    root = upstream_root(config)
    if bool(config["upstream"].get("build_cython", True)):
        build_cython_if_needed(root)

    n_w = int(gw_config.get("n_w", 10))
    n_k = int(gw_config.get("n_k", 51))
    configured_workers = gw_config.get("n_workers")
    n_workers = (
        os.cpu_count() or 1 if configured_workers is None else int(configured_workers)
    )
    if n_workers <= 0:
        n_workers = os.cpu_count() or 1
    cutoff_ratio = float(gw_config.get("cutoff_ratio", 0.9))
    wmin = gw_config.get("wmin")
    wmax = gw_config.get("wmax")
    wmin = None if wmin is None else float(wmin)
    wmax = None if wmax is None else float(wmax)

    output_dir(config).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, copied_config_path(config, config_path))

    temp_context = None
    if bool(run_config.get("keep_workdir", False)):
        workdir = output_dir(config) / f"{run_id(config)}_gw_work"
        if workdir.exists() and bool(run_config.get("overwrite", False)):
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temp_context = tempfile.TemporaryDirectory(prefix=f"{run_id(config)}_gw_")
        workdir = Path(temp_context.name)

    try:
        simulation = load_field_evolution_hdf5(input_h5)
        gw = compute_gw_spectrum(
            root, workdir, simulation, wmin, wmax, n_w, n_k, cutoff_ratio, n_workers
        )
        write_gw_spectrum_hdf5(
            output_h5, config, raw_yaml, simulation, gw, input_h5, n_workers
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(f"Wrote {output_h5}")


if __name__ == "__main__":
    main()

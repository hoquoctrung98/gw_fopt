from __future__ import annotations

from typing import List, Optional, Self, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .generic_potential import GenericPotential


@njit
def _spatial_derivative_numba(phi: np.ndarray, dz: float) -> np.ndarray:
    """
    Second-order central difference with reflect padding, matching:

        phi_padded = np.pad(phi, ((0,0),(1,1)), mode="reflect")
        return (phi_padded[:,:-2] - 2*phi_padded[:,1:-1] + phi_padded[:,2:]) / dz**2

    Reflect padding means:
        left ghost  = phi[:, 1]   (index 1 mirrors index 0)
        right ghost = phi[:, -2]  (index -2 mirrors index -1)

    So the boundary stencils are:
        result[:, 0]  = (2*phi[:,1] - 2*phi[:,0]) / dz^2
        result[:,-1]  = (2*phi[:,-2] - 2*phi[:,-1]) / dz^2
    """
    n_fields, n_z = phi.shape
    result = np.empty((n_fields, n_z), dtype=np.float64)
    dz2 = dz * dz

    for f in range(n_fields):
        result[f, 0] = (2.0 * phi[f, 1] - 2.0 * phi[f, 0]) / dz2
        for z in range(1, n_z - 1):
            result[f, z] = (phi[f, z - 1] - 2.0 * phi[f, z] + phi[f, z + 1]) / dz2
        result[f, n_z - 1] = (2.0 * phi[f, n_z - 2] - 2.0 * phi[f, n_z - 1]) / dz2

    return result


@njit
def _evolvepi_numba(
    phi: np.ndarray,
    pi: np.ndarray,
    dV_dphi: np.ndarray,
    spatial_deriv: np.ndarray,
    s: float,
    ds: float,
) -> np.ndarray:
    """
    Numba kernel for the pi update:

        damping = 1 - 2*ds / (s + ds)
        forcing = s*ds/(s+ds) * (-dV0(phi.T).T + spatial_derivative(phi))
        return pi * damping + forcing
    """
    denom = s + ds
    damping = 1.0 - 2.0 * ds / denom
    coef = s * ds / denom

    pi_new = np.empty_like(pi)
    for f in range(phi.shape[0]):
        for z in range(phi.shape[1]):
            pi_new[f, z] = pi[f, z] * damping + coef * (
                -dV_dphi[f, z] + spatial_deriv[f, z]
            )
    return pi_new


class PDEBubbleSolver:
    """
    Solver for coupled nonlinear wave equations describing bubble dynamics.

    Intended to be configured via a fluent builder API::

        solver = (
            PDEBubbleSolver(potential)
            .with_grid(z_grid, ds)
            .with_distance(d)
            .with_initial_condition(phi1_initial)
        )
        phi1 = solver.evolve(smax)

    Each builder method validates its inputs immediately and returns ``self``
    so calls can be chained.  ``evolve`` checks that all required configuration
    is present before running.
    """

    def __init__(self, potential: GenericPotential) -> None:
        self.potential: GenericPotential = potential
        self.Ndim: int = potential.Ndim

        # Set by with_grid()
        self.z_grid: Optional[NDArray[np.float64]] = None
        self.ds: Optional[float] = None
        self.dz: Optional[float] = None
        self.n_z: Optional[int] = None

        # Set by with_distance()
        self.d: Optional[float] = None

        # Set by with_initial_condition()
        self.phi1_initial: Optional[NDArray[np.float64]] = None

        # Set by evolve()
        self.history_interval: Optional[int] = None
        self.phi1: Optional[NDArray[np.float64]] = None
        self.phi2: Optional[NDArray[np.float64]] = None
        self.n_s: Optional[int] = None
        self.s_max: Optional[float] = None
        self.s_grid: Optional[NDArray[np.float64]] = None
        self.energy_density: Optional[NDArray[np.float64]] = None
        self.energy_how_often: Optional[int] = None

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def with_grid(
        self,
        z_grid: Union[NDArray[np.float64], List[float]],
        ds: float,
    ) -> Self:
        """
        Set the spatial grid and time step.

        Parameters
        ----------
        z_grid : array-like
            Uniformly-spaced spatial coordinates.
        ds : float
            Time step.  Must satisfy the CFL condition ds <= dz;
            unlike the old interface this is enforced strictly rather
            than silently clamped, so the caller always gets exactly
            the step size they asked for.

        Returns
        -------
        self
        """
        z_grid = np.array(z_grid, dtype=np.float64)
        if len(z_grid) < 2:
            raise ValueError("z_grid must contain at least two points.")

        dz = float(np.abs(z_grid[1] - z_grid[0]))

        if ds > dz:
            raise ValueError(
                f"CFL condition violated: ds={ds} > dz={dz}. "
                f"Choose ds <= {dz} to ensure numerical stability."
            )

        self.z_grid = z_grid
        self.ds = float(ds)
        self.dz = dz
        self.n_z = len(z_grid)

        # Invalidate anything that depended on the old grid
        self.phi1_initial = None
        self.phi1 = None
        self.s_grid = None
        self.n_s = None
        self.s_max = None

        return self

    def with_distance(self, d: float) -> Self:
        """
        Set the initial distance between bubble centres.

        Parameters
        ----------
        d : float
            Bubble separation.

        Returns
        -------
        self
        """
        self.d = float(d)
        return self

    def with_initial_condition(
        self,
        phi1_initial: Union[NDArray[np.float64], List[float]],
    ) -> Self:
        """
        Set the initial field configuration.

        Must be called after ``with_grid`` so that the shape can be
        validated against ``(Ndim, n_z)``.

        Parameters
        ----------
        phi1_initial : array-like
            Initial field values.  Accepted shapes:
            - ``(n_z,)``       for a single-component field (Ndim=1)
            - ``(Ndim, n_z)``  for a multi-component field

        Returns
        -------
        self
        """
        if self.n_z is None:
            raise RuntimeError(
                "Call with_grid() before with_initial_condition() "
                "so the shape can be validated."
            )

        arr = np.array(phi1_initial, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # (1, n_z)

        if arr.shape != (self.Ndim, self.n_z):
            raise ValueError(
                f"phi1_initial has shape {arr.shape}, expected "
                f"({self.Ndim}, {self.n_z}) = (Ndim, n_z)."
            )

        self.phi1_initial = arr

        # Invalidate any previous evolution results
        self.phi1 = None
        self.s_grid = None
        self.n_s = None
        self.s_max = None

        return self

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from working version)
    # ------------------------------------------------------------------

    def _spatial_derivative(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        return _spatial_derivative_numba(phi, self.dz)

    def evolvepi(
        self, phi: NDArray[np.float64], pi: NDArray[np.float64], s: float, ds: float
    ) -> NDArray[np.float64]:
        phi_transposed = phi.T
        dV_dphi = self.potential.dV0(phi_transposed).T
        spatial_deriv = _spatial_derivative_numba(phi, self.dz)
        return _evolvepi_numba(phi, pi, dV_dphi, spatial_deriv, s, ds)

    def evolvepi_first_half_step(
        self, phi_initial: NDArray[np.float64], baby_steps: int = 20
    ) -> NDArray[np.float64]:
        pi = np.zeros_like(phi_initial, dtype=np.float64)
        baby_ds = 0.5 * self.ds / (baby_steps - 1)
        phi = phi_initial.copy()
        for i in range(1, baby_steps):
            pi = self.evolvepi(phi, pi, (i - 1) * baby_ds, baby_ds)
            phi += baby_ds * pi
        return pi

    # ------------------------------------------------------------------
    # Public API (unchanged from working version)
    # ------------------------------------------------------------------

    def _check_ready(self, method: str) -> None:
        """Raise a clear error if required configuration is missing."""
        missing = []
        if self.z_grid is None or self.ds is None:
            missing.append("with_grid(z_grid, ds)")
        if self.d is None:
            missing.append("with_distance(d)")
        if self.phi1_initial is None:
            missing.append("with_initial_condition(phi1_initial)")
        if missing:
            raise RuntimeError(
                f"{method}() called before required configuration. "
                f"Missing: {', '.join(missing)}."
            )

    def evolve(self, smax: float, history_interval: int = 1) -> NDArray[np.float64]:
        self._check_ready("evolve")

        n_steps = int(np.ceil(smax / self.ds))
        n_history = n_steps // history_interval
        n_steps = n_history * history_interval
        self.n_s = n_history + 1
        self.s_max = n_history * self.ds * history_interval
        self.history_interval = history_interval

        phi_init = self.phi1_initial.copy()
        self.phi1 = np.zeros((self.Ndim, self.n_s, self.n_z), dtype=np.float64)
        self.s_grid = np.linspace(0, self.s_max, self.n_s)
        self.phi1[:, 0, :] = phi_init

        pi = self.evolvepi_first_half_step(phi_init)
        phi = phi_init.copy()
        for i in range(1, n_steps + 1):
            if i > 1:
                pi = self.evolvepi(phi, pi, (i - 1) * self.ds, self.ds)
            phi += self.ds * pi
            if i % self.history_interval == 0:
                self.phi1[:, i // self.history_interval, :] = phi
        return self.phi1

    def calculate_energy_density(self, how_often: int = 10) -> NDArray[np.float64]:
        if self.phi1 is None:
            raise RuntimeError("Call evolve() before calculate_energy_density().")

        phiall = self.phi1
        n_s = phiall.shape[1]
        idx = np.arange(0, n_s - 2, how_often // self.history_interval)
        idx = idx[idx < n_s - 2]
        n_idx = len(idx)

        energy_density = np.zeros((n_idx, self.n_z - 1), dtype=np.float64)

        for i in range(n_idx):
            phi_diff_s = (phiall[:, idx[i] + 1, :-1] - phiall[:, idx[i], :-1]) / (
                self.ds * self.history_interval
            )
            kin_s = 0.5 * np.sum(phi_diff_s**2, axis=0)

            phi_diff_z = (phiall[:, idx[i], 1:] - phiall[:, idx[i], :-1]) / self.dz
            kin_z = 0.5 * np.sum(phi_diff_z**2, axis=0)

            phi_flat = phiall[:, idx[i], :-1].T
            pot = self.potential.V0(phi_flat.T)

            energy_density[i] = kin_s + kin_z + pot[0]

        self.energy_density = energy_density
        self.energy_how_often = how_often
        return energy_density

    def compute_phi_region2(self, bubble_type="half"):
        if self.phi1 is None:
            raise RuntimeError("Call evolve() before compute_phi_region2().")

        n_s, Ndim, n_z = self.n_s, self.Ndim, self.n_z
        d = self.d
        self.phi2 = np.zeros((Ndim, n_s, n_z))

        if bubble_type == "one":
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]
                phi0_interp = interp1d(
                    self.z_grid, phi0, kind="linear", fill_value="extrapolate"
                )
                S, Z = np.meshgrid(self.s_grid, self.z_grid, indexing="ij")
                R = np.sqrt(S**2 + Z**2)
                self.phi2[n, :, :] = phi0_interp(R)
        else:  # "half" case
            idx_zcenter = np.argmin(np.abs(self.z_grid))
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]

                phi_right = np.abs(phi0[idx_zcenter:])
                idx_phimid_right = np.argmax(phi_right) + idx_zcenter
                z_shifted_right = (
                    self.z_grid[idx_phimid_right:] - self.z_grid[idx_phimid_right]
                )
                z_shifted_right = np.concatenate(
                    (-z_shifted_right[::-1], z_shifted_right)
                )
                phi0_right = phi0[idx_phimid_right:]
                phi0_right = np.concatenate((phi0_right[::-1], phi0_right))
                phi0_interp_right = interp1d(
                    z_shifted_right,
                    phi0_right,
                    kind="linear",
                    fill_value=(0.0, 0.0),
                    bounds_error=False,
                )

                phi_left = np.abs(phi0[:idx_zcenter])
                idx_phimid_left = np.argmax(phi_left)
                z_shifted_left = (
                    self.z_grid[: idx_phimid_left + 1] - self.z_grid[idx_phimid_left]
                )
                z_shifted_left = np.concatenate(
                    (-z_shifted_left[::-1], z_shifted_left[1:])
                )
                phi0_left = phi0[: idx_phimid_left + 1]
                phi0_left = np.concatenate((phi0_left[::-1], phi0_left[1:]))
                phi0_interp_left = interp1d(
                    z_shifted_left,
                    phi0_left,
                    kind="linear",
                    fill_value=(0.0, 0.0),
                    bounds_error=False,
                )

                S, Z = np.meshgrid(self.s_grid, self.z_grid, indexing="ij")
                r_right = np.sqrt(S**2 + (Z - d / 2) ** 2)
                r_left = np.sqrt(S**2 + (Z + d / 2) ** 2)
                self.phi2[n, :, :] = phi0_interp_right(r_right) + phi0_interp_left(
                    r_left
                )

from __future__ import annotations

from typing import List, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .generic_potential import GenericPotential

# ---------------------------------------------------------------------------
# Numba kernels (module-level, unchanged)
# ---------------------------------------------------------------------------


@njit
def _spatial_derivative_numba(phi: np.ndarray, dz: float) -> np.ndarray:
    """Second-order central difference with reflect padding.

    Equivalent to::

        phi_padded = np.pad(phi, ((0, 0), (1, 1)), mode="reflect")
        return (phi_padded[:, :-2] - 2*phi_padded[:, 1:-1] + phi_padded[:, 2:]) / dz**2

    Reflect padding conventions:

    * left ghost  = ``phi[:, 1]``   (index 1 mirrors index 0)
    * right ghost = ``phi[:, -2]``  (index -2 mirrors index -1)

    Boundary stencils therefore simplify to:

    * ``result[:, 0]  = (2*phi[:,1]   - 2*phi[:,0])  / dz²``
    * ``result[:, -1] = (2*phi[:,-2]  - 2*phi[:,-1]) / dz²``
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
    """Numba kernel for the momentum (π) update.

    Computes::

        damping = 1 - 2*ds / (s + ds)
        forcing = s*ds / (s + ds) * (-dV/dφ + ∇²φ)
        π_new   = π * damping + forcing
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


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class PDEBubbleSolverConfig:
    """Fluent builder that accumulates configuration for a :class:`PDEBubbleSolver`.

    Call the ``with_*`` methods in any order, then call :meth:`build` to obtain
    a fully configured, ready-to-use :class:`PDEBubbleSolver`.  No field inside
    the solver itself is ``Optional``; all required values are validated here,
    before the solver is constructed.

    Typical usage::

        solver = (
            PDEBubbleSolverConfig(potential)
            .with_grid(z_grid, ds)
            .with_distance(d)
            .with_initial_condition(phi1_initial)
            .build()
        )
        phi1 = solver.evolve(smax)

    Parameters
    ----------
    potential:
        Potential object exposing ``V0``, ``dV0``, and ``Ndim``.
    """

    def __init__(self, potential: GenericPotential) -> None:
        self._potential: GenericPotential = potential

        # Accumulated config — all optional until build() is called
        self._z_grid: NDArray[np.float64] | None = None
        self._ds: float | None = None
        self._dz: float | None = None
        self._n_z: int | None = None
        self._d: float | None = None
        self._phi1_initial: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def with_grid(
        self,
        z_grid: Union[NDArray[np.float64], List[float]],
        ds: float,
    ) -> PDEBubbleSolverConfig:
        """Set the spatial grid and time step.

        Parameters
        ----------
        z_grid:
            Uniformly-spaced spatial coordinates.
        ds:
            Time step.  Must satisfy the CFL condition ``ds <= dz``.
            Violations raise ``ValueError`` immediately rather than
            being silently clamped.

        Returns
        -------
        PDEBubbleSolverConfig
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``z_grid`` has fewer than two points, or if the CFL
            condition ``ds <= dz`` is violated.
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

        self._z_grid = z_grid
        self._ds = float(ds)
        self._dz = dz
        self._n_z = len(z_grid)

        # Invalidate initial condition if grid shape changed
        self._phi1_initial = None
        return self

    def with_distance(self, d: float) -> PDEBubbleSolverConfig:
        """Set the initial centre-to-centre distance between the two bubbles.

        Parameters
        ----------
        d:
            Bubble separation (same units as ``z_grid``).

        Returns
        -------
        PDEBubbleSolverConfig
            ``self``, for method chaining.
        """
        self._d = float(d)
        return self

    def with_initial_condition(
        self,
        phi1_initial: Union[NDArray[np.float64], List[float]],
    ) -> PDEBubbleSolverConfig:
        """Set the initial field configuration.

        Must be called after :meth:`with_grid` so that the array shape
        can be validated against ``(Ndim, n_z)``.

        Parameters
        ----------
        phi1_initial:
            Initial field values.  Accepted shapes:

            * ``(n_z,)``       for a single-component field (``Ndim = 1``)
            * ``(Ndim, n_z)``  for a multi-component field

        Returns
        -------
        PDEBubbleSolverConfig
            ``self``, for method chaining.

        Raises
        ------
        RuntimeError
            If :meth:`with_grid` has not been called yet.
        ValueError
            If the array shape does not match ``(Ndim, n_z)``.
        """
        if self._n_z is None:
            raise RuntimeError(
                "Call with_grid() before with_initial_condition() "
                "so the array shape can be validated."
            )

        arr = np.array(phi1_initial, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # promote (n_z,) → (1, n_z)

        ndim = self._potential.Ndim
        if arr.shape != (ndim, self._n_z):
            raise ValueError(
                f"phi1_initial has shape {arr.shape}, "
                f"expected ({ndim}, {self._n_z}) = (Ndim, n_z)."
            )

        self._phi1_initial = arr
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> PDEBubbleSolver:
        """Validate that all required fields are set and construct the solver.

        Returns
        -------
        PDEBubbleSolver
            A fully configured, immutable solver instance.

        Raises
        ------
        RuntimeError
            If any of :meth:`with_grid`, :meth:`with_distance`, or
            :meth:`with_initial_condition` have not been called.
        """
        if (
            self._z_grid is None
            or self._ds is None
            or self._dz is None
            or self._n_z is None
        ):
            raise RuntimeError(
                "Cannot build PDEBubbleSolver: missing configuration from with_grid(z_grid, ds)"
            )
        if self._d is None:
            raise RuntimeError(
                "Cannot build PDEBubbleSolver: missing configuration from with_initial_condition(phi1_initial)"
            )
        if self._phi1_initial is None:
            raise RuntimeError(
                "Cannot build PDEBubbleSolver: missing configuration from with_initial_condition(phi1_initial)"
            )

        return PDEBubbleSolver(
            potential=self._potential,
            z_grid=self._z_grid,
            ds=self._ds,
            dz=self._dz,
            n_z=self._n_z,
            d=self._d,
            phi1_initial=self._phi1_initial,
        )


# ---------------------------------------------------------------------------
# Solver  (no Optional fields)
# ---------------------------------------------------------------------------


class PDEBubbleSolver:
    """Solver for coupled nonlinear wave equations describing bubble dynamics.

    Do **not** instantiate this class directly.  Use :class:`PDEBubbleSolverConfig`
    to accumulate configuration and call :meth:`PDEBubbleSolverConfig.build` to
    obtain a ready-to-use instance::

        solver = (
            PDEBubbleSolverConfig(potential)
            .with_grid(z_grid, ds)
            .with_distance(d)
            .with_initial_condition(phi1_initial)
            .build()
        )
        phi1 = solver.evolve(smax)

    All constructor arguments are required and non-optional; the builder
    guarantees they are valid before this class is ever instantiated.

    Parameters
    ----------
    potential:
        Potential object exposing ``V0``, ``dV0``, and ``Ndim``.
    z_grid:
        Uniformly-spaced spatial coordinates, shape ``(n_z,)``.
    ds:
        Time step (CFL-valid, i.e. ``ds <= dz``).
    dz:
        Spatial step derived from ``z_grid``.
    n_z:
        Number of spatial grid points.
    d:
        Initial centre-to-centre distance between the two bubbles.
    phi1_initial:
        Initial field configuration, shape ``(Ndim, n_z)``.
    """

    def __init__(
        self,
        potential: GenericPotential,
        z_grid: NDArray[np.float64],
        ds: float,
        dz: float,
        n_z: int,
        d: float,
        phi1_initial: NDArray[np.float64],
    ) -> None:
        # Core configuration — all non-optional
        self.potential: GenericPotential = potential
        self.Ndim: int = potential.Ndim
        self.z_grid: NDArray[np.float64] = z_grid
        self.ds: float = ds
        self.dz: float = dz
        self.n_z: int = n_z
        self.d: float = d
        self.phi1_initial: NDArray[np.float64] = phi1_initial

        # Evolution results — populated by evolve() and post-processing methods
        self.history_interval: int | None = None
        self.phi1: NDArray[np.float64] | None = None
        self.phi2: NDArray[np.float64] | None = None
        self.n_s: int | None = None
        self.s_max: float = np.nan
        self.s_grid: NDArray[np.float64] | None = None
        self.energy_density: NDArray[np.float64] | None = None
        self.energy_how_often: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def evolvepi(
        self,
        phi: NDArray[np.float64],
        pi: NDArray[np.float64],
        s: float,
        ds: float,
    ) -> NDArray[np.float64]:
        """Advance the conjugate momentum π by one time step.

        Parameters
        ----------
        phi:
            Current field configuration, shape ``(Ndim, n_z)``.
        pi:
            Current momentum configuration, shape ``(Ndim, n_z)``.
        s:
            Current conformal time.
        ds:
            Time step to advance by.

        Returns
        -------
        NDArray of shape ``(Ndim, n_z)``
            Updated momentum field.
        """
        dV_dphi = self.potential.dV0(phi.T).T
        spatial_deriv = _spatial_derivative_numba(phi, self.dz)
        return _evolvepi_numba(phi, pi, dV_dphi, spatial_deriv, s, ds)

    def evolvepi_first_half_step(
        self,
        phi_initial: NDArray[np.float64],
        baby_steps: int = 20,
    ) -> NDArray[np.float64]:
        """Bootstrap π from rest using small sub-steps over the first half-step.

        Starting from ``π = 0``, this method integrates the equation of
        motion over ``baby_steps`` sub-steps each of size
        ``ds/2 / (baby_steps - 1)``, accumulating both φ and π.  The result
        is a consistent momentum field at ``s = ds/2``, suitable as the
        starting value for the main leapfrog loop.

        Parameters
        ----------
        phi_initial:
            Initial field configuration, shape ``(Ndim, n_z)``.
        baby_steps:
            Number of sub-steps used in the bootstrap.

        Returns
        -------
        NDArray of shape ``(Ndim, n_z)``
            Bootstrapped momentum field at ``s = ds/2``.
        """
        pi = np.zeros_like(phi_initial, dtype=np.float64)
        baby_ds = 0.5 * self.ds / (baby_steps - 1)
        phi = phi_initial.copy()
        for i in range(1, baby_steps):
            pi = self.evolvepi(phi, pi, (i - 1) * baby_ds, baby_ds)
            phi += baby_ds * pi
        return pi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        smax: float,
        history_interval: int = 1,
    ) -> NDArray[np.float64]:
        """Integrate the field equations from ``s = 0`` to ``s = smax``.

        Uses a leapfrog (Störmer–Verlet) scheme.  Field snapshots are stored
        every *history_interval* steps; the full trajectory is available
        afterwards via ``self.phi1``.

        Parameters
        ----------
        smax:
            Maximum conformal time to integrate to.
        history_interval:
            Store a snapshot every this many time steps.  ``1`` stores
            every step; larger values reduce memory usage.

        Returns
        -------
        NDArray of shape ``(Ndim, n_history + 1, n_z)``
            Full field trajectory, including the initial condition at index 0.
        """
        n_steps = int(np.ceil(smax / self.ds))
        n_history = n_steps // history_interval
        n_steps = n_history * history_interval
        self.n_s = n_history + 1
        self.s_max = n_history * self.ds * history_interval
        self.history_interval = history_interval

        phi_init = self.phi1_initial.copy()
        try:
            self.phi1 = np.full(
                (self.Ndim, self.n_s, self.n_z), np.nan, dtype=np.float64
            )
        except MemoryError:
            n_bytes = self.Ndim * self.n_s * self.n_z * 8
            raise MemoryError(
                f"Insufficient memory to allocate phi1 of shape "
                f"({self.Ndim}, {self.n_s}, {self.n_z}) "
                f"({n_bytes / 1e9:.2f} GB required)."
            )
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

    def calculate_energy_density(
        self,
        how_often: int = 10,
    ) -> NDArray[np.float64]:
        """Compute the energy density at regular intervals of the stored trajectory.

        The energy density at each stored snapshot index *i* is::

            ε = ½(∂φ/∂s)² + ½(∂φ/∂z)² + V(φ)

        Time derivatives are estimated by first-order finite differences
        between adjacent snapshots; spatial derivatives use first-order
        differences between adjacent grid points.  Results are defined on
        the mid-points of the spatial grid (length ``n_z - 1``).

        Parameters
        ----------
        how_often:
            Compute the energy density every this many *original* time steps.

        Returns
        -------
        NDArray of shape ``(n_idx, n_z - 1)``
            Energy density at each selected time index and each spatial
            mid-point.

        Raises
        ------
        RuntimeError
            If :meth:`evolve` has not been called yet.
        """
        if self.phi1 is None or self.history_interval is None:
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

    def compute_phi_region2(
        self,
        bubble_type: str = "half",
    ) -> NDArray[np.float64]:
        """Reconstruct the two-bubble field configuration in region 2.

        Using the initial profile from region 1 (stored in ``phi1_initial``
        via ``self.phi1[:, 0, :]``), this method extrapolates the field into
        the space-like region between the two bubbles by assuming each bubble
        expands spherically.

        Two modes are supported:

        ``"one"``
            A single spherical bubble centred at the origin.  The radial
            profile is read directly from the initial condition and mapped via
            ``R = sqrt(s² + z²)``.

        ``"half"`` *(default)*
            Two bubbles at ``z = ±d/2``.  For each bubble the profile is
            mirrored around its wall peak and mapped via
            ``R = sqrt(s² + (z ∓ d/2)²)``.  The two contributions are summed.

        Parameters
        ----------
        bubble_type:
            ``"one"`` or ``"half"``.

        Returns
        -------
        NDArray of shape ``(Ndim, n_s, n_z)``
            Reconstructed field stored in ``self.phi2``.

        Raises
        ------
        RuntimeError
            If :meth:`evolve` has not been called yet.
        """
        if (
            self.phi1 is None
            or self.history_interval is None
            or self.n_s is None
            or self.s_grid is None
        ):
            raise RuntimeError("Call evolve() before compute_phi_region2().")

        n_s, Ndim, n_z = self.n_s, self.Ndim, self.n_z
        d = self.d

        try:
            self.phi2 = np.full((Ndim, n_s, n_z), np.nan, dtype=np.float64)
        except MemoryError:
            n_bytes = Ndim * n_s * n_z * 8
            raise MemoryError(
                f"Insufficient memory to allocate phi2 of shape "
                f"({Ndim}, {n_s}, {n_z}) "
                f"({n_bytes / 1e9:.2f} GB required)."
            )

        if bubble_type == "one":
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]
                phi0_interp = interp1d(
                    self.z_grid, phi0, kind="linear", fill_value="extrapolate"
                )
                S, Z = np.meshgrid(self.s_grid, self.z_grid, indexing="ij")
                R = np.sqrt(S**2 + Z**2)
                self.phi2[n, :, :] = phi0_interp(R)

        else:  # "half" — two bubbles at ±d/2
            idx_zcenter = np.argmin(np.abs(self.z_grid))
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]

                # --- right bubble (centred at +d/2) ---
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

                # --- left bubble (centred at -d/2) ---
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

        return self.phi2

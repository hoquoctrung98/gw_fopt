#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <hoquoctrung98@gmail.com>
"""

# import matplotlib.pyplot as plt
# import numpy as np
# from cosmoTransitions import pathDeformation
# from scipy.interpolate import interp1d


# class LatticeSetup:
#     def __init__(self, potential):
#         self.potential = potential
#         self.alpha = 3
#         self.phi_tv = None
#         self.phi_fv = None
#         self.gamma = None
#         self.d = None

#     def with_tunnelling_phi(self, phi_tv, phi_fv):
#         """Set the phi_tv and phi_fv for tunneling with validation."""
#         self.phi_tv = np.array(phi_tv)  # Shape: (Ndim,)
#         self.phi_fv = np.array(phi_fv)  # Shape: (Ndim,)
#         if self.phi_tv.shape != (self.potential.Ndim,) or self.phi_fv.shape != (
#             self.potential.Ndim,
#         ):
#             raise ValueError(
#                 f"phi_tv and phi_fv must have shape ({self.potential.Ndim},)"
#             )
#         return self

#     def with_profiles(self, npoints=1000):
#         """Compute the tunneling profiles for all fields."""
#         if self.phi_tv is None or self.phi_fv is None:
#             raise ValueError(
#                 "phi_tv and phi_fv must be set using with_tunnelling_phi before finding profiles."
#             )

#         profiles = pathDeformation.fullTunneling(
#             [self.phi_tv, self.phi_fv],
#             self.potential.V0,
#             self.potential.dV0,
#             tunneling_init_params={"alpha": self.alpha},
#             tunneling_findProfile_params={"npoints": npoints},
#             deformation_deform_params={"verbose": False},
#         )
#         return profiles.Phi, profiles.profile1D.R

#     def compute_radii(self, npoints=1000):
#         """Compute the radius for each field profile, returning an array of size Ndim."""
#         phi, r = self.with_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

#         # Compute the norm of each field component at the center
#         phi00 = np.array(
#             [phi[0, n] for n in range(self.potential.Ndim)]
#         )  # Shape: (Ndim,)

#         # Compute radii based on where the norm is half the center value for each field
#         radii = np.zeros(self.potential.Ndim)
#         for n in range(self.potential.Ndim):
#             idx = (np.abs(phi[:, n] - phi00[n] / 2)).argmin()
#             radii[n] = r[idx]

#         return radii

#     def compute_inner_outer_radii(self, npoints=1000):
#         """Compute the inner and outer radii for each field profile, returning two arrays of size Ndim."""
#         phi, r = self.with_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

#         # Compute the norm of each field component at the center
#         phi00 = np.array(
#             [phi[0, n] for n in range(self.potential.Ndim)]
#         )  # Shape: (Ndim,)

#         # Compute inner and outer radii
#         inner_radii = np.zeros(self.potential.Ndim)
#         outer_radii = np.zeros(self.potential.Ndim)

#         for n in range(self.potential.Ndim):
#             outer_radii[n] = r[
#                 (np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(1 / 2)))).argmin()
#             ]
#             inner_radii[n] = r[
#                 (np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(-1 / 2)))).argmin()
#             ]

#         return inner_radii, outer_radii

#     def interpolate_profiles(self, z, z0, npoints=1000, decay_rate=0.5):
#         """
#         Interpolate the field profiles onto the z-axis, centered at z0, with exponential decay to phi_metaMin.

#         Parameters:
#         - z: Real array (1D) of z values where the profile is interpolated
#         - z0: The center of the profile on the z-axis
#         - npoints: Number of points to use in with_profiles (default 1000)
#         - decay_rate: Rate of exponential decay towards phi_metaMin (default 0.5)

#         Returns:
#         - phi_z: Interpolated profiles with shape (len(z), Ndim)
#         """
#         # Get the radial profiles
#         phi, r = self.with_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

#         # Compute the radial distance from the center z0
#         r_new = np.abs(z - z0)  # Shape: (len(z),)

#         # Maximum radial distance in the original profile
#         r_max = r[-1]

#         # Interpolate each field component
#         phi_z = np.zeros((len(z), self.potential.Ndim))  # Shape: (len(z), Ndim)
#         for n in range(self.potential.Ndim):
#             # Create an interpolator for the n-th field component within the radial range
#             interpolator = interp1d(
#                 r, phi[:, n], kind="cubic", fill_value="extrapolate", bounds_error=False
#             )

#             # Interpolate within the range [0, r_max]
#             phi_z[:, n] = interpolator(r_new)

#             # Apply exponential decay towards phi_metaMin for points outside r_max
#             mask_outside = r_new > r_max
#             if np.any(mask_outside):
#                 # Compute the distance beyond r_max
#                 r_excess = r_new[mask_outside] - r_max
#                 # Value at the boundary (r_max)
#                 phi_boundary = phi[-1, n]
#                 # Exponential decay from phi_boundary to phi_metaMin
#                 decay = (phi_boundary - self.phi_fv[n]) * np.exp(-decay_rate * r_excess)
#                 phi_z[mask_outside, n] = self.phi_fv[n] + decay

#         return phi_z

#     def plot_profiles(self, ax, npoints=1000, show_radii=True, kwargs_list=None):
#         """Plot all field profiles with vertical dashed lines at the radii and filled regions between inner and outer radii."""
#         if kwargs_list is not None:
#             if len(kwargs_list) != self.potential.Ndim:
#                 raise ValueError(
#                     f"kwargs_list should have length {self.potential.Ndim}"
#                 )
#         else:
#             kwargs_list = [{} for _ in range(self.potential.Ndim)]

#         phi, r = self.with_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

#         # Compute inner and outer radii for each field
#         inner_radii, outer_radii = self.compute_inner_outer_radii(npoints)

#         # Plot each field profile and fill between inner and outer radii
#         for n in range(self.potential.Ndim):
#             # Plot the profile and get the actual line object
#             (line,) = ax.plot(r, phi[:, n], **kwargs_list[n])
#             line_color = line.get_color()

#             if show_radii:
#                 # Draw vertical dashed lines at inner and outer radii in the same color
#                 ax.axvline(
#                     x=inner_radii[n],
#                     color=line_color,
#                     linestyle="--",
#                     alpha=0.5,
#                 )
#                 ax.axvline(
#                     x=outer_radii[n],
#                     color=line_color,
#                     linestyle="--",
#                     alpha=0.5,
#                 )

#             # Fill the region between inner and outer radii with transparency
#             ax.fill_between(
#                 r,
#                 0,
#                 phi[:, n],
#                 where=(r >= inner_radii[n]) & (r <= outer_radii[n]),
#                 color=line_color,
#                 alpha=0.3,
#                 interpolate=True,
#             )

#     def with_boost(self, gamma):
#         self.gamma = gamma
#         self.d = None
#         # d = 2 * self.gamma * self.r0
#         # self.d = d
#         return self

#     def with_distance(self, d):
#         self.d = d
#         self.gamma = None
#         # self.gamma = self.d / (2 * self.r0)
#         return self

#     def two_bubbles(
#         self,
#         layout="positive half",
#         npoints=1000,
#         dz_max=0.1,
#         scale_dz=1.0,
#         scale_z=3.0,
#     ):
#         """
#         Compute the z_grid and interpolated profiles for two bubbles configuration.

#         Parameters:
#         - layout: "full" for two bubbles at ±d, "positive half" for one bubble at d/2,
#                 "negative half" for one bubble at -d/2 (default "positive half")
#         - npoints: Number of points to use in with_profiles (default 1000)

#         Returns:
#         - z_grid: Array of z values (range depends on layout)
#         - phi_z: Interpolated profiles (sum of two for "full", single for "positive half" or "negative half")
#         - d: Distance between bubble centers (for "full") or twice the center position (for "half" types)
#         """
#         # Compute the radii and inner/outer radii
#         radii = self.compute_radii(npoints)
#         inner_radii, outer_radii = self.compute_inner_outer_radii(npoints)

#         # Largest profile radius
#         r0 = np.max(radii)

#         d = 0.0
#         # Distance between the two bubbles (or reference distance for "half" types)
#         if self.gamma is not None:
#             d = 2 * self.gamma * r0
#             self.d = d
#         elif (self.gamma is None) and (self.d is not None):
#             self.gamma = self.d / (2 * r0)
#             d = self.d
#         else:
#             raise ValueError("Either gamma or d must be set.")

#         # Compute wall width at collision (using the field with the largest r0)
#         idx_max_r0 = np.argmax(radii)
#         r_in = inner_radii[idx_max_r0]
#         r_out = outer_radii[idx_max_r0]
#         l_wall_hit = np.sqrt(r_out**2 + (d / 2) ** 2 - r0**2) - np.sqrt(
#             r_in**2 + (d / 2) ** 2 - r0**2
#         )

#         # Compute dz for the lattice
#         dz = l_wall_hit / 10
#         dz *= scale_dz
#         dz = min(dz, dz_max)

#         if layout == "full":
#             # Create z_grid from -3d to 3d with step dz
#             z_grid = np.arange(-scale_z * d, scale_z * d + dz, dz)
#             # Interpolate profiles centered at z0 = -d and z0 = d, then sum them
#             phi_z1 = self.interpolate_profiles(z_grid, z0=-d / 2, npoints=npoints)
#             phi_z2 = self.interpolate_profiles(z_grid, z0=d / 2, npoints=npoints)
#             phi_z = phi_z1 + phi_z2
#         elif layout == "positive half":
#             # Create z_grid from 0 to 3d with step dz
#             z_grid = np.arange(0, scale_z * d + dz, dz)
#             # Interpolate a single profile centered at z0 = d/2
#             phi_z = self.interpolate_profiles(z_grid, z0=d / 2, npoints=npoints)
#         elif layout == "negative half":
#             # Create z_grid from -3d to 0 with step dz
#             z_grid = np.arange(-scale_z * d, 0 + dz, dz)
#             # Interpolate a single profile centered at z0 = -d/2
#             phi_z = self.interpolate_profiles(z_grid, z0=-d / 2, npoints=npoints)
#         else:
#             raise ValueError(
#                 "layout must be either 'full', 'positive half', or 'negative half'"
#             )

#         return z_grid, phi_z, d

from __future__ import annotations

import numpy as np
from cosmoTransitions import pathDeformation
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .generic_potential import GenericPotential


class LatticeSetup:
    """
    Constructs a 1D spatial lattice for simulating collisions between two
    cosmological vacuum-decay bubbles.

    The typical usage pattern is a fluent method chain::

        setup = (
            LatticeSetup(potential)
            .with_tunnelling_phi(phi_tv, phi_fv)
            .with_profiles(npoints=1000)
            .with_boost(gamma=2.0)          # or .with_distance(d=10.0)
        )
        z_grid, phi_z, d = setup.two_bubbles(layout="full")

    Attributes
    ----------
    potential : GenericPotential
        Potential object exposing ``V0``, ``dV0``, and ``Ndim``.
    phi_tv : NDArray[np.float64] | None
        Field values at the true vacuum, shape ``(Ndim,)``.
    phi_fv : NDArray[np.float64] | None
        Field values at the false vacuum, shape ``(Ndim,)``.
    alpha : int
        Spatial-dimension parameter passed to *CosmoTransitions*
        (``3`` = O(3)-symmetric spherical bubble in 3-D space).
    profiles : object | None
        Raw output of ``pathDeformation.fullTunneling`` after
        ``with_profiles`` has been called.
    radii : NDArray[np.float64] | None
        Characteristic half-maximum radius for each field, shape ``(Ndim,)``.
    inner_radii : NDArray[np.float64] | None
        Inner edge of the bubble wall for each field, shape ``(Ndim,)``.
    outer_radii : NDArray[np.float64] | None
        Outer edge of the bubble wall for each field, shape ``(Ndim,)``.
    gamma : float | None
        Lorentz boost factor at collision.  Mutually derived with ``d``.
    d : float | None
        Separation between the two bubble centres.  Mutually derived with
        ``gamma``.
    """

    def __init__(self, potential: GenericPotential) -> None:
        self.potential = potential
        self.phi_tv: NDArray[np.float64] | None = None
        self.phi_fv: NDArray[np.float64] | None = None
        self.alpha: int = 3

        # Populated by with_profiles()
        self.profiles: object | None = None
        self.radii: NDArray[np.float64] | None = None
        self.inner_radii: NDArray[np.float64] | None = None
        self.outer_radii: NDArray[np.float64] | None = None

        # Populated by with_boost() or with_distance()
        self.gamma: float | None = None
        self.d: float | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def with_tunnelling_phi(
        self,
        phi_tv: list[float] | NDArray[np.float64],
        phi_fv: list[float] | NDArray[np.float64],
    ) -> LatticeSetup:
        """Set the field values at the true and false vacua.

        Parameters
        ----------
        phi_tv:
            Field values at the *true* vacuum (bubble interior),
            shape ``(Ndim,)``.
        phi_fv:
            Field values at the *false* vacuum (ambient background),
            shape ``(Ndim,)``.

        Returns
        -------
        LatticeSetup
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If either array does not have shape ``(Ndim,)``.
        """
        self.phi_tv = np.array(phi_tv, dtype=float)
        self.phi_fv = np.array(phi_fv, dtype=float)
        expected = (self.potential.Ndim,)
        if self.phi_tv.shape != expected or self.phi_fv.shape != expected:
            raise ValueError(
                f"phi_tv and phi_fv must have shape ({self.potential.Ndim},)"
            )
        return self

    def with_profiles(self, npoints: int = 1000) -> LatticeSetup:
        """Solve the bounce equation and cache the resulting field profiles.

        This method uses *CosmoTransitions* to compute the O(3)-symmetric
        tunneling solution between ``phi_fv`` and ``phi_tv``.  After the
        profiles are computed, ``compute_radii`` and
        ``compute_inner_outer_radii`` are called automatically and their
        results stored in ``self.radii``, ``self.inner_radii``, and
        ``self.outer_radii``.

        Parameters
        ----------
        npoints:
            Number of radial grid points used by the profile solver.

        Returns
        -------
        LatticeSetup
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``with_tunnelling_phi`` has not been called first.
        """
        if self.phi_tv is None or self.phi_fv is None:
            raise ValueError(
                "phi_tv and phi_fv must be set via with_tunnelling_phi() "
                "before calling with_profiles()."
            )

        self.profiles = pathDeformation.fullTunneling(
            [self.phi_tv, self.phi_fv],
            self.potential.V0,
            self.potential.dV0,
            tunneling_init_params={"alpha": self.alpha},
            tunneling_findProfile_params={"npoints": npoints},
            deformation_deform_params={"verbose": False},
        )

        # Cache derived geometry so downstream methods need no npoints arg.
        self.radii = self._compute_radii()
        self.inner_radii, self.outer_radii = self._compute_inner_outer_radii()

        return self

    # ------------------------------------------------------------------
    # Geometry setters — require profiles to be available
    # ------------------------------------------------------------------

    def with_boost(self, gamma: float) -> LatticeSetup:
        """Set the collision geometry via the Lorentz boost factor.

        The bubble-centre separation is derived as ``d = 2 * gamma * r0``,
        where ``r0`` is the largest characteristic radius across all fields.

        Parameters
        ----------
        gamma:
            Lorentz boost factor of the bubble wall at the moment of
            collision (dimensionless, ``>= 1``).

        Returns
        -------
        LatticeSetup
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``with_profiles()`` has not been called first.
        """
        if self.radii is None:
            raise ValueError(
                "Radii are not available.  Call with_profiles() before with_boost()."
            )
        r0: float = float(np.max(self.radii))
        self.gamma = gamma
        self.d = 2.0 * gamma * r0
        return self

    def with_distance(self, d: float) -> LatticeSetup:
        """Set the collision geometry via the centre-to-centre separation.

        The Lorentz boost factor is back-computed as
        ``gamma = d / (2 * r0)``.

        Parameters
        ----------
        d:
            Distance between the two bubble centres (in field-space units).

        Returns
        -------
        LatticeSetup
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``with_profiles()`` has not been called first.
        """
        if self.radii is None:
            raise ValueError(
                "Radii are not available.  Call with_profiles() before with_distance()."
            )
        r0: float = float(np.max(self.radii))
        self.d = d
        self.gamma = d / (2.0 * r0)
        return self

    # ------------------------------------------------------------------
    # Internal geometry helpers (no longer need npoints)
    # ------------------------------------------------------------------

    def _get_phi_r(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the cached ``(Phi, R)`` arrays from the stored profiles.

        Returns
        -------
        phi : NDArray of shape ``(npoints, Ndim)``
            Field values along the radial profile.
        r : NDArray of shape ``(npoints,)``
            Corresponding radial coordinates.

        Raises
        ------
        RuntimeError
            If ``with_profiles()`` has not been called yet.
        """
        if self.profiles is None:
            raise RuntimeError("No profiles cached.  Call with_profiles() first.")
        return self.profiles.Phi, self.profiles.profile1D.R

    def _compute_radii(self) -> NDArray[np.float64]:
        """Compute the characteristic (half-maximum) radius for each field.

        For each field component *n*, the radius is defined as the point
        where ``phi[:, n]`` first equals half its central value
        ``phi[0, n] / 2``.

        Returns
        -------
        NDArray of shape ``(Ndim,)``
            Half-maximum radii for each field component.
        """
        phi, r = self._get_phi_r()
        phi0 = phi[0, :]  # central values, shape (Ndim,)

        radii = np.zeros(self.potential.Ndim)
        for n in range(self.potential.Ndim):
            idx = int(np.abs(phi[:, n] - phi0[n] / 2.0).argmin())
            radii[n] = r[idx]
        return radii

    def _compute_inner_outer_radii(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the inner and outer edges of the bubble wall for each field.

        The wall is defined by the tanh-based thresholds:

        * **outer** radius: where ``phi[:, n]`` equals
          ``phi0[n]/2 * (1 - tanh(+1/2))``  (≈ 0.18 × phi0[n]/2)
        * **inner** radius: where ``phi[:, n]`` equals
          ``phi0[n]/2 * (1 - tanh(-1/2))``  (≈ 0.82 × phi0[n]/2)

        Returns
        -------
        inner_radii : NDArray of shape ``(Ndim,)``
        outer_radii : NDArray of shape ``(Ndim,)``
        """
        phi, r = self._get_phi_r()
        phi0 = phi[0, :]  # shape (Ndim,)

        inner_radii = np.zeros(self.potential.Ndim)
        outer_radii = np.zeros(self.potential.Ndim)

        for n in range(self.potential.Ndim):
            outer_radii[n] = r[
                np.abs(phi[:, n] - phi0[n] / 2.0 * (1 - np.tanh(0.5))).argmin()
            ]
            inner_radii[n] = r[
                np.abs(phi[:, n] - phi0[n] / 2.0 * (1 - np.tanh(-0.5))).argmin()
            ]

        return inner_radii, outer_radii

    # ------------------------------------------------------------------
    # Profile interpolation
    # ------------------------------------------------------------------

    def interpolate_profiles(
        self,
        z: NDArray[np.float64],
        z0: float,
        decay_rate: float = 0.5,
    ) -> NDArray[np.float64]:
        """Map the radial bounce profile onto a 1-D spatial axis.

        The radial coordinate is converted to ``r_new = |z - z0|``.
        Within the computed radial range the profile is interpolated
        cubically; beyond it the field decays exponentially toward the
        false-vacuum value ``phi_fv``.

        Parameters
        ----------
        z:
            1-D array of spatial positions along the collision axis.
        z0:
            Centre of the bubble on the *z*-axis.
        decay_rate:
            Exponential-decay rate controlling how quickly the field
            relaxes to ``phi_fv`` outside the profile's radial range.

        Returns
        -------
        NDArray of shape ``(len(z), Ndim)``
            Interpolated field values at each *z* position.
        """
        if (self.phi_fv is None) or (self.phi_tv is None):
            raise ValueError(
                "Tunnelling points of the potential are not initialized. Call with_tunnelling_phi before interpolate_profiles()"
            )

        phi, r = self._get_phi_r()
        r_new: NDArray[np.float64] = np.abs(z - z0)
        r_max: float = float(r[-1])

        phi_z = np.zeros((len(z), self.potential.Ndim))
        for n in range(self.potential.Ndim):
            interpolator = interp1d(
                r,
                phi[:, n],
                kind="cubic",
                fill_value="extrapolate",
                bounds_error=False,
            )
            phi_z[:, n] = interpolator(r_new)

            # Exponential decay to phi_fv beyond the profile boundary
            mask_outside = r_new > r_max
            if np.any(mask_outside):
                r_excess = r_new[mask_outside] - r_max
                phi_boundary = float(phi[-1, n])
                decay = (phi_boundary - self.phi_fv[n]) * np.exp(-decay_rate * r_excess)
                phi_z[mask_outside, n] = self.phi_fv[n] + decay

        return phi_z

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_profiles(
        self,
        ax: Axes,
        show_radii: bool = True,
        kwargs_list: list[dict] | None = None,
    ) -> None:
        """Plot all field profiles, optionally annotating the wall region.

        For each field component a line is drawn showing ``phi(r)`` vs
        ``r``.  When *show_radii* is ``True``, dashed vertical lines mark
        the inner and outer wall radii and the region between them is
        shaded semi-transparently.

        Parameters
        ----------
        ax:
            Matplotlib ``Axes`` object on which to draw.
        show_radii:
            Whether to add dashed wall-boundary lines and filled
            shading between the inner and outer radii.
        kwargs_list:
            List of ``Ndim`` keyword-argument dicts forwarded to
            ``ax.plot`` for each field component.  Pass ``None`` to
            use default styling.

        Raises
        ------
        ValueError
            If *kwargs_list* has a length other than ``Ndim``.
        RuntimeError
            If ``with_profiles()`` has not been called yet.
        """
        if kwargs_list is not None and len(kwargs_list) != self.potential.Ndim:
            raise ValueError(f"kwargs_list must have length {self.potential.Ndim}.")
        kwargs_list = kwargs_list or [{} for _ in range(self.potential.Ndim)]

        if (
            (self.inner_radii is None)
            or (self.outer_radii is None)
            or (self.radii is None)
        ):
            raise ValueError(
                "Bubble profiles are not initialized. Call with_profiles before two_bubbles()"
            )

        phi, r = self._get_phi_r()

        for n in range(self.potential.Ndim):
            (line,) = ax.plot(r, phi[:, n], **kwargs_list[n])
            color = line.get_color()

            if show_radii:
                ax.axvline(
                    x=self.inner_radii[n], color=color, linestyle="--", alpha=0.5
                )
                ax.axvline(
                    x=self.outer_radii[n], color=color, linestyle="--", alpha=0.5
                )

            ax.fill_between(
                r,
                0,
                phi[:, n],
                where=(r >= self.inner_radii[n]) & (r <= self.outer_radii[n]),
                color=color,
                alpha=0.3,
                interpolate=True,
            )

    # ------------------------------------------------------------------
    # Lattice assembly
    # ------------------------------------------------------------------

    def two_bubbles(
        self,
        layout: str = "positive half",
        dz_max: float = 0.1,
        scale_dz: float = 1.0,
        scale_z: float = 3.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Assemble a 1-D lattice containing two colliding bubble profiles.

        The method determines an appropriate grid spacing ``dz`` from the
        projected wall width at the moment of collision, then interpolates
        one or both bubble profiles onto the grid.

        The grid spacing is chosen so that the bubble wall contains at
        least ten points at the collision point::

            l_wall = sqrt(r_out² + (d/2)² - r0²) - sqrt(r_in² + (d/2)² - r0²)
            dz     = min(l_wall / 10 * scale_dz, dz_max)

        Parameters
        ----------
        layout:
            Spatial coverage and number of bubbles:

            * ``"full"``           – grid from ``-scale_z*d`` to
              ``+scale_z*d``; profiles at ``±d/2`` are *summed*.
            * ``"positive half"``  – grid from ``0`` to ``+scale_z*d``;
              single profile at ``+d/2``.
            * ``"negative half"``  – grid from ``-scale_z*d`` to ``0``;
              single profile at ``-d/2``.

        dz_max:
            Hard upper bound on the lattice spacing (in field-space units).
        scale_dz:
            Multiplicative scale factor applied to the raw ``dz`` estimate
            before the ``dz_max`` cap.
        scale_z:
            Controls the extent of the grid relative to *d*.

        Returns
        -------
        z_grid : NDArray of shape ``(N,)``
            Spatial coordinates of each lattice site.
        phi_z : NDArray of shape ``(N, Ndim)``
            Interpolated field values at each lattice site.
        d : float
            Centre-to-centre separation between the two bubbles.

        Raises
        ------
        ValueError
            If ``with_boost()`` or ``with_distance()`` has not been called,
            or if *layout* is not one of the accepted strings.
        RuntimeError
            If ``with_profiles()`` has not been called yet.
        """
        if self.d is None or self.gamma is None:
            raise ValueError(
                "Bubble geometry is not set.  Call with_boost() or "
                "with_distance() before two_bubbles()."
            )

        if (
            (self.inner_radii is None)
            or (self.outer_radii is None)
            or (self.radii is None)
        ):
            raise ValueError(
                "Bubble profiles are not initialized. Call with_profiles before two_bubbles()"
            )

        d: float = self.d
        r0: float = float(np.max(self.radii))
        idx_max_r0: int = int(np.argmax(self.radii))

        # Projected wall thickness at the collision point
        r_in = self.inner_radii[idx_max_r0]
        r_out = self.outer_radii[idx_max_r0]
        l_wall_hit = np.sqrt(r_out**2 + (d / 2) ** 2 - r0**2) - np.sqrt(
            r_in**2 + (d / 2) ** 2 - r0**2
        )

        dz: float = min(l_wall_hit / 10.0 * scale_dz, dz_max)

        if layout == "full":
            z_grid = np.arange(-scale_z * d, scale_z * d + dz, dz)
            phi_z = self.interpolate_profiles(
                z_grid, z0=-d / 2
            ) + self.interpolate_profiles(z_grid, z0=+d / 2)
        elif layout == "positive half":
            z_grid = np.arange(0.0, scale_z * d + dz, dz)
            phi_z = self.interpolate_profiles(z_grid, z0=d / 2)
        elif layout == "negative half":
            z_grid = np.arange(-scale_z * d, 0.0 + dz, dz)
            phi_z = self.interpolate_profiles(z_grid, z0=-d / 2)
        else:
            raise ValueError(
                "layout must be 'full', 'positive half', or 'negative half'."
            )

        return z_grid, phi_z, d

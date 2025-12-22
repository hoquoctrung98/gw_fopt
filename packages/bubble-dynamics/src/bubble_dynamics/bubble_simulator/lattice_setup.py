#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <hoquoctrung98@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np
from cosmoTransitions import pathDeformation
from scipy.interpolate import interp1d


class LatticeSetup:
    def __init__(self, potential):
        self.potential = potential
        self.phi_tv = None  # Initialize as None
        self.phi_fv = None  # Initialize as None
        self.alpha = 3
        self.gamma = None
        self.d = None

    def set_tunnelling_phi(self, phi_tv, phi_fv):
        """Set the phi_tv and phi_fv for tunneling with validation."""
        self.phi_tv = np.array(phi_tv)  # Shape: (Ndim,)
        self.phi_fv = np.array(phi_fv)  # Shape: (Ndim,)
        if self.phi_tv.shape != (self.potential.Ndim,) or self.phi_fv.shape != (
            self.potential.Ndim,
        ):
            raise ValueError(
                f"phi_tv and phi_fv must have shape ({self.potential.Ndim},)"
            )

    def find_profiles(self, npoints=1000):
        """Compute the tunneling profiles for all fields."""
        if self.phi_tv is None or self.phi_fv is None:
            raise ValueError(
                "phi_tv and phi_fv must be set using set_tunnelling_phi before finding profiles."
            )

        bubble = pathDeformation.fullTunneling(
            [self.phi_tv, self.phi_fv],
            self.potential.V0,
            self.potential.dV0,
            tunneling_init_params={"alpha": self.alpha},
            tunneling_findProfile_params={"npoints": npoints},
            deformation_deform_params={"verbose": False},
        )
        return bubble.Phi, bubble.profile1D.R

    def compute_radii(self, npoints=1000):
        """Compute the radius for each field profile, returning an array of size Ndim."""
        phi, r = self.find_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

        # Compute the norm of each field component at the center
        phi00 = np.array(
            [phi[0, n] for n in range(self.potential.Ndim)]
        )  # Shape: (Ndim,)

        # Compute radii based on where the norm is half the center value for each field
        radii = np.zeros(self.potential.Ndim)
        for n in range(self.potential.Ndim):
            idx = (np.abs(phi[:, n] - phi00[n] / 2)).argmin()
            radii[n] = r[idx]

        return radii

    def compute_inner_outer_radii(self, npoints=1000):
        """Compute the inner and outer radii for each field profile, returning two arrays of size Ndim."""
        phi, r = self.find_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

        # Compute the norm of each field component at the center
        phi00 = np.array(
            [phi[0, n] for n in range(self.potential.Ndim)]
        )  # Shape: (Ndim,)

        # Compute inner and outer radii
        inner_radii = np.zeros(self.potential.Ndim)
        outer_radii = np.zeros(self.potential.Ndim)

        for n in range(self.potential.Ndim):
            outer_radii[n] = r[
                (np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(1 / 2)))).argmin()
            ]
            inner_radii[n] = r[
                (np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(-1 / 2)))).argmin()
            ]

        return inner_radii, outer_radii

    def interpolate_profiles(self, z, z0, npoints=1000, decay_rate=0.5):
        """
        Interpolate the field profiles onto the z-axis, centered at z0, with exponential decay to phi_metaMin.

        Parameters:
        - z: Real array (1D) of z values where the profile is interpolated
        - z0: The center of the profile on the z-axis
        - npoints: Number of points to use in find_profiles (default 1000)
        - decay_rate: Rate of exponential decay towards phi_metaMin (default 0.5)

        Returns:
        - phi_z: Interpolated profiles with shape (len(z), Ndim)
        """
        # Get the radial profiles
        phi, r = self.find_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

        # Compute the radial distance from the center z0
        r_new = np.abs(z - z0)  # Shape: (len(z),)

        # Maximum radial distance in the original profile
        r_max = r[-1]

        # Interpolate each field component
        phi_z = np.zeros((len(z), self.potential.Ndim))  # Shape: (len(z), Ndim)
        for n in range(self.potential.Ndim):
            # Create an interpolator for the n-th field component within the radial range
            interpolator = interp1d(
                r, phi[:, n], kind="cubic", fill_value="extrapolate", bounds_error=False
            )

            # Interpolate within the range [0, r_max]
            phi_z[:, n] = interpolator(r_new)

            # Apply exponential decay towards phi_metaMin for points outside r_max
            mask_outside = r_new > r_max
            if np.any(mask_outside):
                # Compute the distance beyond r_max
                r_excess = r_new[mask_outside] - r_max
                # Value at the boundary (r_max)
                phi_boundary = phi[-1, n]
                # Exponential decay from phi_boundary to phi_metaMin
                decay = (phi_boundary - self.phi_fv[n]) * np.exp(-decay_rate * r_excess)
                phi_z[mask_outside, n] = self.phi_fv[n] + decay

        return phi_z

    def plot_profiles(self, npoints=1000):
        """Plot all field profiles with vertical dashed lines at the radii and filled regions between inner and outer radii."""
        phi, r = self.find_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)

        # Compute inner and outer radii for each field
        inner_radii, outer_radii = self.compute_inner_outer_radii(npoints)

        # Create figure and axis
        fig, ax = plt.subplots()

        # Define colors using cm.tab10 with cycling
        colors = [plt.cm.tab10(i % 10) for i in range(self.potential.Ndim)]

        # Plot each field profile and fill between inner and outer radii
        for n in range(self.potential.Ndim):
            # Plot the profile
            ax.plot(r, phi[:, n], label=f"Field {n}", color=colors[n])

            # Draw vertical dashed lines at inner and outer radii in the same color
            ax.axvline(
                x=inner_radii[n],
                color=colors[n],
                linestyle="--",
                alpha=0.5,
                label=rf"$R^\text{{in}}_{n} = {inner_radii[n]:.2e}$" if n == 0 else "",
            )
            ax.axvline(
                x=outer_radii[n],
                color=colors[n],
                linestyle="--",
                alpha=0.5,
                label=rf"$R^\text{{out}}_{n} = {outer_radii[n]:.2e}$" if n == 0 else "",
            )

            # Fill the region between inner and outer radii with transparency
            ax.fill_between(
                r,
                0,
                phi[:, n],
                where=(r >= inner_radii[n]) & (r <= outer_radii[n]),
                color=colors[n],
                alpha=0.3,
                interpolate=True,
            )

        # # Set xlim and ylim based on profile range
        ax.set_xlim(left=0, right=r[-1])
        # ax.set_ylim(0, max(phi.max(), phi.min()) * 1.1)  # Extend slightly beyond max/min phi

        # Labels and legend
        ax.set_xlabel("r")
        ax.set_ylabel("Field profiles")
        ax.set_title(
            rf"Field profiles, $L_w={np.max(outer_radii - inner_radii):.2e}$, $R_c={np.max(self.compute_radii()):.2f}$"
        )
        ax.grid(True)

        return fig, ax

    def set_gamma(self, gamma):
        self.gamma = gamma
        self.d = None

    def set_d(self, d):
        self.d = d
        self.gamma = None

    def two_bubbles(
        self,
        layout="positive half",
        npoints=1000,
        dz_max=0.1,
        scale_dz=1.0,
        scale_z=3.0,
    ):
        """
        Compute the z_arr and interpolated profiles for two bubbles configuration.

        Parameters:
        - layout: "full" for two bubbles at ±d, "positive half" for one bubble at d/2,
                "negative half" for one bubble at -d/2 (default "positive half")
        - npoints: Number of points to use in find_profiles (default 1000)

        Returns:
        - z_arr: Array of z values (range depends on layout)
        - phi_z: Interpolated profiles (sum of two for "full", single for "positive half" or "negative half")
        - d: Distance between bubble centers (for "full") or twice the center position (for "half" types)
        """
        # Compute the radii and inner/outer radii
        radii = self.compute_radii(npoints)
        inner_radii, outer_radii = self.compute_inner_outer_radii(npoints)

        # Largest profile radius
        r0 = np.max(radii)

        d = 0.0
        # Distance between the two bubbles (or reference distance for "half" types)
        if self.gamma is not None:
            d = 2 * self.gamma * r0
            self.d = d
        elif (self.gamma is None) and (self.d is not None):
            self.gamma = self.d / (2 * r0)
            d = self.d
        else:
            raise ValueError("Either gamma or d must be set.")

        # Compute wall width at collision (using the field with the largest r0)
        idx_max_r0 = np.argmax(radii)
        r_in = inner_radii[idx_max_r0]
        r_out = outer_radii[idx_max_r0]
        l_wall_hit = np.sqrt(r_out**2 + (d / 2) ** 2 - r0**2) - np.sqrt(
            r_in**2 + (d / 2) ** 2 - r0**2
        )

        # Compute dz for the lattice
        dz = l_wall_hit / 10
        dz *= scale_dz
        dz = min(dz, dz_max)

        if layout == "full":
            # Create z_arr from -3d to 3d with step dz
            z_arr = np.arange(-scale_z * d, scale_z * d + dz, dz)
            # Interpolate profiles centered at z0 = -d and z0 = d, then sum them
            phi_z1 = self.interpolate_profiles(z_arr, z0=-d / 2, npoints=npoints)
            phi_z2 = self.interpolate_profiles(z_arr, z0=d / 2, npoints=npoints)
            phi_z = phi_z1 + phi_z2
        elif layout == "positive half":
            # Create z_arr from 0 to 3d with step dz
            z_arr = np.arange(0, scale_z * d + dz, dz)
            # Interpolate a single profile centered at z0 = d/2
            phi_z = self.interpolate_profiles(z_arr, z0=d / 2, npoints=npoints)
        elif layout == "negative half":
            # Create z_arr from -3d to 0 with step dz
            z_arr = np.arange(-scale_z * d, 0 + dz, dz)
            # Interpolate a single profile centered at z0 = -d/2
            phi_z = self.interpolate_profiles(z_arr, z0=-d / 2, npoints=npoints)
        else:
            raise ValueError(
                "layout must be either 'full', 'positive half', or 'negative half'"
            )

        return z_arr, phi_z, d

"""
@author: Quoc Trung Ho <qho@sissa.it>
"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import psweep as ps
    import dask
    import yaml

    import scipy.integrate
    from scipy.interpolate import interp1d
    from scipy.signal import argrelextrema  # For finding local extrema
    import pandas as pd
    import sys
    from collections import namedtuple

    from typing import List, Tuple, Iterable, Union, Any, Optional, Callable

    sys.path.append("../")  # add the qball package to the python path
    import os
    # sys.path.append(os.path.realpath('..'))

    from cosmoTransitions import pathDeformation
    from cosmoTransitions.generic_potential import generic_potential
    import u_integrand
    from joblib import Parallel, delayed
    import traceback

    from pygw_fopt.utils import sample
    from pygw_fopt.potetials import U1Potential, QuarticPotential
    import time

    import uuid
    import shutil

    class GenericPotential(generic_potential):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Base class for a general potential function in multiple variables.
            """
            generic_potential.__init__(self, *args, **kwargs)
            self.params_str = None

        def V0(self, X: np.ndarray) -> float:
            """
            Compute the potential at tree level V0 at given points.
            
            Parameters:
                X (np.ndarray): A NumPy array of shape (..., Ndim), where the last axis represents the field variables.
            
            Returns:
                float: The potential value at X.
            
            Raises:
                NotImplementedError: If the method is not implemented in a subclass.
            """
            raise NotImplementedError("The potential function V0 must be defined in a subclass.")

        def dV0(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            """
            Compute the gradient of the potential V0 numerically using central finite differences.
            If a subclass defines an explicit dV method, it will override this behavior.
            
            Parameters:
                X (np.ndarray): A NumPy array of shape (..., Ndim) representing input points.
                eps (float): Step size for finite difference approximation.
            
            Returns:
                np.ndarray: A NumPy array of the same shape as X, containing the gradient of V0.
            """
            X = np.asarray(X, dtype=float)
            grad = np.zeros_like(X)
            for i in range(X.shape[-1]):
                dX = np.zeros_like(X)
                dX[..., i] = eps  # Perturb only the i-th field component
                
                V_plus = self.V0(X + dX)
                V_minus = self.V0(X - dX)
                grad[..., i] = (V_plus - V_minus) / (2 * eps)  # Central difference
            return grad

        def d2V0(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            """
            Compute the Hessian matrix (second derivatives) of the potential V0 numerically.
            If a subclass defines an explicit d2V method, it will override this behavior.
            
            Parameters:
                X (np.ndarray): A NumPy array of shape (..., Ndim) representing input points.
                eps (float): Step size for finite difference approximation.
            
            Returns:
                np.ndarray: A NumPy array of shape (Ndim, Ndim) representing the Hessian matrix.
            """
            X = np.asarray(X, dtype=float)
            n = X.shape[-1]
            hessian = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    dX_i = np.zeros_like(X)
                    dX_j = np.zeros_like(X)
                    dX_i[..., i] = eps
                    dX_j[..., j] = eps
                    if i == j:
                        # Second derivative w.r.t. the same variable
                        V_pp = self.V0(X + dX_i)
                        V_mm = self.V0(X - dX_i)
                        V_0 = self.V0(X)
                        hessian[i, j] = (V_pp - 2 * V_0 + V_mm) / (eps**2)
                    else:
                        # Mixed partial derivative
                        V_pq = self.V0(X + dX_i + dX_j)
                        V_p = self.V0(X + dX_i - dX_j)
                        V_q = self.V0(X - dX_i + dX_j)
                        V_mq = self.V0(X - dX_i - dX_j)
                        hessian[i, j] = (V_pq - V_p - V_q + V_mq) / (4 * eps**2)
            return hessian

    # working
    class LatticeSetup:
        def __init__(self, potential):
            self.potential = potential
            self.phi_absMin = None  # Initialize as None
            self.phi_metaMin = None  # Initialize as None
            self.alpha = 3

        def set_tunnelling_phi(self, phi_absMin, phi_metaMin):
            """Set the phi_absMin and phi_metaMin for tunneling with validation."""
            self.phi_absMin = np.array(phi_absMin)  # Shape: (Ndim,)
            self.phi_metaMin = np.array(phi_metaMin)  # Shape: (Ndim,)
            if self.phi_absMin.shape != (self.potential.Ndim,) or self.phi_metaMin.shape != (self.potential.Ndim,):
                raise ValueError(f"phi_absMin and phi_metaMin must have shape ({self.potential.Ndim},)")

        def find_profiles(self, npoints=1000):
            """Compute the tunneling profiles for all fields."""
            if self.phi_absMin is None or self.phi_metaMin is None:
                raise ValueError("phi_absMin and phi_metaMin must be set using set_tunnelling_phi before finding profiles.")
            
            bubble = pathDeformation.fullTunneling(
                [self.phi_absMin, self.phi_metaMin],
                self.potential.V0,
                self.potential.dV0,
                tunneling_init_params={"alpha": self.alpha},
                tunneling_findProfile_params={"npoints": npoints}
            )
            return bubble.Phi, bubble.profile1D.R

        def compute_radii(self, npoints=1000):
            """Compute the radius for each field profile, returning an array of size Ndim."""
            phi, r = self.find_profiles(npoints)  # phi: (npoints, Ndim), r: (npoints,)
            
            # Compute the norm of each field component at the center
            phi00 = np.array([phi[0, n] for n in range(self.potential.Ndim)])  # Shape: (Ndim,)
            
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
            phi00 = np.array([phi[0, n] for n in range(self.potential.Ndim)])  # Shape: (Ndim,)
            
            # Compute inner and outer radii
            inner_radii = np.zeros(self.potential.Ndim)
            outer_radii = np.zeros(self.potential.Ndim)
            
            for n in range(self.potential.Ndim):
                outer_radii[n] = r[(np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(1/2)))).argmin()]
                inner_radii[n] = r[(np.abs(phi[:, n] - phi00[n] / 2 * (1 - np.tanh(-1/2)))).argmin()]
            
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
                interpolator = interp1d(r, phi[:, n], kind='cubic', fill_value="extrapolate", bounds_error=False)
                
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
                    decay = (phi_boundary - self.phi_metaMin[n]) * np.exp(-decay_rate * r_excess)
                    phi_z[mask_outside, n] = self.phi_metaMin[n] + decay
            
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
                ax.plot(r, phi[:, n], label=f'Field {n}', color=colors[n])
                
                # Draw vertical dashed lines at inner and outer radii in the same color
                ax.axvline(x=inner_radii[n], color=colors[n], linestyle='--', alpha=0.5, label=f'Inner Radius Field {n}' if n == 0 else "")
                ax.axvline(x=outer_radii[n], color=colors[n], linestyle='--', alpha=0.5, label=f'Outer Radius Field {n}' if n == 0 else "")
                
                # Fill the region between inner and outer radii with transparency
                ax.fill_between(r, 0, phi[:, n], where=(r >= inner_radii[n]) & (r <= outer_radii[n]), 
                            color=colors[n], alpha=0.3, interpolate=True)
            
            # # Set xlim and ylim based on profile range
            # ax.set_xlim(0, r[-1])
            # ax.set_ylim(0, max(phi.max(), phi.min()) * 1.1)  # Extend slightly beyond max/min phi
            
            # Labels and legend
            ax.set_xlabel('r')
            ax.set_ylabel('phi')
            ax.set_title('Field Profiles with Inner and Outer Radii')
            ax.legend()
            ax.grid(True)
            
            return fig, ax

        def two_bubbles(self, gamma, type="positive half", npoints=1000, dz_max = 0.1, scale_dz = 1.0, scale_z = 3.0):
            """
            Compute the z_arr and interpolated profiles for two bubbles configuration.
            
            Parameters:
            - gamma: Scaling factor to compute the distance between bubbles
            - type: "full" for two bubbles at ±d, "positive half" for one bubble at d/2, 
                    "negative half" for one bubble at -d/2 (default "positive half")
            - npoints: Number of points to use in find_profiles (default 1000)
            
            Returns:
            - z_arr: Array of z values (range depends on type)
            - phi_z: Interpolated profiles (sum of two for "full", single for "positive half" or "negative half")
            - d: Distance between bubble centers (for "full") or twice the center position (for "half" types)
            """
            # Compute the radii and inner/outer radii
            radii = self.compute_radii(npoints)
            inner_radii, outer_radii = self.compute_inner_outer_radii(npoints)
            
            # Largest profile radius
            r0 = np.max(radii)
            
            # Distance between the two bubbles (or reference distance for "half" types)
            d = 2 * gamma * r0
            
            # Compute wall width at collision (using the field with the largest r0)
            idx_max_r0 = np.argmax(radii)
            r_in = inner_radii[idx_max_r0]
            r_out = outer_radii[idx_max_r0]
            l_wall_hit = (np.sqrt(r_out**2 + (d/2)**2 - r0**2) - 
                        np.sqrt(r_in**2 + (d/2)**2 - r0**2))
            
            # Compute dz for the lattice
            dz = l_wall_hit / 10
            dz *= scale_dz
            dz = min(dz, dz_max)
            
            if type == "full":
                # Create z_arr from -3d to 3d with step dz
                z_arr = np.arange(-scale_z * d, scale_z * d + dz, dz)
                # Interpolate profiles centered at z0 = -d and z0 = d, then sum them
                phi_z1 = self.interpolate_profiles(z_arr, z0=-d, npoints=npoints)
                phi_z2 = self.interpolate_profiles(z_arr, z0=d, npoints=npoints)
                phi_z = phi_z1 + phi_z2
            elif type == "positive half":
                # Create z_arr from 0 to 3d with step dz
                z_arr = np.arange(0, scale_z * d + dz, dz)
                # Interpolate a single profile centered at z0 = d/2
                phi_z = self.interpolate_profiles(z_arr, z0=d/2, npoints=npoints)
            elif type == "negative half":
                # Create z_arr from -3d to 0 with step dz
                z_arr = np.arange(-scale_z * d, 0 + dz, dz)
                # Interpolate a single profile centered at z0 = -d/2
                phi_z = self.interpolate_profiles(z_arr, z0=-d/2, npoints=npoints)
            else:
                raise ValueError("type must be either 'full', 'positive half', or 'negative half'")
            
            return z_arr, phi_z, d

    # working
    class PDEBubbleSolver:
        """Class to solve a coupled set of nonlinear wave equations for bubble dynamics."""
        
        def __init__(self, d, phi_initial, z_grid, ds, dz, potential, history_interval=1):
            # distance between bubble centers
            self.d = d
            self.Ndim = potential.Ndim
            self.phi = np.array(phi_initial, dtype=np.float64)
            if self.phi.ndim == 1:  # Reshape 1D input to (Ndim, n_z)
                self.phi = self.phi[np.newaxis, :]
            if self.phi.shape[0] != self.Ndim:
                raise ValueError(f"phi_initial must have shape ({self.Ndim}, n_z), got {self.phi.shape}")
            self.z = np.array(z_grid, dtype=np.float64)
            if ds >= dz:
                self.ds = dz/2.
            else:
                self.ds = ds
            self.dz = dz
            self.potential = potential
            self.n_z = len(z_grid)
            self.history_interval = history_interval
            self.phi_history = None
            self.n_s = None  # Number of history points
            self.s_max = None  # Maximum simulation time
            self.energy_density = None  # Store energy density
            self.energy_how_often = None  # Store how_often parameter
        
        def _spatial_derivative(self, phi):
            phi_padded = np.pad(phi, ((0, 0), (1, 1)), mode='reflect')
            return (phi_padded[:, :-2] - 2 * phi_padded[:, 1:-1] + phi_padded[:, 2:]) / (self.dz**2)
        
        def evolvepi(self, phi, pi, s, ds):
            damping = (1 - 2 * ds / (s + ds))
            # Transpose phi to (n_z, Ndim) for dV0
            phi_transposed = phi.T  # Shape: (n_z, Ndim)
            forcing = s * ds / (s + ds) * (-self.potential.dV0(phi_transposed).T + self._spatial_derivative(phi))
            return pi * damping + forcing
        
        def evolvepi_first_half_step(self, baby_steps=20):
            pi = np.zeros_like(self.phi, dtype=np.float64)
            baby_ds = 0.5 * self.ds / (baby_steps - 1)
            phi = self.phi.copy()
            for i in range(1, baby_steps):
                pi = self.evolvepi(phi, pi, (i-1)*baby_ds, baby_ds)
                phi += baby_ds * pi
            return pi
        
        def evolve(self, smax, verbose=False):
            n_steps = int(np.ceil(smax / self.ds))
            n_history = (n_steps + self.history_interval - 1) // self.history_interval + 1
            self.n_s = n_history  # Store the number of history points
            self.s_max = smax  # Store the maximum simulation time
            self.phi_history = np.zeros((n_history, self.Ndim, self.n_z), dtype=np.float64)
            self.phi_history[0] = self.phi
            
            pi = self.evolvepi_first_half_step()
            for i in range(1, n_steps + 1):
                if verbose and i % 1000 == 0:
                    print(f"Step {i}: s = {i*self.ds}", flush=True)
                if i > 1:
                    pi = self.evolvepi(self.phi, pi, (i-1)*self.ds, self.ds)
                self.phi += self.ds * pi
                if i % self.history_interval == 0:
                    self.phi_history[i // self.history_interval] = self.phi
            return self.phi_history
        
        def calculate_energy_density(self, how_often=10):
            """Calculate energy density efficiently to avoid memory errors."""
            phiall = self.phi_history  # Shape: (n_history, Ndim, n_z)
            n_s = phiall.shape[0]
            idx = np.arange(0, n_s-2, how_often // self.history_interval)
            idx = idx[idx < n_s-2]
            n_idx = len(idx)
            
            # Pre-allocate output array
            energy_density = np.zeros((n_idx, self.n_z-1), dtype=np.float64)
            
            # Compute terms incrementally
            for i in range(n_idx):
                # Kinetic term in s-direction
                phi_diff_s = (phiall[idx[i]+1, :, :-1] - phiall[idx[i], :, :-1]) / (self.ds * self.history_interval)
                kin_s = 0.5 * np.sum(phi_diff_s**2, axis=0)  # Sum over Ndim
                
                # Kinetic term in z-direction
                phi_diff_z = (phiall[idx[i], :, 1:] - phiall[idx[i], :, :-1]) / self.dz
                kin_z = 0.5 * np.sum(phi_diff_z**2, axis=0)  # Sum over Ndim
                
                # Potential term (flatten to 1D for V0, transpose to match dV0 expectation)
                phi_flat = phiall[idx[i], :, :-1].T  # Shape: (n_z-1, Ndim)
                pot = self.potential.V0(phi_flat.T)  # Transpose back after V0 if needed, but V0 should handle (n_z-1, Ndim)
                
                # Combine terms
                energy_density[i] = kin_s + kin_z + pot[0]  # Take first element if V0 returns scalar per point
            
            # Store the results
            self.energy_density = energy_density
            self.energy_how_often = how_often
            return energy_density
        
        def calculate_energy(self):
            phiall = self.phi_history
            n_s = phiall.shape[0]
            energy = np.zeros(n_s - 2, dtype=np.float64)
            kin_s = 0.5 * self.dz * ((phiall[2:] - phiall[:-2]) / (2 * self.ds * self.history_interval))**2
            kin_z = 0.5 * self.dz * ((phiall[1:, :, 1:] - phiall[1:, :, :-1]) / self.dz)**2
            # Transpose phi for V0
            pot = self.dz * self.potential.V0(phiall[1:-1].transpose(0, 2, 1).reshape(-1, self.Ndim).T).T.reshape(self.Ndim, n_s-2, self.n_z)
            energy = np.sum(np.sum(kin_s, axis=2) + np.sum(kin_z, axis=2) + pot, axis=1)
            return energy
        
        def plot_field_evolution(self, field_idx=0, field_name=None, title=""):
            phi_scaled = self.phi_history[:, field_idx, :]
            s = np.arange(0, len(phi_scaled) * self.ds * self.history_interval, self.ds * self.history_interval)
            fig, ax = plt.subplots()
            im = ax.imshow(phi_scaled[::-1], cmap='RdBu_r', extent=[min(self.z), max(self.z), 0, max(s)])
            ax.set_xlabel('z')
            ax.set_ylabel('s', rotation=0)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(
                r'$\Phi_{%d}$' % (field_idx+1) if field_name is None else field_name,
                rotation=0,
                labelpad=-40,  # Increase to move label further above the color bar
                x=1,  # Center the label horizontally
                y=1.1  # Adjust upward to fine-tune vertical position
            )
            if field_name is None:
                ax.set_title(f"Field {field_idx + 1} {title}".strip())
            else:
                ax.set_title(f"Field = {field_name} {title}".strip())
            im.set_clim(1-np.max(phi_scaled), np.max(phi_scaled))
            return fig, ax
        
        def plot_norm_field_evolution(self, title=""):
            """
            Plot the evolution of the norm of all fields, defined as sqrt(sum(phi_i^2)) for i in range(Ndim).
            """
            # Compute the norm: sqrt(sum(phi_i^2)) over all fields
            phi_squared = np.sum(self.phi_history**2, axis=1)  # Shape: (n_history, n_z)
            phi_norm = np.sqrt(phi_squared)  # Shape: (n_history, n_z)
            
            # Time axis (s)
            s = np.arange(0, len(phi_norm) * self.ds * self.history_interval, self.ds * self.history_interval)
            
            # Create the plot
            fig, ax = plt.subplots()
            im = ax.imshow(phi_norm[::-1], cmap='RdBu_r', extent=[min(self.z), max(self.z), 0, max(s)])
            ax.set_xlabel('z')
            ax.set_ylabel('s', rotation=0)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(
                r'$\sqrt{\sum_{i} \Phi_i^2}$',
                rotation=0,
                labelpad=-40,  # Increase to move label further above the color bar
                x=1,  # Center the label horizontally
                y=1.1  # Adjust upward to fine-tune vertical position
            )
            ax.set_title(f"Norm of Fields {title}".strip())
            im.set_clim(0, np.max(phi_norm))  # Norm is non-negative, so start from 0
            return fig, ax
        
        def plot_energy_density(self, title=""):
            """Plot the energy density using stored self.energy_density and self.s_max."""
            if self.energy_density is None or self.s_max is None:
                raise ValueError("Energy density must be calculated and evolve must be called before plotting.")
            
            fig, ax = plt.subplots()
            im = ax.imshow(self.energy_density[::-1], cmap='RdBu_r', extent=[min(self.z), max(self.z), 0, self.s_max])
            ax.set_ylabel('s')
            ax.set_xlabel('z')
            ax.set_title(f"Energy Density {title}".strip())
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(
                r'$\rho$',
                rotation=0,
                labelpad=-40,  # Increase to move label further above the color bar
                x=1,  # Center the label horizontally
                y=1.1  # Adjust upward to fine-tune vertical position
            )
            return fig, ax
        
        def plot_max_rho_grad(self, title=""):
            """
            Plot the maximum of rho_grad over all z at each s step, where rho_grad = |d phi / dz|^2 / 2.
            
            Parameters:
                title (str): Custom title to append to the plot title.
            
            Returns:
                fig, ax: Matplotlib figure and axis objects.
            """
            if self.phi_history is None or self.s_max is None:
                raise ValueError("Evolve must be called before plotting max rho_grad.")
            
            # Shape of phi_history: (n_history, Ndim, n_z)
            # Compute d phi / dz for each s step and field
            phi_diff_z = (self.phi_history[:, :, 1:] - self.phi_history[:, :, :-1]) / self.dz  # Shape: (n_history, Ndim, n_z-1)
            
            # Compute rho_grad = |d phi / dz|^2 / 2
            # First, square the derivatives and sum over fields (Ndim axis)
            phi_diff_z_squared = np.sum(phi_diff_z**2, axis=1)  # Shape: (n_history, n_z-1)
            rho_grad = 0.5 * phi_diff_z_squared  # Shape: (n_history, n_z-1)
            
            # Compute the maximum over z for each s step
            max_rho_grad = np.max(rho_grad, axis=1)  # Shape: (n_history,)
            
            # Time axis (s)
            s = np.arange(0, len(max_rho_grad) * self.ds * self.history_interval, self.ds * self.history_interval)
            
            # Create the plot
            fig, ax = plt.subplots()
            ax.plot(s, max_rho_grad, label='Max rho_grad')
            ax.set_xlabel('s')
            ax.set_ylabel(r'Max $|\partial \phi / \partial z|^2 / 2$')
            ax.set_title(f"Maximum Rho Grad {title}".strip())
            ax.grid(True)
            ax.legend()
            return fig, ax


    # Define the namedtuple with dE_dlogw
    GwSpectrumData = namedtuple('GwSpectrumData', ['dE_dlogw', 'klist', 'integrand_klist'])

    class GravitationalWaveCalculator:
        def __init__(self, solver, d, type='half', cutoff_ratio=1.0):
            self.solver = solver
            self.d = d
            self.type = type
            self.cutoff_ratio = cutoff_ratio
            # Store full phi_history with shape (n_s, Ndim, n_z)
            self.phi_history = solver.phi_history  # Shape: (n_s, Ndim, n_z)
            self.Ndim = solver.potential.Ndim
            self.z = solver.z
            self.ds = solver.ds
            self.dz = abs(self.z[1] - self.z[0])
            self.smax = (len(self.phi_history) - 1) * self.ds
            self.slist = np.linspace(0, self.smax, len(self.phi_history))
            self.n_s = len(self.slist)
            self.n_z = len(self.z)
            self.t_cut = self.cutoff_ratio * self.smax
            self.t_0 = (self.smax - self.t_cut) / 4
            self.phi2_history = self._compute_phi_region2()
            # Precompute derivatives for each field
            self.dphi_dz = [(self.phi_history[:, n, 1:] - self.phi_history[:, n, :-1]) / self.dz for n in range(self.Ndim)]
            self.dphi_ds = [(self.phi_history[1:, n, :] - self.phi_history[:-1, n, :]) / self.ds for n in range(self.Ndim)]
            self.dphi_dz2 = [(self.phi2_history[:, n, 1:] - self.phi2_history[:, n, :-1]) / self.dz for n in range(self.Ndim)]
            self.dphi_ds2 = [(self.phi2_history[1:, n, :] - self.phi2_history[:-1, n, :]) / self.ds for n in range(self.Ndim)]
            self.xz_dphi_dz = [np.array([[self._xz_dphi_dz(i_s, i_z, n) for i_z in range(1, self.n_z)] 
                                        for i_s in range(1, self.n_s)]) for n in range(self.Ndim)]
            self.xz_dphi_ds = [np.array([[self._xz_dphi_ds(i_s, i_z, n) for i_z in range(1, self.n_z)] 
                                        for i_s in range(1, self.n_s)]) for n in range(self.Ndim)]
            self.xz_dphi_dz2 = [np.array([[self._xz_dphi_dz2(i_s, i_z, n) for i_z in range(1, self.n_z)] 
                                        for i_s in range(1, self.n_s)]) for n in range(self.Ndim)]
            self.xz_dphi_ds2 = [np.array([[self._xz_dphi_ds2(i_s, i_z, n) for i_z in range(1, self.n_z)] 
                                        for i_s in range(1, self.n_s)]) for n in range(self.Ndim)]
            # Precompute common factors
            self.s_valid = self.slist[1:]
            self.i_s_valid = np.arange(1, self.n_s)
            self.s_offset = self.s_valid - 0.5 * self.ds
            self.zz_weights = self.i_s_valid**2 * self.ds**3 * np.where(self.i_s_valid == self.n_s - 1, 0.5, 1.0)
            self.xandy_weights = 0.5 * self.s_offset**2 * self.ds
            self.xz_weights = self.s_offset**2 * self.ds
            self.z_weights = np.array([0.5 if i == 0 or i == self.n_z - 1 else 1.0 for i in range(self.n_z)])

        def _compute_phi_region2(self):
            n_s = self.n_s
            n_z = self.n_z
            ds = self.ds
            dz = self.dz
            phi2_history = np.zeros((n_s, self.Ndim, n_z))
            if self.type == 'one':
                for n in range(self.Ndim):
                    phi0_interp = interp1d(self.z, self.phi_history[0, n, :], kind='linear', fill_value='extrapolate')
                    phi2_history[:, n, :] = np.fromfunction(
                        lambda i_s, i_z: phi0_interp(np.sqrt((i_s * ds)**2 + (i_z * dz)**2)),
                        (n_s, n_z)
                    )
            else:
                for n in range(self.Ndim):
                    # phimid = np.argmax(np.abs(self.phi_history[0, n, :]))
                    # z_shifted = self.z[phimid:] - self.z[phimid]
                    # phi0 = self.phi_history[0, n, phimid:]
                    # phi0_interp = interp1d(z_shifted, phi0, kind='linear', fill_value='extrapolate')
                    # phi2_history[:, n, :] = np.fromfunction(
                    #     lambda i_s, i_z: (phi0_interp(np.sqrt((i_s * ds)**2 + (self.z[0] + i_z * dz - self.d / 2)**2)) +
                    #                       phi0_interp(np.sqrt((i_s * ds)**2 + (self.z[0] + i_z * dz + self.d / 2)**2))),
                    #     (n_s, n_z)
                    # )
                    idx_zcenter = np.argmin(np.abs(self.z))
                    if np.argmax(np.abs(self.phi_history[0, n, idx_zcenter:])) != 0:
                        idx_phimid_right = np.argmax(np.abs(self.phi_history[0, n, idx_zcenter:]))
                    else:
                        idx_phimid_right = idx_zcenter
                    idx_phimid_right = np.argmax(np.abs(self.phi_history[0, n, idx_zcenter:])) + idx_zcenter
                    z_shifted_right = self.z[idx_phimid_right:] - self.z[idx_phimid_right]
                    z_shifted_right = np.concatenate((-z_shifted_right[::-1], z_shifted_right))
                    phi0_right = self.phi_history[0, n, idx_phimid_right:]
                    phi0_right = np.concatenate((phi0_right[::-1], phi0_right))
                    phi0_interp_right = interp1d(z_shifted_right, phi0_right, kind='linear', fill_value=(0., 0.), bounds_error=False)

                    if np.argmax(np.abs(self.phi_history[0, n, :idx_zcenter])) != 0:
                        idx_phimid_left = np.argmax(np.abs(self.phi_history[0, n, :idx_zcenter]))
                    else:
                        idx_phimid_left = idx_zcenter
                    z_shifted_left = self.z[:idx_phimid_left] - self.z[idx_phimid_left]
                    z_shifted_left = np.concatenate((-z_shifted_left[::-1], z_shifted_left))
                    phi0_left = self.phi_history[0, n, :idx_phimid_left]
                    phi0_left = np.concatenate((phi0_left[::-1], phi0_left))
                    phi0_interp_left = interp1d(z_shifted_left, phi0_left, kind='linear', fill_value=(0., 0.), bounds_error=False)

                    phi2_history[:, n, :] = np.fromfunction(
                        lambda i_s, i_z: (phi0_interp_right(np.sqrt((i_s * ds)**2 + (self.z[0] + i_z * dz - self.d / 2)**2)) +
                                            phi0_interp_left(np.sqrt((i_s * ds)**2 + (self.z[0] + i_z * dz + self.d / 2)**2))),
                        (n_s, n_z)
                    )
            return phi2_history

        def _u_max(self, s):
            return 1 + (self.t_cut + 7 * self.t_0) / s

        def _integral_u_quad_region1(self, u_int_func, s, k, w, i_s):
            integrand = scipy.LowLevelCallable.from_cython(u_integrand, 'integrand')
            sign = -1
            res1_real = scipy.integrate.quad(
                integrand, 1, self._u_max(s),
                args=(w, k, s, sign, 1, self.t_cut, self.t_0, u_int_func)
            )
            res1_imag = scipy.integrate.quad(
                integrand, 1, self._u_max(s),
                args=(w, k, s, sign, -1, self.t_cut, self.t_0, u_int_func)
            )
            return res1_real[0] + res1_imag[0] * 1j

        def _integral_u_quad_region2(self, u_int_func, s, k, w, i_s):
            integrand = scipy.LowLevelCallable.from_cython(u_integrand, 'integrand')
            sign = 1
            res2_real = scipy.integrate.quad(
                integrand, 0, self._u_max(s),
                args=(w, k, s, sign, 1, self.t_cut, self.t_0, u_int_func)
            )
            res2_imag = scipy.integrate.quad(
                integrand, 0, self._u_max(s),
                args=(w, k, s, sign, -1, self.t_cut, self.t_0, u_int_func)
            )
            return res2_real[0] + res2_imag[0] * 1j

        def _xz_dphi_dz(self, i_s, i_z, n):
            return 1 / (2 * self.dz) * (
                self.phi_history[i_s, n, i_z] - self.phi_history[i_s, n, i_z - 1] +
                self.phi_history[i_s - 1, n, i_z] - self.phi_history[i_s - 1, n, i_z - 1]
            )

        def _xz_dphi_ds(self, i_s, i_z, n):
            return 1 / (2 * self.ds) * (
                self.phi_history[i_s, n, i_z] - self.phi_history[i_s - 1, n, i_z] +
                self.phi_history[i_s, n, i_z - 1] - self.phi_history[i_s - 1, n, i_z - 1]
            )

        def _xz_dphi_dz2(self, i_s, i_z, n):
            return 1 / (2 * self.dz) * (
                self.phi2_history[i_s, n, i_z] - self.phi2_history[i_s, n, i_z - 1] +
                self.phi2_history[i_s - 1, n, i_z] - self.phi2_history[i_s - 1, n, i_z - 1]
            )

        def _xz_dphi_ds2(self, i_s, i_z, n):
            return 1 / (2 * self.ds) * (
                self.phi2_history[i_s, n, i_z] - self.phi2_history[i_s - 1, n, i_z] +
                self.phi2_history[i_s, n, i_z - 1] - self.phi2_history[i_s - 1, n, i_z - 1]
            )

        def _compute_k_integral(self, w, k, k_idx, cos_wkz_shifted, sin_wkz_shifted, cos_wkz_full):
            if k == 1 or k == -1:
                return (w, k_idx, 0.0)

            INTEGRAL_DU_XX = 1
            INTEGRAL_DU_YY = 2
            INTEGRAL_DU_ZZ = 3
            INTEGRAL_DU_XZ = 4

            # ZZ Integral
            u_zz_region1 = np.array([self._integral_u_quad_region1(INTEGRAL_DU_ZZ, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_valid, self.i_s_valid)])
            u_zz_region2 = np.array([self._integral_u_quad_region2(INTEGRAL_DU_ZZ, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_valid, self.i_s_valid)])
            zz_region1 = self.dz * 2 * np.sum([np.sum(cos_wkz_shifted * dphi_dz[1:]**2, axis=1) 
                                            for dphi_dz in self.dphi_dz], axis=0)
            zz_region2 = self.dz * 2 * np.sum([np.sum(cos_wkz_shifted * dphi_dz2[1:]**2, axis=1) 
                                            for dphi_dz2 in self.dphi_dz2], axis=0)
            zz_result = u_zz_region1 * zz_region1 + u_zz_region2 * zz_region2
            zz_integral = np.sum(zz_result * self.zz_weights)

            # Xandy Integral
            u_xx_region1 = np.array([self._integral_u_quad_region1(INTEGRAL_DU_XX, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            u_xx_region2 = np.array([self._integral_u_quad_region2(INTEGRAL_DU_XX, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            u_yy_region1 = np.array([self._integral_u_quad_region1(INTEGRAL_DU_YY, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            u_yy_region2 = np.array([self._integral_u_quad_region2(INTEGRAL_DU_YY, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            xandy_region1 = self.dz * 2 * np.sum([np.sum(cos_wkz_full * dphi_ds**2 * self.z_weights, axis=1) 
                                                for dphi_ds in self.dphi_ds], axis=0)
            xandy_region2 = self.dz * 2 * np.sum([np.sum(cos_wkz_full * dphi_ds2**2 * self.z_weights, axis=1) 
                                                for dphi_ds2 in self.dphi_ds2], axis=0)
            u_xandy_1 = u_xx_region1 * k**2 - u_yy_region1
            u_xandy_2 = u_xx_region2 * k**2 - u_yy_region2
            xandy_integral = np.sum((u_xandy_1 * xandy_region1 + u_xandy_2 * xandy_region2) * self.xandy_weights)

            # XZ Integral
            u_xz_region1 = np.array([self._integral_u_quad_region1(INTEGRAL_DU_XZ, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            u_xz_region2 = np.array([self._integral_u_quad_region2(INTEGRAL_DU_XZ, s, k, w, i_s) 
                                    for s, i_s in zip(self.s_offset, self.i_s_valid)])
            xz_region1 = self.dz * 2 * 1j * np.sum([np.sum(sin_wkz_shifted * xz_dphi_ds * xz_dphi_dz, axis=1) 
                                                    for xz_dphi_ds, xz_dphi_dz in zip(self.xz_dphi_ds, self.xz_dphi_dz)], axis=0)
            xz_region2 = self.dz * 2 * 1j * np.sum([np.sum(sin_wkz_shifted * xz_dphi_ds2 * xz_dphi_dz2, axis=1) 
                                                    for xz_dphi_ds2, xz_dphi_dz2 in zip(self.xz_dphi_ds2, self.xz_dphi_dz2)], axis=0)
            xz_integral = np.sum(1j * (u_xz_region1 * xz_region1 + u_xz_region2 * xz_region2) * self.xz_weights)

            integrand_value = (abs(zz_integral * (1 - k**2) + xandy_integral - 2 * xz_integral * k * np.sqrt(1 - k**2)))**2
            return (w, k_idx, integrand_value * w**3 * 2 * np.pi)

        def compute_gw_integral(self, wlist, n_k, num_processes=None):
            wlist = np.asarray(wlist)
            klist = np.linspace(0, 1, n_k)
            dk = klist[1] - klist[0]
            total_tasks = len(wlist) * n_k

            z_shifted = self.z[1:] - 0.5 * self.dz
            z_full = self.z
            cos_wkz_shifted = {w: {k: np.cos(w * k * z_shifted) for k in klist} for w in wlist}
            sin_wkz_shifted = {w: {k: np.sin(w * k * z_shifted) for k in klist} for w in wlist}
            cos_wkz_full = {w: {k: np.cos(w * k * z_full) for k in klist} for w in wlist}

            if num_processes is None:
                import multiprocessing
                num_processes = min(multiprocessing.cpu_count(), total_tasks)
            else:
                import multiprocessing
                num_processes = min(multiprocessing.cpu_count(), max(1, num_processes), total_tasks)

            tasks = [(w, k, i_k, cos_wkz_shifted[w][k], sin_wkz_shifted[w][k], cos_wkz_full[w][k]) 
                    for w in wlist for i_k, k in enumerate(klist) if k != 1 and k != -1]

            if num_processes > 1 and len(tasks) > 1:
                # Use joblib for parallel execution
                results = Parallel(n_jobs=num_processes)(
                    delayed(self._compute_k_integral)(w, k, k_idx, cos_wkz_shifted, sin_wkz_shifted, cos_wkz_full)
                    for w, k, k_idx, cos_wkz_shifted, sin_wkz_shifted, cos_wkz_full in tasks
                )
            else:
                results = [self._compute_k_integral(*task) for task in tasks]

            integrand_dict = {w: np.zeros(n_k) for w in wlist}
            for w, k_idx, integrand_value in results:
                integrand_dict[w][k_idx] = integrand_value

            final_results = []
            for w in wlist:
                integrand_values = integrand_dict[w]
                dE_dlogw = 0.0
                for i_k, value in enumerate(integrand_values):
                    if i_k == 0 or i_k == len(klist) - 1:
                        dE_dlogw += value * dk
                    else:
                        dE_dlogw += 2 * value * dk
                final_results.append(GwSpectrumData(dE_dlogw=dE_dlogw, klist=klist, integrand_klist=integrand_values))

            return final_results

    def scan_gw_u1(pset):
        try:
            # parameters for potential
            delta_V = pset["delta_V"]
            kappa = pset["kappa"]
            v = pset["v"]
            u1potential = U1Potential(delta_V=delta_V, kappa=kappa, v=v)
            # parameters for bubbles
            gamma = pset["gamma"]
            phase = pset["phase"]
            # parameters for GW computations
            #FIXME
            cutoff_ratio = pset["cutoff_ratio"]
            wlist = np.geomspace(pset["wmin"], pset["wmax"], pset["n_w"])
            n_k = pset["n_k"]
            num_processes = pset["num_processes"]

            # Set up the tunnelling profiles
            setup = LatticeSetup(u1potential)

            # The left bubble has phase=0
            phi_absMin0 = np.array([np.sqrt(2)*v, 0.0])
            phi_metaMin0 = np.array([0.0, 0.0])
            setup.set_tunnelling_phi(phi_absMin=phi_absMin0, phi_metaMin=phi_metaMin0)
            initial0 = setup.two_bubbles(gamma, type="negative half")
            z0, phi_initial0, d0 = initial0

            # The right bubble has phase given above
            phi_absMin_phase = np.array([np.sqrt(2)*v*np.cos(phase), np.sqrt(2)*v*np.sin(phase)])
            phi_metaMin_phase = np.array([0.0, 0.0])
            setup.set_tunnelling_phi(phi_absMin=phi_absMin_phase, phi_metaMin=phi_metaMin_phase)
            initial_phase = setup.two_bubbles(gamma, type="positive half")
            z_phase, phi_initial_phase, d_phase = initial_phase

            # Combine the left and right bubbles into a unique initial profile
            d = max(d0, d_phase)
            z = np.linspace(z0[0], z_phase[-1], len(z0) + len(z_phase))
            dz = abs(z[1] - z[0])
            phi_initial = np.concatenate((phi_initial0.T, phi_initial_phase.T), axis=1)

            # Setting the range of evolution
            smax = d * 2.4
            ds = dz*(1 - 1e-1)

            # Evolution of 2 bubbles with different phase
            solver = PDEBubbleSolver(d, phi_initial, z, ds, dz, u1potential, history_interval=1)
            solver.evolve(smax)

            gw_calc = GravitationalWaveCalculator(solver=solver, d=d, cutoff_ratio=cutoff_ratio)
            gw_result = gw_calc.compute_gw_integral(wlist=wlist, n_k=n_k, num_processes=num_processes)
            df_gw = pd.DataFrame(gw_result)

            ret = {"df_gw": df_gw}
            ret.update(_failed=False, _exc_txt=None)
        except:
            txt = traceback.format_exc()
            print(f"{pset=} failed, traceback:\n{txt}")
            ret = dict(_failed=True, _exc_txt=txt)
        return ret

    # Record the start time
    start_time = time.time()
###############################################################################
### LOADING PARAMETERS FROM INPUT FILE
    input_file = sys.argv[1]
    with open(input_file, 'r') as parameters_file:
        parameters = yaml.safe_load(parameters_file)

    cluster_type = parameters['cluster_type']
    cluster_params = parameters['cluster']
    setup_params = parameters['setup']
    default_scan_params = parameters['default_scan']
    scan_params = parameters['scan']

    if cluster_type == "slurm":
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster
        from distributed.diagnostics.plugin import UploadDirectory
        cluster = SLURMCluster(**cluster_params)
        cluster.scale(**parameters['cluster_scale'])
        client = Client(cluster)
        client.register_plugin(UploadDirectory("/home/qho/workspace/pygw_fopt/pygw_fopt"))
        client.register_plugin(UploadDirectory("/home/qho/workspace/pygw_fopt/cosmoTransitions"))
    elif cluster_type == "local":
        from dask.distributed import Client, LocalCluster
        from distributed.diagnostics.plugin import UploadDirectory
        dask.config.set({'distributed.worker.daemon': False})
        cluster = LocalCluster(**cluster_params)
        client = Client(cluster)
        client.register_plugin(UploadDirectory("~/working/projects/SISSA_projects/code/pygw_fopt/pygw_fopt"))

    par_delta_V = ps.plist("delta_V", [setup_params["delta_V"]])
    par_kappa = ps.plist("kappa", [setup_params["kappa"]])
    par_v = ps.plist("v", [setup_params["v"]])
    par_gamma = ps.plist("gamma", [setup_params["gamma"]])
    par_phase = ps.plist("phase", [setup_params["phase_mod_pi"]*np.pi])
    par_cutoff_ratio = ps.plist("cutoff_ratio", [setup_params["cutoff_ratio"]])
    par_n_k = ps.plist("n_k", [setup_params["n_k"]])
    par_wmin = ps.plist("wmin", [setup_params["wmin"]])
    par_wmax = ps.plist("wmax", [setup_params["wmax"]])
    par_n_w = ps.plist("n_w", [setup_params["n_w"]])
    par_num_processes = ps.plist("num_processes", [setup_params["num_processes"]])

    par_study = ps.plist("study", [scan_params["study"]])
    if "phase_mod_pi" in scan_params:
        if scan_params["phase_mod_pi"] is not None:
            par_phase = ps.plist("phase", sample(*scan_params["phase_mod_pi"])*np.pi)
        else:
            par_phase = ps.plist("phase", sample(*default_scan_params["phase_mod_pi"])*np.pi)

    if scan_params["grid"] == "full":
        # Full parameter list
        params_gw_u1 = ps.pgrid(par_delta_V, par_kappa, par_v, par_gamma, par_phase, par_cutoff_ratio, par_n_k, par_wmin, par_wmax, par_n_w, par_num_processes)
    elif scan_params["grid"] == "star":
        # Parameter list around central value par_central
        params_gw_u1 = ps.stargrid(const=setup_params, vary=[par_phase])

    calc_dir = "./outputs/gw_u1/calc_" + scan_params["study"] + "_" + scan_params["description"]
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    database_dir = os.path.join(calc_dir, str(uuid.uuid4()))
    print("###SCANNING GW ...")
    print(f"    + Input file: {input_file}")
    print(f"    + Output dir: {database_dir}")
    # run the parameter scan and save database results
    df_gw = ps.run(scan_gw_u1, params_gw_u1, skip_dups=True, dask_client=client, calc_dir=calc_dir, save=False, backup=False)
    ps.df_write(os.path.join(database_dir, "database.pickle"), df_gw)
    # copy file of input parameters to the database directory
    shutil.copy2(input_file, os.path.join(database_dir, "inputs.yaml"))
    print("###SCANNING COMPLETED")

    # Record the end time
    end_time = time.time()
    # Calculate total time taken in seconds
    execution_time = end_time - start_time

    # Convert the execution time into a struct_time in GMT format
    readable_time = time.gmtime(execution_time)

    # Format it into hours:minutes:seconds
    formatted_time = time.strftime('%H:%M:%S', readable_time)

    # Now handle the logging part
    with open(os.path.join(database_dir, "run.log"), "w") as log_file:
        log_file.write(f"Start time: {time.strftime('%Y/%m/%d-%a-%H:%M:%S', time.localtime(start_time))}\n")
        log_file.write(f"End time: {time.strftime('%Y/%m/%d-%a-%H:%M:%S', time.localtime(end_time))}\n")
        log_file.write(f"Total execution time: {execution_time:.2f} (seconds) ==> {formatted_time} (hours:minutes:seconds)\n")
        log_file.write("=" * 50 + "\n")  # Add separator for readability
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.colors import LogNorm
import pickle

class TwoBubblesEvolutionVisualizer:
    def __init__(self, phi1, d, s_grid, z_grid):
        """
        Initialize the TwoBubblesEvolutionVisualizer.
        
        Parameters:
        - phi1: 3D array, the field to analyze.
        - d: Parameter defining the offset of the 45-degree line.
        - s_grid: 1D array of s coordinates.
        - z_grid: 1D array of z coordinates.
        """
        self.phi1 = phi1
        self.d = d
        self.s_grid = s_grid
        self.z_grid = z_grid
        self.window_width = None
        self.dz = np.abs(z_grid[1] - z_grid[0])
        self.ds = np.abs(s_grid[1] - s_grid[0])
        self.s_max = max(s_grid)
        self.integration_width = None
        self.s_col = None
        self.sigma = None
        self.s_valid_indices = None
        self.s_coords_valid = None
        self.z_max_indices = None
        self.z_max_coords = None
        self.max_rho_grad = None

        # Compute rho_grad
        self.rho_grad = 0.5 * np.sum(np.gradient(self.phi1, self.dz, axis=2, edge_order=2)**2, axis=0)
        self.rho_grad = np.maximum(self.rho_grad, 1e-10)

        # Compute s_col: time where rho_grad at z=0 is maximum
        z_zero_idx = np.argmin(np.abs(self.z_grid))  # Find index where z ≈ 0
        rho_grad_z0 = self.rho_grad[:, z_zero_idx]
        self.s_col = self.s_grid[np.argmax(rho_grad_z0)]

    def set_width(self, integration_width, window_width=None):
        """
        Set the width of the integration region.
        
        Parameters:
        - integration_width: Width of the integration region around the max rho_grad path.
        - window_width: Width of the window around z = s - s_col for s >= s_col
        """
        self.integration_width = integration_width
        if window_width is not None:
            self.window_width = window_width
        else:
            self.window_width = integration_width
        self.sigma = None
        self.s_valid_indices = None
        self.s_coords_valid = None
        self.z_max_indices = None
        self.z_max_coords = None
        self.max_rho_grad = None

    def compute_max_gradient_energy_density(self):
        """
        Compute the maximum of rho_grad for each s:
        - For s < s_col, find max rho_grad where z >= 0 and z < d/2.
        - For s >= s_col, find max rho_grad within window_width around z = s - s_col.
        
        Returns:
        - s_coords_valid: 1D array of valid s coordinates.
        - z_max_coords: 1D array of z coordinates corresponding to max rho_grad.
        - max_rho_grad: 1D array of maximum rho_grad values.
        """
        # Initialize arrays
        self.s_valid_indices = np.arange(self.rho_grad.shape[0])
        self.s_coords_valid = self.s_grid[self.s_valid_indices]
        self.z_max_indices = np.zeros_like(self.s_valid_indices, dtype=int)
        self.max_rho_grad = np.zeros_like(self.s_coords_valid)
        
        # Find indices where z >= 0 and z < d/2
        z_pre_col_mask = (self.z_grid >= 0) & (self.z_grid < self.d / 2)
        
        for i, s_idx in enumerate(self.s_valid_indices):
            s = self.s_grid[s_idx]
            if s < self.s_col:
                # For s < s_col, max rho_grad where z >= 0 and z < d/2
                valid_z_mask = z_pre_col_mask
                valid_z_indices = np.where(valid_z_mask)[0]
            else:
                # For s >= s_col, max rho_grad within window_width around z = s - s_col
                z_target = s - self.s_col
                valid_z_mask = (self.z_grid >= z_target - self.window_width / 2) & \
                               (self.z_grid <= z_target + self.window_width / 2)
                valid_z_indices = np.where(valid_z_mask)[0]
            
            if len(valid_z_indices) > 0:
                rho_grad_s = self.rho_grad[s_idx, valid_z_indices]
                max_idx = np.argmax(rho_grad_s)
                self.z_max_indices[i] = valid_z_indices[max_idx]
                self.max_rho_grad[i] = rho_grad_s[max_idx]
            else:
                self.z_max_indices[i] = 0
                self.max_rho_grad[i] = 1e-10  # Fallback value
        
        self.z_max_coords = self.z_grid[self.z_max_indices]
        
        return self.s_coords_valid, self.z_max_coords, self.max_rho_grad

    def compute_surface_tension(self):
        """
        Compute the surface tension sigma by integrating rho_grad over the specified width
        centered around the path of maximum rho_grad.
        """
        if self.z_max_coords is None:
            self.compute_max_gradient_energy_density()
        
        n_width = int(self.integration_width / self.dz)  # Number of grid points for half-width
        
        # Extract rho_grad values within the width around max rho_grad path
        integrand = []
        for s_idx, z_idx in zip(self.s_valid_indices, self.z_max_indices):
            z_lower = max(0, z_idx - n_width)
            z_upper = min(self.rho_grad.shape[1], z_idx + n_width + 1)
            integrand.append(self.rho_grad[s_idx, z_lower:z_upper])
        
        # Pad integrand arrays to uniform length
        max_length = max(len(arr) for arr in integrand)
        integrand = np.array([
            np.pad(arr, (0, max_length - len(arr)), mode='constant')
            if len(arr) < max_length else arr[:max_length]
            for arr in integrand
        ])
        self.sigma = np.trapz(integrand, dx=self.dz, axis=1)

    def export_surface_tension(self, file_path):
        """
        Export surface tension data to a pickle file.
        
        Parameters:
        - file_path: String, path (including filename) where the pickle file will be saved.
        
        Raises:
        - ValueError: If compute_surface_tension has not been called.
        """
        if self.sigma is None:
            raise ValueError("Must call compute_surface_tension before exporting surface tension.")
        
        data = {
            'd': self.d,
            'width': self.integration_width,
            'window_width': self.window_width,
            's_grid': self.s_grid,
            's_col': self.s_col,
            'sigma': self.sigma,
            'z_max_coords': self.z_max_coords
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def plot_gradient_energy_density(self, plot_boundaries=False, cutoff=None):
        """
        Plot the gradient energy density with a logarithmic colorbar and optional max rho_grad path.
        
        Parameters:
        - plot_boundaries: Boolean flag to plot the max rho_grad path and boundaries (default: False).
        - cutoff: Optional float, values of rho_grad below this are set to cutoff for plotting (default: None).
                 If None, set to 5 orders of magnitude below the max rho_grad (rounded down to nearest power of 10).
        
        Returns:
        - fig: Matplotlib figure object.
        - ax: Matplotlib axes object.
        
        Raises:
        - ValueError: If plot_boundaries=True and compute_max_gradient_energy_density has not been called.
        """
        if plot_boundaries and self.z_max_coords is None:
            raise ValueError("Must call compute_max_gradient_energy_density before plotting boundaries.")

        fig, ax = plt.subplots()
        if cutoff is None:
            max_rho_grad = np.max(self.rho_grad)
            base_power = 10 ** np.floor(np.log10(max_rho_grad))
            cutoff = base_power / 1e5
            cutoff = max(cutoff, 1e-10)

        if cutoff is not None:
            plot_data = np.maximum(self.rho_grad, cutoff)
            vmin = max(cutoff, 1e-10)
        else:
            plot_data = self.rho_grad
            vmin = max(np.min(self.rho_grad), 1e-10)

        im = ax.imshow(
            plot_data[::-1],
            extent=[min(self.z_grid), max(self.z_grid), 0, self.s_max],
            cmap=mpl.cm.inferno.reversed(),
            norm=LogNorm(vmin=vmin, vmax=np.max(self.rho_grad))
        )
        ax.set_ylabel('s')
        ax.set_xlabel('z')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r'$\rho$',
            rotation=0,
            labelpad=-40,
            x=1.2,
            y=1.1
        )

        if plot_boundaries:
            z_center = self.z_max_coords
            z_lower = z_center - self.integration_width / 2
            z_upper = z_center + self.integration_width / 2
            s_plot = self.s_coords_valid
            ax.plot(z_center, s_plot, color='g', linestyle='-', linewidth=1)
            ax.plot(z_lower, s_plot, color='g', linestyle='--', linewidth=1)
            ax.plot(z_upper, s_plot, color='g', linestyle='--', linewidth=1)

        return fig, ax

    def plot_surface_tension(self, small_s_range=(3e-1, 7e-1), large_s_range=(1.05, 1.3), normalized=False):
        """
        Plot the surface tension sigma with power-law fits for small and large s regions.
        
        Parameters:
        - small_s_range: Tuple (min, max) for small s/s_col fitting range.
        - large_s_range: Tuple (min, max) for large s/s_col fitting range.
        
        Returns:
        - fig: Matplotlib figure object.
        - ax: Matplotlib axes object.
        
        Raises:
        - ValueError: If compute_surface_tension has not been called.
        """
        if self.sigma is None:
            raise ValueError("Must call compute_surface_tension before plotting surface tension.")

        if normalized:
            sigma = self.sigma / self.sigma.max()
        else:
            sigma = self.sigma

        s_scaled = self.s_coords_valid / self.s_col

        def power_law(s_scaled, a, b):
            return a * (s_scaled**b)

        small_s_mask = (s_scaled >= small_s_range[0]) & (s_scaled <= small_s_range[1])
        s_small = s_scaled[small_s_mask]
        sigma_small = sigma[small_s_mask]
        popt_small, _ = curve_fit(power_law, s_small, sigma_small, p0=[1.0, 1.0])

        large_s_mask = (s_scaled >= large_s_range[0]) & (s_scaled <= large_s_range[1])
        s_large = s_scaled[large_s_mask]
        sigma_large = sigma[large_s_mask]
        popt_large, _ = curve_fit(power_law, s_large, sigma_large, p0=[1.0, 1.0])

        sigma_fit_small = power_law(s_small, *popt_small)
        sigma_fit_large = power_law(s_large, *popt_large)

        fig, ax = plt.subplots()
        ax.plot(s_scaled, sigma, 'k-', label='Surface tension')
        ax.fill_between(
            s_scaled, 0, sigma, where=small_s_mask, color='lightblue', alpha=0.5,
            label='Small s fit region'
        )
        ax.fill_between(
            s_scaled, 0, sigma, where=large_s_mask, color='lightcoral', alpha=0.5,
            label='Large s fit region'
        )
        ax.plot(
            s_small, sigma_fit_small, 'b--',
            label=rf'Small $s/s_{{\text{{col}}}}$ fit: {popt_small[0]:.2f} $(s/s_{{\text{{col}}}})^{{{popt_small[1]:.2f}}}$'
        )
        ax.plot(
            s_large, sigma_fit_large, 'r--',
            label=rf'Large $s/s_{{\text{{col}}}}$ fit: {popt_large[0]:.2f} $(s/s_{{\text{{col}}}})^{{{popt_large[1]:.2f}}}$'
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r'$s/s_{\text{col}}$')
        ax.set_ylabel(r'$\sigma$')
        ax.set_title(
            rf"Surface Tension $\sigma = \int_{{z_{{\text{{wall}}}}-\Delta}}^{{z_{{\text{{wall}}}}+\Delta}} dz \rho_{{\text{{grad}}}}^{{\text{{wall}}}}$ "
            rf", $d={self.d:.2f}$, $\Delta = {{{self.integration_width:.2f}}}$"
        )
        ax.grid(True)
        ax.legend()
        ax.set_xlim(left=1e-1)

        return fig, ax

    def plot_field_evolution(
        self,
        field_idx: int = 0,
        field_name: Optional[str] = None,
        title: str = "",
        max_npoints_plot: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        phi_scaled = self.phi1[field_idx, :, :]  # Shape: (n_s, n_z)
        s = self.s_grid
        z = self.z_grid
        
        if max_npoints_plot is not None:
            max_s_points, max_z_points = max_npoints_plot
            if len(s) > max_s_points:
                s_step = max(1, len(s) // max_s_points)
                s = s[::s_step]
                phi_scaled = phi_scaled[::s_step, :]
            if len(z) > max_z_points:
                z_step = max(1, len(z) // max_z_points)
                z = z[::z_step]
                phi_scaled = phi_scaled[:, ::z_step]
        
        fig, ax = plt.subplots()
        im = ax.imshow(phi_scaled[::-1], cmap='RdBu_r', extent=[min(z), max(z), 0, max(s)], **kwargs)
        ax.set_xlabel('z')
        ax.set_ylabel('s', rotation=0)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r'$\Phi_{%d}$' % (field_idx+1) if field_name is None else field_name,
            rotation=0,
            labelpad=-40,
            x=1,
            y=1.1
        )
        ax.set_title(f"Field {field_idx + 1} {title}".strip() if field_name is None else f"Field = {field_name} {title}".strip())
        im.set_clim(1-np.max(phi_scaled), np.max(phi_scaled))
        return fig, ax

    def plot_norm_field_evolution(
        self,
        title: str = "",
        max_npoints_plot: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        phi_squared = np.sum(self.phi1**2, axis=0)  # Shape: (n_s, n_z)
        phi_norm = np.sqrt(phi_squared)
        s = self.s_grid
        z = self.z_grid
        
        if max_npoints_plot is not None:
            max_s_points, max_z_points = max_npoints_plot
            if len(s) > max_s_points:
                s_step = max(1, len(s) // max_s_points)
                s = s[::s_step]
                phi_norm = phi_norm[::s_step, :]
            if len(z) > max_z_points:
                z_step = max(1, len(z) // max_z_points)
                z = z[::z_step]
                phi_norm = phi_norm[:, ::z_step]
        
        fig, ax = plt.subplots()
        im = ax.imshow(phi_norm[::-1], cmap='RdBu_r', extent=[min(z), max(z), 0, max(s)], **kwargs)
        ax.set_xlabel('z')
        ax.set_ylabel('s', rotation=0)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r'$\sqrt{\sum_{i} \Phi_i^2}$',
            rotation=0,
            labelpad=-40,
            x=1.2,
            y=1.1
        )
        ax.set_title(f"Norm of Fields {title}".strip())
        im.set_clim(0, np.max(phi_norm))
        return fig, ax
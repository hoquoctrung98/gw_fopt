import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .filter_dataframe import filter_dataframe


class SurfaceTensionFitter:
    def __init__(
        self,
        df_surface_tension,
        filter_dict,
        d_values,
        xi_powers,
        nonlinear_cutoff_ratio,
        x_fit_right,
        Lw,
    ):
        """
        Initialize the SurfaceTensionFitter directly from the surface tension DataFrame.

        Parameters:
        -----------
        df_surface_tension : pandas.DataFrame
            DataFrame containing surface tension data with columns including:
            'couplings, width, d, s_grid, sigma
        filter_dict : dict
            Fixed filter parameters (e.g., {"couplings": 0.2, "width": 2}).
            Must NOT contain the key 'd' — this will be varied.
        d_values : list or array-like
            List of bubble wall thickness values (d) to analyze.
        xi_powers : array-like
            Powers of xi in the theoretical prediction (e.g. [0, 1]).
        nonlinear_cutoff_ratio : float
            Ratio defining the nonlinear regime cutoff.
        x_fit_right : float
            Right boundary of the fitting range in s/s_col.
        Lw : float
            Characteristic length scale (R_outer - R_inner) of the field profile.
        """
        if "d" in filter_dict:
            raise ValueError(
                "filter_dict must not contain the key 'd' — it is varied separately."
            )

        self.df_surface_tension = df_surface_tension
        self.filter_dict = filter_dict
        self.d_values = np.asarray(d_values)
        self.xi_powers = np.asarray(xi_powers)
        self.nonlinear_cutoff_ratio = nonlinear_cutoff_ratio
        self.x_fit_right = x_fit_right
        self.Lw = Lw
        self.potential = "quartic"

        # Results storage
        self.results = []
        self.x_common = None
        self.mean_y = None
        self.err_y = None
        self.x_fit = None
        self.mean_y_fit = None
        self.err_y_fit = None
        self.a_0 = None
        self.a_mid = None
        self.a_env = None
        self.chi2_min = None
        self.NDOF = None
        self.chi2_ndof_min = None
        self.err_a_0 = None
        self.err_a_mid = None
        self.chi2_grid = None
        self.delta_chi2_ndof = None

    def _extract_surface_tension_data(self, d):
        """
        Extract x_data = 2s/d and y_data = sigma for a given d.
        Skips the first point (s=0).
        """
        current_filter = self.filter_dict.copy()
        current_filter["d"] = d
        row = filter_dataframe(self.df_surface_tension, current_filter).iloc[0]

        s_grid = np.asarray(row["s_grid"])
        sigma = np.asarray(row["sigma"])

        x_data = s_grid * 2 / d
        y_data = sigma
        return x_data[1:], y_data[1:]  # drop s=0 point

    def _fit_func(self, x, a_0, a_mid):
        """Fitting function: σ/σ₀ ~ a₀/x^(ξ₀+2) + a_mid/x^(ξ₁+2)"""
        return a_0 / (x ** (self.xi_powers[0] + 2)) + a_mid / (
            x ** (self.xi_powers[1] + 2)
        )

    def _chi2(self, params, data):
        a, b = params
        x, y, y_err = data[:, 0], data[:, 1], data[:, 2]
        model = self._fit_func(x, a, b)
        return np.sum(((model - y) ** 2) / y_err**2)

    def fit_data(self, minimization_method="COBYQA"):
        """Main fitting routine over all d values."""
        for d in self.d_values:
            x_data, y_data = self._extract_surface_tension_data(d)

            # Rough estimate of σ₀ from small-x
            small_mask = (x_data >= 0.5) & (x_data <= 0.8)
            if not np.any(small_mask):
                raise ValueError(f"No data in small-x range [0.5, 0.8] for d={d}")
            sigma0 = np.mean(y_data[small_mask])

            self.results.append(
                {"d": d, "sigma0": sigma0, "x_data": x_data, "y_data": y_data}
            )

        # === Build common x-grid and interpolate normalized σ/σ₀ ===
        df_res = pd.DataFrame(self.results)
        longest_idx = df_res["x_data"].apply(len).idxmax()
        self.x_common = df_res.loc[longest_idx, "x_data"]

        interpolated = []
        for _, row in df_res.iterrows():
            f = interp1d(
                row["x_data"],
                row["y_data"] / row["sigma0"],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated.append(f(self.x_common))
        interpolated = np.array(interpolated)

        self.mean_y = np.nanmean(interpolated, axis=0)
        self.err_y = np.nanmax(interpolated, axis=0) - np.nanmin(interpolated, axis=0)

        # === Define fitting window ===
        x_fit_left = max(
            1 + 2 * self.nonlinear_cutoff_ratio * self.Lw / d for d in self.d_values
        )
        fit_mask = (self.x_common > x_fit_left) & (self.x_common < self.x_fit_right)
        self.x_fit = self.x_common[fit_mask]
        self.mean_y_fit = self.mean_y[fit_mask]
        self.err_y_fit = self.err_y[fit_mask] + 1e-12  # avoid zero errors

        # === Perform global fit ===
        data = np.column_stack([self.x_fit, self.mean_y_fit, self.err_y_fit])

        constraints = [
            {"type": "ineq", "fun": lambda p: p[0]},  # a_0 >= 0
            {"type": "ineq", "fun": lambda p: p[1]},  # a_mid >= 0
            {"type": "ineq", "fun": lambda p: 1 - p[0] - p[1]},  # a_env >= 0
        ]

        res = minimize(
            self._chi2,
            x0=[0.2, 0.1],
            args=(data,),
            constraints=constraints,
            method=minimization_method,
            options={"maxiter": 1000},
        )

        self.a_0, self.a_mid = res.x
        self.a_env = 1.0 - self.a_0 - self.a_mid
        self.chi2_min = res.fun
        self.NDOF = len(self.x_fit) - 2
        self.chi2_ndof_min = self.chi2_min / self.NDOF

    # === The rest of the methods remain exactly the same ===

    def compute_confidence_region(self, n_points=200):
        """Compute 1σ contour in (a_0, a_mid) plane without displaying."""
        a0_grid = np.linspace(0, 1, n_points)
        amid_grid = np.linspace(0, 1, n_points)
        A0, Amid = np.meshgrid(a0_grid, amid_grid)

        data = np.column_stack([self.x_fit, self.mean_y_fit, self.err_y_fit])
        chi2_grid = np.vectorize(lambda a0, amid: self._chi2([a0, amid], data))(
            A0, Amid
        )

        self.chi2_grid = chi2_grid
        self.delta_chi2_ndof = chi2_grid / self.NDOF - self.chi2_ndof_min

        fig, ax = plt.subplots()
        cs = ax.contour(A0, Amid, self.delta_chi2_ndof, levels=[1.0])
        paths = cs.allsegs[0]  # list of arrays
        if paths:
            verts = np.vstack(paths)
            self.err_a_0 = max(np.abs(verts[:, 0] - self.a_0))
            self.err_a_mid = max(np.abs(verts[:, 1] - self.a_mid))
        else:
            self.err_a_0 = self.err_a_mid = np.nan
            print("Warning: No closed 1σ contour found.")
        plt.close(fig)

    def plot_surface_tension(self, fig, ax):
        """Plot normalized surface tension curves and global fit."""

        for res in self.results:
            ax.plot(
                res["x_data"],
                res["y_data"] / res["sigma0"],
                "o",
                ms=2,
                alpha=0.6,
                label=f"d = {res['d']}",
            )

        ax.plot(self.x_common, self.mean_y, "k-", lw=2, label=r"Mean $\sigma/\sigma_0$")
        ax.fill_between(
            self.x_common,
            self.mean_y - self.err_y / 2,
            self.mean_y + self.err_y / 2,
            color="gray",
            alpha=0.3,
        )

        model_fit = self._fit_func(self.x_fit, self.a_0, self.a_mid)
        ax.plot(self.x_fit, model_fit, "r--", lw=2, label="Best fit")
        ax.fill_between(
            self.x_fit, self.mean_y.min(), model_fit, color="red", alpha=0.3, zorder=5
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$s/s_{\rm col}$")
        ax.set_ylabel(r"$\sigma / \sigma_0$")
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend()

        x_fit_left = max(
            1 + 2 * self.nonlinear_cutoff_ratio * self.Lw / d for d in self.d_values
        )
        ax.set_title(
            rf"Fit to mean $\sigma/\sigma_0$ for ${x_fit_left:.2f} \leq s_\text{{fit}}/s_\text{{col}} \leq {self.x_fit_right:.2f}$"
            + "\n"
            + rf"$\xi = [{', '.join(f'{x:}' for x in self.xi_powers[:-1])}$"
            + r"$, \infty]$"
            + rf", $a_\xi = [{self.a_0:.2e}, {self.a_mid:.2e}, {self.a_env:.2e}]$, $\chi^2 = {self.chi2_min:.2e}$, $\chi_\nu^2 = {self.chi2_ndof_min:.2e}$",
            fontsize=14,
        )

        return fig, ax

    def plot_confidence_region(self, n_points=200):
        if self.chi2_grid is None:
            self.compute_confidence_region(n_points)

        a0_grid = np.linspace(0, 1, n_points)
        amid_grid = np.linspace(0, 1, n_points)
        A0, Amid = np.meshgrid(a0_grid, amid_grid)

        fig, ax = plt.subplots(figsize=(8, 6))
        levels = np.arange(0, 3.1, 0.5)
        cf = ax.contourf(A0, Amid, self.delta_chi2_ndof, levels=levels, cmap="RdYlBu_r")
        fig.colorbar(cf, label=r"$\Delta\chi^2_\nu$")

        ax.contour(
            A0, Amid, self.delta_chi2_ndof, levels=[1.0], colors="black", linewidths=2
        )

        ax.plot(
            self.a_0,
            self.a_mid,
            "o",
            color="white",
            markeredgecolor="black",
            markersize=8,
        )

        # Forbidden region a_0 + a_mid > 1
        ax.contourf(
            A0, Amid, A0 + Amid, levels=[1, 2], colors="gray", alpha=0.3, hatches=["//"]
        )

        ax.set_xlabel(r"$a_0$")
        ax.set_ylabel(r"$a_{\rm mid}$")
        ax.set_title(
            rf"Confidence region – $a_0 = {self.a_0:.3f} \pm {self.err_a_0:.3f}$, "
            rf"$a_{{\rm mid}} = {self.a_mid:.3f} \pm {self.err_a_mid:.3f}$"
        )
        ax.grid(True, ls="--", alpha=0.5)
        return fig, ax

    def get_best_fit_params(self):
        return self.a_0, self.a_mid, self.a_env

    def get_best_fit_chi2(self):
        return self.chi2_min

    def get_ndof(self):
        return self.NDOF

    def get_err_params(self):
        if self.err_a_0 is None:
            self.compute_confidence_region()
        return self.err_a_0, self.err_a_mid

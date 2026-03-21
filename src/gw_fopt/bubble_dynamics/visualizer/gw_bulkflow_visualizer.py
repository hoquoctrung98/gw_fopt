from typing import List, Optional, Self, Sequence, Tuple, Union

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .get_colors import get_colors_from_cmap
from .mean_error_type import MeanErrorType


class GwBulkflowVisualizer:
    """
    Visualize averaged gravitational wave energy density spectra from precomputed data.

    This class assumes that directional averaging (over 'direction_index') has already
    been performed, so each history contributes one spectrum.
    """

    def __init__(
        self,
        w_arr: np.ndarray,
        gw_arr: np.ndarray,
        coefficients_sets: List[Sequence[float]],
        powers_sets: List[Sequence[Optional[float]]],
    ):
        """
        Initialize the visualizer with precomputed spectral data.

        Parameters
        ----------
        w_arr : np.ndarray
            1D array of angular frequencies (unscaled simulation units).
        gw_data : np.ndarray
            3D array of shape `(n_histories, n_sets, n_frequencies)` containing
            unscaled GW energy density spectra for each history and coefficient/power set.
        coefficients_sets : list of sequences
            Coefficient sets used for labeling (e.g., [[1.0], [1.0]]).
        powers_sets : list of sequences
            Power sets used for labeling (e.g., [[None], [3.0]]).
        """
        self.w_arr = np.asarray(w_arr)
        if self.w_arr.ndim != 1:
            raise ValueError("w_arr must be a 1D array.")
        self.gw_data = np.asarray(gw_arr)
        if self.gw_data.ndim != 3:
            raise ValueError("gw_data must be 3D: (n_histories, n_sets, n_freq).")
        if self.gw_data.shape[-1] != len(self.w_arr):
            raise ValueError("Last dimension of gw_data must match length of w_arr.")
        if len(coefficients_sets) != len(powers_sets):
            raise ValueError("coefficients_sets and powers_sets must have same length.")
        if len(coefficients_sets) != self.gw_data.shape[1]:
            raise ValueError("Number of sets must match gw_data.shape[1].")

        self.coefficients_sets = coefficients_sets
        self.powers_sets = powers_sets

        # Plot state
        self.scale_factors = (1.0, 1.0)
        self.labels = None
        self.colors = None
        self.linestyles = None
        self.w_plot = None
        self.gw_plot = None
        self.gw_mean = None
        self.gw_lower = None
        self.gw_upper = None

        # Fitting results
        self.lmfit_results = None
        self.fit_ranges = None
        self.p0 = None

    def generate_labels(self) -> List[str]:
        """
        Generate plot labels based on coefficient/power configurations.

        Returns
        -------
        labels : list of str
            Human-readable LaTeX labels for each set.
        """
        labels = []
        for b_set, p_set in zip(self.coefficients_sets, self.powers_sets):
            b = np.ravel(b_set)
            p = np.ravel(p_set)
            if np.allclose(b, 0) or (len(p) == 1 and p[0] is None):
                labels.append("Envelope Approximation")
            else:
                b_str = ", ".join(f"{x:.2f}" for x in b)
                xi_str = ", ".join(f"{x - 3:.2f}" for x in p if x is not None)
                labels.append(f"$b_j=({b_str}), \\xi_j=({xi_str})$")

        return labels

    def setup(
        self,
        scale_factors: Tuple[float, float] = (1.0, 1.0),
        error_bars: MeanErrorType = MeanErrorType.SEM,
        scale_error: float = 1.0,
        labels: Optional[List[str]] = None,
        colors: Optional[List] = None,
        linestyles: Optional[List[str]] = None,
    ) -> Self:
        """
        Setup the plot state and compute statistics.

        Parameters
        ----------
        scale_factors : tuple (w_scale, omega_scale)
            Multiplicative factors applied to frequency and amplitude for unit conversion.
        error_bars : MeanErrorType
            Type of error to display.
        scale_error : float, default=1.0
            Additional multiplicative factor applied to error bands.
        labels : list of str, optional
            Custom labels for each set.
        colors : list, optional
            Custom colors for each set.
        linestyles : list of str, optional
            Custom linestyles for each set.

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        self.scale_factors = scale_factors

        # Apply scaling for physical units
        self.w_plot = scale_factors[0] * self.w_arr
        self.gw_plot = scale_factors[1] * self.gw_data

        # Compute statistics
        self.gw_mean = self.gw_plot.mean(axis=0)  # (n_sets, n_freq)
        gw_data_std = self.gw_plot.std(axis=0)
        n_histories = self.gw_plot.shape[0]
        log_gw = np.log(self.gw_plot)
        log_mean = log_gw.mean(axis=0)
        self.gw_mean = np.exp(log_mean)
        log_std = log_gw.std(axis=0, ddof=1)

        if error_bars == MeanErrorType.SEM:
            err = gw_data_std / np.sqrt(n_histories)
            self.gw_lower = self.gw_mean - scale_error * err
            self.gw_upper = self.gw_mean + scale_error * err
        elif error_bars == MeanErrorType.STD:
            self.gw_lower = self.gw_mean - scale_error * gw_data_std
            self.gw_upper = self.gw_mean + scale_error * gw_data_std
        elif error_bars == MeanErrorType.ABS:
            self.gw_lower = self.gw_plot.min(axis=0)
            self.gw_upper = self.gw_plot.max(axis=0)
        else:
            raise ValueError(f"Unsupported error_bars: {error_bars}")

        # Generate labels and colors
        self.labels = labels if labels is not None else self.generate_labels()
        self.colors = (
            colors
            if colors is not None
            else get_colors_from_cmap(cmap="Set1", n_colors=len(self.labels))
        )
        self.linestyles = (
            linestyles if linestyles is not None else ["-"] * len(self.labels)
        )

        return self

    def plot_fit_regions(
        self,
        ax: plt.Axes,
        ir_range: Tuple[float, float] = (1e-2, 1e-1),
        uv_range: Tuple[float, float] = (5e1, 1e3),
    ) -> Self:
        """
        Plot IR and UV fit regions.

        Parameters
        ----------
        ir_range, uv_range : tuple of float
            Frequency ranges for IR and UV fit regions (in physical units after scaling).

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        ax.axvspan(*ir_range, color="gray", alpha=0.15, zorder=0)
        ax.axvspan(*uv_range, color="gray", alpha=0.25, zorder=0)
        return self

    def plot_gw_mean(self, ax: plt.Axes, **kwargs) -> Self:
        """
        Plot the mean GW spectra.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to `ax.plot`.

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        for spectrum, label, color, ls in zip(
            self.gw_mean, self.labels, self.colors, self.linestyles
        ):
            ax.plot(
                self.w_plot,
                spectrum,
                label=label,
                color=color,
                linestyle=ls,
                **kwargs,
            )
        return self

    def plot_gw_error(self, ax: plt.Axes, **kwargs) -> Self:
        """
        Plot error bands for GW spectra.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to `ax.fill_between`.

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        for i, color in enumerate(self.colors):
            ax.fill_between(
                self.w_plot,
                self.gw_lower[i],
                self.gw_upper[i],
                color=color,
                **kwargs,
            )
        return self

    @staticmethod
    def double_power_law_spectrum(w, spectrum_peak, w_peak, a_ir, a_uv):
        """Double power-law spectrum model.
        return (
            spectrum_peak
            * (a_uv + a_ir)
            * w_peak**a_uv
            * w**a_ir
            / (a_ir * w_peak ** (a_ir + a_uv) + a_uv * w ** (a_ir + a_uv))
        )"""
        w = np.asarray(w, dtype=float)

        # Early out for invalid inputs
        if not (np.all(w > 0) and w_peak > 0 and spectrum_peak > 0):
            return np.full_like(w, np.nan)

        # Check coefficients that go into log
        if a_ir <= 0 or a_uv <= 0 or (a_uv + a_ir) <= 0:
            return np.full_like(w, np.nan)

        # Now safe to take logs
        log_w = np.log(w)
        log_wp = np.log(w_peak)

        # log(numerator)
        log_num = (
            np.log(spectrum_peak) + np.log(a_uv + a_ir) + a_uv * log_wp + a_ir * log_w
        )

        # log(denominator) — use logaddexp for stability
        log_term1 = np.log(a_ir) + (a_ir + a_uv) * log_wp
        log_term2 = np.log(a_uv) + (a_ir + a_uv) * log_w
        log_denom = np.logaddexp(log_term1, log_term2)

        log_result = log_num - log_denom

        return np.exp(log_result)

    def get_fit_results(
        self,
        fit_ranges: Union[Tuple[float, float], List[Tuple[float, float]]],
        model: Optional[lmfit.Model] = None,
        params_list: Optional[List[lmfit.Parameters]] = None,
        **kwargs_fit,
    ) -> Self:
        """
        Fit GW spectra using lmfit with optional custom model/parameters.

        Parameters
        ----------
        fit_ranges : tuple or list of tuples
            Frequency ranges for fitting (in scaled physical units).
        model : lmfit.Model, optional
            Custom model to use. If None, uses default double_power_law_spectrum.
        params_list : list of lmfit.Parameters, optional
            List of parameter templates, one per dataset (n_sets).
            Each template defines parameter names, initial values, and constraints.
            If None, creates default bounded params for each set.
            If provided, length must match n_sets.

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        if self.gw_lower is None or self.gw_upper is None:
            raise RuntimeError(
                "Error bands not computed. Call setup() before get_fit_results()."
            )

        n_sets = len(self.gw_mean)
        self.fit_ranges = fit_ranges

        # Normalize fit_ranges to list
        if isinstance(fit_ranges, tuple) and len(fit_ranges) == 2:
            fit_ranges_normalized = [fit_ranges] * n_sets
        elif isinstance(fit_ranges, list):
            if len(fit_ranges) != n_sets:
                raise ValueError(
                    f"Length of fit_ranges list ({len(fit_ranges)}) must match "
                    f"number of sets ({n_sets})."
                )
            fit_ranges_normalized = fit_ranges
        else:
            raise TypeError("fit_ranges must be tuple or list of tuples.")

        # Validate params_list length if provided
        if params_list is not None:
            if not isinstance(params_list, list):
                raise TypeError(
                    "params_list must be a list of lmfit.Parameters or None."
                )
            if len(params_list) != n_sets:
                raise ValueError(
                    f"Length of params_list ({len(params_list)}) must match "
                    f"number of sets ({n_sets})."
                )
            # Validate each element is lmfit.Parameters
            for i, p in enumerate(params_list):
                if not isinstance(p, lmfit.Parameters):
                    raise TypeError(
                        f"params_list[{i}] must be lmfit.Parameters, got {type(p)}."
                    )

        # Prepare default model if not provided
        if model is None:
            model = lmfit.Model(
                GwBulkflowVisualizer.double_power_law_spectrum, independent_vars=["w"]
            )

        self.lmfit_results = []

        for i, (gw_data, orig_label, fit_range) in enumerate(
            zip(self.gw_mean, self.labels, fit_ranges_normalized)
        ):
            w_data = self.w_plot

            # Mask data to fit range
            if fit_range is not None:
                mask = (w_data >= fit_range[0]) & (w_data <= fit_range[1])
                w_fit = w_data[mask]
                gw_fit = gw_data[mask]
            else:
                mask = np.ones_like(w_data, dtype=bool)
                w_fit = w_data
                gw_fit = gw_data

            # Compute weights from error bands
            sigma_full = (self.gw_upper[i] - self.gw_lower[i]) / 2.0
            sigma_fit = sigma_full[mask]
            min_valid_sigma = (
                1e-12 * np.max(sigma_fit) if np.max(sigma_fit) > 0 else 1e-20
            )
            sigma_fit = np.where(sigma_fit > 0, sigma_fit, min_valid_sigma)

            # Prepare parameters: use params_list[i] or create defaults
            if params_list is not None:
                # Clone template to avoid mutating original
                params_i = params_list[i].copy()
            else:
                # Default initial guesses from data peak (for default model only)
                idx_peak = np.argmax(gw_fit)
                params_i = lmfit.Parameters()
                params_i.add("spectrum_peak", value=gw_fit[idx_peak], min=1e-30)
                params_i.add("w_peak", value=w_fit[idx_peak], min=1e-30)
                params_i.add("a_ir", value=3.0, min=1e-6)
                params_i.add("a_uv", value=4.0, min=1e-6)

            # Perform fit
            result = model.fit(
                gw_fit,
                params_i,
                w=w_fit,
                weights=1.0 / sigma_fit,
                **kwargs_fit,
            )

            lmfit_result_entry = {
                "original_label": orig_label,
                "fit_range": fit_range,
                "lmfit_result": result,  # Single source of truth for all fit info
            }
            self.lmfit_results.append(lmfit_result_entry)
        self.lmfit_results = pd.DataFrame(self.lmfit_results)

        return self

    def plot_gw_fit(
        self,
        ax: plt.Axes,
        fit_params_labels: Optional[dict[str, str]] = None,
        show_fit_range: bool = False,
        show_fit_values: bool = True,
        show_fit_errors: bool = True,
        ci_sigma: float = 1.0,
        params_per_line: int = 2,
        allow_vary: bool = True,
        w_plot: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Self:
        """
        Plot fitted spectra using lmfit.MinimizerResult for evaluation.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to plot on.
        fit_params_labels : dict[str, str], optional
            Mapping from parameter names to LaTeX labels.
            Example: {"spectrum_peak": r"$\Omega_{\rm peak}$", "a_ir": r"$a_{\rm IR}$"}
            If None, uses default dict mapping for common parameter names.
        show_fit_range : bool, default=False
            Highlight fitting regions if True.
        show_fit_errors : bool, default=True
            If True, append asymmetric 1σ errors as ^{+high}_{-low} next to each parameter value.
        params_per_line : int, default=2
            Number of parameters to display per line in the legend label.
        allow_vary : bool, default=True
            If False, only show varied parameters in the labels.
        **kwargs : dict
            Additional keyword arguments for ax.plot.

        Returns
        -------
        self : GwBulkflowVisualizer
            Returns self for chaining.
        """
        if self.w_plot is None or self.gw_mean is None:
            raise RuntimeError(
                "Plot state not initialized. Call setup() before plot_gw_fit()."
            )
        if self.lmfit_results is None:
            raise RuntimeError("Call method get_fit_results first")
        if w_plot is None:
            w_plot = np.logspace(
                np.log10(self.w_plot.min()), np.log10(self.w_plot.max()), 500
            )

        # Default label mapping for common parameters
        default_param_labels_dict = {
            "spectrum_peak": r"$\Omega_{\rm peak}$",
            "w_peak": r"$\omega_{\rm peak}/\beta$",
            "a_ir": r"$a_{\rm IR}$",
            "a_uv": r"$a_{\rm UV}$",
        }

        # Handle fit_params_labels (dict only now)
        if fit_params_labels is None:
            param_labels = default_param_labels_dict
        elif isinstance(fit_params_labels, dict):
            param_labels = fit_params_labels
        else:
            raise TypeError("fit_params_labels must be dict[str, str] or None")

        # Optional: show fit ranges
        if show_fit_range and self.fit_ranges is not None:
            n_sets = len(self.gw_mean)
            if isinstance(self.fit_ranges, tuple):
                ax.axvspan(
                    self.fit_ranges[0],
                    self.fit_ranges[1],
                    color="gray",
                    alpha=0.05,
                    zorder=1,
                    label="Fit range" if n_sets == 1 else None,
                )
            elif isinstance(self.fit_ranges, list):
                for rng, color in zip(self.fit_ranges, self.colors):
                    if rng is not None:
                        ax.axvspan(
                            rng[0],
                            rng[1],
                            color=color,
                            alpha=0.05,
                            zorder=1,
                        )

        # Plot each fitted spectrum
        for i, fit_entry in self.lmfit_results.iterrows():
            color = self.colors[i]
            result: lmfit.model.ModelResult = fit_entry["lmfit_result"]
            ci_out = result.conf_interval(sigmas=[ci_sigma])

            # Dynamically extract parameters from lmfit result
            params = result.params
            if allow_vary:
                param_items = [(name, p) for name, p in params.items()]
            else:
                param_items = [(name, p) for name, p in params.items() if p.vary]

            if show_fit_values:
                # Build label with dynamic parameter names and values
                label_parts = [f"Fitting {fit_entry['original_label']}\n"]
                for idx, (param_name, param_obj) in enumerate(param_items):
                    param_value = param_obj.value

                    # Get display name from label mapping
                    display_name = param_labels.get(param_name, param_name)

                    # Add error bars if requested and available
                    if show_fit_errors:
                        stderr = param_obj.stderr

                        if stderr is not None and np.isfinite(stderr) and stderr > 0:
                            # Compute asymmetric errors using profile likelihood if available
                            # Otherwise use symmetric stderr
                            ci_param = ci_out[param_name]
                            # ci is typically [(conf_level, lower, upper), ...]
                            # Get confidence interval corresponding to sigma
                            low, _, high = sorted([ci[1] for ci in ci_param])
                            err_high = high - param_value
                            err_low = param_value - low
                            # LaTeX superscript/subscript format: ^{+high}_{-low}
                            value_str = rf"${param_value:.2e}^{{+{err_high:.2e}}}_{{-{err_low:.2e}}}$, "
                        else:
                            value_str = rf"${param_value:.2e}$,  "
                    else:
                        value_str = rf"${param_value:.2e}$,  "

                    label_parts.append(f"{display_name}={value_str}")

                    # Insert newline after every params_per_line parameters
                    if (idx + 1) % params_per_line == 0 and idx < len(param_items) - 1:
                        label_parts.append("\n")
                # Join parts, handling newlines properly
                label = " ".join(label_parts).replace(" \n", "\n")
            else:
                label = f"Fitting {fit_entry['original_label']}"

            # Evaluate fit on fine grid using lmfit's built-in eval
            spectrum_fitted = result.eval(w=w_plot)

            # Plot with merged kwargs
            default_kwargs = {
                "color": color,
                "label": label,
                "linestyle": self.linestyles[i],
            }
            merged_kwargs = {**default_kwargs, **kwargs}
            ax.plot(w_plot, spectrum_fitted, **merged_kwargs)

        return self

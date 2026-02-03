import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class PowerLawFitter:
    """
    Class to fit power-law models to data and visualize results on a log-log scale.
    """

    def __init__(
        self,
        x_data,
        y_data,
        fitting_powers,
        weights=None,
        x_range=None,
        small_x_range=None,
        large_x_range=None,
        match_condition=False,
    ):
        """
        Initialize the fitter with data and parameters.

        Parameters:
        - x_data: Array of positive x values
        - y_data: Array of positive y values
        - fitting_powers: List of power-law exponents for large x (e.g., [-2, -3, -4])
        - weights: Array of weights for large x fit (default: 1/y_data**3)
        - x_range: Tuple (xmin, xmax) for plot (default: [min(x_data), max(x_data)])
        - small_x_range: Tuple (xmin, xmax) for small x fit (default: None, skips fit)
        - large_x_range: Tuple (xmin, xmax) for large x fit (default: None, uses all data)
        - match_condition: If True, constrain sum of large x coefficients to small x 'a'

        Raises:
        - ValueError: If inputs are invalid
        """
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.fitting_powers = np.asarray(fitting_powers)
        self.weights = weights
        self.x_range = (
            x_range if x_range is not None else (np.min(x_data), np.max(x_data))
        )
        self.small_x_range = small_x_range
        self.large_x_range = large_x_range
        self.match_condition = match_condition
        self.fit_results = {"small": None, "large": None}
        self.fit_info = {}  # Store diagnostic info for get_info

        self._validate_inputs()
        self._fit()

    def _validate_inputs(self):
        """Validate input data and parameters."""
        if len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must have the same length")
        if len(self.x_data) == 0:
            raise ValueError("x_data and y_data must not be empty")
        if len(self.fitting_powers) == 0:
            raise ValueError("powers list must not be empty")
        if np.any(self.x_data <= 0) or np.any(self.y_data <= 0):
            raise ValueError("x_data and y_data must be positive for log-log plot")
        if np.any(np.isnan(self.x_data)) or np.any(np.isnan(self.y_data)):
            raise ValueError("x_data and y_data must not contain NaN")
        if self.match_condition and self.small_x_range is None:
            raise ValueError("match_condition=True requires small_x_range")
        if not (self.x_range[0] <= self.x_range[1] and self.x_range[0] > 0):
            raise ValueError("x_range must be valid and positive")
        if self.small_x_range is not None:
            if not (
                self.small_x_range[0] <= self.small_x_range[1]
                and self.small_x_range[0] > 0
            ):
                raise ValueError("small_x_range must be valid and positive")
        if self.large_x_range is not None:
            if not (
                self.large_x_range[0] <= self.large_x_range[1]
                and self.large_x_range[0] > 0
            ):
                raise ValueError("large_x_range must be valid and positive")

    def _fit_small_range(self):
        """Fit y = a * x^b in the small x range."""
        mask_small = (self.x_data >= self.small_x_range[0]) & (
            self.x_data <= self.small_x_range[1]
        )
        if not np.any(mask_small):
            raise ValueError("No data points in small_x_range")

        x_small = self.x_data[mask_small]
        y_small = self.y_data[mask_small]

        def model_small(x, a, b):
            return a * x**b

        try:
            popt_small, pcov_small = curve_fit(
                model_small, x_small, y_small, p0=[1.0, 1.0], bounds=(0, np.inf)
            )
        except RuntimeError as e:
            raise RuntimeError(f"Small x range fitting failed: {e}")

        def fitted_model_small(x):
            return model_small(x, *popt_small)

        self.fit_results["small"] = {
            "popt": popt_small,
            "pcov": pcov_small,
            "model": fitted_model_small,
        }
        self.fit_info["small_fit"] = (
            f"Small x fit: y = {popt_small[0]:.4e} x^{popt_small[1]:.4e}"
        )

    def _fit_large_range(self):
        """Fit sum(A_i * x^powers[i]) in the large x range with all A_i > 0."""
        # Filter data
        if self.large_x_range is not None:
            mask_large = (self.x_data >= self.large_x_range[0]) & (
                self.x_data <= self.large_x_range[1]
            )
            if not np.any(mask_large):
                raise ValueError("No data points in large_x_range")
            x_large = self.x_data[mask_large]
            y_large = self.y_data[mask_large]
            weights_large = (
                self.weights[mask_large] if self.weights is not None else None
            )
        else:
            x_large = self.x_data
            y_large = self.y_data
            weights_large = self.weights

        # Set default weights
        if weights_large is None:
            weights_large = 1.0 / (y_large**3)
        else:
            weights_large = np.asarray(weights_large)
            if len(weights_large) != len(y_large):
                raise ValueError("weights must match filtered y_data length")
            if np.any(weights_large <= 0):
                raise ValueError("weights must be positive")

        if self.match_condition:
            if self.fit_results["small"] is None:
                raise ValueError(
                    "match_condition=True requires a successful small x fit"
                )
            a_small = self.fit_results["small"]["popt"][0]

            if len(self.fitting_powers) == 1:

                def model_large(x, *params):
                    return a_small * x ** self.fitting_powers[0]

                popt_large = [a_small]
                pcov_large = np.zeros((1, 1))

                def fitted_model_large(x):
                    return model_large(x, *popt_large)

                self.fit_results["large"] = {
                    "popt": popt_large,
                    "pcov": pcov_large,
                    "model": fitted_model_large,
                }
                terms = [
                    f"{popt_large[i]:.4e} x^{self.fitting_powers[i]}"
                    for i in range(len(self.fitting_powers))
                ]
                self.fit_info["large_fit"] = f"Large x fit: y = {' + '.join(terms)}"
                self.fit_info["sum_coeffs"] = (
                    f"Sum of large x coefficients: {sum(popt_large):.4e}, expected: {a_small:.4e}"
                )
                if self.fit_results["small"] is not None:
                    y_small_at_x1 = self.fit_results["small"]["model"](1.0)
                    y_large_at_x1 = self.fit_results["large"]["model"](1.0)
                    self.fit_info["at_x1"] = (
                        f"At x=1: small fit y = {y_small_at_x1:.4e}, large fit y = {y_large_at_x1:.4e}"
                    )
            else:

                def model_large(x, *b_params):
                    b_params = np.asarray(b_params)
                    a_params = []
                    product = 1.0
                    for i in range(len(b_params)):
                        a_i = a_small * b_params[i] * product
                        a_params.append(a_i)
                        product *= 1 - b_params[i]
                    a_n_plus_1 = a_small - sum(a_params)
                    a_params.append(a_n_plus_1)
                    return sum(a * x**p for a, p in zip(a_params, self.fitting_powers))

                # p0_large
                mask_x1 = np.abs(x_large - 1.0) < 0.1
                if np.any(mask_x1):
                    y_at_x1 = np.mean(y_large[mask_x1])
                    b_init = min(0.9, a_small / (y_at_x1 * len(self.fitting_powers)))
                else:
                    b_init = 0.5 / len(self.fitting_powers)
                p0_large = np.full(len(self.fitting_powers) - 1, b_init)
                bounds_lower = np.full(len(self.fitting_powers) - 1, 1e-10)
                bounds_upper = np.full(len(self.fitting_powers) - 1, 1.0 - 1e-10)

                try:
                    popt_b, pcov_b = curve_fit(
                        model_large,
                        x_large,
                        y_large,
                        p0=p0_large,
                        sigma=1.0 / np.sqrt(weights_large),
                        absolute_sigma=True,
                        bounds=(bounds_lower, bounds_upper),
                        maxfev=10000,
                    )
                    # Compute a_params
                    a_params = []
                    product = 1.0
                    for i in range(len(popt_b)):
                        a_i = a_small * popt_b[i] * product
                        a_params.append(a_i)
                        product *= 1 - popt_b[i]
                    a_n_plus_1 = a_small - sum(a_params)
                    popt_large = np.append(a_params, a_n_plus_1)

                    # Numerical Jacobian for pcov_large
                    def a_params_func(b):
                        a = []
                        prod = 1.0
                        for bb in b:
                            ai = a_small * bb * prod
                            a.append(ai)
                            prod *= 1 - bb
                        a.append(a_small - sum(a))
                        return np.array(a)

                    delta = 1e-8
                    n_a = len(self.fitting_powers)
                    n_b = len(popt_b)
                    J = np.zeros((n_a, n_b))
                    for j in range(n_b):
                        b_pert = popt_b.copy()
                        b_pert[j] += delta
                        a_pert = a_params_func(b_pert)
                        b_pert[j] = popt_b[j] - delta
                        a_mert = a_params_func(b_pert)
                        J[:, j] = (a_pert - a_mert) / (2 * delta)
                    pcov_large = J @ pcov_b @ J.T

                    def fitted_model_large(x):
                        return sum(
                            a * x**p for a, p in zip(popt_large, self.fitting_powers)
                        )

                    self.fit_results["large"] = {
                        "popt": popt_large,
                        "pcov": pcov_large,
                        "model": fitted_model_large,
                    }
                    terms = [
                        f"{popt_large[i]:.4e} x^{self.fitting_powers[i]}"
                        for i in range(len(self.fitting_powers))
                    ]
                    self.fit_info["large_fit"] = f"Large x fit: y = {' + '.join(terms)}"
                    self.fit_info["sum_coeffs"] = (
                        f"Sum of large x coefficients: {sum(popt_large):.4e}, expected: {a_small:.4e}"
                    )
                    if self.fit_results["small"] is not None:
                        y_small_at_x1 = self.fit_results["small"]["model"](1.0)
                        y_large_at_x1 = self.fit_results["large"]["model"](1.0)
                        self.fit_info["at_x1"] = (
                            f"At x=1: small fit y = {y_small_at_x1:.4e}, large fit y = {y_large_at_x1:.4e}"
                        )
                except (RuntimeError, ValueError) as e:
                    raise RuntimeError(
                        f"Large x range fitting failed with constraint: {e}"
                    )
        else:

            def model_large(x, *params):
                if np.any(np.array(params) <= 0):
                    raise ValueError("Negative coefficient detected")
                return sum(
                    param * x**power
                    for param, power in zip(params, self.fitting_powers)
                )

            p0_large = np.full(len(self.fitting_powers), 0.1)
            bounds_lower = np.full(len(self.fitting_powers), 1e-10)
            bounds_upper = np.full(len(self.fitting_powers), np.inf)

            try:
                popt_large, pcov_large = curve_fit(
                    model_large,
                    x_large,
                    y_large,
                    p0=p0_large,
                    sigma=1.0 / np.sqrt(weights_large),
                    absolute_sigma=True,
                    bounds=(bounds_lower, bounds_upper),
                )

                def fitted_model_large(x):
                    return sum(
                        a * x**p for a, p in zip(popt_large, self.fitting_powers)
                    )

                self.fit_results["large"] = {
                    "popt": popt_large,
                    "pcov": pcov_large,
                    "model": fitted_model_large,
                }
                terms = [
                    f"{popt_large[i]:.4e} x^{self.fitting_powers[i]}"
                    for i in range(len(self.fitting_powers))
                ]
                self.fit_info["large_fit"] = f"Large x fit: y = {' + '.join(terms)}"
            except RuntimeError as e:
                raise RuntimeError(f"Large x range fitting failed: {e}")

    def _fit(self):
        """Perform fitting for small and large x ranges."""
        if self.small_x_range is not None:
            self._fit_small_range()
        self._fit_large_range()

    def get_normalized_coeffs_with_cov(self):
        """
        Return the normalized coefficients (large popt / small a) and their covariance.
        For match_condition=True, only the first n-1 coefficients have uncertainties.

        Returns:
        --------
        coeffs : np.ndarray
            Normalized coefficients.
        pcov_coeffs : np.ndarray
            Covariance matrix of the normalized coefficients (last row/column zero for match_condition=True).
        """
        if self.fit_results["small"] is None or self.fit_results["large"] is None:
            raise ValueError("Both small and large fits required for normalized coeffs")
        a_small = self.fit_results["small"]["popt"][0]
        var_a_small = self.fit_results["small"]["pcov"][0, 0]
        popt_large = self.fit_results["large"]["popt"]
        pcov_large = self.fit_results["large"]["pcov"]
        n = len(popt_large)
        coeffs = popt_large / a_small
        pcov_coeffs = np.zeros_like(pcov_large)
        if self.match_condition:
            for i in range(n - 1):
                for j in range(n - 1):
                    if i == j:
                        pcov_coeffs[i, i] = (1 / a_small**2 * pcov_large[i, i]) + (
                            popt_large[i] ** 2 / a_small**4 * var_a_small
                        )
                    else:
                        pcov_coeffs[i, j] = 1 / a_small**2 * pcov_large[i, j]
            # Last coefficient has no variance since it's determined by sum constraint
            pcov_coeffs[n - 1, :] = 0
            pcov_coeffs[:, n - 1] = 0
        else:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        pcov_coeffs[i, i] = (1 / a_small**2 * pcov_large[i, i]) + (
                            popt_large[i] ** 2 / a_small**4 * var_a_small
                        )
                    else:
                        pcov_coeffs[i, j] = 1 / a_small**2 * pcov_large[i, j]
        return coeffs, pcov_coeffs

    def get_rss(self):
        """
        Compute the residual sum of squares (RSS) for the large-x fit.

        Returns:
        - float: RSS value
        """
        if self.fit_results["large"] is None:
            raise ValueError("Large x fit not performed")
        mask_large = (
            (self.x_data >= self.large_x_range[0])
            & (self.x_data <= self.large_x_range[1])
            if self.large_x_range is not None
            else np.ones_like(self.x_data, dtype=bool)
        )
        x_large = self.x_data[mask_large]
        y_large = self.y_data[mask_large]
        y_pred = self.fit_results["large"]["model"](x_large)
        return np.sum((y_large - y_pred) ** 2)

    def get_tss(self):
        """
        Compute the total sum of squares (TSS) for the large-x fit.

        Returns:
        - float: TSS value
        """
        mask_large = (
            (self.x_data >= self.large_x_range[0])
            & (self.x_data <= self.large_x_range[1])
            if self.large_x_range is not None
            else np.ones_like(self.x_data, dtype=bool)
        )
        y_large = self.y_data[mask_large]
        return np.sum((y_large - np.mean(y_large)) ** 2)

    def get_fit_metrics(self):
        """
        Compute AIC, BIC, and adjusted R-squared for the large-x fit.

        Returns:
        - dict: Contains 'rss', 'aic', 'bic', 'r2', 'r2_adj', 'n_params'
        """
        rss = self.get_rss()
        tss = self.get_tss()
        n = np.sum(
            (self.x_data >= self.large_x_range[0])
            & (self.x_data <= self.large_x_range[1])
            if self.large_x_range is not None
            else np.ones_like(self.x_data, dtype=bool)
        )
        k = (
            len(self.fitting_powers) - 1
            if self.match_condition
            else len(self.fitting_powers)
        )

        # AIC
        aic = 2 * k + n * np.log(rss / n)

        # BIC
        bic = k * np.log(n) + n * np.log(rss / n)

        # R-squared and Adjusted R-squared
        r2 = 1 - (rss / tss)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        return {
            "rss": rss,
            "aic": aic,
            "bic": bic,
            "r2": r2,
            "r2_adj": r2_adj,
            "n_params": k,
        }

    def compare_models(self, fitting_powers_list, metric="bic"):
        """
        Compare models with different fitting_powers using the specified metric.

        Parameters:
        - fitting_powers_list: List of lists/arrays of fitting_powers to compare (e.g., [[-2], [-2, -3], [-2, -3, -4]])
        - metric: str, 'aic' or 'bic' to select the best model

        Returns:
        - dict: Best fitting_powers and corresponding metrics
        """
        if metric not in ["aic", "bic"]:
            raise ValueError("metric must be 'aic' or 'bic'")

        results = []
        original_fitting_powers = self.fitting_powers
        original_fit_results = self.fit_results
        original_fit_info = self.fit_info

        for fitting_powers in fitting_powers_list:
            # Update fitting_powers and re-run _fit
            self.fitting_powers = np.asarray(fitting_powers)
            self.fit_results = {"small": self.fit_results["small"], "large": None}
            self.fit_info = {}
            try:
                self._fit_large_range()
                metrics = self.get_fit_metrics()
                results.append({"fitting_powers": fitting_powers, "metrics": metrics})
            except Exception as e:
                results.append(
                    {"fitting_powers": fitting_powers, "metrics": {"error": str(e)}}
                )

        # Restore original state
        self.fitting_powers = original_fitting_powers
        self.fit_results = original_fit_results
        self.fit_info = original_fit_info

        # Select best model based on metric (skip models with errors)
        valid_results = [r for r in results if "error" not in r["metrics"]]
        if not valid_results:
            raise ValueError("No models fitted successfully")
        best_result = min(valid_results, key=lambda x: x["metrics"][metric])

        return {
            "best_fitting_powers": best_result["fitting_powers"],
            "best_metrics": best_result["metrics"],
            "all_results": results,
        }

    def plot_model_comparison(self, fitting_powers_list, metric="bic"):
        """
        Visualize the comparison of models with different fitting_powers.

        Parameters:
        - fitting_powers_list: List of lists/arrays of fitting_powers to compare
        - metric: str, 'aic' or 'bic' to plot

        Returns:
        - fig: Matplotlib figure object
        - ax: Matplotlib axes object
        """
        if metric not in ["aic", "bic"]:
            raise ValueError("metric must be 'aic' or 'bic'")

        # Run comparison
        comparison = self.compare_models(fitting_powers_list, metric=metric)

        # Prepare data for plotting
        n_params = []
        metric_values = []
        labels = []
        for result in comparison["all_results"]:
            if "error" in result["metrics"]:
                continue
            n_params.append(result["metrics"]["n_params"])
            metric_values.append(result["metrics"][metric])
            labels.append(f"[{', '.join(map(str, result['fitting_powers']))}]")

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            range(len(n_params)),
            metric_values,
            tick_label=labels,
            color="tab:blue",
            alpha=0.7,
        )

        # Highlight best model
        best_idx = labels.index(
            f"[{', '.join(map(str, comparison['best_fitting_powers']))}]"
        )
        bars[best_idx].set_color("tab:orange")
        bars[best_idx].set_label(
            f"Best model ({metric.upper()} = {comparison['best_metrics'][metric]:.2f})"
        )

        # Plot settings
        ax.set_xlabel("Fitting Powers")
        ax.set_ylabel(metric.upper())
        ax.set_title(
            f"Model Comparison by {metric.upper()} (Best: {', '.join(map(str, comparison['best_fitting_powers']))})"
        )
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()

        return fig, ax

    def plot(self, small_plot_to_x1=True, large_plot_from_x1=True):
        """
        Plot the data and fitted models on a log-log scale.

        Parameters:
        - small_plot_to_x1: If True, plot small x fit up to x=1; else, full small_x_range
        - large_plot_from_x1: If True, plot large x fit from x=1; else, full large_x_range

        Returns:
        - fig: Matplotlib figure object
        - ax: Matplotlib axes object
        """
        fig, ax = plt.subplots()
        ax.plot(
            self.x_data,
            self.y_data,
            color="red",
            label="Data",
            marker="o",
            ms=2,
            linestyle="none",
        )

        if self.fit_results["small"] is not None:
            x_max = 1.0 if small_plot_to_x1 else self.small_x_range[1]
            x_small_fit = np.geomspace(self.small_x_range[0], x_max, 50)
            y_small_fit = self.fit_results["small"]["model"](x_small_fit)
            ax.plot(x_small_fit, y_small_fit, color="green", lw=2, label="Small x Fit")

        large_plot_range = (
            self.large_x_range if self.large_x_range is not None else self.x_range
        )
        x_min = 1.0 if large_plot_from_x1 else large_plot_range[0]
        x_large_fit = np.geomspace(x_min, large_plot_range[1], 50)
        y_large_fit = self.fit_results["large"]["model"](x_large_fit)
        ax.plot(x_large_fit, y_large_fit, color="blue", lw=2, label="Large x Fit")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(self.x_range)
        ax.grid(True, which="both", ls="--")
        ax.legend()
        return fig, ax

    def get_models(self):
        """
        Return the fitted models.

        Returns:
        - dict: {'small': callable or None, 'large': callable}
        """
        return {
            "small": self.fit_results["small"]["model"]
            if self.fit_results["small"] is not None
            else None,
            "large": self.fit_results["large"]["model"],
        }

    def get_formatted_string(self):
        """
        Return formatted string for powers and coefficients: [-p_i-2], a_ξ = [a_i].

        Returns:
        - str: Formatted string
        """
        if self.fit_results["large"] is None:
            return ""
        fitting_coefficients = self.fit_results["large"]["popt"]
        formatted_string = (
            "[" + ", ".join([f"{x}" for x in (-self.fitting_powers - 2)]) + "]"
        )
        formatted_string += r"$,~ a_\xi = $"
        formatted_string += (
            "[" + ", ".join([f"{x:.2e}" for x in fitting_coefficients]) + "]"
        )
        return formatted_string

    def get_info(self):
        """
        Return diagnostic information about the fits.

        Returns:
        - str: Formatted string containing small x fit, large x fit, sum of large x coefficients,
               and values at x=1 (if applicable).
        """
        info_lines = []
        if "small_fit" in self.fit_info:
            info_lines.append(self.fit_info["small_fit"])
        if "sum_coeffs" in self.fit_info:
            info_lines.append(self.fit_info["sum_coeffs"])
        if "at_x1" in self.fit_info:
            info_lines.append(self.fit_info["at_x1"])
        if "large_fit" in self.fit_info:
            info_lines.append(self.fit_info["large_fit"])
        return "\n".join(info_lines)

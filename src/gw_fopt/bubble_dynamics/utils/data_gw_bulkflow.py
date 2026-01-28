import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from gw_fopt.bubble_gw import utils

from .filter_dataframe import filter_dataframe


class DataGwBulkFlow:
    def __init__(self, df, d0=1.0):
        """Initialize method

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing bulk flow data with columns '_pset_id', 'w_sample',
            'c_bulkflow', 'c_envelope', 'powers_sets_sample'
        d0 : float
            The bubble nucleation corresponding to df
        """
        self.d0 = d0
        self.df = filter_dataframe(df, {"d": d0})
        self.cos_thetak_arr = np.sort(self.df.cos_theta_k.values)
        self.w_arr = utils.sample(*self.df["w_sample"][0])

    def de_dlogw_dcosthetak(self, xi_powers, xi_coefficients, check_sum=True):
        """
        Extract and process gravitational wave bulk flow data from a DataFrame to compute
        a weighted average GW energy density spectrum.

        Parameters:
        -----------
        xi_powers : list or np.ndarray
            List or array of power indices (numbers) or None values to process. None indicates use of 'c_envelope'.
        xi_coefficients : list or np.ndarray
            List or array of weights for each entry in xi_powers. Must match length of xi_powers.
        check_sum : bool, optional
            If True, check if xi_coefficients sum to 1 (default: True).

        Returns:
        --------
        omega_gw_avg : numpy.ndarray
            Averaged GW energy density spectrum, with non-positive values set to 1e-50.

        Raises:
        -------
        ValueError
            If input validations fail (e.g., invalid frequencies, coefficients, or array shapes).
        """
        if not isinstance(xi_powers, (list, np.ndarray)) or not isinstance(
            xi_coefficients, (list, np.ndarray)
        ):
            raise ValueError(
                "xi_powers and xi_coefficients must be lists or numpy arrays"
            )
        xi_powers = np.asarray(xi_powers, dtype=object)
        xi_coefficients = np.asarray(xi_coefficients)
        if len(xi_powers) != len(xi_coefficients):
            raise ValueError("xi_powers and xi_coefficients must have the same length")
        if check_sum and not np.isclose(np.sum(xi_coefficients), 1.0):
            raise ValueError("xi_coefficients must sum to 1")
        if not all(p is None or isinstance(p, (int, float)) for p in xi_powers):
            raise ValueError("xi_powers must contain numbers or None")

        w_arr = self.w_arr
        de_dlnw_dcosthetak = []

        for _, row in self.df.iterrows():
            Cplus_sum = np.zeros_like(w_arr, dtype=complex)
            Cminus_sum = np.zeros_like(w_arr, dtype=complex)

            powers_available = None
            if any(p is not None for p in xi_powers):
                powers_available = list(utils.sample(*row["powers_sets_sample"]))

            for p, coeff in zip(xi_powers, xi_coefficients):
                if p is None:
                    c_matrix = np.asarray(row["c_envelope"])
                    if c_matrix.ndim != 2 or c_matrix.shape[0] != 2:
                        raise ValueError(
                            f"c_envelope for _pset_id {row['_pset_id']} must be a 2D array with shape[0]=2"
                        )
                    Cplus_sum += coeff * c_matrix[0, :]
                    Cminus_sum += coeff * c_matrix[1, :]
                else:
                    power = p + 3
                    if powers_available is None:
                        raise ValueError(
                            "powers_sets_sample required for non-None powers"
                        )
                    if power not in powers_available:
                        raise ValueError(
                            f"Power {power} not in powers_sets_sample for _pset_id {row['_pset_id']}"
                        )
                    power_idx = powers_available.index(power)
                    c_matrix = np.asarray(row["c_bulkflow"])
                    if c_matrix.ndim == 2 and c_matrix.shape[0] == 2:
                        c_matrix = c_matrix[:, np.newaxis, :]
                    elif c_matrix.ndim != 3 or c_matrix.shape[0] != 2:
                        raise ValueError(
                            f"c_bulkflow for _pset_id {row['_pset_id']} must be a 2D or 3D array with shape[0]=2"
                        )
                    Cplus_sum += coeff * c_matrix[0, power_idx, :]
                    Cminus_sum += coeff * c_matrix[1, power_idx, :]

            spectrum = (
                8
                * np.pi
                * w_arr**3
                * (np.abs(Cplus_sum) ** 2 + np.abs(Cminus_sum) ** 2)
            )
            de_dlnw_dcosthetak.append(spectrum)

        de_dlnw_dcosthetak = np.array(de_dlnw_dcosthetak)
        return de_dlnw_dcosthetak

    def de_dlogw(self, xi_powers, xi_coefficients, check_sum=True):
        """
        Extract and process gravitational wave bulk flow data from a DataFrame to compute
        a weighted average GW energy density spectrum.

        Parameters:
        -----------
        xi_powers : list or np.ndarray
            List or array of power indices (numbers) or None values to process. None indicates use of 'c_envelope'.
        xi_coefficients : list or np.ndarray
            List or array of weights for each entry in xi_powers. Must match length of xi_powers.
        check_sum : bool, optional
            If True, check if xi_coefficients sum to 1 (default: True).

        Returns:
        --------
        omega_gw_avg : numpy.ndarray
            Averaged GW energy density spectrum, with non-positive values set to 1e-50.

        Raises:
        -------
        ValueError
            If input validations fail (e.g., invalid frequencies, coefficients, or array shapes).
        """
        de_dlogw_dcosthetak = self.de_dlogw_dcosthetak(
            xi_powers=xi_powers,
            xi_coefficients=xi_coefficients,
            check_sum=check_sum,
        )
        return 2.0 * np.trapezoid(de_dlogw_dcosthetak, x=self.cos_thetak_arr, axis=0)

    def de_dlogw_interp(
        self, xi_powers, xi_coefficients, check_sum=True, d=None, rho_vacuum=1.0
    ):
        """
        Extract and process gravitational wave bulk flow data from a DataFrame to compute
        a weighted average GW energy density spectrum.

        Parameters:
        -----------
        d : float
            bubble nucleation where we need to compute the GW spectrum
            If None, d=self.d0
        rho_vacuum: float
            The difference of vacuum energy density between TV and FV of the potential
        xi_powers : list or np.ndarray
            List or array of power indices (numbers) or None values to process. None indicates use of 'c_envelope'.
        xi_coefficients : list or np.ndarray
            List or array of weights for each entry in xi_powers. Must match length of xi_powers.
        check_sum : bool, optional
            If True, check if xi_coefficients sum to 1 (default: True).

        Returns:
        --------
        omega_gw_avg : numpy.ndarray
            Averaged GW energy density spectrum, with non-positive values set to 1e-50.

        Raises:
        -------
        ValueError
            If input validations fail (e.g., invalid frequencies, coefficients, or array shapes).
        """
        if d is None:
            d = self.d0
        de_dlogw = (
            (d / self.d0) ** 5
            * rho_vacuum**2
            * self.de_dlogw(
                xi_powers=xi_powers,
                xi_coefficients=xi_coefficients,
                check_sum=check_sum,
            )
        )
        return CubicSpline(self.w_arr / d, de_dlogw)

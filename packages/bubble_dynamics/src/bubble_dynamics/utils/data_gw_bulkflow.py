import numpy as np
import pandas as pd
from .filter_dataframe import filter_dataframe
from bubble_gw.utils import sample

class DataGwBulkFlow():
    def __init__(self, df):
        """Initialize method

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing bulk flow data with columns '_pset_id', 'w_sample',
            'c_bulkflow', 'c_envelope', 'powers_sets_sample'
        """
        self.df = df
        self.df_filtered = df

    def set_filter(self, filter_dict):
        self.df_filtered = filter_dataframe(self.df, filter_dict)

    def extract_gw_bulkflow(self, xi_powers, xi_coefficients, check_sum=True):
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
        if not isinstance(xi_powers, (list, np.ndarray)) or not isinstance(xi_coefficients, (list, np.ndarray)):
            raise ValueError("xi_powers and xi_coefficients must be lists or numpy arrays")
        xi_powers = np.asarray(xi_powers, dtype=object)
        xi_coefficients = np.asarray(xi_coefficients)
        if len(xi_powers) != len(xi_coefficients):
            raise ValueError("xi_powers and xi_coefficients must have the same length")
        if check_sum and not np.isclose(np.sum(xi_coefficients), 1.0):
            raise ValueError("xi_coefficients must sum to 1")
        if not all(p is None or isinstance(p, (int, float)) for p in xi_powers):
            raise ValueError("xi_powers must contain numbers or None")

        w_arr = None
        de_dlnw_dcosthetak_arr = []

        for _, row in self.df_filtered.iterrows():
            w_arr_from_sample = np.asarray(sample(*row['w_sample']))
            if w_arr_from_sample.ndim != 1 or np.any(w_arr_from_sample <= 0):
                raise ValueError(f"w_arr for _pset_id {row['_pset_id']} must be a 1D array of positive frequencies")
            if w_arr is None:
                w_arr = w_arr_from_sample
            elif not np.allclose(w_arr, w_arr_from_sample):
                raise ValueError(f"Inconsistent w_sample across df_bulkflow rows for _pset_id {row['_pset_id']}")

            Cplus_sum = np.zeros_like(w_arr, dtype=complex)
            Cminus_sum = np.zeros_like(w_arr, dtype=complex)

            powers_available = None
            if any(p is not None for p in xi_powers):
                powers_available = list(sample(*row['powers_sets_sample']))

            for p, coeff in zip(xi_powers, xi_coefficients):
                if p is None:
                    c_matrix = np.asarray(row['c_envelope'])
                    if c_matrix.ndim != 2 or c_matrix.shape[0] != 2:
                        raise ValueError(f"c_envelope for _pset_id {row['_pset_id']} must be a 2D array with shape[0]=2")
                    Cplus_sum += coeff * c_matrix[0, :]
                    Cminus_sum += coeff * c_matrix[1, :]
                else:
                    power = p + 3
                    if powers_available is None:
                        raise ValueError("powers_sets_sample required for non-None powers")
                    if power not in powers_available:
                        raise ValueError(f"Power {power} not in powers_sets_sample for _pset_id {row['_pset_id']}")
                    power_idx = powers_available.index(power)
                    c_matrix = np.asarray(row['c_bulkflow'])
                    if c_matrix.ndim == 2 and c_matrix.shape[0] == 2:
                        c_matrix = c_matrix[:, np.newaxis, :]
                    elif c_matrix.ndim != 3 or c_matrix.shape[0] != 2:
                        raise ValueError(f"c_bulkflow for _pset_id {row['_pset_id']} must be a 2D or 3D array with shape[0]=2")
                    Cplus_sum += coeff * c_matrix[0, power_idx, :]
                    Cminus_sum += coeff * c_matrix[1, power_idx, :]

            de_dlnw_dcosthetak = 8 * np.pi * w_arr**3 * (np.abs(Cplus_sum)**2 + np.abs(Cminus_sum)**2)
            de_dlnw_dcosthetak_arr.append(de_dlnw_dcosthetak)

        de_dlnw_dcosthetak_arr = np.array(de_dlnw_dcosthetak_arr)
        return de_dlnw_dcosthetak_arr

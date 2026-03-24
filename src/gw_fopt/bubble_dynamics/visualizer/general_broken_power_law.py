import lmfit
import numpy as np
from numba import njit


@njit(cache=True)
def _log1pexp_stable(x):
    """
    Stable scalar computation of log(1 + exp(x)).
    Equivalent to np.logaddexp(0, x), but Numba-friendly.
    """
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    else:
        return np.log1p(np.exp(x))


@njit(cache=True)
def general_broken_power_law_numba(
    w, amplitude, wscales, widths, powers, windex_amplitude
):
    """
    Numba-optimized version of general_broken_power_law.

    Parameters
    ----------
    w : 1D float64 array
    amplitude : float
    wscales : 1D float64 array, length n
    widths : 1D float64 array, length n
    powers : 1D float64 array, length n+1
    windex_amplitude : int
        1-based index, same convention as your original code
    """
    n_w = w.shape[0]
    n_scales = wscales.shape[0]

    result = np.empty(n_w, dtype=np.float64)

    w_amplitude = wscales[windex_amplitude - 1]

    # Initialize log_result = powers[0] * log(w / w_amplitude)
    for i in range(n_w):
        result[i] = powers[0] * np.log(w[i] / w_amplitude)

    # Add each transition contribution
    for k in range(n_scales):
        expv = 1.0 / widths[k]
        dpowers = powers[k + 1] - powers[k]

        log_ratio_wm = expv * np.log(w_amplitude / wscales[k])
        log_den = _log1pexp_stable(log_ratio_wm)  # scalar

        coeff = widths[k] * dpowers

        for i in range(n_w):
            log_ratio_w = expv * np.log(w[i] / wscales[k])
            log_num = _log1pexp_stable(log_ratio_w)
            result[i] += coeff * (log_num - log_den)

    # Exponentiate and multiply by amplitude
    for i in range(n_w):
        result[i] = amplitude * np.exp(result[i])

    return result


@njit(cache=True)
def asymptotic_broken_power_law_numba(
    w, amplitude, wscales, widths, powers, windex_amplitude, region
):
    """
    Numba-optimized version of asymptotic_broken_power_law.

    region is 0-indexed, same as your original function.
    """
    n_w = w.shape[0]
    n = wscales.shape[0]
    k = region

    w_amplitude = wscales[windex_amplitude - 1]

    # Compute amplitude prefactor (renamed to avoid shadowing input amplitude)
    amp_prefactor = 1.0

    for j in range(n):
        wscale = wscales[j]
        width = widths[j]
        dpowers = powers[j + 1] - powers[j]

        ratio_m = (w_amplitude / wscale) ** (1.0 / width)

        # Denominator factor applies to all transitions
        amp_prefactor /= (1.0 + ratio_m) ** (width * dpowers)

        # Fired transitions: j < k
        if j < k:
            amp_prefactor *= (w_amplitude / wscale) ** dpowers

    total_amp = amplitude * amp_prefactor
    slope = powers[k]

    result = np.empty(n_w, dtype=np.float64)
    for i in range(n_w):
        result[i] = total_amp * (w[i] / w_amplitude) ** slope

    return result


def general_broken_power_law(w, amplitude, wscales, widths, powers, windex_amplitude):
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    wscales = np.ascontiguousarray(np.asarray(wscales, dtype=np.float64))
    widths = np.ascontiguousarray(np.asarray(widths, dtype=np.float64))
    powers = np.ascontiguousarray(np.asarray(powers, dtype=np.float64))

    return general_broken_power_law_numba(
        w, amplitude, wscales, widths, powers, windex_amplitude
    )


def asymptotic_broken_power_law(
    w, amplitude, wscales, widths, powers, windex_amplitude, region
):
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    wscales = np.ascontiguousarray(np.asarray(wscales, dtype=np.float64))
    widths = np.ascontiguousarray(np.asarray(widths, dtype=np.float64))
    powers = np.ascontiguousarray(np.asarray(powers, dtype=np.float64))

    return asymptotic_broken_power_law_numba(
        w, amplitude, wscales, widths, powers, windex_amplitude, region
    )


class TriplePowerLawFittingInput:
    def __init__(self, params_list):
        self.params_list = params_list
        self.model = lmfit.Model(self.__call__, independent_vars=["w"])
        self.fitting_function_latex = r"$\Delta(\omega) = \widehat{\Delta} \left(\dfrac{\omega}{\widehat{\omega}_2}\right)^{p_\text{IR1}} \left( \dfrac{1+\left(\omega/\widehat{\omega}_1\right)^{1/\varepsilon_1}} {1+\left(\widehat{\omega}_2/\widehat{\omega}_1\right)^{1/\varepsilon_1}} \right)^{\varepsilon_1(p_\text{IR2}-p_\text{IR1})} \left( \dfrac{1+\left(\omega/\widehat{\omega}_2\right)^{1/\varepsilon_2}} {2} \right)^{\varepsilon_2(p_\text{UV}-p_\text{IR2})}$"
        self.fit_params_labels = {
            "amplitude": r"$\widehat{\Delta}$",
            "wscale1": r"$\widehat{\omega}_1 / \beta$",
            "wscale2": r"$\widehat{\omega}_2 / \beta$",
            "width1": r"$\varepsilon_1$",
            "width2": r"$\varepsilon_2$",
            "power1": r"$p_\text{IR1}$",
            "power2": r"$p_\text{IR2}$",
            "power3": r"$p_\text{UV}$",
        }

    def __call__(
        self, w, amplitude, wscale1, wscale2, width1, width2, power1, power2, power3
    ):
        return general_broken_power_law(
            w=w,
            amplitude=amplitude,
            wscales=[wscale1, wscale2],
            widths=[width1, width2],
            powers=[power1, power2, power3],
            windex_amplitude=2,
        )


class DoublePowerLawFittingInput:
    def __init__(self, params_list):
        self.params_list = params_list
        self.model = lmfit.Model(self.__call__, independent_vars=["w"])
        self.fitting_function_latex = r"$\Delta(\omega) = \widehat{\Delta} \left( \dfrac{\omega}{\widehat{\omega}} \right)^{p_\text{IR}} \left( \dfrac{1 + \left(\omega/\widehat{\omega}\right)^{1/\varepsilon}}{2} \right)^{\varepsilon(p_\text{UV} - p_\text{IR})}$"
        self.fit_params_labels = {
            "amplitude": r"$\widehat{\Delta}$",
            "wscale": r"$\widehat{\omega} / \beta$",
            "width": r"$\varepsilon$",
            "power1": r"$p_\text{IR}$",
            "power2": r"$p_\text{UV}$",
        }

    def __call__(
        self,
        w,
        amplitude,
        wscale,
        width,
        power1,
        power2,
    ):
        return general_broken_power_law(
            w=w,
            amplitude=amplitude,
            wscales=[wscale],
            widths=[width],
            powers=[power1, power2],
            windex_amplitude=1,
        )

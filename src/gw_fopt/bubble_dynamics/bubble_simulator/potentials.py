from typing import Tuple

import numpy as np
from numba import njit
from scipy.signal import argrelextrema

from .generic_potential import GenericPotential


# Here |phi|^2 = (phi_re^2+phi_im^2)/2 for correct normalization of kinetic term
class U1Potential(GenericPotential):
    def __init__(self, delta_V, kappa, v):
        self.Ndim = 2
        self.params = delta_V, kappa, v
        self.params_str = rf"$\Delta V={delta_V}, \kappa={kappa}, v={v}$"
        self.name = "u1"
        self.name_latex = r"$\dfrac{V(\Phi)}{\Delta V} = 1 + \kappa \dfrac{|\Phi|^2}{v^2} + \dfrac{|\Phi|^4}{v^4} \left[ (\kappa + 2)\log\left( \dfrac{|\Phi|^2}{v^2} \right) - (\kappa + 1) \right]$"
        self.rho_vacuum = 2

    @staticmethod
    @njit
    def V0_numba(
        phi_re: np.ndarray,
        phi_im: np.ndarray,
        delta_V: float,
        kappa: float,
        v: float,
    ) -> np.ndarray:
        """
        Compute V0 for arrays of field values.
        phi_re, phi_im: shape (n_points,) or any broadcastable shape.
        """
        phi2 = (phi_re**2 + phi_im**2) / 2.0
        v2 = v * v
        v4 = v2 * v2
        result = np.empty_like(phi2)
        for i in range(phi2.size):
            p2 = phi2.flat[i]
            if p2 < 1e-100:
                p2 = 1e-100
            rval = (
                1.0
                + kappa * p2 / v2
                + (p2 * p2 / v4) * ((kappa + 2.0) * np.log(p2 / v2) - (kappa + 1.0))
            )
            result.flat[i] = 2.0 * delta_V * rval
        return result

    @staticmethod
    @njit
    def dV0_numba(
        phi_re: np.ndarray,
        phi_im: np.ndarray,
        delta_V: float,
        kappa: float,
        v: float,
    ) -> np.ndarray:
        """
        Compute dV/d(phi_re) and dV/d(phi_im).
        Returns array of shape (*phi_re.shape, 2): [..., 0] = dV/dphi_re,
                                                    [..., 1] = dV/dphi_im.
        """
        v2 = v * v
        v4 = v2 * v2
        out = np.empty(phi_re.shape + (2,), dtype=np.float64)
        for i in range(phi_re.size):
            re = phi_re.flat[i]
            im = phi_im.flat[i]
            p2 = (re * re + im * im) / 2.0
            if p2 < 1e-100:
                p2 = 1e-100
            tmp = -kappa * (p2 - v2) + 2.0 * (2.0 + kappa) * p2 * np.log(p2 / v2)
            scale = 2.0 * delta_V / v4
            out.flat[2 * i] = scale * tmp * re
            out.flat[2 * i + 1] = scale * tmp * im
        return out

    # ------------------------------------------------------------------
    # Python wrappers — same call signatures as the original so all
    # existing call sites work without modification.
    # ------------------------------------------------------------------

    def V0(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (..., 2)  where [..., 0]=phi_re, [..., 1]=phi_im.
        Returns array of shape (...,).
        """
        delta_V, kappa, v = self.params
        phi_re = np.ascontiguousarray(X[..., 0])
        phi_im = np.ascontiguousarray(X[..., 1])
        return self.V0_numba(phi_re, phi_im, delta_V, kappa, v)

    def dV0(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (..., 2).
        Returns array of same shape (..., 2).
        """
        delta_V, kappa, v = self.params
        phi_re = np.ascontiguousarray(X[..., 0])
        phi_im = np.ascontiguousarray(X[..., 1])
        return self.dV0_numba(phi_re, phi_im, delta_V, kappa, v)

    def plot_potential(
        self, ax, phi_range, fixed_phi_im=0.0, num_points=1000, **kwargs_plot
    ):
        """
        Plot the potential V0(phi_re, phi_im=fixed_phi_im) over a given range and identify local maxima/minima.

        Parameters:
        - ax: Matplotlib figure and axis objects
        - phi_range: Tuple (phi_re_min, phi_re_max) specifying the range of phi_re
        - fixed_phi_im: The fixed value of phi_im (default 0.0)
        - num_points: Number of points to evaluate the potential (default 1000)
        """
        phi_re = np.linspace(phi_range[0], phi_range[1], num_points)

        phi = np.zeros((num_points, self.Ndim))
        phi[:, 0] = phi_re
        phi[:, 1] = fixed_phi_im

        V = self.V0(phi)

        minima_idx = argrelextrema(V, np.less)[0]
        maxima_idx = argrelextrema(V, np.greater)[0]

        ax.plot(phi_re, V, **kwargs_plot)

        if len(minima_idx) > 0:
            ax.scatter(phi_re[minima_idx], V[minima_idx], color="green", zorder=5)
        if len(maxima_idx) > 0:
            ax.scatter(phi_re[maxima_idx], V[maxima_idx], color="red", zorder=5)

        ax.set_xlabel("$\\phi_{\\text{re}}$")
        ax.set_ylabel("$V_0(\\phi)$", rotation=0, labelpad=15)
        ax.set_xlim(phi_range[0], phi_range[1])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()


class QuarticPotential:
    def __init__(self, c2: float, c3: float, c4: float):
        self.name = "quartic"
        self.name_latex = r"$V(\phi) = \dfrac{c_2}{2} \phi^2 - \dfrac{c_3}{3} \phi^3 + \dfrac{c_4}{4} \phi^4$"
        self.couplings_latex = [r"$c_1$", r"$c_2$", r"$c_3$"]
        self.Ndim = 1
        self.params: Tuple[float, float, float] = (c2, c3, c4)

        lambdabar = 9 * c4 * c2 / (2 * c3**2)
        self.phi_fv = np.array([0.0])
        self.phi_max = np.array([(c3 - np.sqrt(c3**2 - 4 * c4 * c2)) / (2 * c4)])
        self.phi_tv = np.array([(c3 + np.sqrt(c3**2 - 4 * c4 * c2)) / (2 * c4)])
        self.m2_fv = c2
        self.m2_tv = (
            c2 * (9 + 3 * np.sqrt(9 - 8 * lambdabar) - 8 * lambdabar) / (4 * lambdabar)
        )
        self.potential_fv = np.array([0.0])
        self.potential_tv = np.array([(self.m2_fv**2 - self.m2_tv**2) / (12 * c4)])
        self.rho_vacuum = self.potential_fv - self.potential_tv

    @staticmethod
    @njit
    def V0_numba(phi: np.ndarray, c2: float, c3: float, c4: float) -> np.ndarray:
        """Compute V(φ) for array of field values. phi: (Ndim, n_points)"""
        return c2 * phi**2 / 2.0 - c3 * phi**3 / 3.0 + c4 * phi**4 / 4.0

    @staticmethod
    @njit
    def dV0_numba(phi: np.ndarray, c2: float, c3: float, c4: float) -> np.ndarray:
        """Compute dV/dφ for array of field values. phi: (Ndim, n_points)"""
        return c2 * phi - c3 * phi**2 + c4 * phi**3

    @staticmethod
    @njit
    def d2V0_numba(phi: np.ndarray, c2: float, c3: float, c4: float) -> np.ndarray:
        """Compute d²V/dφ² for array of field values. phi: (Ndim, n_points)"""
        return c2 - 2 * c3 * phi + 3 * c4 * phi**2

    # Python wrappers that bind parameters for convenience
    def V0(self, phi: np.ndarray) -> np.ndarray:
        c2, c3, c4 = self.params
        return self.V0_numba(phi, c2, c3, c4)

    def dV0(self, phi: np.ndarray) -> np.ndarray:
        c2, c3, c4 = self.params
        return self.dV0_numba(phi, c2, c3, c4)

    def d2V0(self, phi: np.ndarray) -> np.ndarray:
        c2, c3, c4 = self.params
        return self.d2V0_numba(phi, c2, c3, c4)

    def plot_potential(self, ax, phi_range, num_points=1000, **kwargs_plot):
        phi = np.linspace(phi_range[0], phi_range[1], num_points)
        V = self.V0(phi)

        # Find local minima and maxima
        minima_idx = argrelextrema(V, np.less)[0]
        maxima_idx = argrelextrema(V, np.greater)[0]

        ax.plot(phi, V, **kwargs_plot)
        # Scatter plot for minima and maxima if they exist
        if len(minima_idx) > 0:
            ax.scatter(phi[minima_idx], V[minima_idx], color="green", zorder=5)
        if len(maxima_idx) > 0:
            ax.scatter(phi[maxima_idx], V[maxima_idx], color="red", zorder=5)


class GouldQuarticPotential(QuarticPotential):
    def __init__(self, lambdabar):
        c2 = 2 * lambdabar / 9.0
        c3 = 1.0
        c4 = 1.0
        self.lambdabar = lambdabar
        QuarticPotential.__init__(self, c2=c2, c3=c3, c4=c4)
        self.name = "quartic_gould"
        self.name_latex = r"$V(\phi) = \dfrac{\overline{\lambda}}{9} \phi^2 - \dfrac{1}{3} \phi^3 + \dfrac{1}{4} \phi^4$"
        self.couplings_latex = [rf"$\overline{{\lambda}} = {lambdabar:.2f}$"]


class TobyQuarticPotential(QuarticPotential):
    def __init__(self, lambdabar):
        upsilon = 3 + np.sqrt(9 - 8 * lambdabar)
        c2 = 1.0
        c3 = 3 * upsilon / (4 * lambdabar)
        c4 = upsilon**2 / (8 * lambdabar)
        QuarticPotential.__init__(self, c2=c2, c3=c3, c4=c4)
        self.name = "quartic_toby"

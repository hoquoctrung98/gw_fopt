from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

InitialFieldStatus = Literal["one_bubble", "two_bubbles"]
IntegrationMethod = Literal[
    "gauss_legendre",
    "newton_cotes",
    "g7k15",
    "g10k21",
    "g15k31",
    "g20k41",
    "g25k51",
    "g30k61",
    "gauss_kronrod",
    "g7k15r",
    "g10k21r",
    "g15k31r",
    "g20k41r",
    "g25k51r",
    "g30k61r",
    "gauss_kronrod_relative",
]

class GravitationalWaveCalculator:
    def __init__(
        self,
        initial_field_status: InitialFieldStatus | str,
        phi1: npt.ArrayLike,
        phi2: npt.ArrayLike,
        z_grid: npt.ArrayLike,
        ds: float,
        ratio_t_cut: float | None = ...,
        ratio_t_0: float | None = ...,
    ) -> None: ...
    def set_num_threads(self, num_threads: int) -> None: ...
    def set_integration_params(
        self,
        method: IntegrationMethod | str,
        n: int | None = ...,
        tol: float | None = ...,
        max_iter: int | None = ...,
    ) -> None: ...
    def set_integral_params(self, tol: float, max_iter: int) -> None: ...
    def compute_averaged_gw_spectrum(
        self,
        w_arr: list[float],
        cos_thetak_arr: list[float],
    ) -> npt.NDArray[np.float64]: ...
    def compute_angular_gw_spectrum(
        self,
        w_arr: list[float],
        cos_thetak_arr: list[float],
    ) -> npt.NDArray[np.float64]: ...
    def compute_t_tensor(
        self,
        w_arr: list[float],
        cos_thetak_arr: list[float],
    ) -> npt.NDArray[np.complex128]: ...
    @property
    def phi1(self) -> npt.NDArray[np.float64]: ...
    @property
    def phi2(self) -> npt.NDArray[np.float64]: ...
    @property
    def dphi1_dz(self) -> npt.NDArray[np.float64]: ...
    @property
    def dphi1_ds(self) -> npt.NDArray[np.float64]: ...
    @property
    def dphi2_dz(self) -> npt.NDArray[np.float64]: ...
    @property
    def dphi2_ds(self) -> npt.NDArray[np.float64]: ...
    @property
    def xz_deriv_dphi_dz(self) -> npt.NDArray[np.float64]: ...
    @property
    def xz_deriv_dphi_ds(self) -> npt.NDArray[np.float64]: ...
    @property
    def xz_deriv_dphi_dz2(self) -> npt.NDArray[np.float64]: ...
    @property
    def xz_deriv_dphi_ds2(self) -> npt.NDArray[np.float64]: ...
    @property
    def zz_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def rr_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def xz_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def z_grid(self) -> npt.NDArray[np.float64]: ...
    @property
    def s_grid(self) -> npt.NDArray[np.float64]: ...
    @property
    def ds(self) -> float: ...
    @property
    def dz(self) -> float: ...
    @property
    def n_s(self) -> int: ...
    @property
    def n_z(self) -> int: ...
    @property
    def t_cut(self) -> float: ...
    @property
    def t_0(self) -> float: ...
    @property
    def integration_method(self) -> str: ...
    @property
    def s_offset(self) -> npt.NDArray[np.float64]: ...
    @property
    def n_fields(self) -> int: ...

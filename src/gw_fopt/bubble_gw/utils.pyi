from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

SampleType = Literal["uniform", "linear", "log", "exp"]

def sample(
    start: float,
    stop: float,
    n_sample: int,
    n_grid: int,
    n_iter: int,
    sample_type: SampleType | str,
    base: float = ...,
) -> npt.NDArray[np.float64]: ...
def sample_arr(
    start: float,
    stop: float,
    n_sample: int,
    n_grid: int,
    n_iter: int,
    sample_type: SampleType | str,
    base: float = ...,
) -> npt.NDArray[np.float64]: ...

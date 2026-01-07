use bubble_gw::many_bubbles::bubbles_nucleation::FixedRateNucleation;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "FixedNucleationRate")]
#[derive(Clone, Debug)]
pub struct PyFixedNucleationRate {
    pub inner: FixedRateNucleation,
}

#[pymethods]
impl PyFixedNucleationRate {
    /// Create a new FixedNucleationRate strategy.
    ///
    /// Args:
    ///     beta (float): Inverse timescale of rate growth.
    ///     gamma0 (float): Base nucleation rate density (bubbles / volume /
    /// time).     t0 (float): Reference time (rate = gamma0 at t = t0).
    ///     d_p0 (float): Target probability per step (~0.1–0.5).
    ///     seed (Optional[int]): RNG seed for reproducibility.
    ///
    /// Note:
    ///     If seed is None, a new random seed is used each time `nucleate` is
    /// called.
    #[new]
    #[pyo3(signature = (beta, gamma0, t0, d_p0, seed=None))]
    fn new(beta: f64, gamma0: f64, t0: f64, d_p0: f64, seed: Option<u64>) -> PyResult<Self> {
        if beta.is_nan() || gamma0.is_nan() || t0.is_nan() || d_p0.is_nan() {
            return Err(PyValueError::new_err("Parameters must be finite"));
        }
        if d_p0 <= 0.0 || d_p0 > 1.0 {
            return Err(PyValueError::new_err("d_p0 must be in (0, 1]"));
        }
        if gamma0 < 0.0 {
            return Err(PyValueError::new_err("gamma0 must be non-negative"));
        }

        Ok(PyFixedNucleationRate {
            inner: FixedRateNucleation {
                beta,
                gamma0,
                t0,
                d_p0,
                seed,
            },
        })
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FixedRateNucleation(beta={}, gamma0={}, t0={}, d_p0={}, seed={:?})",
            self.inner.beta, self.inner.gamma0, self.inner.t0, self.inner.d_p0, self.inner.seed
        ))
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    #[getter]
    fn gamma0(&self) -> f64 {
        self.inner.gamma0
    }

    #[getter]
    fn t0(&self) -> f64 {
        self.inner.t0
    }

    #[getter]
    fn d_p0(&self) -> f64 {
        self.inner.d_p0
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }
}

use bubble_gw::many_bubbles::bubbles_nucleation::{
    FixedRateNucleation,
    NucleationStrategy,
    VolumeRemainingMethod,
};
use bubble_gw::many_bubbles::lattice::{BoundaryConditions, BuiltInLattice};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_many_bubbles::py_lattice::{PyCartesian, PyEmpty, PyParallelepiped, PySpherical};
use crate::py_many_bubbles::py_lattice_bubbles::PyLatticeBubbles;

#[pyclass(name = "FixedNucleationRate")]
#[derive(Debug)]
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
    ///     d_p0 (float): Target probability per step (~0.01–0.1).
    ///     seed (Optional[int]): RNG seed for reproducibility.
    ///     method (str): "approximation" or "montecarlo".
    ///     n_points (int, optional): Number of Monte Carlo samples (default:
    /// 10000).
    ///
    /// Note:
    ///     If seed is None, a new random seed is used.
    #[new]
    #[pyo3(signature = (beta, gamma0, t0, d_p0, seed=None, method="approximation", n_points=10000, max_attempts=None, volume_remaining_fraction_cutoff=None))]
    fn new(
        beta: f64,
        gamma0: f64,
        t0: f64,
        d_p0: f64,
        seed: Option<u64>,
        method: &str,
        n_points: usize,
        max_attempts: Option<usize>,
        volume_remaining_fraction_cutoff: Option<f64>,
    ) -> PyResult<Self> {
        if beta.is_nan() || gamma0.is_nan() || t0.is_nan() || d_p0.is_nan() {
            return Err(PyValueError::new_err("Parameters must be finite"));
        }
        if d_p0 <= 0.0 || d_p0 > 1.0 {
            return Err(PyValueError::new_err("d_p0 must be in (0, 1]"));
        }
        if gamma0 < 0.0 {
            return Err(PyValueError::new_err("gamma0 must be non-negative"));
        }
        if n_points == 0 && method.to_lowercase() == "montecarlo" {
            return Err(PyValueError::new_err("n_points must be > 0 for Monte Carlo"));
        }

        let volume_method = match method.to_lowercase().as_str() {
            "approximation" => VolumeRemainingMethod::Approximation,
            "montecarlo" => VolumeRemainingMethod::MonteCarlo { n_points },
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'approximation' or 'montecarlo'",
                ));
            },
        };

        Ok(PyFixedNucleationRate {
            inner: FixedRateNucleation::new(
                beta,
                gamma0,
                t0,
                d_p0,
                seed,
                volume_method,
                max_attempts,
                volume_remaining_fraction_cutoff,
            ),
        })
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FixedNucleationRate(beta={}, gamma0={}, t0={}, d_p0={}, seed={:?}, method={})",
            self.inner.beta,
            self.inner.gamma0,
            self.inner.t0,
            self.inner.d_p0,
            self.inner.seed,
            match &self.inner.volume_method {
                VolumeRemainingMethod::Approximation => "approximation",
                VolumeRemainingMethod::MonteCarlo { .. } => "montecarlo",
            }
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

    #[getter]
    fn n_points(&self) -> usize {
        match self.inner.volume_method {
            VolumeRemainingMethod::Approximation => 0,
            VolumeRemainingMethod::MonteCarlo { n_points } => n_points,
        }
    }

    #[getter]
    fn method(&self) -> String {
        match &self.inner.volume_method {
            VolumeRemainingMethod::Approximation => "approximation".to_string(),
            VolumeRemainingMethod::MonteCarlo { .. } => "montecarlo".to_string(),
        }
    }

    #[getter]
    fn time_history(&self) -> Vec<f64> {
        self.inner.time_history.clone()
    }

    #[getter]
    fn volume_remaining_history(&self) -> Vec<f64> {
        self.inner.volume_remaining_history.clone()
    }

    #[pyo3(signature = (lattice, boundary_condition = "periodic"))]
    fn nucleate(
        &mut self,
        lattice: &Bound<'_, PyAny>,
        boundary_condition: &str,
    ) -> PyResult<PyLatticeBubbles> {
        // Extract builtin lattice from concrete Python object
        let lattice: BuiltInLattice = if let Ok(l) = lattice.extract::<PyParallelepiped>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PyCartesian>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PySpherical>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PyEmpty>() {
            l.builtin
        } else {
            return Err(PyValueError::new_err(
                "Expected a lattice instance: ParallelepipedLattice, CartesianLattice, SphericalLattice, or EmptyLattice",
            ));
        };

        let boundary_condition = match boundary_condition.to_lowercase().as_str() {
            "periodic" => BoundaryConditions::Periodic,
            "reflection" => BoundaryConditions::Reflection,
            "none" => BoundaryConditions::None,
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid boundary condition. Expected 'periodic' or 'reflection'.",
                ));
            },
        };

        let lattice_bubbles = PyLatticeBubbles {
            inner: self
                .inner
                .nucleate(&lattice, boundary_condition)
                .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?,
        };

        Ok(lattice_bubbles)
    }
}

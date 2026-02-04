use bubble_gw::many_bubbles::bubbles_nucleation::{FixedRateNucleation, FixedRateNucleationError};
use bubble_gw::many_bubbles::lattice::{BoundaryConditions, BuiltInLattice};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_many_bubbles::py_lattice::{PyCartesian, PyEmpty, PyParallelepiped, PySpherical};
use crate::py_many_bubbles::py_lattice_bubbles::PyLatticeBubbles;

// --- New Result Struct ---
#[pyclass(name = "FixedRateNucleationResult")]
pub struct PyFixedRateNucleationResult {
    #[pyo3(get)]
    pub lattice_bubbles: PyLatticeBubbles,
    #[pyo3(get)]
    pub time_history: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub volume_false_vacuum_history: Py<PyArray1<f64>>,
}

#[pyclass(name = "FixedNucleationRate")]
#[derive(Debug)]
pub struct PyFixedNucleationRate {
    pub inner: FixedRateNucleation,
}

#[pymethods]
impl PyFixedNucleationRate {
    /// Create a new FixedNucleationRate strategy (Monte Carlo only).
    ///
    /// Args:
    ///     beta (float): Inverse timescale of rate growth.
    ///     gamma0 (float): Base nucleation rate density (bubbles / volume / time).
    ///     t0 (float): Reference time (rate = gamma0 at t = t0).
    ///     d_p0 (float): Target probability per step (~0.01–0.1).
    ///     seed (Optional[int]): RNG seed for reproducibility.
    ///     n_points (int): Number of Monte Carlo samples (default: 10000).
    ///     max_time_iterations (Optional[int]): Max time steps (default: 1_000_000).
    ///     cutoff_fraction_false_vacuum (Optional[float]): Volume cutoff fraction (default: 0.01).
    ///
    /// Note:
    ///     If seed is None, a new random seed is used from OS entropy.
    #[new]
    #[pyo3(signature = (
        beta,
        gamma0,
        t0,
        d_p0,
        seed=None,
        n_points=10000,
        max_time_iterations=None,
        cutoff_fraction_false_vacuum=None
    ))]
    fn new(
        beta: f64,
        gamma0: f64,
        t0: f64,
        d_p0: f64,
        seed: Option<u64>,
        n_points: usize,
        max_time_iterations: Option<usize>,
        cutoff_fraction_false_vacuum: Option<f64>,
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
        if n_points == 0 {
            return Err(PyValueError::new_err("n_points must be > 0"));
        }

        let inner = FixedRateNucleation::new(
            beta,
            gamma0,
            t0,
            d_p0,
            seed,
            n_points,
            max_time_iterations,
            cutoff_fraction_false_vacuum,
        )
        .map_err(|e| match e {
            FixedRateNucleationError::RngInitializationError(err) => {
                PyValueError::new_err(format!("Failed to initialize RNG: {}", err))
            },
            FixedRateNucleationError::LatticeBubblesError(err) => {
                PyValueError::new_err(format!("Lattice bubbles error: {}", err))
            },
        })?;

        Ok(PyFixedNucleationRate { inner })
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FixedNucleationRate(beta={}, gamma0={}, t0={}, d_p0={}, seed={:?}, n_points={})",
            self.inner.beta,
            self.inner.gamma0,
            self.inner.t0,
            self.inner.d_p0,
            self.inner.seed,
            self.inner.n_points
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
        self.inner.n_points
    }

    #[pyo3(signature = (lattice, boundary_condition = "periodic"))]
    fn nucleate(
        &mut self,
        py: Python,
        lattice: &Bound<'_, PyAny>,
        boundary_condition: &str,
    ) -> PyResult<PyFixedRateNucleationResult> {
        // Extract builtin lattice
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
                    "Invalid boundary condition. Expected 'periodic', 'reflection', or 'none'.",
                ));
            },
        };

        let result = self
            .inner
            .nucleate(&lattice, boundary_condition)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        let py_result = PyFixedRateNucleationResult {
            lattice_bubbles: PyLatticeBubbles {
                inner: result.lattice_bubbles,
            },
            time_history: PyArray1::from_array(py, &result.time_history).into(),
            volume_false_vacuum_history: PyArray1::from_array(
                py,
                &result.volume_false_vacuum_history,
            )
            .into(),
        };

        Ok(py_result)
    }
}

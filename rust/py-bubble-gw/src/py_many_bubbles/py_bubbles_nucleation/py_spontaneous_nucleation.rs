use bubble_gw::many_bubbles::bubbles_nucleation::{NucleationStrategy, SpontaneousNucleation};
use bubble_gw::many_bubbles::lattice::{BoundaryConditions, BuiltInLattice};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_many_bubbles::py_lattice::{PyCartesian, PyEmpty, PyParallelepiped, PySpherical};
use crate::py_many_bubbles::py_lattice_bubbles::PyLatticeBubbles;

#[pyclass(name = "SpontaneousNucleation")]
#[derive(Clone)]
pub struct PySpontaneousNucleation {
    pub(crate) inner: SpontaneousNucleation,
}

#[pymethods]
impl PySpontaneousNucleation {
    /// Create a new SpontaneousNucleation strategy.
    ///
    /// Args:
    ///     n_bubbles (int): number of bubbles to be nucleated
    ///     t0 (float): Reference time (rate = gamma0 at t = t0).
    ///     seed (Optional[int]): RNG seed for reproducibility.
    ///
    /// Note:
    ///     If seed is None, a new random seed is used each time `nucleate` is
    /// called.
    #[new]
    #[pyo3(signature = (n_bubbles, t0 = 0.0, seed = None))]
    fn new(n_bubbles: usize, t0: f64, seed: Option<u64>) -> Self {
        Self {
            inner: SpontaneousNucleation::new(n_bubbles, t0, seed),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpontaneousNucleation(n_bubbles={}, t0={}, seed={})",
            self.inner.n_bubbles,
            self.inner.t0,
            match self.inner.seed {
                Some(s) => format!("Some({})", s),
                None => "None".to_string(),
            }
        )
    }

    #[getter]
    fn n_bubbles(&self) -> usize {
        self.inner.n_bubbles
    }

    #[getter]
    fn t0(&self) -> f64 {
        self.inner.t0
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
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

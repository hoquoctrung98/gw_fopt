use bubble_gw::many_bubbles::bubbles_nucleation::SpontaneousNucleation;
use pyo3::prelude::*;

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
            inner: SpontaneousNucleation {
                n_bubbles,
                t0,
                seed,
            },
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
}

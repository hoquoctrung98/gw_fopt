use bubble_gw::many_bubbles::bubbles_nucleation::SpontaneousNucleation;
use pyo3::prelude::*;

#[pyclass(name = "UniformAtFixedTime")]
#[derive(Clone)]
pub struct PyUniformAtFixedTime {
    pub(crate) inner: SpontaneousNucleation,
}

#[pymethods]
impl PyUniformAtFixedTime {
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
            "UniformAtFixedTime(n_bubbles={}, t0={}, seed={})",
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

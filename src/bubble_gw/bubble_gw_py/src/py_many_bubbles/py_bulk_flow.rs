use bubble_gw_rs::many_bubbles::bulk_flow::{BubbleIndex, BulkFlow, BulkFlowError};
use ndarray::Array2;
use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::Python;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

// Newtype wrapper for BulkFlowError to satisfy orphan rules
#[derive(Debug)]
struct PyBulkFlowError(BulkFlowError);

// Convert BulkFlowError to PyBulkFlowError
impl From<BulkFlowError> for PyBulkFlowError {
    fn from(err: BulkFlowError) -> Self {
        PyBulkFlowError(err)
    }
}

// Convert PyBulkFlowError to PyErr for Python exception handling
impl From<PyBulkFlowError> for PyErr {
    fn from(err: PyBulkFlowError) -> PyErr {
        match err.0 {
            BulkFlowError::InvalidIndex { index, max } => {
                PyIndexError::new_err(format!("Index {} out of bounds for max {}", index, max))
            }
            BulkFlowError::UninitializedField(field) => {
                PyValueError::new_err(format!("Field '{}' is not initialized", field))
            }
            BulkFlowError::InvalidResolution(msg) => {
                PyValueError::new_err(format!("Invalid resolution: {}", msg))
            }
            BulkFlowError::InvalidTimeRange { begin, end } => PyValueError::new_err(format!(
                "Invalid time range: t_begin={} > t_end={}",
                begin, end
            )),
            BulkFlowError::ArrayShapeMismatch(msg) => {
                PyValueError::new_err(format!("Array shape mismatch: {}", msg))
            }
            BulkFlowError::ThreadPoolBuildError(msg) => {
                PyValueError::new_err(format!("Building thread pool unsucessfully: {}", msg))
            }
            BulkFlowError::BubbleFormedInsideBubble { a, b } => {
                let a_str = match a {
                    BubbleIndex::Interior(i) => format!("Interior({})", i),
                    BubbleIndex::Exterior(i) => format!("Exterior({})", i),
                    BubbleIndex::None => "None".to_string(),
                };
                let b_str = match b {
                    BubbleIndex::Interior(i) => format!("Interior({})", i),
                    BubbleIndex::Exterior(i) => format!("Exterior({})", i),
                    BubbleIndex::None => "None".to_string(),
                };
                PyValueError::new_err(format!(
                    "Bubble {} is formed inside bubble {} at initial time (overlapping light cones)",
                    a_str, b_str
                ))
            }
        }
    }
}

#[pyclass(name = "BulkFlow")]
pub struct PyBulkFlow {
    inner: BulkFlow,
}

#[pymethods]
impl PyBulkFlow {
    #[new]
    #[pyo3(signature = (bubbles_interior, bubbles_exterior = None))]
    pub fn new(
        bubbles_interior: PyReadonlyArray2<f64>,
        bubbles_exterior: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let bubbles_interior = bubbles_interior.to_owned_array();
        let bubbles_exterior =
            bubbles_exterior.map_or(Array2::zeros((0, 4)), |arr| arr.to_owned_array());
        let bulk_flow =
            BulkFlow::new(bubbles_interior, bubbles_exterior).map_err(PyBulkFlowError)?;
        Ok(PyBulkFlow { inner: bulk_flow })
    }

    #[getter]
    pub fn bubbles_interior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubbles_interior()).into())
    }

    #[getter]
    pub fn bubbles_exterior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubbles_exterior()).into())
    }

    pub fn set_bubbles_interior(
        &mut self,
        bubbles_interior: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_bubbles_interior(bubbles_interior.to_owned_array())
            .map_err(PyBulkFlowError)?;
        Ok(())
    }

    pub fn set_bubbles_exterior(
        &mut self,
        bubbles_exterior: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_bubbles_exterior(bubbles_exterior.to_owned_array())
            .map_err(PyBulkFlowError)?;
        Ok(())
    }

    #[getter]
    pub fn delta(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, self.inner.delta()).into())
    }

    #[getter]
    pub fn delta_squared(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.delta_squared()).into())
    }

    pub fn set_delta(&mut self, delta: PyReadonlyArray3<f64>) -> PyResult<()> {
        let delta = delta.to_owned_array();
        self.inner.set_delta(delta);
        Ok(())
    }

    #[getter]
    pub fn coefficients_sets(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.coefficients_sets()).into())
    }

    pub fn set_coefficients_sets(
        &mut self,
        coefficients_sets: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let coefficients_sets = coefficients_sets.to_owned_array();
        self.inner.set_coefficients_sets(coefficients_sets);
        Ok(())
    }

    #[getter]
    pub fn get_powers_sets(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.powers_sets()).into())
    }

    pub fn set_powers_sets(&mut self, powers_sets: PyReadonlyArray2<f64>) -> PyResult<()> {
        let powers_sets = powers_sets.to_owned_array();
        self.inner.set_powers_sets(powers_sets);
        Ok(())
    }

    #[getter]
    pub fn active_sets(&self, py: Python) -> PyResult<Py<PyArray1<bool>>> {
        Ok(PyArray1::from_array(py, self.inner.active_sets()).into())
    }

    pub fn set_active_sets(&mut self, active_sets: PyReadonlyArray1<bool>) -> PyResult<()> {
        let active_sets = active_sets.to_owned_array();
        self.inner.set_active_sets(active_sets);
        Ok(())
    }

    #[getter]
    fn cos_thetax(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let cos_thetax = self
            .inner
            .cos_thetax()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_array(py, cos_thetax).into())
    }

    #[getter]
    fn phix(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let phix = self
            .inner
            .phix()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_array(py, phix).into())
    }

    #[pyo3(signature = (coefficients_sets, powers_sets, damping_factor = None))]
    pub fn set_gradient_scaling_params(
        &mut self,
        coefficients_sets: Vec<Vec<f64>>,
        powers_sets: Vec<Vec<f64>>,
        damping_factor: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_gradient_scaling_params(coefficients_sets, powers_sets, damping_factor)
            .map_err(PyBulkFlowError)?;
        Ok(())
    }

    pub fn compute_first_colliding_bubble(
        &self,
        py: Python,
        a_idx: usize,
    ) -> PyResult<Py<PyArray2<i32>>> {
        let first_bubble = self
            .inner
            .compute_first_colliding_bubble(a_idx)
            .map_err(PyBulkFlowError)?;
        let first_bubble_int = first_bubble.mapv(|bi| match bi {
            BubbleIndex::None => -1i32,
            BubbleIndex::Interior(i) => i as i32,
            BubbleIndex::Exterior(i) => -((i as i32) + 2),
        });
        Ok(PyArray2::from_array(py, &first_bubble_int).into())
    }

    pub fn compute_delta_tab(
        &self,
        py: Python,
        a_idx: usize,
        first_bubble: PyReadonlyArray2<i32>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let first_bubble = first_bubble.to_owned_array().mapv(|i: i32| {
            if i == -1 {
                BubbleIndex::None
            } else if i >= 0 {
                BubbleIndex::Interior(i as usize)
            } else {
                BubbleIndex::Exterior((-i - 2) as usize)
            }
        });
        let delta_tab_grid = self
            .inner
            .compute_delta_tab(a_idx, first_bubble.view())
            .map_err(PyBulkFlowError)?;
        Ok(PyArray2::from_array(py, &delta_tab_grid).into())
    }

    pub fn compute_collision_status(
        &self,
        py: Python,
        a_idx: usize,
        t: f64,
        first_bubble: PyReadonlyArray2<i32>,
        delta_tab_grid: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<i32>>> {
        let first_bubble = first_bubble.to_owned_array().mapv(|i: i32| {
            if i == -1 {
                BubbleIndex::None
            } else if i >= 0 {
                BubbleIndex::Interior(i as usize)
            } else {
                BubbleIndex::Exterior((-i - 2) as usize)
            }
        });
        let delta_tab_grid = delta_tab_grid.to_owned_array();
        let collision_status = self
            .inner
            .compute_collision_status(a_idx, t, first_bubble.view(), delta_tab_grid.view())
            .map_err(PyBulkFlowError)?;
        let collision_status_int = collision_status.mapv(|s| s as i32);
        Ok(PyArray2::from_array(py, &collision_status_int).into())
    }

    pub fn compute_c_integral_fixed_bubble(
        &mut self,
        py: Python,
        a_idx: usize,
        w_arr: PyReadonlyArray1<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let c_matrix = self
            .inner
            .compute_c_integral_fixed_bubble(a_idx, w_arr.view(), t_begin, t_end, n_t)
            .map_err(PyBulkFlowError)?;
        let c_matrix_numpy = c_matrix.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    pub fn compute_c_integrand_fixed_bubble(
        &self,
        a_idx: usize,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let integrand = self
            .inner
            .compute_c_integrand_fixed_bubble(a_idx, w_arr.view(), t_begin, t_end, n_t)
            .map_err(PyBulkFlowError)?;
        let integrand_numpy = integrand.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_array(py, &integrand_numpy).into())
    }

    pub fn compute_c_integrand(
        &self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let integrand = self
            .inner
            .compute_c_integrand(w_arr.view(), t_begin, t_end, n_t)
            .map_err(PyBulkFlowError)?;
        let integrand_numpy = integrand.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_array(py, &integrand_numpy).into())
    }

    pub fn compute_c_integral(
        &mut self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let c_matrix = self
            .inner
            .compute_c_integral(w_arr.view(), t_begin, t_end, n_t)
            .map_err(PyBulkFlowError)?;
        let c_matrix_numpy = c_matrix.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    #[pyo3(signature = (n_cos_thetax, n_phix, precompute_first_bubbles = true))]
    pub fn set_resolution(
        &mut self,
        n_cos_thetax: usize,
        n_phix: usize,
        precompute_first_bubbles: bool,
    ) -> PyResult<()> {
        self.inner
            .set_resolution(n_cos_thetax, n_phix, precompute_first_bubbles)
            .map_err(PyBulkFlowError)?;
        Ok(())
    }
}

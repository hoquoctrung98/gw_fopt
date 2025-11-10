use bubble_gw_rs::many_bubbles::bulk_flow::{BubbleIndex, BulkFlow, BulkFlowError};
use ndarray::Array2;
use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Local Python-facing error — 100% owned by this crate
#[derive(Error, Debug)]
pub enum PyBulkFlowError {
    #[error("Index {index} out of bounds (max: {max})")]
    InvalidIndex { index: usize, max: usize },

    #[error("Field '{0}' is not initialized")]
    UninitializedField(String),

    #[error("Invalid resolution: {0}")]
    InvalidResolution(String),

    #[error("Invalid time range: t_begin={begin} > t_end={end}")]
    InvalidTimeRange { begin: f64, end: f64 },

    #[error("Array shape mismatch: {0}")]
    ArrayShapeMismatch(String),

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuildError(String),

    #[error("Bubble {a} is formed inside bubble {b} at initial time (overlapping light cones)")]
    BubbleFormedInsideBubble { a: BubbleIndex, b: BubbleIndex },
}

impl From<BulkFlowError> for PyBulkFlowError {
    fn from(err: BulkFlowError) -> Self {
        match err {
            BulkFlowError::InvalidIndex { index, max } => {
                PyBulkFlowError::InvalidIndex { index, max }
            }
            BulkFlowError::UninitializedField(field) => PyBulkFlowError::UninitializedField(field),
            BulkFlowError::InvalidResolution(msg) => PyBulkFlowError::InvalidResolution(msg),
            BulkFlowError::InvalidTimeRange { begin, end } => {
                PyBulkFlowError::InvalidTimeRange { begin, end }
            }
            BulkFlowError::ArrayShapeMismatch(msg) => PyBulkFlowError::ArrayShapeMismatch(msg),
            BulkFlowError::ThreadPoolBuildError(e) => {
                PyBulkFlowError::ThreadPoolBuildError(e.to_string())
            }
            BulkFlowError::BubbleFormedInsideBubble { a, b } => {
                PyBulkFlowError::BubbleFormedInsideBubble { a, b }
            }
        }
    }
}

impl From<PyBulkFlowError> for PyErr {
    fn from(err: PyBulkFlowError) -> Self {
        match err {
            PyBulkFlowError::InvalidIndex { .. } => PyIndexError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

type PyResult<T> = Result<T, PyBulkFlowError>;

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
        let bubbles_exterior = bubbles_exterior
            .map(|arr| arr.to_owned_array())
            .unwrap_or_else(|| Array2::zeros((0, 4)));

        let bulk_flow = BulkFlow::new(bubbles_interior, bubbles_exterior)?;
        Ok(PyBulkFlow { inner: bulk_flow })
    }

    #[getter]
    pub fn bubbles_interior(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, self.inner.bubbles_interior()).into()
    }

    #[getter]
    pub fn bubbles_exterior(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, self.inner.bubbles_exterior()).into()
    }

    pub fn set_bubbles_interior(
        &mut self,
        bubbles_interior: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_bubbles_interior(bubbles_interior.to_owned_array())?;
        Ok(())
    }

    pub fn set_bubbles_exterior(
        &mut self,
        bubbles_exterior: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_bubbles_exterior(bubbles_exterior.to_owned_array())?;
        Ok(())
    }

    #[getter]
    pub fn delta(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, self.inner.delta()).into()
    }

    #[getter]
    pub fn delta_squared(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, self.inner.delta_squared()).into()
    }

    pub fn set_delta(&mut self, delta: PyReadonlyArray3<f64>) {
        self.inner.set_delta(delta.to_owned_array());
    }

    #[getter]
    pub fn coefficients_sets(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, self.inner.coefficients_sets()).into()
    }

    pub fn set_coefficients_sets(&mut self, coefficients_sets: PyReadonlyArray2<f64>) {
        self.inner
            .set_coefficients_sets(coefficients_sets.to_owned_array());
    }

    #[getter]
    pub fn get_powers_sets(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, self.inner.powers_sets()).into()
    }

    pub fn set_powers_sets(&mut self, powers_sets: PyReadonlyArray2<f64>) {
        self.inner.set_powers_sets(powers_sets.to_owned_array());
    }

    #[getter]
    pub fn active_sets(&self, py: Python) -> Py<PyArray1<bool>> {
        PyArray1::from_array(py, self.inner.active_sets()).into()
    }

    pub fn set_active_sets(&mut self, active_sets: PyReadonlyArray1<bool>) {
        self.inner.set_active_sets(active_sets.to_owned_array());
    }

    #[getter]
    fn cos_thetax(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let cos_thetax = self.inner.cos_thetax()?;
        Ok(PyArray1::from_array(py, cos_thetax).into())
    }

    #[getter]
    fn phix(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let phix = self.inner.phix()?;
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
            .set_gradient_scaling_params(coefficients_sets, powers_sets, damping_factor)?;
        Ok(())
    }

    pub fn compute_first_colliding_bubble(
        &self,
        py: Python,
        a_idx: usize,
    ) -> PyResult<Py<PyArray2<i32>>> {
        let first_bubble = self.inner.compute_first_colliding_bubble(a_idx)?;
        let first_bubble_int = first_bubble.mapv(|bi| match bi {
            BubbleIndex::None => -1,
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
        let delta_tab_grid = self.inner.compute_delta_tab(a_idx, first_bubble.view())?;
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
        let collision_status = self.inner.compute_collision_status(
            a_idx,
            t,
            first_bubble.view(),
            delta_tab_grid.view(),
        )?;
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
        let c_matrix =
            self.inner
                .compute_c_integral_fixed_bubble(a_idx, w_arr.view(), t_begin, t_end, n_t)?;
        let c_matrix_numpy = c_matrix.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    pub fn compute_c_integrand_fixed_bubble(
        &self,
        py: Python,
        a_idx: usize,
        w_arr: PyReadonlyArray1<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let integrand = self.inner.compute_c_integrand_fixed_bubble(
            a_idx,
            w_arr.view(),
            t_begin,
            t_end,
            n_t,
        )?;
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
            .compute_c_integrand(w_arr.view(), t_begin, t_end, n_t)?;
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
            .compute_c_integral(w_arr.view(), t_begin, t_end, n_t)?;
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
            .set_resolution(n_cos_thetax, n_phix, precompute_first_bubbles)?;
        Ok(())
    }
}

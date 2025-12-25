use crate::py_many_bubbles::PyLatticeBubbles;
use bubble_gw::many_bubbles::bulk_flow::{BulkFlow, BulkFlowError};
use bubble_gw::many_bubbles::lattice::BuiltInLattice;
use bubble_gw::many_bubbles::lattice_bubbles::BubbleIndex;
use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Python-facing error
#[derive(Error, Debug)]
pub enum PyBubblesError {
    #[error("Array shape mismatch: {0}")]
    ArrayShapeMismatch(String),

    #[error("Bubble {a} is formed inside bubble {b} at initial time (overlapping light cones)")]
    BubbleFormedInsideBubble { a: BubbleIndex, b: BubbleIndex },

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse float error at {path}:{line}: '{value}'")]
    ParseFloat {
        path: String,
        line: usize,
        value: String,
    },

    #[error("Invalid row in {path}:{line}: expected 4 columns, got {got}")]
    InvalidColumnCount {
        path: String,
        line: usize,
        got: usize,
    },

    #[error("Empty bubble file: {0}")]
    EmptyFile(String),
}

/// Python-facing error
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

    #[error(transparent)]
    PyErr(#[from] pyo3::PyErr),

    #[error("Bubbles Error")]
    BubblesError,
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
            BulkFlowError::BubblesError(..) => PyBulkFlowError::BubblesError,
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
    inner: BulkFlow<BuiltInLattice>,
}

#[pymethods]
impl PyBulkFlow {
    #[new]
    #[pyo3(signature = (lattice))]
    pub fn new(lattice: PyLatticeBubbles) -> PyResult<Self> {
        let bulk_flow = BulkFlow::new(lattice.inner)?;
        Ok(PyBulkFlow { inner: bulk_flow })
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> PyResult<()> {
        self.inner.set_num_threads(num_threads)?;
        Ok(())
    }

    #[getter]
    pub fn bubbles_interior(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, &self.inner.bubbles_interior().to_array2()).into()
    }

    #[getter]
    pub fn bubbles_exterior(&self, py: Python) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, &self.inner.bubbles_exterior().to_array2()).into()
    }

    // #[getter]
    // pub fn delta_squared(&self, py: Python) -> Py<PyArray2<f64>> {
    //     let foo = self.inner.delta_squared().to_pyarray(py).to_owned_array();
    //     PyArray2::from_array(py, &foo).into()
    // }

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
        let delta_tab_grid = self.inner.compute_delta_tab(a_idx, &first_bubble)?;
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
        let collision_status =
            self.inner
                .compute_collision_status(a_idx, t, &first_bubble, &delta_tab_grid)?;
        let collision_status_int = collision_status.mapv(|s| s as i32);
        Ok(PyArray2::from_array(py, &collision_status_int).into())
    }

    pub fn compute_c_integral_fixed_bubble(
        &mut self,
        py: Python,
        a_idx: usize,
        w_arr: Vec<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let c_matrix = self
            .inner
            .compute_c_integral_fixed_bubble(a_idx, &w_arr, t_begin, t_end, n_t)?;
        let c_matrix_numpy = c_matrix.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    pub fn compute_c_integrand_fixed_bubble(
        &self,
        py: Python,
        a_idx: usize,
        w_arr: Vec<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let integrand = self
            .inner
            .compute_c_integrand_fixed_bubble(a_idx, &w_arr, t_begin, t_end, n_t)?;
        let integrand_numpy = integrand.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_array(py, &integrand_numpy).into())
    }

    #[pyo3(signature = (w_arr, *, t_begin=None, t_end, n_t, selected_bubbles=None))]
    pub fn compute_c_integrand(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<PyReadonlyArray1<usize>>,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let selected_vec: Option<Vec<usize>> = selected_bubbles
            .map(|arr| {
                let array = arr.to_owned_array();
                if !array.is_standard_layout() {
                    return Err(PyValueError::new_err(
                        "selected_bubbles must be a contiguous 1-D array in C order",
                    ));
                }
                Ok(array.to_vec())
            })
            .transpose()?;

        let selected_slice: Option<&[usize]> = selected_vec.as_deref();

        let integrand = self
            .inner
            .compute_c_integrand(&w_arr, t_begin, t_end, n_t, selected_slice)
            .map_err(|e| match e {
                BulkFlowError::InvalidIndex { index, max } => {
                    PyBulkFlowError::InvalidIndex { index, max }
                }
                _ => PyBulkFlowError::from(e),
            })?;

        let integrand_numpy = integrand.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_owned_array(py, integrand_numpy).into())
    }

    #[pyo3(signature = (w_arr, *, t_begin=None, t_end, n_t, selected_bubbles=None))]
    pub fn compute_c_integral(
        &mut self,
        py: Python,
        w_arr: Vec<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<PyReadonlyArray1<usize>>,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let selected_vec: Option<Vec<usize>> = selected_bubbles
            .map(|arr| {
                let array = arr.to_owned_array();
                if !array.is_standard_layout() {
                    return Err(PyValueError::new_err(
                        "selected_bubbles must be a contiguous 1-D array in C order",
                    ));
                }
                Ok(array.to_vec())
            })
            .transpose()?;

        let selected_slice: Option<&[usize]> = selected_vec.as_deref();

        let c_matrix = self
            .inner
            .compute_c_integral(&w_arr, t_begin, t_end, n_t, selected_slice)
            .map_err(|e| match e {
                BulkFlowError::InvalidIndex { index, max } => {
                    PyBulkFlowError::InvalidIndex { index, max }
                }
                _ => PyBulkFlowError::from(e),
            })?;

        let c_matrix_numpy = c_matrix.mapv(|c| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_owned_array(py, c_matrix_numpy).into())
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

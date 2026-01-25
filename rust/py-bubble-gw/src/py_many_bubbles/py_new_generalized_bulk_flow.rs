use bubble_gw::many_bubbles::lattice::BuiltInLattice;
use bubble_gw::many_bubbles::lattice_bubbles::BubbleIndex;
use bubble_gw::many_bubbles::new_generalized_bulk_flow::{
    GeneralizedBulkFlow,
    GeneralizedBulkFlowError,
};
use numpy::{
    Complex64 as NumpyComplex64,
    PyArray1,
    PyArray2,
    PyArray3,
    PyArray4,
    PyArrayMethods,
    PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

use crate::py_many_bubbles::py_lattice_bubbles::PyLatticeBubbles;

// ——————————————————————————————————————————————————————
// Error conversion
// ——————————————————————————————————————————————————————

#[derive(Error, Debug)]
pub enum PyGeneralizedBulkFlowError {
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

    #[error("Bubble {a} is formed inside bubble {b}")]
    BubbleFormedInsideBubble { a: BubbleIndex, b: BubbleIndex },

    #[error(transparent)]
    PyErr(#[from] pyo3::PyErr),

    #[error("Bubbles Error")]
    BubblesError,
}

impl From<GeneralizedBulkFlowError> for PyGeneralizedBulkFlowError {
    fn from(err: GeneralizedBulkFlowError) -> Self {
        match err {
            GeneralizedBulkFlowError::InvalidIndex { index, max } => {
                PyGeneralizedBulkFlowError::InvalidIndex { index, max }
            },
            GeneralizedBulkFlowError::UninitializedField(field) => {
                PyGeneralizedBulkFlowError::UninitializedField(field)
            },
            GeneralizedBulkFlowError::InvalidResolution(msg) => {
                PyGeneralizedBulkFlowError::InvalidResolution(msg)
            },
            GeneralizedBulkFlowError::InvalidTimeRange { begin, end } => {
                PyGeneralizedBulkFlowError::InvalidTimeRange { begin, end }
            },
            GeneralizedBulkFlowError::ArrayShapeMismatch(msg) => {
                PyGeneralizedBulkFlowError::ArrayShapeMismatch(msg)
            },
            GeneralizedBulkFlowError::ThreadPoolBuildError(e) => {
                PyGeneralizedBulkFlowError::ThreadPoolBuildError(e.to_string())
            },
            GeneralizedBulkFlowError::BubbleFormedInsideBubble { a, b } => {
                PyGeneralizedBulkFlowError::BubbleFormedInsideBubble { a, b }
            },
            GeneralizedBulkFlowError::BubblesError(_) => PyGeneralizedBulkFlowError::BubblesError,
        }
    }
}

impl From<PyGeneralizedBulkFlowError> for PyErr {
    fn from(err: PyGeneralizedBulkFlowError) -> Self {
        match err {
            PyGeneralizedBulkFlowError::InvalidIndex { .. } => {
                PyIndexError::new_err(err.to_string())
            },
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

type PyResult<T> = Result<T, PyGeneralizedBulkFlowError>;

// ——————————————————————————————————————————————————————
// Python class
// ——————————————————————————————————————————————————————

#[pyclass(name = "NewGeneralizedBulkFlow")]
pub struct PyNewGeneralizedBulkFlow {
    inner: GeneralizedBulkFlow<BuiltInLattice>,
}

#[pymethods]
impl PyNewGeneralizedBulkFlow {
    #[new]
    #[pyo3(signature = (lattice_bubbles))]
    pub fn new(lattice_bubbles: PyLatticeBubbles) -> PyResult<Self> {
        let bulk_flow = GeneralizedBulkFlow::new(lattice_bubbles.inner)?;
        Ok(PyNewGeneralizedBulkFlow { inner: bulk_flow })
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

    // ————————————————————————————————————————
    // Tensorized computation methods
    // ————————————————————————————————————————

    /// Compute the time-integrated C tensor for one bubble.
    ///
    /// Returns shape: (6, n_sets, n_w)
    pub fn compute_c_tensor_fixed_bubble(
        &mut self,
        py: Python,
        a_idx: usize,
        w_arr: Vec<f64>,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let c_tensor = self
            .inner
            .compute_c_tensor_fixed_bubble(a_idx, &w_arr, t_begin, t_end, n_t)?;

        // Convert HermitianTensor3<Array2<Complex64>> → (6, n_sets, n_w)
        let n_sets = c_tensor.xx().shape()[0];
        let n_w = c_tensor.xx().shape()[1];
        let mut out = Vec::with_capacity(6 * n_sets * n_w);
        for comp in [
            c_tensor.xx(),
            c_tensor.xy(),
            c_tensor.xz(),
            c_tensor.yy(),
            c_tensor.yz(),
            c_tensor.zz(),
        ] {
            out.extend(comp.iter().map(|&c| NumpyComplex64::new(c.re, c.im)));
        }
        let out_array = ndarray::Array3::from_shape_vec((6, n_sets, n_w), out)
            .map_err(|e| PyGeneralizedBulkFlowError::ArrayShapeMismatch(e.to_string()))?;
        Ok(PyArray3::from_array(py, &out_array).into())
    }

    /// Compute the time-dependent C tensor integrand for one bubble.
    ///
    /// Returns shape: (6, n_sets, n_w, n_t)
    pub fn compute_c_tensor_integrand_fixed_bubble(
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
            .compute_c_tensor_integrand_fixed_bubble(a_idx, &w_arr, t_begin, t_end, n_t)?;

        // HermitianTensor3<Array3<Complex64>> → (6, n_sets, n_w, n_t)
        let n_sets = integrand.xx().shape()[0];
        let n_w = integrand.xx().shape()[1];
        let n_t = integrand.xx().shape()[2];
        let mut out = Vec::with_capacity(6 * n_sets * n_w * n_t);
        for comp in [
            integrand.xx(),
            integrand.xy(),
            integrand.xz(),
            integrand.yy(),
            integrand.yz(),
            integrand.zz(),
        ] {
            out.extend(comp.iter().map(|&c| NumpyComplex64::new(c.re, c.im)));
        }
        let out_array = ndarray::Array4::from_shape_vec((6, n_sets, n_w, n_t), out)
            .map_err(|e| PyGeneralizedBulkFlowError::ArrayShapeMismatch(e.to_string()))?;
        Ok(PyArray4::from_array(py, &out_array).into())
    }

    /// Compute the time-integrated C tensor, summed over selected bubbles.
    ///
    /// Returns shape: (6, n_sets, n_w)
    #[pyo3(signature = (w_arr, *, t_begin=None, t_end, n_t, selected_bubbles=None))]
    pub fn compute_c_tensor(
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

        let c_tensor = self
            .inner
            .compute_c_tensor(&w_arr, t_begin, t_end, n_t, selected_slice)?;

        let n_sets = c_tensor.xx().shape()[0];
        let n_w = c_tensor.xx().shape()[1];
        let mut out = Vec::with_capacity(6 * n_sets * n_w);
        for comp in [
            c_tensor.xx(),
            c_tensor.xy(),
            c_tensor.xz(),
            c_tensor.yy(),
            c_tensor.yz(),
            c_tensor.zz(),
        ] {
            out.extend(comp.iter().map(|&c| NumpyComplex64::new(c.re, c.im)));
        }
        let out_array = ndarray::Array3::from_shape_vec((6, n_sets, n_w), out)
            .map_err(|e| PyGeneralizedBulkFlowError::ArrayShapeMismatch(e.to_string()))?;
        Ok(PyArray3::from_owned_array(py, out_array).into())
    }

    /// Compute the time-dependent C tensor integrand, summed over selected
    /// bubbles.
    ///
    /// Returns shape: (6, n_sets, n_w, n_t)
    #[pyo3(signature = (w_arr, *, t_begin=None, t_end, n_t, selected_bubbles=None))]
    pub fn compute_c_tensor_integrand(
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

        let integrand =
            self.inner
                .compute_c_tensor_integrand(&w_arr, t_begin, t_end, n_t, selected_slice)?;

        let n_sets = integrand.xx().shape()[0];
        let n_w = integrand.xx().shape()[1];
        let n_t = integrand.xx().shape()[2];
        let mut out = Vec::with_capacity(6 * n_sets * n_w * n_t);
        for comp in [
            integrand.xx(),
            integrand.xy(),
            integrand.xz(),
            integrand.yy(),
            integrand.yz(),
            integrand.zz(),
        ] {
            out.extend(comp.iter().map(|&c| NumpyComplex64::new(c.re, c.im)));
        }
        let out_array = ndarray::Array4::from_shape_vec((6, n_sets, n_w, n_t), out)
            .map_err(|e| PyGeneralizedBulkFlowError::ArrayShapeMismatch(e.to_string()))?;
        Ok(PyArray4::from_owned_array(py, out_array).into())
    }

    // ————————————————————————————————————————
    // Legacy compatibility methods
    // ————————————————————————————————————————

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
}

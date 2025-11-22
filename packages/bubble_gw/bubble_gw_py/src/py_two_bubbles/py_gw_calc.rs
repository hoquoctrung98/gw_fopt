use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

use bubble_gw_rs::two_bubbles::gw_calc::{
    GWCalcError, GravitationalWaveCalculator, InitialFieldStatus,
};

#[derive(Error, Debug)]
pub enum PyGWCalcError {
    #[error("phi1 and phi2 must have the same shape: phi1={phi1:?}, phi2={phi2:?}")]
    ShapeMismatch { phi1: Vec<usize>, phi2: Vec<usize> },

    #[error("w_arr and cos_thetak_arr must have the same length: {w_len} vs {k_len}")]
    LengthMismatch { w_len: usize, k_len: usize },

    #[error("Tolerance must be positive: got {0}")]
    InvalidTolerance(f64),

    #[error("Maximum iterations must be > 0: got {0}")]
    InvalidMaxIter(u32),

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuild(String),

    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("Invalid cutoff: t_cut={t_cut}, t_0={t_0}, smax={smax}")]
    InvalidCutoff { t_cut: f64, t_0: f64, smax: f64 },

    #[error("Invalid initial_field_status: {0}")]
    InvalidInitialFieldStatus(String),
}

impl From<GWCalcError> for PyGWCalcError {
    fn from(err: GWCalcError) -> Self {
        match err {
            GWCalcError::ShapeMismatch { phi1, phi2 } => {
                PyGWCalcError::ShapeMismatch { phi1, phi2 }
            }
            GWCalcError::LengthMismatch { w_len, k_len } => {
                PyGWCalcError::LengthMismatch { w_len, k_len }
            }
            GWCalcError::InvalidTolerance(t) => PyGWCalcError::InvalidTolerance(t),
            GWCalcError::InvalidMaxIter(n) => PyGWCalcError::InvalidMaxIter(n),
            GWCalcError::ThreadPoolBuildError(e) => PyGWCalcError::ThreadPoolBuild(e.to_string()),
            GWCalcError::IntegrationFailed(s) => PyGWCalcError::IntegrationFailed(s),
            GWCalcError::InvalidCutoff { t_cut, t_0, smax } => {
                PyGWCalcError::InvalidCutoff { t_cut, t_0, smax }
            }
        }
    }
}

impl From<PyGWCalcError> for PyErr {
    fn from(err: PyGWCalcError) -> Self {
        match err {
            PyGWCalcError::ShapeMismatch { .. }
            | PyGWCalcError::LengthMismatch { .. }
            | PyGWCalcError::InvalidTolerance(_)
            | PyGWCalcError::InvalidMaxIter(_)
            | PyGWCalcError::InvalidCutoff { .. }
            | PyGWCalcError::InvalidInitialFieldStatus(_) => PyValueError::new_err(err.to_string()),
            PyGWCalcError::ThreadPoolBuild(_) | PyGWCalcError::IntegrationFailed(_) => {
                PyRuntimeError::new_err(err.to_string())
            }
        }
    }
}

type PyResult<T> = Result<T, PyGWCalcError>;

#[pyclass(name = "GravitationalWaveCalculator")]
pub struct PyGravitationalWaveCalculator {
    inner: GravitationalWaveCalculator,
}

#[pymethods]
impl PyGravitationalWaveCalculator {
    #[new]
    #[pyo3(signature = (initial_field_status, phi1, phi2, z_grid, ds, ratio_t_cut = None, ratio_t_0 = None, num_threads = None))]
    fn new(
        initial_field_status: &str,
        phi1: PyReadonlyArray3<f64>,
        phi2: PyReadonlyArray3<f64>,
        z_grid: PyReadonlyArray1<f64>,
        ds: f64,
        ratio_t_cut: Option<f64>,
        ratio_t_0: Option<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let phi1 = phi1.to_owned_array();
        let phi2 = phi2.to_owned_array();
        let z_grid = z_grid.to_owned_array();

        let initial_field_status = match initial_field_status.to_lowercase().as_str() {
            "one_bubble" => InitialFieldStatus::OneBubble,
            "two_bubbles" => InitialFieldStatus::TwoBubbles,
            _ => {
                return Err(PyGWCalcError::InvalidInitialFieldStatus(
                    initial_field_status.to_string(),
                ));
            }
        };

        let inner = GravitationalWaveCalculator::new(
            initial_field_status,
            phi1,
            phi2,
            z_grid,
            ds,
            ratio_t_cut,
            ratio_t_0,
            num_threads,
        )?;

        Ok(Self { inner })
    }

    fn set_integral_params(&mut self, tol: f64, max_iter: u32) -> PyResult<()> {
        self.inner.set_integral_params(tol, max_iter)?; // Clean
        Ok(())
    }

    #[pyo3(signature = (w_arr, cos_thetak_arr))]
    fn compute_averaged_gw_spectrum(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        cos_thetak_arr: Vec<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let results = self
            .inner
            .compute_averaged_gw_spectrum(w_arr, cos_thetak_arr)?;

        Ok(PyArray1::from_vec(py, results).into())
    }

    #[pyo3(signature = (w_arr, cos_thetak_arr))]
    fn compute_angular_gw_spectrum(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        cos_thetak_arr: Vec<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let spectrum_2d = self
            .inner
            .compute_angular_gw_spectrum(w_arr, cos_thetak_arr)?;
        Ok(PyArray2::from_array(py, &spectrum_2d).into())
    }

    #[pyo3(signature = (w_arr, cos_thetak_arr))]
    fn compute_t_tensor(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        cos_thetak_arr: Vec<f64>,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let t_tensor = self.inner.compute_t_tensor(w_arr, cos_thetak_arr)?;
        Ok(PyArray3::from_array(py, &t_tensor).into())
    }

    #[getter]
    fn phi1(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.phi1).into()
    }

    #[getter]
    fn phi2(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.phi2).into()
    }

    #[getter]
    fn dphi1_dz(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.dphi1_dz_sq).into()
    }

    #[getter]
    fn dphi1_ds(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.dphi1_ds_sq).into()
    }

    #[getter]
    fn dphi2_dz(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.dphi2_dz_sq).into()
    }

    #[getter]
    fn dphi2_ds(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.dphi2_ds_sq).into()
    }

    #[getter]
    fn xz_deriv_dphi_dz(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi1_dz).into()
    }

    #[getter]
    fn xz_deriv_dphi_ds(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi1_ds).into()
    }

    #[getter]
    fn xz_deriv_dphi_dz2(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi2_dz).into()
    }

    #[getter]
    fn xz_deriv_dphi_ds2(&self, py: Python) -> Py<PyArray3<f64>> {
        PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi2_ds).into()
    }

    #[getter]
    fn zz_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.precomputed.zz_weights).into()
    }

    #[getter]
    fn rr_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.precomputed.rr_weights).into()
    }

    #[getter]
    fn xz_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.precomputed.xz_weights).into()
    }

    #[getter]
    fn z_grid(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.lattice.z_grid).into()
    }

    #[getter]
    fn s_grid(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.lattice.s_grid).into()
    }

    #[getter]
    fn ds(&self) -> f64 {
        self.inner.lattice.ds
    }

    #[getter]
    fn dz(&self) -> f64 {
        self.inner.lattice.dz
    }

    #[getter]
    fn n_s(&self) -> usize {
        self.inner.lattice.n_s
    }

    #[getter]
    fn n_z(&self) -> usize {
        self.inner.lattice.n_z
    }

    #[getter]
    fn t_cut(&self) -> f64 {
        self.inner.config.t_cut
    }

    #[getter]
    fn t_0(&self) -> f64 {
        self.inner.config.t_0
    }

    #[getter]
    fn tol(&self) -> f64 {
        self.inner.tol
    }

    #[getter]
    fn max_iter(&self) -> u32 {
        self.inner.max_iter
    }

    #[getter]
    fn s_offset(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.s_offset).into()
    }

    #[getter]
    fn n_fields(&self) -> usize {
        self.inner.n_fields
    }
}

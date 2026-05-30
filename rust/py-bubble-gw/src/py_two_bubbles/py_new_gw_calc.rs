use bubble_gw::two_bubbles::Integral;
use bubble_gw::two_bubbles::new_gw_calc::{
    ExponentialTimeCutoff,
    GWCalcError,
    GravitationalWaveCalculator,
    InitialFieldStatus,
};
use numpy::{
    Complex64 as NumpyComplex64,
    PyArray1,
    PyArray2,
    PyArray3,
    PyArrayMethods,
    PyReadonlyArray1,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PyNewGWCalcError {
    #[error("phi1 and phi2 must have the same shape: phi1={phi1:?}, phi2={phi2:?}")]
    ShapeMismatch { phi1: Vec<usize>, phi2: Vec<usize> },

    #[error("w_arr and cos_thetak_arr must have the same length: {w_len} vs {k_len}")]
    LengthMismatch { w_len: usize, k_len: usize },

    #[error("Tolerance must be positive: got {0}")]
    InvalidTolerance(f64),

    #[error("Maximum iterations must be > 0: got {0}")]
    InvalidMaxIter(u32),

    #[error("Gauss-Legendre integration order must be in 2..=30: got {0}")]
    InvalidGaussLegendreOrder(usize),

    #[error("Newton-Cotes integration requires at least 1 subinterval: got {0}")]
    InvalidNewtonCotesOrder(usize),

    #[error("The selected integration method does not accept tolerance/max_iter parameters")]
    IntegralParamsUnsupported,

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuild(String),

    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("Invalid cutoff: t_cut={t_cut}, t_0={t_0}, smax={smax}")]
    InvalidCutoff { t_cut: f64, t_0: f64, smax: f64 },

    #[error("Invalid initial_field_status: {0}")]
    InvalidInitialFieldStatus(String),

    #[error("Invalid integration method: {0}")]
    InvalidIntegrationMethod(String),
}

impl From<GWCalcError> for PyNewGWCalcError {
    fn from(err: GWCalcError) -> Self {
        match err {
            GWCalcError::ShapeMismatch { phi1, phi2 } => {
                PyNewGWCalcError::ShapeMismatch { phi1, phi2 }
            },
            GWCalcError::LengthMismatch { w_len, k_len } => {
                PyNewGWCalcError::LengthMismatch { w_len, k_len }
            },
            GWCalcError::InvalidTolerance(tol) => PyNewGWCalcError::InvalidTolerance(tol),
            GWCalcError::InvalidMaxIter(max_iter) => PyNewGWCalcError::InvalidMaxIter(max_iter),
            GWCalcError::InvalidGaussLegendreOrder(n) => {
                PyNewGWCalcError::InvalidGaussLegendreOrder(n)
            },
            GWCalcError::InvalidNewtonCotesOrder(n) => PyNewGWCalcError::InvalidNewtonCotesOrder(n),
            GWCalcError::IntegralParamsUnsupported => PyNewGWCalcError::IntegralParamsUnsupported,
            GWCalcError::ThreadPoolBuildError(e) => {
                PyNewGWCalcError::ThreadPoolBuild(e.to_string())
            },
            GWCalcError::IntegrationFailed(s) => PyNewGWCalcError::IntegrationFailed(s),
            GWCalcError::InvalidCutoff { t_cut, t_0, smax } => {
                PyNewGWCalcError::InvalidCutoff { t_cut, t_0, smax }
            },
        }
    }
}

impl From<PyNewGWCalcError> for PyErr {
    fn from(err: PyNewGWCalcError) -> Self {
        match err {
            PyNewGWCalcError::ShapeMismatch { .. }
            | PyNewGWCalcError::LengthMismatch { .. }
            | PyNewGWCalcError::InvalidTolerance(_)
            | PyNewGWCalcError::InvalidMaxIter(_)
            | PyNewGWCalcError::InvalidGaussLegendreOrder(_)
            | PyNewGWCalcError::InvalidNewtonCotesOrder(_)
            | PyNewGWCalcError::IntegralParamsUnsupported
            | PyNewGWCalcError::InvalidCutoff { .. }
            | PyNewGWCalcError::InvalidInitialFieldStatus(_)
            | PyNewGWCalcError::InvalidIntegrationMethod(_) => {
                PyValueError::new_err(err.to_string())
            },
            PyNewGWCalcError::ThreadPoolBuild(_) | PyNewGWCalcError::IntegrationFailed(_) => {
                PyRuntimeError::new_err(err.to_string())
            },
        }
    }
}

type PyResult<T> = Result<T, PyNewGWCalcError>;

fn parse_initial_field_status(initial_field_status: &str) -> PyResult<InitialFieldStatus> {
    match initial_field_status.to_lowercase().as_str() {
        "one_bubble" => Ok(InitialFieldStatus::OneBubble),
        "two_bubbles" => Ok(InitialFieldStatus::TwoBubbles),
        _ => Err(PyNewGWCalcError::InvalidInitialFieldStatus(initial_field_status.to_string())),
    }
}

fn parse_integration_method(
    method: &str,
    n: Option<usize>,
    tol: Option<f64>,
    max_iter: Option<u32>,
) -> PyResult<Integral> {
    let method = method.to_lowercase().replace(['-', ' '], "_");
    let tol = tol.unwrap_or(1e-5);
    let max_iter = max_iter.unwrap_or(20);

    match method.as_str() {
        "gauss_legendre" => Ok(Integral::GaussLegendre(n.unwrap_or(16))),
        "newton_cotes" => Ok(Integral::NewtonCotes(n.unwrap_or(6))),
        "g7k15" => Ok(Integral::G7K15(tol, max_iter)),
        "g10k21" => Ok(Integral::G10K21(tol, max_iter)),
        "g15k31" => Ok(Integral::G15K31(tol, max_iter)),
        "g20k41" => Ok(Integral::G20K41(tol, max_iter)),
        "g25k51" => Ok(Integral::G25K51(tol, max_iter)),
        "g30k61" | "gauss_kronrod" => Ok(Integral::G30K61(tol, max_iter)),
        "g7k15r" => Ok(Integral::G7K15R(tol, max_iter)),
        "g10k21r" => Ok(Integral::G10K21R(tol, max_iter)),
        "g15k31r" => Ok(Integral::G15K31R(tol, max_iter)),
        "g20k41r" => Ok(Integral::G20K41R(tol, max_iter)),
        "g25k51r" => Ok(Integral::G25K51R(tol, max_iter)),
        "g30k61r" | "gauss_kronrod_relative" => Ok(Integral::G30K61R(tol, max_iter)),
        _ => Err(PyNewGWCalcError::InvalidIntegrationMethod(method)),
    }
}

fn format_integration_method(method: Integral) -> String {
    match method {
        Integral::GaussLegendre(n) => format!("gauss_legendre(n={n})"),
        Integral::NewtonCotes(n) => format!("newton_cotes(n={n})"),
        Integral::G7K15(tol, max_iter) => format!("g7k15(tol={tol}, max_iter={max_iter})"),
        Integral::G10K21(tol, max_iter) => format!("g10k21(tol={tol}, max_iter={max_iter})"),
        Integral::G15K31(tol, max_iter) => format!("g15k31(tol={tol}, max_iter={max_iter})"),
        Integral::G20K41(tol, max_iter) => format!("g20k41(tol={tol}, max_iter={max_iter})"),
        Integral::G25K51(tol, max_iter) => format!("g25k51(tol={tol}, max_iter={max_iter})"),
        Integral::G30K61(tol, max_iter) => format!("g30k61(tol={tol}, max_iter={max_iter})"),
        Integral::G7K15R(tol, max_iter) => format!("g7k15r(tol={tol}, max_iter={max_iter})"),
        Integral::G10K21R(tol, max_iter) => format!("g10k21r(tol={tol}, max_iter={max_iter})"),
        Integral::G15K31R(tol, max_iter) => format!("g15k31r(tol={tol}, max_iter={max_iter})"),
        Integral::G20K41R(tol, max_iter) => format!("g20k41r(tol={tol}, max_iter={max_iter})"),
        Integral::G25K51R(tol, max_iter) => format!("g25k51r(tol={tol}, max_iter={max_iter})"),
        Integral::G30K61R(tol, max_iter) => format!("g30k61r(tol={tol}, max_iter={max_iter})"),
    }
}

#[pyclass(name = "NewGravitationalWaveCalculator")]
pub struct PyNewGravitationalWaveCalculator {
    inner: GravitationalWaveCalculator<ExponentialTimeCutoff>,
}

#[pymethods]
impl PyNewGravitationalWaveCalculator {
    #[new]
    #[pyo3(signature = (initial_field_status, phi1, phi2, z_grid, ds, ratio_t_cut = None, ratio_t_0 = None))]
    fn new(
        initial_field_status: &str,
        phi1: PyReadonlyArray3<f64>,
        phi2: PyReadonlyArray3<f64>,
        z_grid: PyReadonlyArray1<f64>,
        ds: f64,
        ratio_t_cut: Option<f64>,
        ratio_t_0: Option<f64>,
    ) -> PyResult<Self> {
        let phi1 = phi1.to_owned_array();
        let phi2 = phi2.to_owned_array();
        let z_grid = z_grid.to_owned_array();
        let initial_field_status = parse_initial_field_status(initial_field_status)?;
        let smax = (phi1.shape()[1] - 1) as f64 * ds;
        let time_cutoff = ExponentialTimeCutoff::new(smax, ratio_t_cut, ratio_t_0);

        let inner = GravitationalWaveCalculator::new(
            initial_field_status,
            phi1,
            phi2,
            z_grid,
            ds,
            time_cutoff,
        )?;

        Ok(Self { inner })
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> PyResult<()> {
        self.inner.set_num_threads(num_threads)?;
        Ok(())
    }

    #[pyo3(signature = (method, n = None, tol = None, max_iter = None))]
    fn set_integration_params(
        &mut self,
        method: &str,
        n: Option<usize>,
        tol: Option<f64>,
        max_iter: Option<u32>,
    ) -> PyResult<()> {
        let method = parse_integration_method(method, n, tol, max_iter)?;
        self.inner.set_integration_params(method)?;
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
        self.inner.time_cutoff.t_cut
    }

    #[getter]
    fn t_0(&self) -> f64 {
        self.inner.time_cutoff.t_0
    }

    #[getter]
    fn integration_method(&self) -> String {
        format_integration_method(self.inner.quadrature.method)
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

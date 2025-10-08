use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

use bubble_gw_rs::two_bubbles::gw_calc::{GravitationalWaveCalculator, InitialFieldStatus};

#[pyclass(name = "GravitationalWaveCalculator")]
pub struct PyGravitationalWaveCalculator {
    inner: GravitationalWaveCalculator,
}

#[pymethods]
impl PyGravitationalWaveCalculator {
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

        let initial_field_status = match initial_field_status.to_lowercase().as_str() {
            "one_bubble" => InitialFieldStatus::OneBubble,
            "two_bubbles" => InitialFieldStatus::TwoBubbles,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid initial_field_status".to_string(),
                ))
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
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Ok(PyGravitationalWaveCalculator { inner })
    }

    #[pyo3(name = "set_integral_params")]
    fn set_integral_params(&mut self, tol: f64, max_iter: u32) -> PyResult<()> {
        self.inner
            .set_integral_params(tol, max_iter)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(())
    }

    #[pyo3(signature = (w_arr, n_k, num_threads = None))]
    fn compute_averaged_gw_spectrum(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        n_k: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let results = self
            .inner
            .compute_averaged_gw_spectrum(&w_arr, n_k, num_threads)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Error: {}", e))
            })?;
        Ok(PyArray1::from_vec(py, results).into())
    }

    #[pyo3(signature = (w_arr, k_arr, num_threads = None))]
    fn compute_t_tensor(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        k_arr: Vec<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let results = self
            .inner
            .compute_t_tensor(&w_arr, &k_arr, num_threads)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Error: {}", e))
            })?;

        // Validate the result length
        let n = w_arr.len();
        if results.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Expected {} tensor results, got {}",
                n,
                results.len()
            )));
        }

        // Convert Vec<(Complex64, Complex64, Complex64, Complex64)> to a flat Vec<f64>
        let flat_results: Vec<f64> = results
            .into_iter()
            .flat_map(|(t_xx, t_yy, t_zz, t_xz)| {
                vec![
                    t_xx.re, t_xx.im, t_yy.re, t_yy.im, t_zz.re, t_zz.im, t_xz.re, t_xz.im,
                ]
            })
            .collect();

        // Create a NumPy array of shape (n, 8)
        let array = unsafe {
            let array = PyArray2::new(py, [n, 8], false);
            for (i, chunk) in flat_results.chunks(8).enumerate() {
                for (j, &val) in chunk.iter().enumerate() {
                    *array.uget_mut([i, j]) = val;
                }
            }
            array
        };

        Ok(array.into())
    }

    #[pyo3(signature = (w_arr, cos_thetak_arr, num_threads = None))]
    fn compute_angular_gw_spectrum(
        &self,
        py: Python,
        w_arr: Vec<f64>,
        cos_thetak_arr: Vec<f64>,
        num_threads: Option<usize>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let results = self
            .inner
            .compute_angular_gw_spectrum(&w_arr, &cos_thetak_arr, num_threads)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Error: {}", e))
            })?;
        Ok(PyArray1::from_vec(py, results).into())
    }

    #[getter]
    fn phi1(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.phi1).into())
    }

    #[getter]
    fn phi2(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.phi2).into())
    }

    #[getter]
    fn dphi1_dz(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.dphi1_dz_sq).into())
    }

    #[getter]
    fn dphi1_ds(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.dphi1_ds_sq).into())
    }

    #[getter]
    fn dphi2_dz(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.dphi2_dz_sq).into())
    }

    #[getter]
    fn dphi2_ds(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.dphi2_ds_sq).into())
    }

    #[getter]
    fn xz_deriv_dphi_dz(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi1_dz).into())
    }

    #[getter]
    fn xz_deriv_dphi_ds(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi1_ds).into())
    }

    #[getter]
    fn xz_deriv_dphi_dz2(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi2_dz).into())
    }

    #[getter]
    fn xz_deriv_dphi_ds2(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, &self.inner.precomputed.xz_deriv_dphi2_ds).into())
    }

    #[getter]
    fn zz_weights(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.precomputed.zz_weights).into())
    }

    #[getter]
    fn rr_weights(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.precomputed.rr_weights).into())
    }

    #[getter]
    fn xz_weights(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.precomputed.xz_weights).into())
    }

    #[getter]
    fn z_grid(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.lattice.z_grid).into())
    }

    #[getter]
    fn s_grid(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.lattice.s_grid).into())
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
    fn t_cut(&self) -> PyResult<f64> {
        Ok(self.inner.config.t_cut)
    }

    #[getter]
    fn t_0(&self) -> PyResult<f64> {
        Ok(self.inner.config.t_0)
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
    fn s_offset(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_array(py, &self.inner.s_offset).into())
    }

    #[getter]
    fn n_fields(&self) -> usize {
        self.inner.n_fields
    }
}

use bubble_gw_rs::many_bubbles::bulk_flow::{BubbleIndex, BulkFlow, CollisionStatus, Segment};
use ndarray::Array2;
use num_complex::Complex64 as RustComplex64;
use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        let interior_four_vectors = bubbles_interior.to_owned_array();
        let exterior_four_vectors =
            bubbles_exterior.map_or(Array2::zeros((0, 4)), |arr| arr.to_owned_array());
        Ok(PyBulkFlow {
            inner: BulkFlow::new(interior_four_vectors, exterior_four_vectors),
        })
    }

    #[getter]
    pub fn get_four_vectors_interior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubbles_interior()).into())
    }

    #[getter]
    pub fn get_four_vectors_exterior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubbles_exterior()).into())
    }

    #[setter]
    pub fn set_four_vectors_interior(
        &mut self,
        four_vectors: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let four_vectors = four_vectors.to_owned_array();
        self.inner.set_interior_four_vectors(four_vectors);
        Ok(())
    }

    #[setter]
    pub fn set_four_vectors_exterior(
        &mut self,
        four_vectors: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let four_vectors = four_vectors.to_owned_array();
        self.inner.set_exterior_four_vectors(four_vectors);
        Ok(())
    }

    #[getter]
    pub fn get_delta(&self, py: Python) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_array(py, self.inner.delta()).into())
    }

    #[setter]
    pub fn set_delta(&mut self, delta: PyReadonlyArray3<f64>) -> PyResult<()> {
        let delta = delta.to_owned_array();
        self.inner.set_delta(delta);
        Ok(())
    }

    #[getter]
    pub fn get_coefficients_sets(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.coefficients_sets()).into())
    }

    #[setter]
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

    #[setter]
    pub fn set_powers_sets(&mut self, powers_sets: PyReadonlyArray2<f64>) -> PyResult<()> {
        let powers_sets = powers_sets.to_owned_array();
        self.inner.set_powers_sets(powers_sets);
        Ok(())
    }

    #[getter]
    pub fn get_active_sets(&self, py: Python) -> PyResult<Py<PyArray1<bool>>> {
        Ok(PyArray1::from_array(py, self.inner.active_sets()).into())
    }

    #[setter]
    pub fn set_active_sets(&mut self, active_sets: PyReadonlyArray1<bool>) -> PyResult<()> {
        let active_sets = active_sets.to_owned_array();
        self.inner.set_active_sets(active_sets);
        Ok(())
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
            .map_err(|e| PyValueError::new_err(e))
    }

    pub fn compute_b_matrix(
        &self,
        py: Python,
        cos_thetax: f64,
        segments: Vec<(f64, f64, f64, i32, i32)>,
        t_nc_grid: PyReadonlyArray2<f64>,
        idx: usize,
        t: f64,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let ta = self.inner.bubbles_interior()[[0, 0]];
        let delta_ta = t - ta;
        let t_nc_grid = t_nc_grid.to_owned_array();
        let rust_segments: Vec<Segment> = segments
            .into_iter()
            .map(|s| {
                let collision_status = match s.4 {
                    0 => CollisionStatus::NeverCollided,
                    1 => CollisionStatus::AlreadyCollided,
                    2 => CollisionStatus::NotYetCollided,
                    _ => return Err(PyValueError::new_err("Invalid collision status")),
                };
                let bubble_index = if s.3 == -1 {
                    BubbleIndex::None
                } else if s.3 >= 0 {
                    BubbleIndex::Interior(s.3 as usize)
                } else {
                    BubbleIndex::Exterior((-s.3 - 2) as usize)
                };
                Ok(Segment {
                    cos_thetax: s.0,
                    phi_lower: s.1,
                    phi_upper: s.2,
                    bubble_index,
                    collision_status,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (b_plus_arr, b_minus_arr) = self.inner.compute_b_matrix(
            cos_thetax,
            &rust_segments,
            t_nc_grid.view(),
            idx,
            delta_ta,
        );
        Ok((
            PyArray1::from_array(py, &b_plus_arr).into(),
            PyArray1::from_array(py, &b_minus_arr).into(),
        ))
    }

    pub fn compute_a_matrix(
        &self,
        py: Python,
        a_idx: usize,
        w_arr: PyReadonlyArray1<f64>,
        t: f64,
        first_bubble: PyReadonlyArray2<i32>,
        delta_tab_grid: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<NumpyComplex64>>, Py<PyArray2<NumpyComplex64>>)> {
        let w_arr = w_arr.to_owned_array();
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
        let (a_plus, a_minus) = self.inner.compute_a_matrix(
            a_idx,
            w_arr.view(),
            t,
            first_bubble.view(),
            delta_tab_grid.view(),
        );
        let a_plus_numpy = a_plus.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        let a_minus_numpy = a_minus.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        Ok((
            PyArray2::from_array(py, &a_plus_numpy).into(),
            PyArray2::from_array(py, &a_minus_numpy).into(),
        ))
    }

    pub fn compute_first_colliding_bubble(
        &self,
        py: Python,
        a_idx: usize,
    ) -> PyResult<Py<PyArray2<i32>>> {
        let first_bubble = self.inner.compute_first_colliding_bubble(a_idx);
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
        let delta_tab_grid = self.inner.compute_delta_tab(a_idx, first_bubble.view());
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
        );
        let collision_status_int = collision_status.mapv(|s| s as i32);
        Ok(PyArray2::from_array(py, &collision_status_int).into())
    }

    // pub fn generate_segments(
    //     &self,
    //     first_bubble: PyReadonlyArray2<i32>,
    //     collision_status: PyReadonlyArray2<i32>,
    // ) -> PyResult<Vec<(f64, f64, f64, i32, i32)>> {
    //     let first_bubble = first_bubble.to_owned_array().mapv(|i: i32| {
    //         if i == -1 {
    //             BubbleIndex::None
    //         } else if i >= 0 {
    //             BubbleIndex::Interior(i as usize)
    //         } else {
    //             BubbleIndex::Exterior((-i - 2) as usize)
    //         }
    //     });
    //     let collision_status = collision_status.to_owned_array();
    //     let segments = self
    //         .inner
    //         .generate_segments(first_bubble.view(), collision_status.view());
    //     let py_segments: Vec<(f64, f64, f64, i32, i32)> = segments
    //         .into_iter()
    //         .map(|s| {
    //             let status_int = match s.collision_status {
    //                 CollisionStatus::NeverCollided => 0,
    //                 CollisionStatus::AlreadyCollided => 1,
    //                 CollisionStatus::NotYetCollided => 2,
    //             };
    //             let bubble_int = match s.bubble_index {
    //                 BubbleIndex::None => -1,
    //                 BubbleIndex::Interior(i) => i as i32,
    //                 BubbleIndex::Exterior(i) => -((i as i32) + 2),
    //             };
    //             (
    //                 s.cos_thetax,
    //                 s.phi_lower,
    //                 s.phi_upper,
    //                 bubble_int,
    //                 status_int,
    //             )
    //         })
    //         .collect();
    //     Ok(py_segments)
    // }

    pub fn compute_c_integrand(
        &self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let integrand = self.inner.compute_c_integrand(w_arr.view(), t_max, n_t);
        let integrand_numpy = integrand.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_array(py, &integrand_numpy).into())
    }

    pub fn compute_c_matrix(
        &mut self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let c_matrix = self.inner.compute_c_matrix(w_arr.view(), t_max, n_t);
        let c_matrix_numpy = c_matrix.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    pub fn compute_c_matrix_fixed_bubble(
        &mut self,
        py: Python,
        a_idx: usize,
        w_arr: PyReadonlyArray1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let c_matrix = self
            .inner
            .compute_c_matrix_fixed_bubble(a_idx, w_arr.view(), t_max, n_t);
        let c_matrix_numpy = c_matrix.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
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
            .set_resolution(n_cos_thetax, n_phix, precompute_first_bubbles);
        Ok(())
    }
}

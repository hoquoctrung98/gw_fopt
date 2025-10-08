use bubble_gw_rs::many_bubbles::bubble_formation::{
    BubbleFormationSimulator, Lattice, LatticeType, ManualNucleation, NucleationStrategy,
    PoissonNucleation, SimulationState,
};
use bubble_gw_rs::many_bubbles::bulk_flow::{BubbleIndex, BulkFlow, CollisionStatus, Segment};
use ndarray::Array2;
use num_complex::Complex64 as RustComplex64;
use numpy::{
    Complex64 as NumpyComplex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

#[pyclass(name = "Lattice")]
pub struct PyLattice {
    inner: Lattice,
}

#[pymethods]
impl PyLattice {
    #[new]
    fn new(lattice_type: &str, sizes: Vec<f64>, n: usize) -> PyResult<Self> {
        let inner = Lattice::new(lattice_type, sizes, n).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyLattice { inner })
    }

    fn get_volume(&self) -> f64 {
        self.inner.get_volume()
    }

    fn generate_grid(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let (grid, _cell_map) = self.inner.generate_grid();
        Ok(PyArray2::from_array(py, &grid).into())
    }

    fn get_lattice_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.get_lattice_bounds()
    }

    #[getter]
    fn lattice_type(&self) -> String {
        match self.inner.lattice_type {
            LatticeType::Cartesian => "cartesian".to_string(),
            LatticeType::Sphere => "sphere".to_string(),
        }
    }

    #[getter]
    fn sizes(&self) -> Vec<f64> {
        self.inner.sizes.to_vec()
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
}

#[pyclass(name = "PoissonNucleation")]
pub struct PyPoissonNucleation {
    inner: PoissonNucleation,
}

#[pymethods]
impl PyPoissonNucleation {
    #[new]
    fn new(params: &Bound<PyDict>) -> PyResult<Self> {
        let mut rust_params = BTreeMap::new();
        for item in params.items() {
            let (key, value) = item.extract::<(String, f64)>()?;
            rust_params.insert(key, value);
        }
        let inner = PoissonNucleation::new(rust_params).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyPoissonNucleation { inner })
    }
}

#[pyclass(name = "ManualNucleation")]
pub struct PyManualNucleation {
    inner: ManualNucleation,
}

#[pymethods]
impl PyManualNucleation {
    #[new]
    fn new(schedule: Vec<(f64, Vec<Vec<f64>>)>) -> PyResult<Self> {
        let rust_schedule: Vec<(f64, Vec<[f64; 3]>)> = schedule
            .into_iter()
            .map(|(t, points)| {
                let rust_points = points
                    .into_iter()
                    .map(|vec| {
                        if vec.len() != 3 {
                            return Err(PyValueError::new_err(
                                "Each point must have exactly 3 coordinates",
                            ));
                        }
                        Ok([vec[0], vec[1], vec[2]])
                    })
                    .collect::<PyResult<Vec<[f64; 3]>>>()?;
                Ok((t, rust_points))
            })
            .collect::<PyResult<Vec<(f64, Vec<[f64; 3]>)>>>()?;
        let inner = ManualNucleation::new(rust_schedule).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyManualNucleation { inner })
    }
}

#[pyclass(name = "BubbleFormationSimulator")]
pub struct PyBubbleFormationSimulator {
    inner: BubbleFormationSimulator,
}

#[pymethods]
impl PyBubbleFormationSimulator {
    #[new]
    #[pyo3(signature = (lattice, vw = 0.5, dt = 0.1, nucleation_strategy = None, seed=None))]
    fn new(
        lattice: &PyLattice,
        vw: f64,
        dt: f64,
        nucleation_strategy: Option<Py<PyAny>>,
        seed: Option<u64>,
        py: Python,
    ) -> PyResult<Self> {
        let strategy: Option<NucleationStrategy> = match nucleation_strategy {
            Some(obj) => {
                if let Ok(poisson) = obj.extract::<PyRef<PyPoissonNucleation>>(py) {
                    Some(NucleationStrategy::Poisson(poisson.inner.clone()))
                } else if let Ok(manual) = obj.extract::<PyRef<PyManualNucleation>>(py) {
                    Some(NucleationStrategy::Manual(manual.inner.clone()))
                } else {
                    return Err(PyValueError::new_err(
                        "nucleation_strategy must be PoissonNucleation or ManualNucleation",
                    ));
                }
            }
            None => None,
        };
        let inner = BubbleFormationSimulator::new(lattice.inner.clone(), vw, dt, strategy, seed)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyBubbleFormationSimulator { inner })
    }

    fn run_simulation(&mut self, t_final: f64, verbose: bool) -> PyResult<()> {
        self.inner
            .run_simulation(t_final, verbose)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    fn get_boundary_intersecting_bubbles(&self, t: f64) -> Vec<([f64; 3], f64)> {
        self.inner
            .get_boundary_intersecting_bubbles(t)
            .into_iter()
            .map(|(idx, t)| (self.inner.get_center(idx), t))
            .collect()
    }

    fn generate_exterior_bubbles(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let exterior_bubbles = self.inner.generate_exterior_bubbles();
        Ok(PyArray2::from_array(py, &exterior_bubbles).into())
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt()
    }

    #[getter]
    fn v_remaining(&self) -> f64 {
        self.inner.v_remaining()
    }

    fn get_valid_points(&mut self, t: Option<f64>, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let points = self.inner.get_valid_points(t);
        Ok(PyArray2::from_array(py, &points).into())
    }

    fn update_remaining_volume_bulk(&mut self, t: f64) -> f64 {
        let valid_points = self.inner.get_valid_points(Some(t));
        self.inner.update_remaining_volume_bulk(t, &valid_points)
    }

    fn get_bubbles(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let bubbles_array = self.inner.get_bubbles();
        Ok(PyArray2::from_array(py, &bubbles_array).into())
    }

    #[getter]
    fn lattice(&self) -> PyLattice {
        PyLattice {
            inner: self.inner.lattice().clone(),
        }
    }

    #[getter]
    fn vw(&self) -> f64 {
        self.inner.vw()
    }

    #[getter]
    fn grid(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.grid()).into())
    }

    #[getter]
    fn v_total(&self) -> f64 {
        self.inner.v_total()
    }
}

#[pyclass(name = "BulkFlow")]
pub struct PyBulkFlow {
    inner: BulkFlow,
}

#[pymethods]
impl PyBulkFlow {
    #[new]
    #[pyo3(signature = (bubble_four_vectors_interior, bubble_four_vectors_exterior = None))]
    pub fn new(
        bubble_four_vectors_interior: PyReadonlyArray2<f64>,
        bubble_four_vectors_exterior: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let interior_four_vectors = bubble_four_vectors_interior.to_owned_array();
        let exterior_four_vectors =
            bubble_four_vectors_exterior.map_or(Array2::zeros((0, 4)), |arr| arr.to_owned_array());
        Ok(PyBulkFlow {
            inner: BulkFlow::new(interior_four_vectors, exterior_four_vectors),
        })
    }

    #[getter]
    pub fn get_four_vectors_interior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubble_four_vectors_interior()).into())
    }

    #[getter]
    pub fn get_four_vectors_exterior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(PyArray2::from_array(py, self.inner.bubble_four_vectors_exterior()).into())
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
        let ta = self.inner.bubble_four_vectors_interior()[[0, 0]];
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

    pub fn generate_segments(
        &self,
        first_bubble: PyReadonlyArray2<i32>,
        collision_status: PyReadonlyArray2<i32>,
    ) -> PyResult<Vec<(f64, f64, f64, i32, i32)>> {
        let first_bubble = first_bubble.to_owned_array().mapv(|i: i32| {
            if i == -1 {
                BubbleIndex::None
            } else if i >= 0 {
                BubbleIndex::Interior(i as usize)
            } else {
                BubbleIndex::Exterior((-i - 2) as usize)
            }
        });
        let collision_status = collision_status.to_owned_array();
        let segments = self
            .inner
            .generate_segments(first_bubble.view(), collision_status.view());
        let py_segments: Vec<(f64, f64, f64, i32, i32)> = segments
            .into_iter()
            .map(|s| {
                let status_int = match s.collision_status {
                    CollisionStatus::NeverCollided => 0,
                    CollisionStatus::AlreadyCollided => 1,
                    CollisionStatus::NotYetCollided => 2,
                };
                let bubble_int = match s.bubble_index {
                    BubbleIndex::None => -1,
                    BubbleIndex::Interior(i) => i as i32,
                    BubbleIndex::Exterior(i) => -((i as i32) + 2),
                };
                (
                    s.cos_thetax,
                    s.phi_lower,
                    s.phi_upper,
                    bubble_int,
                    status_int,
                )
            })
            .collect();
        Ok(py_segments)
    }

    pub fn compute_c_integrand(
        &self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        tmax: f64,
        n_time_points: usize,
    ) -> PyResult<Py<PyArray4<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let integrand = self
            .inner
            .compute_c_integrand(w_arr.view(), tmax, n_time_points);
        let integrand_numpy = integrand.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray4::from_array(py, &integrand_numpy).into())
    }

    pub fn compute_c_matrix(
        &mut self,
        py: Python,
        w_arr: PyReadonlyArray1<f64>,
        tmax: f64,
        n_time_points: usize,
    ) -> PyResult<Py<PyArray3<NumpyComplex64>>> {
        let w_arr = w_arr.to_owned_array();
        let c_matrix = self
            .inner
            .compute_c_matrix(w_arr.view(), tmax, n_time_points);
        let c_matrix_numpy = c_matrix.mapv(|c: RustComplex64| NumpyComplex64::new(c.re, c.im));
        Ok(PyArray3::from_array(py, &c_matrix_numpy).into())
    }

    /// Sets the angular resolution for the angular grids.
    /// Must be called before any computation methods.
    #[pyo3(signature = (n_cos_thetax, n_phix))]
    pub fn set_resolution(&mut self, n_cos_thetax: usize, n_phix: usize) -> PyResult<()> {
        self.inner.set_resolution(n_cos_thetax, n_phix);
        Ok(())
    }
}

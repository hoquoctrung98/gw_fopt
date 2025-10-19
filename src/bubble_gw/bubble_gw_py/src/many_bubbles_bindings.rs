use bubble_gw_rs::many_bubbles::bubble_formation::{
    BubbleFormationSimulator, Lattice, LatticeType, ManualNucleation, PoissonNucleation,
    SimulationEndStatus,
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

#[pyclass(name = "Lattice")]
pub struct PyLattice {
    inner: Lattice,
}

#[pymethods]
impl PyLattice {
    #[new]
    fn new(lattice_type: &str, sizes: Vec<f64>, n_grid: usize) -> PyResult<Self> {
        let inner =
            Lattice::new(lattice_type, sizes, n_grid).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyLattice { inner })
    }

    fn get_volume(&self) -> f64 {
        self.inner.get_volume()
    }

    fn generate_grid(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let grid = self.inner.generate_grid();
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
        self.inner.n_grid
    }
}

#[pyclass(name = "PoissonNucleation")]
pub struct PyPoissonNucleation {
    inner: PoissonNucleation,
}

#[pymethods]
impl PyPoissonNucleation {
    #[new]
    fn new(gamma0: f64, beta: f64, t0: f64, dp0: f64) -> PyResult<Self> {
        let inner =
            PoissonNucleation::new(gamma0, beta, t0, dp0).map_err(|e| PyValueError::new_err(e))?;
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
    fn new(schedule: Vec<(f64, Vec<Vec<f64>>)>, dt: f64) -> PyResult<Self> {
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
        let inner =
            ManualNucleation::new(rust_schedule, dt).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyManualNucleation { inner })
    }
}

#[pyclass(name = "BubbleFormationSimulator")]
pub struct PyBubbleFormationSimulator {
    inner: BubbleFormationSimulatorWrapper,
}

enum BubbleFormationSimulatorWrapper {
    Poisson(BubbleFormationSimulator<PoissonNucleation>),
    Manual(BubbleFormationSimulator<ManualNucleation>),
}

impl PyBubbleFormationSimulator {
    fn get_inner(&self) -> &BubbleFormationSimulatorWrapper {
        &self.inner
    }

    fn get_inner_mut(&mut self) -> &mut BubbleFormationSimulatorWrapper {
        &mut self.inner
    }
}

#[pymethods]
impl PyBubbleFormationSimulator {
    #[new]
    #[pyo3(signature = (lattice, vw = 0.5, nucleation_strategy = None, seed=None))]
    fn new(
        lattice: &PyLattice,
        vw: f64,
        nucleation_strategy: Option<Py<PyAny>>,
        seed: Option<u64>,
        py: Python,
    ) -> PyResult<Self> {
        let inner = match nucleation_strategy {
            Some(obj) => {
                if let Ok(poisson) = obj.extract::<PyRef<PyPoissonNucleation>>(py) {
                    let simulator = BubbleFormationSimulator::new(
                        lattice.inner.clone(),
                        vw,
                        poisson.inner.clone(),
                        seed,
                    )
                    .map_err(|e| PyValueError::new_err(e))?;
                    BubbleFormationSimulatorWrapper::Poisson(simulator)
                } else if let Ok(manual) = obj.extract::<PyRef<PyManualNucleation>>(py) {
                    let simulator = BubbleFormationSimulator::new(
                        lattice.inner.clone(),
                        vw,
                        manual.inner.clone(),
                        seed,
                    )
                    .map_err(|e| PyValueError::new_err(e))?;
                    BubbleFormationSimulatorWrapper::Manual(simulator)
                } else {
                    return Err(PyValueError::new_err(
                        "nucleation_strategy must be PoissonNucleation or ManualNucleation",
                    ));
                }
            }
            None => {
                let poisson = PoissonNucleation::new(0.1, 1.0, 0.0, 0.1)
                    .map_err(|e| PyValueError::new_err(e))?;
                let simulator =
                    BubbleFormationSimulator::new(lattice.inner.clone(), vw, poisson, seed)
                        .map_err(|e| PyValueError::new_err(e))?;
                BubbleFormationSimulatorWrapper::Poisson(simulator)
            }
        };
        Ok(PyBubbleFormationSimulator { inner })
    }

    fn set_seed(&mut self, seed: u64) {
        match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.set_seed(seed),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.set_seed(seed),
        }
    }

    #[pyo3(signature = (t_max=None, min_volume_remaining_fraction=None, max_time_iterations=None))]
    fn run_simulation(
        &mut self,
        t_max: Option<f64>,
        min_volume_remaining_fraction: Option<f64>,
        max_time_iterations: Option<usize>,
    ) -> PyResult<()> {
        match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => {
                sim.run_simulation(t_max, min_volume_remaining_fraction, max_time_iterations);
            }
            BubbleFormationSimulatorWrapper::Manual(sim) => {
                sim.run_simulation(t_max, min_volume_remaining_fraction, max_time_iterations);
            }
        }
        Ok(())
    }

    #[getter]
    fn end_status(&self, py: Python) -> PyResult<Option<Py<PyDict>>> {
        let status = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => &sim.end_status,
            BubbleFormationSimulatorWrapper::Manual(sim) => &sim.end_status,
        };
        match status {
            Some(status) => {
                let dict = PyDict::new(py);
                match status {
                    SimulationEndStatus::TimeLimitReached {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "TimeLimitReached")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    }
                    SimulationEndStatus::VolumeFractionReached {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "VolumeFractionReached")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    }
                    SimulationEndStatus::MaxTimeIterationsReached {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "MaxTimeIterationsReached")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    }
                    SimulationEndStatus::VolumeDepleted {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "VolumeDepleted")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    }
                }
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    fn get_boundary_intersecting_bubbles(&self, t: f64) -> Vec<([f64; 3], f64)> {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim
                .get_boundary_intersecting_bubbles(t)
                .into_iter()
                .map(|(idx, t)| (sim.get_center(idx), t))
                .collect(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim
                .get_boundary_intersecting_bubbles(t)
                .into_iter()
                .map(|(idx, t)| (sim.get_center(idx), t))
                .collect(),
        }
    }

    fn generate_exterior_bubbles(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let exterior_bubbles = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.generate_exterior_bubbles(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.generate_exterior_bubbles(),
        };
        Ok(PyArray2::from_array(py, &exterior_bubbles).into())
    }

    #[getter]
    fn volume_remaining(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.volume_remaining(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.volume_remaining(),
        }
    }

    fn get_valid_points(&mut self, t: Option<f64>, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let points = match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.get_valid_points(t),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.get_valid_points(t),
        };
        Ok(PyArray2::from_array(py, &points).into())
    }

    fn get_bubbles(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let bubbles_array = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.get_bubbles(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.get_bubbles(),
        };
        Ok(PyArray2::from_array(py, &bubbles_array).into())
    }

    fn get_volume_history(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let volume_history = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.get_volume_history(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.get_volume_history(),
        };
        let history_vec: Vec<f64> = volume_history
            .into_iter()
            .flat_map(|(dt, vol)| vec![dt, vol])
            .collect();
        let history_array = Array2::from_shape_vec((history_vec.len() / 2, 2), history_vec)
            .map_err(|e| PyValueError::new_err(format!("Failed to create history array: {}", e)))?;
        Ok(PyArray2::from_array(py, &history_array).into())
    }

    #[getter]
    fn lattice(&self) -> PyLattice {
        let lattice = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.lattice().clone(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.lattice().clone(),
        };
        PyLattice { inner: lattice }
    }

    #[getter]
    fn vw(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.vw(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.vw(),
        }
    }

    #[getter]
    fn grid(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let grid = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.grid(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.grid(),
        };
        Ok(PyArray2::from_array(py, grid).into())
    }

    #[getter]
    fn volume_total(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.volume_total(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.volume_total(),
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

    #[pyo3(signature = (n_cos_thetax, n_phix))]
    pub fn set_resolution(&mut self, n_cos_thetax: usize, n_phix: usize) -> PyResult<()> {
        self.inner.set_resolution(n_cos_thetax, n_phix);
        Ok(())
    }
}

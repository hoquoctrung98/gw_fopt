use bubble_gw::many_bubbles_legacy::bubble_formation::{
    BubbleFormationSimulator,
    ManualNucleation,
    PoissonNucleation,
    SimulationEndStatus,
};
use bubble_gw::many_bubbles_legacy::lattice::{
    BoundaryConditions,
    Lattice,
    LatticeType,
    generate_bubbles_exterior,
};
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
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
        let lattice_type = match lattice_type.to_lowercase().as_str() {
            "cartesian" => {
                if sizes.len() != 3 {
                    return Err("For cartesian lattice, sizes must have length 3".to_string())
                        .map_err(PyValueError::new_err);
                }
                LatticeType::Cartesian {
                    sizes: [sizes[0], sizes[1], sizes[2]],
                }
            },
            "sphere" => {
                if sizes.len() != 1 {
                    return Err("For sphere lattice, sizes must have length 1".to_string())
                        .map_err(PyValueError::new_err);
                }
                LatticeType::Sphere { radius: sizes[0] }
            },
            _ => return Err("Invalid lattice_type".to_string()).map_err(PyValueError::new_err),
        };
        let inner = Lattice::new(lattice_type, n_grid).map_err(PyValueError::new_err)?;
        Ok(PyLattice { inner })
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn generate_grid(&self, py: Python) -> Py<PyArray2<f64>> {
        let grid = self.inner.generate_grid();
        PyArray2::from_array(py, &grid).into()
    }

    fn lattice_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.lattice_bounds()
    }

    #[getter]
    fn lattice_type(&self) -> String {
        match self.inner.lattice_type {
            LatticeType::Cartesian { .. } => "cartesian".to_string(),
            LatticeType::Sphere { .. } => "sphere".to_string(),
        }
    }

    #[getter]
    fn sizes(&self) -> Vec<f64> {
        match self.inner.lattice_type {
            LatticeType::Cartesian { sizes } => sizes.to_vec(),
            LatticeType::Sphere { radius } => vec![radius],
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n_grid
    }
}

#[pyfunction]
#[pyo3(name = "generate_bubbles_exterior")]
pub fn py_generate_bubbles_exterior(
    py: Python,
    lattice: &PyLattice,
    bubbles_interior: PyReadonlyArray2<f64>,
    boundary_condition: &str, // "periodic" or "reflective"
) -> PyResult<Py<PyArray2<f64>>> {
    // Parse boundary condition
    let bc = match boundary_condition.to_lowercase().as_str() {
        "periodic" => BoundaryConditions::Periodic,
        "reflective" => BoundaryConditions::Reflection,
        _ => {
            return Err(PyValueError::new_err(
                "boundary_condition must be 'periodic' or 'reflective'",
            ));
        },
    };

    // Validate bubbles
    let bubbles_array = bubbles_interior.to_owned_array();
    if bubbles_array.shape().get(1) != Some(&4) {
        return Err(PyValueError::new_err("bubbles_interior must have shape (N, 4)"));
    }

    // Call the new function
    let result = generate_bubbles_exterior(&lattice.inner, &bubbles_array, bc);

    Ok(PyArray2::from_array(py, &result).into())
}

#[pyclass(name = "PoissonNucleation")]
pub struct PyPoissonNucleation {
    inner: PoissonNucleation,
}

#[pymethods]
impl PyPoissonNucleation {
    #[new]
    fn new(gamma0: f64, beta: f64, t0: f64, dp0: f64) -> PyResult<Self> {
        let inner = PoissonNucleation::new(gamma0, beta, t0, dp0).map_err(PyValueError::new_err)?;
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
        let inner = ManualNucleation::new(rust_schedule, dt).map_err(PyValueError::new_err)?;
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
    #[pyo3(signature = (lattice, vw = 1.0, nucleation_strategy = None, seed=None))]
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
                    .map_err(PyValueError::new_err)?;
                    BubbleFormationSimulatorWrapper::Poisson(simulator)
                } else if let Ok(manual) = obj.extract::<PyRef<PyManualNucleation>>(py) {
                    let simulator = BubbleFormationSimulator::new(
                        lattice.inner.clone(),
                        vw,
                        manual.inner.clone(),
                        seed,
                    )
                    .map_err(PyValueError::new_err)?;
                    BubbleFormationSimulatorWrapper::Manual(simulator)
                } else {
                    return Err(PyValueError::new_err(
                        "nucleation_strategy must be PoissonNucleation or ManualNucleation",
                    ));
                }
            },
            None => {
                let poisson =
                    PoissonNucleation::new(0.1, 1.0, 0.0, 0.1).map_err(PyValueError::new_err)?;
                let simulator =
                    BubbleFormationSimulator::new(lattice.inner.clone(), vw, poisson, seed)
                        .map_err(PyValueError::new_err)?;
                BubbleFormationSimulatorWrapper::Poisson(simulator)
            },
        };
        Ok(PyBubbleFormationSimulator { inner })
    }

    fn set_seed(&mut self, seed: u64) {
        match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.set_seed(seed),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.set_seed(seed),
        }
    }

    #[pyo3(signature = (t_end=None, min_volume_remaining_fraction=None, max_time_iterations=None))]
    fn run_simulation(
        &mut self,
        t_end: Option<f64>,
        min_volume_remaining_fraction: Option<f64>,
        max_time_iterations: Option<usize>,
    ) {
        match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => {
                sim.run_simulation(t_end, min_volume_remaining_fraction, max_time_iterations);
            },
            BubbleFormationSimulatorWrapper::Manual(sim) => {
                sim.run_simulation(t_end, min_volume_remaining_fraction, max_time_iterations);
            },
        }
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
                    },
                    SimulationEndStatus::VolumeFractionReached {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "VolumeFractionReached")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    },
                    SimulationEndStatus::MaxTimeIterationsReached {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "MaxTimeIterationsReached")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    },
                    SimulationEndStatus::VolumeDepleted {
                        t_end,
                        volume_remaining_fraction,
                        time_iteration,
                    } => {
                        dict.set_item("status", "VolumeDepleted")?;
                        dict.set_item("t_end", t_end)?;
                        dict.set_item("volume_remaining_fraction", volume_remaining_fraction)?;
                        dict.set_item("time_iteration", time_iteration)?;
                    },
                }
                Ok(Some(dict.into()))
            },
            None => Ok(None),
        }
    }

    fn boundary_intersecting_bubbles(&self, t: f64) -> Vec<([f64; 3], f64)> {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim
                .boundary_intersecting_bubbles(t)
                .into_iter()
                .map(|(idx, t)| (sim.centers(idx), t))
                .collect(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim
                .boundary_intersecting_bubbles(t)
                .into_iter()
                .map(|(idx, t)| (sim.centers(idx), t))
                .collect(),
        }
    }

    fn bubbles_interior(&self, py: Python) -> Py<PyArray2<f64>> {
        let bubbles_array = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.bubbles_interior(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.bubbles_interior(),
        };
        PyArray2::from_array(py, &bubbles_array).into()
    }

    #[getter]
    fn volume_remaining(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.volume_remaining(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.volume_remaining(),
        }
    }

    fn get_valid_points(&mut self, t: f64, py: Python) -> Py<PyArray2<f64>> {
        let points = match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.get_valid_points(t),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.get_valid_points(t),
        };
        PyArray2::from_array(py, &points).into()
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
            .map_err(|e| PyValueError::new_err(format!("Failed to create history array: {e}")))?;
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
    fn grid(&self, py: Python) -> Py<PyArray2<f64>> {
        let grid = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.grid(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.grid(),
        };
        PyArray2::from_array(py, grid).into()
    }

    #[getter]
    fn volume_total(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.volume_total(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.volume_total(),
        }
    }
}

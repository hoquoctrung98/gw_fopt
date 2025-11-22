use bubble_gw_rs::many_bubbles::bubble_formation::{
    BoundaryConditions, BubbleFormationSimulator, Lattice, LatticeType, ManualNucleation,
    PoissonNucleation, SimulationEndStatus, generate_bubbles_exterior,
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
        let inner =
            Lattice::new(lattice_type, sizes, n_grid).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyLattice { inner })
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn generate_grid(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let grid = self.inner.generate_grid();
        Ok(PyArray2::from_array(py, &grid).into())
    }

    fn lattice_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.lattice_bounds()
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

// // PyO3 binding for generate_bubbles_exterior function
// #[pyfunction]
// #[pyo3(name = "generate_bubbles_exterior")]
// pub fn py_generate_bubbles_exterior(
//     py: Python,
//     lattice_sizes: Bound<'_, PyAny>,
//     bubbles_interior: PyReadonlyArray2<f64>,
// ) -> PyResult<Py<PyArray2<f64>>> {
//     // Convert lattice_sizes to [f64; 3]
//     let sizes: [f64; 3] = {
//         let vec: Vec<f64> = if let Ok(list) = lattice_sizes.cast::<PyList>() {
//             list.iter()
//                 .map(|item: Bound<PyAny>| item.extract::<f64>())
//                 .collect::<PyResult<Vec<f64>>>()
//         } else if let Ok(tuple) = lattice_sizes.cast::<PyTuple>() {
//             tuple
//                 .iter()
//                 .map(|item: Bound<PyAny>| item.extract::<f64>())
//                 .collect::<PyResult<Vec<f64>>>()
//         } else {
//             return Err(PyValueError::new_err(
//                 "lattice_sizes must be a list or tuple",
//             ));
//         }?;
//
//         if vec.len() != 3 {
//             return Err(PyValueError::new_err(
//                 "lattice_sizes must contain exactly 3 elements",
//             ));
//         }
//         [vec[0], vec[1], vec[2]]
//     };
//
//     // Validate and convert bubbles_interior to Array2<f64>
//     let bubbles_array = bubbles_interior.to_owned_array();
//     if bubbles_array.shape()[1] != 4 {
//         return Err(PyValueError::new_err(
//             "bubbles_interior must have shape (N, 4)",
//         ));
//     }
//
//     // Call the Rust function
//     let result = generate_bubbles_exterior(sizes, bubbles_array);
//
//     // Convert the result to a NumPy array
//     Ok(PyArray2::from_array(py, &result).into())
// }

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
        "reflective" => BoundaryConditions::Reflective,
        _ => {
            return Err(PyValueError::new_err(
                "boundary_condition must be 'periodic' or 'reflective'",
            ));
        }
    };

    // Validate bubbles
    let bubbles_array = bubbles_interior.to_owned_array();
    if bubbles_array.shape().get(1) != Some(&4) {
        return Err(PyValueError::new_err("bubbles_interior must have shape (N, 4)"));
    }

    // Call the new function
    let result = generate_bubbles_exterior(&lattice.inner, bubbles_array, bc);

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

    #[pyo3(signature = (t_end=None, min_volume_remaining_fraction=None, max_time_iterations=None))]
    fn run_simulation(
        &mut self,
        t_end: Option<f64>,
        min_volume_remaining_fraction: Option<f64>,
        max_time_iterations: Option<usize>,
    ) -> PyResult<()> {
        match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => {
                sim.run_simulation(t_end, min_volume_remaining_fraction, max_time_iterations);
            }
            BubbleFormationSimulatorWrapper::Manual(sim) => {
                sim.run_simulation(t_end, min_volume_remaining_fraction, max_time_iterations);
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

    fn bubbles_interior(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let bubbles_array = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.bubbles_interior(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.bubbles_interior(),
        };
        Ok(PyArray2::from_array(py, &bubbles_array).into())
    }

    #[pyo3(name = "bubbles_exterior")]
    fn py_bubbles_exterior(
        &self,
        py: Python,
        boundary_condition: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let bc = match boundary_condition.to_lowercase().as_str() {
            "periodic" => BoundaryConditions::Periodic,
            "reflective" => BoundaryConditions::Reflective,
            _ => {
                return Err(PyValueError::new_err(
                    "boundary_condition must be 'periodic' or 'reflective'",
                ));
            }
        };

        let exterior = match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.bubbles_exterior(bc),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.bubbles_exterior(bc),
        };

        Ok(PyArray2::from_array(py, &exterior).into())
    }

    #[getter]
    fn volume_remaining(&self) -> f64 {
        match self.get_inner() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.volume_remaining(),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.volume_remaining(),
        }
    }

    fn get_valid_points(&mut self, t: f64, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let points = match self.get_inner_mut() {
            BubbleFormationSimulatorWrapper::Poisson(sim) => sim.get_valid_points(t),
            BubbleFormationSimulatorWrapper::Manual(sim) => sim.get_valid_points(t),
        };
        Ok(PyArray2::from_array(py, &points).into())
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

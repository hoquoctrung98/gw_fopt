use bubble_gw::many_bubbles::bubbles::Bubbles;
use bubble_gw::many_bubbles::lattice::{
    BoundaryConditions,
    BuiltInLattice,
    LatticeGeometry,
    TransformationIsometry3,
};
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use crate::py_many_bubbles::py_isometry::PyIsometry3;
use crate::py_many_bubbles::py_lattice::{PyCartesian, PyEmpty, PyParallelepiped, PySpherical};

#[pyclass(name = "LatticeBubbles")]
#[derive(Clone)]
pub struct PyLatticeBubbles {
    pub(crate) inner: LatticeBubbles<BuiltInLattice>,
}

impl PyLatticeBubbles {
    fn from_builtin(
        lattice: BuiltInLattice,
        interior: Array2<f64>,
        exterior: Option<Array2<f64>>,
    ) -> PyResult<Self> {
        // Convert Array2 (N×4) → Vec<Vector4>
        let lb = LatticeBubbles::new(interior, exterior, lattice)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        Ok(Self { inner: lb })
    }
}

// TODO: Add method `new` to create an empty LatticeBubbles
#[pymethods]
impl PyLatticeBubbles {
    #[new]
    #[pyo3(signature = (lattice, bubbles_interior, bubbles_exterior = None))]
    fn new(
        lattice: &Bound<'_, PyAny>,
        bubbles_interior: PyReadonlyArray2<f64>,
        bubbles_exterior: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        // Extract builtin lattice from built-in Python object
        let builtin: BuiltInLattice = if let Ok(l) = lattice.extract::<PyParallelepiped>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PyCartesian>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PySpherical>() {
            l.builtin
        } else if let Ok(l) = lattice.extract::<PyEmpty>() {
            l.builtin
        } else {
            return Err(PyValueError::new_err(
                "Expected a lattice instance: ParallelepipedLattice, CartesianLattice, SphericalLattice, or EmptyLattice",
            ));
        };

        let interior = bubbles_interior.to_owned_array();
        let exterior = bubbles_exterior.map(|a| a.to_owned_array());

        Self::from_builtin(builtin, interior, exterior)
    }

    /// Sorts interior and exterior bubbles in-place by nucleation time `t`.
    fn sort_by_time(&mut self) {
        self.inner.sort_by_time();
    }

    fn __repr__(&self) -> String {
        format!(
            "LatticeBubbles(lattice_type={}, n_interior={}, n_exterior={})",
            match &self.inner.lattice {
                BuiltInLattice::Parallelepiped(_) => "Parallelepiped",
                BuiltInLattice::Cartesian(_) => "Cartesian",
                BuiltInLattice::Spherical(_) => "Spherical",
                BuiltInLattice::Empty(_) => "Empty",
            },
            self.inner.interior.spacetime.len(),
            self.inner.exterior.spacetime.len()
        )
    }

    #[getter]
    fn lattice(&self, py: Python) -> PyResult<Py<PyAny>> {
        let py_obj: Py<PyAny> = match &self.inner.lattice {
            BuiltInLattice::Parallelepiped(l) => {
                Py::new(py, PyParallelepiped::from_concrete(l.clone()))?.into()
            },
            BuiltInLattice::Cartesian(l) => {
                Py::new(py, PyCartesian::from_concrete(l.clone()))?.into()
            },
            BuiltInLattice::Spherical(l) => {
                Py::new(py, PySpherical::from_concrete(l.clone()))?.into()
            },
            BuiltInLattice::Empty(l) => Py::new(py, PyEmpty::from_concrete(l.clone()))?.into(),
        };
        Ok(py_obj)
    }

    #[getter]
    fn bubbles_interior(&self, py: Python) -> Py<PyArray2<f64>> {
        let arr = bubbles_to_array2(&self.inner.interior);
        PyArray2::from_array(py, &arr).into()
    }

    #[getter]
    fn bubbles_exterior(&self, py: Python) -> Py<PyArray2<f64>> {
        let arr = bubbles_to_array2(&self.inner.exterior);
        PyArray2::from_array(py, &arr).into()
    }

    #[getter]
    fn lattice_type(&self) -> &'static str {
        match &self.inner.lattice {
            BuiltInLattice::Parallelepiped(_) => "Parallelepiped",
            BuiltInLattice::Cartesian(_) => "Cartesian",
            BuiltInLattice::Spherical(_) => "Spherical",
            BuiltInLattice::Empty(_) => "Empty",
        }
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.inner.lattice.volume()
    }

    #[getter]
    fn reference_point(&self) -> [f64; 3] {
        let p = self.inner.lattice.reference_point();
        [p.x, p.y, p.z]
    }

    fn parameters(&self) -> Vec<f64> {
        self.inner.lattice.parameters()
    }

    fn contains(&self, points: Vec<[f64; 3]>) -> Vec<bool> {
        let points: Vec<nalgebra::Point3<f64>> =
            points.into_iter().map(nalgebra::Point3::from).collect();
        self.inner.lattice.contains(&points)
    }

    #[pyo3(name = "transform")]
    fn py_transform(&self, iso: &PyIsometry3) -> Self {
        let new_lb = self.inner.transform(iso);
        Self { inner: new_lb }
    }

    #[pyo3(signature = (boundary_condition = "periodic"))]
    fn with_boundary_condition(&mut self, boundary_condition: &str) -> PyResult<()> {
        let bc = match boundary_condition.to_lowercase().as_str() {
            "periodic" => BoundaryConditions::Periodic,
            "reflection" => BoundaryConditions::Reflection,
            "none" => BoundaryConditions::None,
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid boundary condition. Expected 'periodic' or 'reflection'.",
                ));
            },
        };

        self.inner.with_boundary_condition(bc);
        Ok(())
    }
}

fn bubbles_to_array2(bubbles: &Bubbles) -> Array2<f64> {
    let n = bubbles.spacetime.len();
    let mut data = Vec::with_capacity(n * 4);
    for v in &bubbles.spacetime {
        data.push(v[0]);
        data.push(v[1]);
        data.push(v[2]);
        data.push(v[3]);
    }
    Array2::from_shape_vec((n, 4), data).unwrap()
}

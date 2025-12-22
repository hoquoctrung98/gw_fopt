use crate::py_many_bubbles_nalgebra::py_lattice::PyConcreteLattice;
use bubble_gw::many_bubbles_nalgebra::lattice::{ConcreteLattice, LatticeGeometry};
use bubble_gw::many_bubbles_nalgebra::lattice_bubbles::LatticeBubbles;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass(name = "LatticeBubbles")]
#[derive(Clone)]
pub struct PyLatticeBubbles {
    pub inner: LatticeBubbles<ConcreteLattice>,
}

// NOTE: Need to add methods to perform Isometry3 on both lattice and bubbles
#[pymethods]
impl PyLatticeBubbles {
    #[new]
    fn new(
        bubbles_interior: PyReadonlyArray2<f64>,
        bubbles_exterior: PyReadonlyArray2<f64>,
        lattice: PyConcreteLattice,
        sort_by_time: bool,
    ) -> PyResult<Self> {
        let inner = LatticeBubbles::new(
            bubbles_interior.to_owned_array(),
            bubbles_exterior.to_owned_array(),
            lattice.inner,
            sort_by_time,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn bubbles_interior(&self, py: Python) -> Py<PyArray2<f64>> {
        let interior = self.inner.interior.to_array2();
        PyArray2::from_array(py, &interior).into()
    }

    #[getter]
    pub fn bubbles_exterior(&self, py: Python) -> Py<PyArray2<f64>> {
        let exterior = self.inner.exterior.to_array2();
        PyArray2::from_array(py, &exterior).into()
    }

    #[getter]
    fn lattice_type(&self) -> &'static str {
        match &self.inner.lattice {
            ConcreteLattice::Parallelepiped(_) => "parallelepiped",
            ConcreteLattice::Cartesian(_) => "cartesian",
            ConcreteLattice::Sphere(_) => "sphere",
            ConcreteLattice::Empty(_) => "empty",
        }
    }

    fn volume(&self) -> f64 {
        self.inner.lattice.volume()
    }

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
        let result = self.inner.lattice.contains(&points);
        result
    }
}

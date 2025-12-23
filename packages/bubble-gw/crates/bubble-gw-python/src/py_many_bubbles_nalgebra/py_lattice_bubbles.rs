use crate::py_many_bubbles_nalgebra::py_lattice::PyBuiltInLattice;
use bubble_gw::many_bubbles_nalgebra::lattice::{BuiltInLattice, LatticeGeometry};
use bubble_gw::many_bubbles_nalgebra::lattice_bubbles::LatticeBubbles;
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass(name = "LatticeBubbles")]
#[derive(Clone)]
pub struct PyLatticeBubbles {
    pub inner: LatticeBubbles<BuiltInLattice>,
}

// NOTE: Need to add methods to perform Isometry3 on both lattice and bubbles
// TODO: create PyLatticeBubblesError instead of passing to PyErr::new
#[pymethods]
impl PyLatticeBubbles {
    #[new]
    fn new(
        bubbles_interior: PyReadonlyArray2<f64>,
        bubbles_exterior: Option<PyReadonlyArray2<f64>>,
        lattice: PyBuiltInLattice,
        sort_by_time: bool,
    ) -> PyResult<Self> {
        // if bubbles_exterior is None, assuming no exterior bubbles are given
        let bubbles_exterior = bubbles_exterior
            .map(|arr| arr.to_owned_array())
            .unwrap_or_else(|| Array2::zeros((0, 4)));
        let bubbles_interior = bubbles_interior.to_owned_array();
        let inner =
            LatticeBubbles::new(bubbles_interior, bubbles_exterior, lattice.inner, sort_by_time)
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
            BuiltInLattice::Parallelepiped(_) => "parallelepiped",
            BuiltInLattice::Cartesian(_) => "cartesian",
            BuiltInLattice::Sphere(_) => "sphere",
            BuiltInLattice::Empty(_) => "empty",
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

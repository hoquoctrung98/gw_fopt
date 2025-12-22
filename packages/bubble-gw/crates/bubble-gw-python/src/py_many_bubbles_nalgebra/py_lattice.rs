use bubble_gw::many_bubbles_nalgebra::lattice::{
    CartesianLattice, ConcreteLattice, EmptyLattice, LatticeGeometry, ParallelepipedLattice,
    SphericalLattice, TransformationIsometry3,
};
use pyo3::prelude::*;

#[pyclass(name = "Lattice")]
#[derive(Clone)]
pub struct PyConcreteLattice {
    pub inner: ConcreteLattice,
}

impl TransformationIsometry3 for PyConcreteLattice {
    fn transform(&self, iso: &nalgebra::Isometry3<f64>) -> Self {
        Self {
            inner: self.inner.transform(iso),
        }
    }

    fn transform_mut(&mut self, iso: &nalgebra::Isometry3<f64>) {
        self.inner.transform_mut(iso);
    }
}

#[pymethods]
impl PyConcreteLattice {
    #[staticmethod]
    fn parallelepiped(origin: [f64; 3], edges: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = nalgebra::Point3::from(origin);
        let edges = [
            nalgebra::Vector3::from(edges[0]),
            nalgebra::Vector3::from(edges[1]),
            nalgebra::Vector3::from(edges[2]),
        ];
        let lattice = ParallelepipedLattice::new(origin, edges);
        Ok(Self {
            inner: ConcreteLattice::Parallelepiped(lattice),
        })
    }

    #[staticmethod]
    fn cartesian(origin: [f64; 3], edges: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = nalgebra::Point3::from(origin);
        let edges = [
            nalgebra::Vector3::from(edges[0]),
            nalgebra::Vector3::from(edges[1]),
            nalgebra::Vector3::from(edges[2]),
        ];
        let lattice = CartesianLattice::new(origin, edges);
        Ok(Self {
            inner: ConcreteLattice::Cartesian(lattice),
        })
    }

    #[staticmethod]
    fn sphere(center: [f64; 3], radius: f64) -> PyResult<Self> {
        let center = nalgebra::Point3::from(center);
        let lattice = SphericalLattice::new(center, radius);
        Ok(Self {
            inner: ConcreteLattice::Sphere(lattice),
        })
    }

    #[staticmethod]
    fn empty() -> Self {
        Self {
            inner: ConcreteLattice::Empty(EmptyLattice {}),
        }
    }

    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume()
    }
}

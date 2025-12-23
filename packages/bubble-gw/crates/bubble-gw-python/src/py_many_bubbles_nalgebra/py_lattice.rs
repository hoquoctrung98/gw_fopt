use bubble_gw::many_bubbles_nalgebra::lattice::{
    BuiltInLattice, CartesianLattice, EmptyLattice, LatticeError, LatticeGeometry,
    ParallelepipedLattice, SphericalLattice, TransformationIsometry3,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PyLatticeError {
    #[error("edges are linearly dependent (zero volume)")]
    DegenerateEdges,

    #[error("edges are not pairwise orthogonal")]
    NonOrthogonalEdges,
}

impl From<LatticeError> for PyLatticeError {
    fn from(err: LatticeError) -> Self {
        match err {
            LatticeError::DegenerateEdges => PyLatticeError::DegenerateEdges,
            LatticeError::NonOrthogonalEdges => PyLatticeError::NonOrthogonalEdges,
        }
    }
}

impl From<PyLatticeError> for PyErr {
    fn from(err: PyLatticeError) -> Self {
        match err {
            PyLatticeError::DegenerateEdges { .. } => PyValueError::new_err(err.to_string()),
            PyLatticeError::NonOrthogonalEdges { .. } => PyValueError::new_err(err.to_string()),
        }
    }
}

type PyResult<T> = Result<T, PyLatticeError>;

#[pyclass(name = "Lattice")]
#[derive(Clone)]
pub struct PyBuiltInLattice {
    pub inner: BuiltInLattice,
}

impl TransformationIsometry3 for PyBuiltInLattice {
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
impl PyBuiltInLattice {
    #[staticmethod]
    fn parallelepiped(origin: [f64; 3], edges: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = nalgebra::Point3::from(origin);
        let edges = [
            nalgebra::Vector3::from(edges[0]),
            nalgebra::Vector3::from(edges[1]),
            nalgebra::Vector3::from(edges[2]),
        ];
        let lattice = ParallelepipedLattice::try_new(origin, edges)?;
        Ok(Self {
            inner: BuiltInLattice::Parallelepiped(lattice),
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
        let lattice = CartesianLattice::try_new(origin, edges)?;
        Ok(Self {
            inner: BuiltInLattice::Cartesian(lattice),
        })
    }

    #[staticmethod]
    fn sphere(center: [f64; 3], radius: f64) -> PyResult<Self> {
        let center = nalgebra::Point3::from(center);
        let lattice = SphericalLattice::new(center, radius);
        Ok(Self {
            inner: BuiltInLattice::Sphere(lattice),
        })
    }

    #[staticmethod]
    fn empty() -> Self {
        Self {
            inner: BuiltInLattice::Empty(EmptyLattice {}),
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

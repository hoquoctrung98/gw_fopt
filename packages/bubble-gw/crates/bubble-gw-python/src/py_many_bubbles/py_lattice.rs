use bubble_gw::many_bubbles::lattice::{
    BuiltInLattice, CartesianLattice, EmptyLattice, LatticeError, LatticeGeometry,
    ParallelepipedLattice, SphericalLattice, TransformationIsometry3,
};
use nalgebra::Isometry3;
use pyo3::{exceptions::PyValueError, prelude::*};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PyLatticeError {
    #[error("Basis vectors are linearly dependent (zero volume)")]
    DegenerateBasis,

    #[error("Basis vectors are not pairwise orthogonal")]
    NonOrthogonalBasis,
}

impl From<LatticeError> for PyLatticeError {
    fn from(err: LatticeError) -> Self {
        match err {
            LatticeError::DegenerateBasis => PyLatticeError::DegenerateBasis,
            LatticeError::NonOrthogonalBasis => PyLatticeError::NonOrthogonalBasis,
        }
    }
}

impl From<PyLatticeError> for PyErr {
    fn from(err: PyLatticeError) -> Self {
        match err {
            PyLatticeError::DegenerateBasis { .. } => PyValueError::new_err(err.to_string()),
            PyLatticeError::NonOrthogonalBasis { .. } => PyValueError::new_err(err.to_string()),
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
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        Self {
            inner: self.inner.transform(iso),
        }
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        self.inner.transform_mut(iso);
    }
}

#[pymethods]
impl PyBuiltInLattice {
    #[staticmethod]
    #[pyo3(name = "Parallelepiped")]
    fn parallelepiped(origin: [f64; 3], basis: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = nalgebra::Point3::from(origin);
        let basis = [
            nalgebra::Vector3::from(basis[0]),
            nalgebra::Vector3::from(basis[1]),
            nalgebra::Vector3::from(basis[2]),
        ];
        let lattice = ParallelepipedLattice::try_new(origin, basis)?;
        Ok(Self {
            inner: BuiltInLattice::Parallelepiped(lattice),
        })
    }

    #[staticmethod]
    #[pyo3(name = "Cartesian")]
    fn cartesian(origin: [f64; 3], basis: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = nalgebra::Point3::from(origin);
        let basis = [
            nalgebra::Vector3::from(basis[0]),
            nalgebra::Vector3::from(basis[1]),
            nalgebra::Vector3::from(basis[2]),
        ];
        let lattice = CartesianLattice::try_new(origin, basis)?;
        Ok(Self {
            inner: BuiltInLattice::Cartesian(lattice),
        })
    }

    #[staticmethod]
    #[pyo3(name = "Spherical")]
    fn spherical(center: [f64; 3], radius: f64) -> PyResult<Self> {
        let center = nalgebra::Point3::from(center);
        let lattice = SphericalLattice::new(center, radius);
        Ok(Self {
            inner: BuiltInLattice::Spherical(lattice),
        })
    }

    #[staticmethod]
    #[pyo3(name = "Empty")]
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

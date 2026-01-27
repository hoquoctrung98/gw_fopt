use bubble_gw::many_bubbles::lattice::{
    BuiltInLattice,
    CartesianLattice,
    EmptyLattice,
    LatticeError,
    LatticeGeometry,
    ParallelepipedLattice,
    SphericalLattice,
    TransformationIsometry3,
};
use nalgebra::{Point3, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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
        PyValueError::new_err(err.to_string())
    }
}

pub type PyResult<T> = Result<T, PyLatticeError>;

// === Concrete Python lattice wrappers ===
// Each stores: concrete type (for methods) + BuiltInLattice (for interop)

#[pyclass(name = "ParallelepipedLattice")]
#[derive(Clone)]
pub struct PyParallelepiped {
    pub(crate) concrete: ParallelepipedLattice,
    pub(crate) builtin: BuiltInLattice,
}

#[pyclass(name = "CartesianLattice")]
#[derive(Clone)]
pub struct PyCartesian {
    pub(crate) concrete: CartesianLattice,
    pub(crate) builtin: BuiltInLattice,
}

#[pyclass(name = "SphericalLattice")]
#[derive(Clone)]
pub struct PySpherical {
    pub(crate) concrete: SphericalLattice,
    pub(crate) builtin: BuiltInLattice,
}

#[pyclass(name = "EmptyLattice")]
#[derive(Clone)]
pub struct PyEmpty {
    pub(crate) concrete: EmptyLattice,
    pub(crate) builtin: BuiltInLattice,
}

// === Helper: construct Py* from concrete lattice ===

impl PyParallelepiped {
    pub(crate) fn from_concrete(l: ParallelepipedLattice) -> Self {
        let builtin = BuiltInLattice::Parallelepiped(l.clone());
        Self {
            concrete: l,
            builtin,
        }
    }
}

impl PyCartesian {
    pub(crate) fn from_concrete(l: CartesianLattice) -> Self {
        let builtin = BuiltInLattice::Cartesian(l.clone());
        Self {
            concrete: l,
            builtin,
        }
    }
}

impl PySpherical {
    pub(crate) fn from_concrete(l: SphericalLattice) -> Self {
        let builtin = BuiltInLattice::Spherical(l.clone());
        Self {
            concrete: l,
            builtin,
        }
    }
}

impl PyEmpty {
    pub(crate) fn from_concrete(l: EmptyLattice) -> Self {
        let builtin = BuiltInLattice::Empty(l.clone());
        Self {
            concrete: l,
            builtin,
        }
    }
}

// === Implementations ===

#[pymethods]
impl PyParallelepiped {
    #[new]
    #[pyo3(signature = (origin, basis))]
    fn py_new(origin: [f64; 3], basis: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = Point3::from(origin);
        let basis = [
            Vector3::from(basis[0]),
            Vector3::from(basis[1]),
            Vector3::from(basis[2]),
        ];
        let concrete = ParallelepipedLattice::try_new(origin, basis)?;
        Ok(Self::from_concrete(concrete))
    }

    fn name(&self) -> String {
        "ParallelepipedLattice".to_string()
    }

    #[staticmethod]
    fn axis_aligned(origin: [f64; 3], lx: f64, ly: f64, lz: f64) -> Self {
        let concrete = ParallelepipedLattice::axis_aligned(Point3::from(origin), lx, ly, lz);
        Self::from_concrete(concrete)
    }

    #[staticmethod]
    fn cube_centered(center: [f64; 3], side: f64) -> Self {
        let concrete = ParallelepipedLattice::cube_centered(Point3::from(center), side);
        Self::from_concrete(concrete)
    }

    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.concrete.origin.coords.into()
    }

    #[getter]
    fn basis(&self) -> [[f64; 3]; 3] {
        self.concrete.basis.map(|v| v.into())
    }

    #[getter]
    fn normalized_basis(&self) -> [[f64; 3]; 3] {
        self.concrete.normalized_basis().map(|v| v.into())
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.concrete.volume()
    }

    fn transform(&self, iso: &crate::py_many_bubbles::py_isometry::PyIsometry3) -> Self {
        let concrete = self.concrete.transform(iso);
        Self::from_concrete(concrete)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.concrete)
    }
}

#[pymethods]
impl PyCartesian {
    #[new]
    #[pyo3(signature = (origin, basis))]
    fn py_new(origin: [f64; 3], basis: [[f64; 3]; 3]) -> PyResult<Self> {
        let origin = Point3::from(origin);
        let basis = [
            Vector3::from(basis[0]),
            Vector3::from(basis[1]),
            Vector3::from(basis[2]),
        ];
        let concrete = CartesianLattice::try_new(origin, basis)?;
        Ok(Self::from_concrete(concrete))
    }

    fn name(&self) -> String {
        "CartesianLattice".to_string()
    }

    #[staticmethod]
    fn with_origin_and_sizes(origin: [f64; 3], sizes: [f64; 3]) -> Self {
        let concrete = CartesianLattice::with_origin_and_sizes(Point3::from(origin), sizes);
        Self::from_concrete(concrete)
    }

    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.concrete.0.origin.coords.into()
    }

    #[getter]
    fn basis(&self) -> [[f64; 3]; 3] {
        self.concrete.0.basis.map(|v| v.into())
    }

    #[getter]
    fn side_lengths(&self) -> [f64; 3] {
        self.concrete.side_lengths()
    }

    #[getter]
    fn normalized_basis(&self) -> [[f64; 3]; 3] {
        self.concrete.normalized_basis().map(|v| v.into())
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.concrete.volume()
    }

    fn transform(&self, iso: &crate::py_many_bubbles::py_isometry::PyIsometry3) -> Self {
        let concrete = self.concrete.transform(iso);
        Self::from_concrete(concrete)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.concrete)
    }
}

#[pymethods]
impl PySpherical {
    #[new]
    fn py_new(center: [f64; 3], radius: f64) -> Self {
        let concrete = SphericalLattice::new(Point3::from(center), radius);
        Self::from_concrete(concrete)
    }

    fn name(&self) -> String {
        "SphericalLattice".to_string()
    }

    #[getter]
    fn center(&self) -> [f64; 3] {
        self.concrete.center.coords.into()
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.concrete.radius
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.concrete.volume()
    }

    fn transform(&self, iso: &crate::py_many_bubbles::py_isometry::PyIsometry3) -> Self {
        let concrete = self.concrete.transform(iso);
        Self::from_concrete(concrete)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.concrete)
    }
}

#[pymethods]
impl PyEmpty {
    #[new]
    fn py_new() -> Self {
        let concrete = EmptyLattice {};
        Self::from_concrete(concrete)
    }

    fn name(&self) -> String {
        "EmptyLattice".to_string()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.concrete.volume()
    }

    fn transform(&self, _iso: &crate::py_many_bubbles::py_isometry::PyIsometry3) -> Self {
        // Empty is invariant
        Self::from_concrete(self.concrete.clone())
    }

    fn __repr__(&self) -> String {
        "EmptyLattice()".to_string()
    }
}

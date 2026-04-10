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

#[pyclass(from_py_object, name = "LatticeBubbles")]
#[derive(Clone, Debug)]
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

    // =========================================================================
    // Explicit Time Translation Bindings
    // =========================================================================

    /// Shifts the nucleation time of all bubbles by a specified amount.
    ///
    /// This method modifies the object in-place.
    ///
    /// Parameters
    /// ----------
    /// t_shift : float
    ///     Amount to add to all bubble times (can be negative).
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Notes
    /// -----
    /// Spacetime intervals are invariant under uniform time translation,
    /// so precomputed collision data remains valid.
    ///
    /// Examples
    /// --------
    /// >>> # Shift forward by 10 time units
    /// >>> lb.translate_time_mut(10.0)
    ///
    /// >>> # Shift backward by 2.5 time units
    /// >>> lb.translate_time_mut(-2.5)
    fn translate_time_mut(&mut self, t_shift: f64) -> PyResult<()> {
        self.inner.translate_time_mut(t_shift);
        Ok(())
    }

    /// Returns a new LatticeBubbles with explicit time translation applied.
    ///
    /// Functional (non-mutating) variant: original object unchanged.
    ///
    /// Parameters
    /// ----------
    /// t_shift : float
    ///     Amount to add to all bubble times.
    ///
    /// Returns
    /// -------
    /// LatticeBubbles
    ///     New instance with translated bubble times.
    ///
    /// Examples
    /// --------
    /// >>> # Get shifted copy without modifying original
    /// >>> lb_shifted = lb.translate_time(5.0)
    fn translate_time(&self, t_shift: f64) -> Self {
        let new_inner = self.inner.translate_time(t_shift);
        Self { inner: new_inner }
    }

    // =========================================================================
    // Auto-Normalization Bindings
    // =========================================================================

    /// Normalizes bubble times so the earliest nucleation occurs at t=0.
    ///
    /// This method modifies the object in-place. Computes the minimum time
    /// across all interior and exterior bubbles, then shifts all times by
    /// `-min_time`.
    ///
    /// Parameters
    /// ----------
    /// None
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Notes
    /// -----
    /// - After normalization: `min(bubbles_interior[:, 0]) == 0.0`
    /// - If no bubbles exist, this is a no-op
    /// - Precomputed spacetime intervals remain valid
    ///
    /// Examples
    /// --------
    /// >>> # Normalize so earliest bubble nucleates at t=0
    /// >>> lb.normalize_time_mut()
    /// >>> assert abs(lb.bubbles_interior[:, 0].min()) < 1e-10
    fn normalize_time_mut(&mut self) -> PyResult<()> {
        self.inner.normalize_time_mut();
        Ok(())
    }

    /// Returns a new LatticeBubbles with time normalization applied.
    ///
    /// Functional variant: returns copy with earliest bubble at t=0.
    ///
    /// Parameters
    /// ----------
    /// None
    ///
    /// Returns
    /// -------
    /// LatticeBubbles
    ///     New instance with normalized bubble times.
    ///
    /// Examples
    /// --------
    /// >>> # Get normalized copy without modifying original
    /// >>> lb_normalized = lb.normalize_time()
    /// >>> # Original lb is unchanged
    fn normalize_time(&self) -> Self {
        let new_inner = self.inner.normalize_time();
        Self { inner: new_inner }
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

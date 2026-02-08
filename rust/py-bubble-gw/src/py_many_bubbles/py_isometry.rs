use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3};
use pyo3::prelude::*;

#[pyclass(from_py_object, name = "Isometry3")]
#[derive(Clone, Debug)]
pub struct PyIsometry3 {
    pub(crate) inner: Isometry3<f64>,
}

impl From<Isometry3<f64>> for PyIsometry3 {
    fn from(inner: Isometry3<f64>) -> Self {
        Self { inner }
    }
}

// For `Transform::transform::<PyIsometry3>(&self, iso)`
impl Into<Isometry3<f64>> for &PyIsometry3 {
    fn into(self) -> Isometry3<f64> {
        self.inner
    }
}

fn rotation_from_matrix(mat: [[f64; 3]; 3]) -> PyResult<Rotation3<f64>> {
    let m = Matrix3::from_row_slice(&[
        mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1],
        mat[2][2],
    ]);
    let quat = UnitQuaternion::from_matrix(&m);
    Ok(Rotation3::from(quat))
}

/// Converts rotation input to Rotation3
fn rotation_from_input(
    euler: Option<[f64; 3]>,
    matrix: Option<[[f64; 3]; 3]>,
) -> PyResult<Rotation3<f64>> {
    match (euler, matrix) {
        (Some(angles), None) => Ok(Rotation3::from_euler_angles(angles[0], angles[1], angles[2])),
        (None, Some(mat)) => rotation_from_matrix(mat),
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            "Specify either euler_angles or rotation_matrix, not both.",
        )),
        (None, None) => Ok(Rotation3::identity()),
    }
}

#[pymethods]
impl PyIsometry3 {
    #[new]
    #[pyo3(signature = (translation=[0.0, 0.0, 0.0], *, euler_angles=None, rotation_matrix=None))]
    fn new(
        translation: [f64; 3],
        euler_angles: Option<[f64; 3]>,
        rotation_matrix: Option<[[f64; 3]; 3]>,
    ) -> PyResult<Self> {
        let t = Translation3::from(Vector3::from(translation));
        let rot = rotation_from_input(euler_angles, rotation_matrix)?;
        let iso = Isometry3::from_parts(t, rot.into()); // ← .into() for Unit<Quaternion>
        Ok(iso.into())
    }

    #[staticmethod]
    fn identity() -> Self {
        Isometry3::identity().into()
    }

    #[staticmethod]
    #[pyo3(signature = (translation))]
    fn from_translation(translation: [f64; 3]) -> Self {
        let t = Translation3::from(Vector3::from(translation));
        Isometry3::from_parts(t, Rotation3::identity().into()).into()
    }

    #[staticmethod]
    #[pyo3(signature = (*, euler_angles=None, rotation_matrix=None))]
    fn from_rotation(
        euler_angles: Option<[f64; 3]>,
        rotation_matrix: Option<[[f64; 3]; 3]>,
    ) -> PyResult<Self> {
        let rot = rotation_from_input(euler_angles, rotation_matrix)?;
        let iso = Isometry3::from_parts(Translation3::identity(), rot.into());
        Ok(iso.into())
    }

    #[staticmethod]
    fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        let rot = Rotation3::from_euler_angles(roll, pitch, yaw);
        Isometry3::from_parts(Translation3::identity(), rot.into()).into()
    }

    #[staticmethod]
    fn from_axis_angle(axis: [f64; 3], angle: f64) -> PyResult<Self> {
        let axis_vec = Vector3::from(axis);
        let norm = axis_vec.norm();
        if norm < f64::EPSILON {
            return Err(pyo3::exceptions::PyValueError::new_err("Axis must be a non-zero vector."));
        }
        let unit_axis = Unit::new_normalize(axis_vec);
        let rot = Rotation3::from_axis_angle(&unit_axis, angle);
        let iso = Isometry3::from_parts(Translation3::identity(), rot.into());
        Ok(iso.into())
    }

    // ─── Instance methods ───────────────────────────────────────

    #[pyo3(name = "transform")]
    fn py_transform(&self, other: &PyIsometry3) -> Self {
        (self.inner * other.inner).into()
    }

    #[pyo3(name = "rotate")]
    #[pyo3(signature = (*, euler_angles=None, rotation_matrix=None))]
    fn py_rotate(
        &self,
        euler_angles: Option<[f64; 3]>,
        rotation_matrix: Option<[[f64; 3]; 3]>,
    ) -> PyResult<Self> {
        let rot = rotation_from_input(euler_angles, rotation_matrix)?;
        let new_rot = rot * self.inner.rotation;
        Ok(Isometry3::from_parts(self.inner.translation, new_rot.into()).into())
    }

    #[pyo3(name = "translate")]
    fn py_translate(&self, translation: [f64; 3]) -> Self {
        let delta = Vector3::from(translation);
        let new_t = self.inner.translation.vector + delta;
        Isometry3::from_parts(Translation3::from(new_t), self.inner.rotation).into()
    }

    // ─── Getters ───────────────────────────────────────────────

    #[getter]
    fn translation(&self) -> [f64; 3] {
        let t = self.inner.translation.vector;
        [t.x, t.y, t.z]
    }

    #[getter]
    fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        let rmat = self.inner.rotation.to_rotation_matrix();
        let m = rmat.matrix();
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ]
    }
    fn __repr__(&self) -> String {
        format!(
            "Isometry3(translation=[{:.3}, {:.3}, {:.3}], rotation=...)",
            self.translation()[0],
            self.translation()[1],
            self.translation()[2],
        )
    }
}

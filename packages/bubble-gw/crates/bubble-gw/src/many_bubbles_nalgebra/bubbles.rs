use nalgebra::{Rotation3, Vector3, Vector4};
use ndarray::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Bubbles {
    pub spacetime: Vec<Vector4<f64>>,
}

impl Bubbles {
    pub fn new(spacetime: Vec<Vector4<f64>>) -> Self {
        Self { spacetime }
    }

    pub fn n_bubbles(&self) -> usize {
        self.spacetime.len()
    }

    pub fn to_array2(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.n_bubbles(), 4), |(i_bubble, i_spacetime)| {
            self.spacetime[i_bubble][i_spacetime]
        })
    }

    pub fn from_array2(array: Array2<f64>) -> Self {
        let n_bubbles = array.nrows();
        assert_eq!(array.ncols(), 4);
        let mat: Vec<Vector4<f64>> = (0..n_bubbles)
            .map(|i_bubble| {
                Vector4::from_row_slice(
                    array
                        .row(i_bubble)
                        .as_slice()
                        .expect("Cannot convert Array2 to slice"),
                )
            })
            .collect();
        Self::new(mat)
    }

    /// Returns a new `Bubbles` with spatial parts rotated by `rotation`.
    ///
    /// The rotation is applied only to the spatial `(x, y, z)` components;
    /// the time component (`t`) is unchanged.
    ///
    /// Accepts any type convertible to `Rotation3<f64>` (e.g., `Matrix3`, `UnitQuaternion`, `Rotation3`, Euler angles via `::from_euler_angles(...)`).
    pub fn rotate_spatial<R: Into<Rotation3<f64>>>(&self, rotation: R) -> Self {
        let rot = rotation.into();
        let rotated_spacetime = self
            .spacetime
            .iter()
            .map(|v| {
                let t = v[0];
                // Extract spatial: indices 1..4 → [x, y, z]
                let spatial = Vector3::from_row_slice(&v.as_slice()[1..4]);
                let rotated = rot * spatial;
                // Rebuild: [t, x', y', z']
                Vector4::from_row_slice(&[t, rotated.x, rotated.y, rotated.z])
            })
            .collect();
        Self::new(rotated_spacetime)
    }

    /// In-place version: rotates the spatial parts of all bubbles.
    ///
    /// More efficient when mutation is acceptable.
    pub fn rotate_spatial_mut<R: Into<Rotation3<f64>>>(&mut self, rotation: R) {
        let rot = rotation.into();
        for v in &mut self.spacetime {
            let t = v[0];
            let spatial = Vector3::from_row_slice(&v.as_slice()[1..4]);
            let rotated = rot * spatial;
            *v = Vector4::from_row_slice(&[t, rotated.x, rotated.y, rotated.z]);
        }
    }
}

use crate::many_bubbles_nalgebra::lattice::TransformationIsometry3;
use nalgebra::{Vector3, Vector4};
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
}

impl TransformationIsometry3 for Bubbles {
    fn transform<I: Into<nalgebra::Isometry3<f64>>>(&self, iso: I) -> Self {
        let iso = iso.into();
        let rot = iso.rotation;
        let trans = iso.translation.vector;

        let spacetime = self
            .spacetime
            .iter()
            .map(|v| {
                // Extract spatial part (x, y, z)
                let spatial = Vector3::new(v[1], v[2], v[3]);
                // Apply isometry: R·p + t
                let transformed_spatial = rot * spatial + trans;
                // Reconstruct with unchanged time
                Vector4::new(
                    v[0],
                    transformed_spatial.x,
                    transformed_spatial.y,
                    transformed_spatial.z,
                )
            })
            .collect();

        Self { spacetime }
    }

    fn transform_mut<I: Into<nalgebra::Isometry3<f64>>>(&mut self, iso: I) {
        let iso = iso.into();
        let rot = iso.rotation;
        let trans = iso.translation.vector;

        for v in &mut self.spacetime {
            let spatial = Vector3::new(v[1], v[2], v[3]);
            let transformed = rot * spatial + trans;
            v[1] = transformed.x;
            v[2] = transformed.y;
            v[3] = transformed.z;
            // v[0] (time) unchanged
        }
    }
}

use std::borrow::Borrow;

use nalgebra::{Isometry3, Point3, Vector3, Vector4};
use rand::Rng;
use rand::rngs::StdRng;

use super::{
    BoundaryConditions,
    GenerateBubblesExterior,
    LatticeGeometry,
    SamplePointsInsideLattice,
    TransformationIsometry3,
};
use crate::many_bubbles::bubbles::Bubbles;

#[derive(Clone, Debug, PartialEq)]
pub struct SphericalLattice {
    pub center: Point3<f64>,
    pub radius: f64,
}

impl SphericalLattice {
    pub fn new(center: Point3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl TransformationIsometry3 for SphericalLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        Self {
            center: iso.into() * self.center,
            radius: self.radius,
        }
    }
    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        self.center = iso.into() * self.center;
    }
}

impl LatticeGeometry for SphericalLattice {
    fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    fn reference_point(&self) -> Point3<f64> {
        self.center
    }

    fn parameters(&self) -> Vec<f64> {
        let c = self.center.coords;
        vec![c.x, c.y, c.z, self.radius]
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        let r2 = self.radius * self.radius;
        points
            .iter()
            .map(|p| (*p - self.center).norm_squared() <= r2)
            .collect()
    }
}

impl GenerateBubblesExterior for SphericalLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        match boundary_condition {
            BoundaryConditions::Periodic => Bubbles::new(Vec::new()),
            BoundaryConditions::Reflection => {
                let interior = bubbles_interior.borrow();
                let mut exterior_spacetime = Vec::with_capacity(interior.spacetime.len() * 2);

                let center = self.center;
                let radius = self.radius;

                for &event in &interior.spacetime {
                    let t = event[0];
                    let p = Point3::new(event[1], event[2], event[3]);
                    let v = p - center; // Vector from center to point
                    let d = v.norm();

                    if d < f64::EPSILON {
                        // Point at center: create two images along x-axis
                        let img1 = Point3::new(center.x + radius, center.y, center.z);
                        let img2 = Point3::new(center.x - radius, center.y, center.z);

                        exterior_spacetime.push(Vector4::new(t, img1.x, img1.y, img1.z));
                        exterior_spacetime.push(Vector4::new(t, img2.x, img2.y, img2.z));
                        continue;
                    }

                    // unit vector from center to p
                    let unit_vec = v / d;

                    // Intersection points with sphere surface
                    let q1 = Point3::from(center.coords + radius * unit_vec); // forward
                    let q2 = Point3::from(center.coords - radius * unit_vec); // backward

                    // Normals at tangent planes (same as radial directions)
                    let n1 = unit_vec;
                    let n2 = -unit_vec;

                    // Reflection across tangent plane at q1:
                    // p1_img = p - 2 * dot(p - q1, n1) * n1
                    let pq1 = p - q1;
                    let dot1 = pq1.x * n1.x + pq1.y * n1.y + pq1.z * n1.z;
                    let p1_img = Point3::new(
                        p.x - 2.0 * dot1 * n1.x,
                        p.y - 2.0 * dot1 * n1.y,
                        p.z - 2.0 * dot1 * n1.z,
                    );

                    // Reflection across tangent plane at q2:
                    // p2_img = p - 2 * dot(p - q2, n2) * n2
                    let pq2 = p - q2;
                    let dot2 = pq2.x * n2.x + pq2.y * n2.y + pq2.z * n2.z;
                    let p2_img = Point3::new(
                        p.x - 2.0 * dot2 * n2.x,
                        p.y - 2.0 * dot2 * n2.y,
                        p.z - 2.0 * dot2 * n2.z,
                    );

                    exterior_spacetime.push(Vector4::new(t, p1_img.x, p1_img.y, p1_img.z));
                    exterior_spacetime.push(Vector4::new(t, p2_img.x, p2_img.y, p2_img.z));
                }

                Bubbles::new(exterior_spacetime)
            },
            BoundaryConditions::None => Bubbles::new(Vec::new()),
        }
    }
}

// Point rejection: perform uniform sampling in cartesian coordinates, and
// reject points that are outside of the sphere
impl SamplePointsInsideLattice for SphericalLattice {
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>> {
        let mut points = Vec::with_capacity(n_points);
        let max_attempts = n_points * 10_000;

        let r = self.radius;

        for _ in 0..max_attempts {
            if points.len() >= n_points {
                break;
            }

            // Sample uniformly in [-1, 1]^3 cube (unit cube centered at origin)
            let sample: Vector3<f64> = Vector3::new(
                rng.random::<f64>() * 2.0 - 1.0,
                rng.random::<f64>() * 2.0 - 1.0,
                rng.random::<f64>() * 2.0 - 1.0,
            );

            // Reject if outside unit ball
            if sample.norm_squared() > 1.0 {
                continue;
            }

            // Scale and translate to lattice
            let pt = Point3::from(self.center.coords + r * sample);
            points.push(pt);
        }

        if points.len() != n_points {
            panic!(
                "Failed to sample {} points in SphericalLattice (radius={}, attempts={})",
                n_points, r, max_attempts
            );
        }

        points
    }
}

// use rand_distr::{Distribution, StandardNormal};
// //See  https://www.johndcook.com/blog/2025/10/11/ball-rng/#:~:text=To%20generate%20a%20random%20point%20in%20a,u$%20can%20over%2Dsample%20points%20near%20the%20origin.
// // Further docs: https://vhartmann.com/ball_sampling/
// //FIXME: This method seems to produce points sparser near the center,
// requires // more testing
// impl SamplePointsInsideLattice for SphericalLattice {
//     fn sample_points(&self, n_points: usize, rng: &mut StdRng) ->
// Vec<Point3<f64>> {         let mut points = Vec::with_capacity(n_points);
//         let max_attempts = n_points * 10_000;
//
//         for _ in 0..max_attempts {
//             if points.len() >= n_points {
//                 break;
//             }
//
//             // ✅ Explicit: sample f64 Gaussian components
//             let x: f64 = StandardNormal.sample(rng);
//             let y: f64 = StandardNormal.sample(rng);
//             let z: f64 = StandardNormal.sample(rng);
//
//             let v = Vector3::new(x, y, z);
//             let norm = v.norm();
//
//             if norm < f64::EPSILON {
//                 continue;
//             }
//
//             // ✅ Explicit: dir is Vector3<f64>, norm is f64 → division is
// unambiguous             let dir: Vector3<f64> = v / norm;
//
//             let u: f64 = rng.random();
//             let r = self.radius * u.cbrt();
//
//             // ✅ Point3<f64> construction is now unambiguous
//             let pt = Point3::new(
//                 self.center.x + r * dir.x,
//                 self.center.y + r * dir.y,
//                 self.center.z + r * dir.z,
//             );
//
//             points.push(pt);
//         }
//
//         if points.len() != n_points {
//             panic!(
//                 "Failed to sample {} points in SphericalLattice (radius={},
// attempts={})",                 n_points, self.radius, max_attempts
//             );
//         }
//
//         points
//     }
// }

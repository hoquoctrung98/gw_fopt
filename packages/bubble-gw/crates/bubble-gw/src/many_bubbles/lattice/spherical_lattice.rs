use std::borrow::Borrow;

use nalgebra::{Isometry3, Point3};
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
            BoundaryConditions::Periodic => {
                // Return empty as requested
                Bubbles::new(Vec::new())
            },

            BoundaryConditions::Reflection => {
                let interior = bubbles_interior.borrow();
                let mut exterior_spacetime = Vec::with_capacity(interior.spacetime.len());

                let center = self.center;
                let radius = self.radius;

                for &event in &interior.spacetime {
                    let t = event[0];
                    let p = nalgebra::Point3::new(event[1], event[2], event[3]);
                    let v = p - center; // Vector3
                    let d = v.norm();

                    let p_exterior = if d < f64::EPSILON {
                        // At center: reflect along x-axis
                        nalgebra::Point3::from(
                            center.coords + nalgebra::Vector3::x() * radius * 2.0,
                        )
                    } else {
                        // Mirror across surface: p' = c + (2*r/d - 1) * (p - c)
                        let scale = 2.0 * radius / d - 1.0;
                        nalgebra::Point3::from(center.coords + v * scale)
                    };

                    exterior_spacetime.push(nalgebra::Vector4::new(
                        t,
                        p_exterior.x,
                        p_exterior.y,
                        p_exterior.z,
                    ));
                }

                Bubbles::new(exterior_spacetime)
            },

            BoundaryConditions::None => Bubbles::new(Vec::new()),
        }
    }
}

impl SamplePointsInsideLattice for SphericalLattice {
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>> {
        let mut points = Vec::with_capacity(n_points);
        let max_attempts = n_points * 1000;

        for _ in 0..max_attempts {
            if points.len() >= n_points {
                break;
            }
            // Uniform in ball: r = R * cbrt(u), direction uniform
            let u = rng.random::<f64>();
            let r = self.radius * u.cbrt();
            let z = rng.random::<f64>() * 2.0 - 1.0;
            let phi = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
            let sin_theta = f64::sqrt(1.0 - z * z);
            let x = r * sin_theta * phi.cos();
            let y = r * sin_theta * phi.sin();
            let z_coord = r * z;
            let pt = Point3::new(self.center.x + x, self.center.y + y, self.center.z + z_coord);
            // Theoretically always inside, but check for floating-point safety
            if self.contains(&[pt])[0] {
                points.push(pt);
            }
        }

        if points.len() != n_points {
            panic!(
                "Failed to sample {} points in SphericalLattice (radius={})",
                n_points, self.radius
            );
        }
        points
    }
}

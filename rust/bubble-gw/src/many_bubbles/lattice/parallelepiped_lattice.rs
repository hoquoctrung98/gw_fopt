use std::borrow::Borrow;

use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Isometry3, Point3, Vector3};
use rand::RngExt;
use rand::rngs::StdRng;

use super::{
    BoundaryConditions,
    GenerateBubblesExterior,
    LatticeError,
    LatticeGeometry,
    SamplePointsInsideLattice,
    TransformationIsometry3,
};
use crate::many_bubbles::bubbles::Bubbles;

#[derive(Clone, Debug, PartialEq)]
pub struct ParallelepipedLattice {
    pub origin: Point3<f64>,
    pub basis: [Vector3<f64>; 3],
}

impl AbsDiffEq for ParallelepipedLattice {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.origin.abs_diff_eq(&other.origin, epsilon)
            && self.basis[0].abs_diff_eq(&other.basis[0], epsilon)
            && self.basis[1].abs_diff_eq(&other.basis[1], epsilon)
            && self.basis[2].abs_diff_eq(&other.basis[2], epsilon)
    }
}

impl RelativeEq for ParallelepipedLattice {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.origin
            .relative_eq(&other.origin, epsilon, max_relative)
            && self.basis[0].relative_eq(&other.basis[0], epsilon, max_relative)
            && self.basis[1].relative_eq(&other.basis[1], epsilon, max_relative)
            && self.basis[2].relative_eq(&other.basis[2], epsilon, max_relative)
    }
}

impl ParallelepipedLattice {
    /// Creates a parallelepiped from origin and basis vectors.
    ///
    /// Checks that the three basis vectors are linearly independent (non-zero
    /// volume). Panics if invalid (in debug builds; release may skip for
    /// performance).
    ///
    /// For fallible version, use `try_new`.
    pub fn new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        match Self::try_new(origin, basis) {
            Ok(l) => l,
            Err(e) => panic!("Invalid ParallelepipedLattice: {}", e),
        }
    }

    /// Fallible constructor: returns `Err` if basis vectors are linearly
    /// dependent.
    pub fn try_new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Result<Self, LatticeError> {
        let mat = nalgebra::Matrix3::from_columns(&basis);
        if mat.determinant().abs() <= f64::EPSILON {
            Err(LatticeError::DegenerateBasis)
        } else {
            Ok(Self { origin, basis })
        }
    }

    /// Creates without validation — for performance-critical code after
    /// verification.
    pub fn new_unchecked(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        Self { origin, basis }
    }

    /// Creates an axis-aligned box `[ox, ox+lx] × [oy, oy+ly] × [oz, oz+lz]`.
    pub fn axis_aligned(origin: Point3<f64>, lx: f64, ly: f64, lz: f64) -> Self {
        Self::new(origin, [Vector3::x() * lx, Vector3::y() * ly, Vector3::z() * lz])
    }

    /// Creates a cube centered at `origin`.
    pub fn cube_centered(origin: Point3<f64>, side: f64) -> Self {
        let h = side / 2.0;
        Self::new(
            origin - Vector3::repeat(h),
            [
                Vector3::x() * side,
                Vector3::y() * side,
                Vector3::z() * side,
            ],
        )
    }

    /// Returns normalized basis vectors directions.
    pub fn normalized_basis(&self) -> [Vector3<f64>; 3] {
        self.basis.map(|e| {
            let n2 = e.norm_squared();
            if n2 > f64::EPSILON {
                e / n2.sqrt()
            } else {
                Vector3::zeros()
            }
        })
    }
}

impl TransformationIsometry3 for ParallelepipedLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        let iso = iso.into();
        Self {
            origin: iso * self.origin,
            basis: [
                iso.rotation * self.basis[0],
                iso.rotation * self.basis[1],
                iso.rotation * self.basis[2],
            ],
        }
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        let iso = iso.into();
        self.origin = iso * self.origin;
        for v in &mut self.basis {
            *v = iso.rotation * *v;
        }
    }
}

impl LatticeGeometry for ParallelepipedLattice {
    fn volume(&self) -> f64 {
        let mat = nalgebra::Matrix3::from_columns(&self.basis);
        mat.determinant().abs()
    }

    fn reference_point(&self) -> Point3<f64> {
        self.origin
    }

    fn parameters(&self) -> Vec<f64> {
        let o = self.origin.coords;
        let [e1, e2, e3] = self.basis;
        vec![
            o.x, o.y, o.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z,
        ]
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        let mat = nalgebra::Matrix3::from_columns(&self.basis);
        let inv = match mat.try_inverse() {
            Some(m) => m,
            None => return vec![false; points.len()], // degenerate box
        };

        points
            .iter()
            .map(|p| {
                // d = p - origin → Vector3
                let d = *p - self.origin; // Vector3

                // uvw = inv * d → Vector3
                let uvw = inv * d; // Vector3

                // Use indexing to avoid field name issues
                uvw[0] >= 0.0
                    && uvw[0] <= 1.0
                    && uvw[1] >= 0.0
                    && uvw[1] <= 1.0
                    && uvw[2] >= 0.0
                    && uvw[2] <= 1.0
            })
            .collect()
    }
}

impl GenerateBubblesExterior for ParallelepipedLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        let interior = bubbles_interior.borrow();
        match boundary_condition {
            BoundaryConditions::Periodic => {
                let mut exterior_spacetime = Vec::new();
                let [e1, e2, e3] = self.basis;

                // Only 6 directions: ±e1, ±e2, ±e3
                let shifts = [e1, -e1, e2, -e2, e3, -e3];

                for &event in &interior.spacetime {
                    for &shift in &shifts {
                        let t = event[0];
                        let x = event[1] + shift.x;
                        let y = event[2] + shift.y;
                        let z = event[3] + shift.z;
                        exterior_spacetime.push(nalgebra::Vector4::new(t, x, y, z));
                    }
                }

                Bubbles::new(exterior_spacetime)
            },

            BoundaryConditions::Reflection => {
                let mut exterior_spacetime = Vec::new();
                let [e1, e2, e3] = self.basis;

                // Compute face centers and outward normals (unit)
                let norm1 = e1.norm();
                let norm2 = e2.norm();
                let norm3 = e3.norm();

                // Unit normals
                let n1 = if norm1 > f64::EPSILON {
                    e1 / norm1
                } else {
                    nalgebra::Vector3::x()
                };
                let n2 = if norm2 > f64::EPSILON {
                    e2 / norm2
                } else {
                    nalgebra::Vector3::y()
                };
                let n3 = if norm3 > f64::EPSILON {
                    e3 / norm3
                } else {
                    nalgebra::Vector3::z()
                };

                // Face points (min and max along each axis)
                let origin = self.origin;
                let p1_max = nalgebra::Point3::from(origin.coords + e1);
                let p2_max = nalgebra::Point3::from(origin.coords + e2);
                let p3_max = nalgebra::Point3::from(origin.coords + e3);

                for &event in &interior.spacetime {
                    let t = event[0];
                    let p = nalgebra::Point3::new(event[1], event[2], event[3]);

                    // Reflect across 6 faces: min/max for each axis
                    // For face with point q and outward normal n: p' = p - 2*((p - q)·n)*n
                    let reflections = [
                        // -e1 face (at origin, outward = -n1)
                        {
                            let d = (p - origin).dot(&(-n1));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n1))
                        },
                        // +e1 face (at p1_max, outward = +n1)
                        {
                            let d = (p - p1_max).dot(&n1);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n1)
                        },
                        // -e2 face
                        {
                            let d = (p - origin).dot(&(-n2));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n2))
                        },
                        // +e2 face
                        {
                            let d = (p - p2_max).dot(&n2);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n2)
                        },
                        // -e3 face
                        {
                            let d = (p - origin).dot(&(-n3));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n3))
                        },
                        // +e3 face
                        {
                            let d = (p - p3_max).dot(&n3);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n3)
                        },
                    ];

                    for rp in reflections {
                        exterior_spacetime.push(nalgebra::Vector4::new(t, rp.x, rp.y, rp.z));
                    }
                }

                Bubbles::new(exterior_spacetime)
            },

            BoundaryConditions::None => Bubbles::new(Vec::new()),
        }
    }
}

impl SamplePointsInsideLattice for ParallelepipedLattice {
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>> {
        let mut points = Vec::with_capacity(n_points);
        let max_attempts = n_points * 1000;

        for _ in 0..max_attempts {
            if points.len() >= n_points {
                break;
            }
            let u = rng.random::<f64>();
            let v = rng.random::<f64>();
            let w = rng.random::<f64>();
            let pt = Point3::from(
                self.origin.coords + u * self.basis[0] + v * self.basis[1] + w * self.basis[2],
            );
            // In parallelepiped, u,v,w∈[0,1] ⇒ pt∈lattice — no rejection needed
            points.push(pt);
        }

        if points.len() != n_points {
            panic!(
                "Failed to sample {} points in ParallelepipedLattice (volume={})",
                n_points,
                self.volume()
            );
        }
        points
    }
}

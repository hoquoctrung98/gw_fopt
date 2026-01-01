use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    BoundaryConditions, BuiltInLattice, GenerateBubblesExterior, LatticeGeometry,
    ParallelepipedLattice, SphericalLattice, TransformationIsometry3,
};
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, NucleationError};
use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::Array2;
use rand::{Rng, SeedableRng, random, rngs::StdRng};

/// Strategy trait for bubble nucleation.
///
/// Implemented per concrete lattice type (e.g. `BuiltInLattice`) due to lack of dyn-compatibility
/// (generic parameter `L` and method `nucleate` is not object-safe).
pub trait NucleationStrategy<L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior>
{
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError>;
}

/// Nucleates `n_bubbles` bubbles at fixed time `t0`, uniformly distributed within the lattice.
#[derive(Clone, Debug)]
pub struct UniformAtFixedTime {
    pub n_bubbles: usize,
    pub t0: f64,
    pub seed: Option<u64>,
}

impl NucleationStrategy<BuiltInLattice> for UniformAtFixedTime {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<BuiltInLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError> {
        let lattice = &lattice_bubbles.lattice;
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };
        let existing_interior = &lattice_bubbles.interior.spacetime;
        let existing_exterior = &lattice_bubbles.exterior.spacetime;

        let interior = match lattice {
            BuiltInLattice::Parallelepiped(l) => {
                self.sample_in_parallelepiped(l, &mut rng, existing_interior, existing_exterior)?
            },
            BuiltInLattice::Cartesian(c) => {
                self.sample_in_parallelepiped(&c.0, &mut rng, existing_interior, existing_exterior)?
            },
            BuiltInLattice::Spherical(s) => {
                self.sample_in_sphere(s, &mut rng, existing_interior, existing_exterior)?
            },
            BuiltInLattice::Empty(_) => return Err(NucleationError::UnsupportedLattice),
        };

        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles =
            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();

        Ok((interior, exterior))
    }
}

impl UniformAtFixedTime {
    fn sample_in_parallelepiped(
        &self,
        lattice: &ParallelepipedLattice,
        rng: &mut StdRng,
        existing_interior: &[Vector4<f64>],
        existing_exterior: &[Vector4<f64>],
    ) -> Result<Array2<f64>, NucleationError> {
        let mut points = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 100;
        for _ in 0..max_attempts {
            if points.len() >= self.n_bubbles {
                break;
            }
            let u = rng.random::<f64>();
            let v = rng.random::<f64>();
            let w = rng.random::<f64>();
            let pt = Point3::from(
                lattice.origin.coords
                    + u * lattice.basis[0]
                    + v * lattice.basis[1]
                    + w * lattice.basis[2],
            );
            let candidate = Vector4::new(self.t0, pt.x, pt.y, pt.z);
            if !lattice.contains(&[pt])[0] {
                continue;
            }
            let conflict = existing_interior
                .iter()
                .chain(existing_exterior.iter())
                .any(|&e| {
                    let d = candidate - e;
                    d[0] >= 0.0 && d.scalar(&d) > 0.0
                });
            if !conflict {
                points.push(pt);
            }
        }
        self.build_array(points)
    }

    fn sample_in_sphere(
        &self,
        lattice: &SphericalLattice,
        rng: &mut StdRng,
        existing_interior: &[Vector4<f64>],
        existing_exterior: &[Vector4<f64>],
    ) -> Result<Array2<f64>, NucleationError> {
        let mut points = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 100;
        for _ in 0..max_attempts {
            if points.len() >= self.n_bubbles {
                break;
            }
            let u = rng.random::<f64>();
            let r = lattice.radius * u.cbrt();
            let z = rng.random::<f64>() * 2.0 - 1.0;
            let phi = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
            let sin_theta = f64::sqrt(1.0 - z * z);
            let x = r * sin_theta * phi.cos();
            let y = r * sin_theta * phi.sin();
            let z = r * z;
            let pt = Point3::new(lattice.center.x + x, lattice.center.y + y, lattice.center.z + z);
            let candidate = Vector4::new(self.t0, pt.x, pt.y, pt.z);
            if !lattice.contains(&[pt])[0] {
                continue;
            }
            let conflict = existing_interior
                .iter()
                .chain(existing_exterior.iter())
                .any(|&e| {
                    let d = candidate - e;
                    d[0] >= 0.0 && d.scalar(&d) > 0.0
                });
            if !conflict {
                points.push(pt);
            }
        }
        self.build_array(points)
    }

    fn build_array(&self, points: Vec<Point3<f64>>) -> Result<Array2<f64>, NucleationError> {
        if points.len() != self.n_bubbles {
            Err(NucleationError::InsufficientBubbles {
                requested: self.n_bubbles,
                generated: points.len(),
            })
        } else {
            Ok(Array2::from_shape_fn((self.n_bubbles, 4), |(i, j)| match j {
                0 => self.t0,
                1 => points[i].x,
                2 => points[i].y,
                3 => points[i].z,
                _ => unreachable!(),
            }))
        }
    }
}

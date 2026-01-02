use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    BoundaryConditions, BuiltInLattice, CartesianLattice, GenerateBubblesExterior, LatticeGeometry,
    ParallelepipedLattice, SphericalLattice, TransformationIsometry3,
};
use crate::many_bubbles::lattice_bubbles::LatticeBubbles;
use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::Array2;
use rand::{Rng, SeedableRng, random, rngs::StdRng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NucleationError {
    #[error("Lattice does not support uniform sampling (e.g., EmptyLattice)")]
    UnsupportedLattice,

    #[error("Failed to generate {requested} bubbles; only {generated} produced")]
    InsufficientBubbles { requested: usize, generated: usize },

    #[error("Bubble at ({x}, {y}, {z}) is outside lattice")]
    BubbleOutsideLattice { x: f64, y: f64, z: f64 },

    #[error("Bubble formed inside existing bubble (causality violation)")]
    BubbleInsideExistingBubble,

    #[error("Strategy configuration error: {0}")]
    InvalidConfig(String),
}

/// Strategy trait for bubble nucleation.
pub trait NucleationStrategy<L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior>
{
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError>;
}

/// Nucleates `n_bubbles` bubbles at fixed time `t0`, uniformly distributed within the lattice.
/// Ensures no two *newly nucleated* bubbles violate causality (i.e., no overlap at formation).
#[derive(Clone, Debug)]
pub struct UniformAtFixedTime {
    pub n_bubbles: usize,
    pub t0: f64,
    pub seed: Option<u64>,
}

impl UniformAtFixedTime {
    fn sample_in_parallelepiped(
        &self,
        lattice: &ParallelepipedLattice,
        rng: &mut StdRng,
    ) -> Result<Array2<f64>, NucleationError> {
        let mut points = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 1000;

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
            let candidate_vec = Vector4::new(self.t0, pt.x, pt.y, pt.z);

            if !lattice.contains(&[pt])[0] {
                continue;
            }

            // Check only against already accepted new bubbles
            let conflict = points.iter().any(|&existing_pt: &Point3<f64>| {
                let existing_vec =
                    Vector4::new(self.t0, existing_pt.x, existing_pt.y, existing_pt.z);
                let delta = candidate_vec - existing_vec;
                delta.scalar(&delta) < 0.0
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
    ) -> Result<Array2<f64>, NucleationError> {
        let mut points = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 1000;

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
            let z_coord = r * z; // avoid name shadowing `z`
            let pt =
                Point3::new(lattice.center.x + x, lattice.center.y + y, lattice.center.z + z_coord);
            let candidate_vec = Vector4::new(self.t0, pt.x, pt.y, pt.z);

            if !lattice.contains(&[pt])[0] {
                continue;
            }

            let conflict = points.iter().any(|&existing_pt: &Point3<f64>| {
                let existing_vec =
                    Vector4::new(self.t0, existing_pt.x, existing_pt.y, existing_pt.z);
                let delta = candidate_vec - existing_vec;
                delta.scalar(&delta) < 0.0
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

        let interior = match lattice {
            BuiltInLattice::Parallelepiped(l) => self.sample_in_parallelepiped(l, &mut rng)?,
            BuiltInLattice::Cartesian(c) => self.sample_in_parallelepiped(&c.0, &mut rng)?,
            BuiltInLattice::Spherical(s) => self.sample_in_sphere(s, &mut rng)?,
            BuiltInLattice::Empty(_) => return Err(NucleationError::UnsupportedLattice),
        };

        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles =
            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();

        Ok((interior, exterior))
    }
}

impl NucleationStrategy<ParallelepipedLattice> for UniformAtFixedTime {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<ParallelepipedLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError> {
        let lattice = &lattice_bubbles.lattice;
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_in_parallelepiped(lattice, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles =
            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();
        Ok((interior, exterior))
    }
}

impl NucleationStrategy<CartesianLattice> for UniformAtFixedTime {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<CartesianLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError> {
        let lattice = &lattice_bubbles.lattice;
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_in_parallelepiped(&lattice.0, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles =
            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();
        Ok((interior, exterior))
    }
}

impl NucleationStrategy<SphericalLattice> for UniformAtFixedTime {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<SphericalLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError> {
        let lattice = &lattice_bubbles.lattice;
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_in_sphere(lattice, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles =
            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();
        Ok((interior, exterior))
    }
}

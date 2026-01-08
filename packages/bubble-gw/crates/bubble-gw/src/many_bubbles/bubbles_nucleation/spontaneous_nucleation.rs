use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{SeedableRng, random};

use super::{GeneralLatticeProperties, NucleationStrategy};
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    BoundaryConditions,
    BuiltInLattice,
    CartesianLattice,
    GenerateBubblesExterior,
    ParallelepipedLattice,
    SphericalLattice,
};
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};

/// Nucleates `n_bubbles` bubbles at fixed time `t0`, uniformly distributed
/// within the lattice. Ensures no two *newly nucleated* bubbles violate
/// causality (i.e., no overlap at formation).
#[derive(Clone, Debug)]
pub struct SpontaneousNucleation {
    pub n_bubbles: usize,
    pub t0: f64,
    pub seed: Option<u64>,
}

impl SpontaneousNucleation {
    /// Sample `n_bubbles` points uniformly in lattice, rejecting any that
    /// violate causality with already-accepted points.
    fn sample_interior<L: GeneralLatticeProperties>(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        rng: &mut StdRng,
    ) -> Result<Array2<f64>, LatticeBubblesError> {
        let lattice = &lattice_bubbles.lattice;
        let mut accepted = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 10_000;

        for _ in 0..max_attempts {
            if accepted.len() >= self.n_bubbles {
                break;
            }

            let candidate_pt = lattice.sample_points(1, rng).into_iter().next().unwrap();
            let candidate_vec =
                Vector4::new(self.t0, candidate_pt.x, candidate_pt.y, candidate_pt.z);

            // Skip if outside lattice (FP edge case)
            if !lattice.contains(&[candidate_pt])[0] {
                continue;
            }

            // Check causality against already accepted *new* points
            let conflict = accepted.iter().any(|&pt: &Point3<f64>| {
                let existing_vec = Vector4::new(self.t0, pt.x, pt.y, pt.z);
                let delta = candidate_vec - existing_vec;
                // In (−,+,+,+) signature: timelike separation ⇒ causality violation
                delta.scalar(&delta) < 0.0
            });

            if !conflict {
                accepted.push(candidate_pt);
            }
        }

        if accepted.len() != self.n_bubbles {
            return Err(LatticeBubblesError::NucleationError(
                format!(
                    "Insufficient Bubbles: requested {}, generated {}",
                    self.n_bubbles,
                    accepted.len()
                )
                .to_string(),
            ));
        }

        Ok(Array2::from_shape_fn((self.n_bubbles, 4), |(i, j)| match j {
            0 => self.t0,
            1 => accepted[i].x,
            2 => accepted[i].y,
            3 => accepted[i].z,
            _ => unreachable!(),
        }))
    }
}

impl NucleationStrategy<BuiltInLattice> for SpontaneousNucleation {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<BuiltInLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), LatticeBubblesError> {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_interior(lattice_bubbles, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior_bubbles = lattice_bubbles
            .lattice
            .generate_bubbles_exterior(&dummy_interior, boundary_condition);
        let exterior = exterior_bubbles.to_array2();

        Ok((interior, exterior))
    }
}

impl NucleationStrategy<ParallelepipedLattice> for SpontaneousNucleation {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<ParallelepipedLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), LatticeBubblesError> {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_interior(lattice_bubbles, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior = lattice_bubbles
            .lattice
            .generate_bubbles_exterior(&dummy_interior, boundary_condition)
            .to_array2();
        Ok((interior, exterior))
    }
}

impl NucleationStrategy<CartesianLattice> for SpontaneousNucleation {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<CartesianLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), LatticeBubblesError> {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_interior(lattice_bubbles, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior = lattice_bubbles
            .lattice
            .generate_bubbles_exterior(&dummy_interior, boundary_condition)
            .to_array2();
        Ok((interior, exterior))
    }
}

impl NucleationStrategy<SphericalLattice> for SpontaneousNucleation {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<SphericalLattice>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), LatticeBubblesError> {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let interior = self.sample_interior(lattice_bubbles, &mut rng)?;
        let dummy_interior = Bubbles::from_array2(interior.clone());
        let exterior = lattice_bubbles
            .lattice
            .generate_bubbles_exterior(&dummy_interior, boundary_condition)
            .to_array2();
        Ok((interior, exterior))
    }
}

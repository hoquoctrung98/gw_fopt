use nalgebra::Vector4;
use nalgebra_spacetime::Lorentzian;
use rand::rngs::StdRng;
use rand::{SeedableRng, random};

use super::{GeneralLatticeProperties, NucleationStrategy};
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};

/// Nucleates `n_bubbles` bubbles at fixed time `t0`, uniformly distributed
/// within the lattice. Ensures no two *newly nucleated* bubbles violate
/// causality (i.e., no overlap at formation).
#[derive(Clone, Debug)]
pub struct SpontaneousNucleation {
    pub n_bubbles: usize,
    pub t0: f64,
    pub seed: Option<u64>,
    rng: StdRng,
}

impl SpontaneousNucleation {
    pub fn new(n_bubbles: usize, t0: f64, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(random::<u64>()),
        };
        Self {
            n_bubbles,
            t0,
            seed,
            rng,
        }
    }

    /// Sample `n_bubbles` points uniformly in lattice, rejecting any that
    /// violate causality with already-accepted points.
    fn sample_interior<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
    ) -> Result<Bubbles, LatticeBubblesError> {
        let mut accepted_spacetime = Vec::with_capacity(self.n_bubbles);
        let max_attempts = self.n_bubbles * 10_000;

        for _ in 0..max_attempts {
            if accepted_spacetime.len() >= self.n_bubbles {
                break;
            }

            let candidate_pt = lattice
                .sample_points(1, &mut self.rng)
                .into_iter()
                .next()
                .unwrap();
            let candidate_vec =
                Vector4::new(self.t0, candidate_pt.x, candidate_pt.y, candidate_pt.z);

            if !lattice.contains(&[candidate_pt])[0] {
                continue;
            }

            // Check causality against already accepted *new* bubbles
            let conflict = accepted_spacetime
                .iter()
                .any(|existing_vec: &Vector4<f64>| {
                    let delta = candidate_vec - existing_vec;
                    delta.scalar(&delta) < 0.0
                });

            if !conflict {
                accepted_spacetime.push(candidate_vec);
            }
        }

        if accepted_spacetime.len() != self.n_bubbles {
            return Err(LatticeBubblesError::NucleationError(
                format!(
                    "Insufficient Bubbles: requested {}, generated {}",
                    self.n_bubbles,
                    accepted_spacetime.len()
                )
                .to_string(),
            ));
        }

        Ok(Bubbles::new(accepted_spacetime))
    }
}

impl<L: GeneralLatticeProperties> NucleationStrategy<L> for SpontaneousNucleation {
    fn nucleate(
        &mut self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Bubbles, Bubbles), LatticeBubblesError> {
        let lattice = &lattice_bubbles.lattice;
        let interior = self.sample_interior(lattice)?;
        let exterior = lattice.generate_bubbles_exterior(&interior, boundary_condition);
        Ok((interior, exterior))
    }
}

use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{BoundaryConditions, GeneralLatticeProperties};
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};
pub mod fixed_rate_nucleation;
pub use fixed_rate_nucleation::FixedRateNucleation;
pub mod spontaneous_nucleation;
pub use spontaneous_nucleation::SpontaneousNucleation;

/// Strategy trait for bubble nucleation.
pub trait NucleationStrategy<L: GeneralLatticeProperties> {
    fn nucleate(
        &mut self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Bubbles, Bubbles), LatticeBubblesError>;
}

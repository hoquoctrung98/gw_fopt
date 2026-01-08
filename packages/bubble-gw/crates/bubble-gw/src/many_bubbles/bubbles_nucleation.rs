use ndarray::Array2;

use crate::many_bubbles::lattice::{BoundaryConditions, GeneralLatticeProperties};
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};
pub mod fixed_rate_nucleation;
pub use fixed_rate_nucleation::FixedRateNucleation;
pub mod spontaneous_nucleation;
pub use spontaneous_nucleation::SpontaneousNucleation;

/// Strategy trait for bubble nucleation.
pub trait NucleationStrategy<L: GeneralLatticeProperties> {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), LatticeBubblesError>;
}

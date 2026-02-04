use crate::many_bubbles::lattice::GeneralLatticeProperties;
pub mod fixed_rate_nucleation;
pub use fixed_rate_nucleation::{
    FixedRateNucleation,
    FixedRateNucleationError,
    FixedRateNucleationMethod,
};
pub mod spontaneous_nucleation;
pub use spontaneous_nucleation::SpontaneousNucleation;

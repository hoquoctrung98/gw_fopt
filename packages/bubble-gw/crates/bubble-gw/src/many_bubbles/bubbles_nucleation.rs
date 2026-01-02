use ndarray::Array2;
use thiserror::Error;

use crate::many_bubbles::lattice::{BoundaryConditions, GeneralLatticeProperties};
use crate::many_bubbles::lattice_bubbles::LatticeBubbles;
pub mod fixed_nucleation_rate;
pub use fixed_nucleation_rate::FixedNucleationRate;
pub mod uniform_at_fixed_time;
pub use uniform_at_fixed_time::UniformAtFixedTime;

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
pub trait NucleationStrategy<L: GeneralLatticeProperties> {
    fn nucleate(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Array2<f64>, Array2<f64>), NucleationError>;
}

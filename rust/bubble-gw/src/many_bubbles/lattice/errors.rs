use thiserror::Error;

#[derive(Debug, Error)]
pub enum LatticeError {
    #[error("Basis vectors are linearly dependent (zero volume)")]
    DegenerateBasis,

    #[error(
        "Basis vectors are not pairwise orthogonal, maybe you want to use parallelepiped lattice instead?"
    )]
    NonOrthogonalBasis,
}

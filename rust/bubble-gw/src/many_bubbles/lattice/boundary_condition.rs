/// Enum representing boundary conditions for the simulation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    Periodic,
    Reflection,
    None,
}

use std::borrow::Borrow;

use nalgebra::{Isometry3, Point3};
use rand::rngs::StdRng;

use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;

pub trait GenerateBubblesExterior: Clone + Sync {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles;
}

/// Trait for 3D lattice geometries supporting rigid transformations.
///
/// All methods are designed for batch efficiency and introspection.
pub trait LatticeGeometry: Clone + Sync {
    /// Returns the volume of the lattice.
    fn volume(&self) -> f64;

    /// Returns a canonical reference point (e.g., origin or center).
    fn reference_point(&self) -> Point3<f64>;

    /// Returns a flat list of parameters needed to reconstruct the lattice.
    ///
    /// Format:
    /// - `CartesianLattice`: `[ox, oy, oz, e1x, e1y, e1z, e2x, e2y, e2z, e3x,
    ///   e3y, e3z]`
    /// - `SphericalLattice`: `[cx, cy, cz, r]`
    fn parameters(&self) -> Vec<f64>;

    /// Batch containment check: returns `Vec<bool>` where `out[i] =
    /// self.contains(points[i])`.
    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool>;
}

pub trait TransformationIsometry3: Clone + Sync {
    /// Transforms self by an isometry, returning a new instance.
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self;

    /// Transforms self in-place by an isometry.
    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I);
}

/// Ability to sample points uniformly from the interior of a lattice.
pub trait SamplePointsInsideLattice {
    /// Sample `n_points` points uniformly from the lattice volume.
    ///
    /// Uses rejection sampling internally: generates candidate points,
    /// accepts only those satisfying `self.contains([p]) == [true]`.
    ///
    /// May panic if lattice has zero volume or sampling fails after too many
    /// attempts.
    ///
    /// # Parameters
    /// - `n_points`: number of points to sample
    /// - `rng`: random number generator
    ///
    /// # Returns
    /// `Vec<Point3<f64>>` of length `n_points`.
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>>;
}

/// Common lattice properties bundled into a trait, useful to reduce boilerplate
pub trait GeneralLatticeProperties:
    LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior + SamplePointsInsideLattice
{
}

// Blanket impl
impl<T> GeneralLatticeProperties for T where
    T: LatticeGeometry
        + TransformationIsometry3
        + GenerateBubblesExterior
        + SamplePointsInsideLattice
{
}

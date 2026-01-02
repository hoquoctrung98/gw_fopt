use std::borrow::Borrow;

use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Isometry3, Point3, Vector3};
use rand::rngs::StdRng;

use super::{
    BoundaryConditions,
    GenerateBubblesExterior,
    LatticeError,
    LatticeGeometry,
    ParallelepipedLattice,
    SamplePointsInsideLattice,
    TransformationIsometry3,
};
use crate::many_bubbles::bubbles::Bubbles;

/// An oriented Cartesian box (i.e., a rectangular parallelepiped with
/// orthogonal basis).
///
/// While stored as a `ParallelepipedLattice`, this type implies orthogonality
/// of basis, enabling optimizations (e.g., faster containment, volume =
/// |e1|*|e2|*|e3|).
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianLattice(pub ParallelepipedLattice);

impl AbsDiffEq for CartesianLattice {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0.abs_diff_eq(&other.0, epsilon)
    }
}

impl RelativeEq for CartesianLattice {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0.relative_eq(&other.0, epsilon, max_relative)
    }
}

impl CartesianLattice {
    /// Creates a Cartesian lattice (orthogonal basis) from origin and basis.
    ///
    /// Checks orthogonality and non-degeneracy.
    /// Panics on failure (debug-friendly).
    pub fn new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        match Self::try_new(origin, basis) {
            Ok(l) => l,
            Err(e) => panic!("Invalid CartesianLattice: {}", e),
        }
    }

    /// Fallible constructor.
    pub fn try_new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Result<Self, LatticeError> {
        // First check linear independence (via parallelepiped)
        ParallelepipedLattice::try_new(origin, basis)?;

        // Then check orthogonality: e_i · e_j ≈ 0 for i ≠ j
        let [e1, e2, e3] = basis;
        let eps = 1e-10; // tolerance for floating-point orthogonality

        if (e1.dot(&e2)).abs() > eps || (e1.dot(&e3)).abs() > eps || (e2.dot(&e3)).abs() > eps {
            return Err(LatticeError::NonOrthogonalBasis);
        }

        Ok(Self(ParallelepipedLattice { origin, basis }))
    }

    /// Fast constructor without checks.
    pub fn new_unchecked(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        Self(ParallelepipedLattice::new_unchecked(origin, basis))
    }

    /// Creates from origin and side lengths (axis-aligned).
    pub fn with_origin_and_sizes(origin: Point3<f64>, sizes: [f64; 3]) -> Self {
        let [lx, ly, lz] = sizes;
        Self::new(origin, [Vector3::x() * lx, Vector3::y() * ly, Vector3::z() * lz])
    }

    /// Creates from center, side lengths, and optional rotation.
    ///
    /// The box is centered at `center`, with side lengths `[lx, ly, lz]`.
    /// If `rotation` is provided, the box is oriented accordingly.
    pub fn centered(
        center: Point3<f64>,
        sizes: [f64; 3],
        rotation: Option<&nalgebra::Rotation3<f64>>,
    ) -> Self {
        let [lx, ly, lz] = sizes;
        let half = Vector3::new(lx / 2.0, ly / 2.0, lz / 2.0);
        let origin = center.coords - half;

        let mut basis = [Vector3::x() * lx, Vector3::y() * ly, Vector3::z() * lz];

        if let Some(rot) = rotation {
            for e in &mut basis {
                *e = *rot * *e;
            }
        }

        // Since rotation preserves orthogonality, we can use `new_unchecked` safely
        Self::new_unchecked(Point3::from(origin), basis)
    }

    /// Returns normalized basis vectors (should be orthonormal).
    pub fn normalized_basis(&self) -> [Vector3<f64>; 3] {
        self.0.normalized_basis()
    }

    /// Returns side lengths (basis norms).
    pub fn side_lengths(&self) -> [f64; 3] {
        self.0.basis.map(|e| e.norm())
    }
}

impl LatticeGeometry for CartesianLattice {
    fn volume(&self) -> f64 {
        self.0.volume()
    }

    fn reference_point(&self) -> Point3<f64> {
        self.0.reference_point()
    }

    fn parameters(&self) -> Vec<f64> {
        self.0.parameters()
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        self.0.contains(points)
    }
}

impl TransformationIsometry3 for CartesianLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        Self(self.0.transform(iso))
    }
    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        self.0.transform(iso);
    }
}

impl GenerateBubblesExterior for CartesianLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        self.0
            .generate_bubbles_exterior(bubbles_interior, boundary_condition)
    }
}

impl SamplePointsInsideLattice for CartesianLattice {
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>> {
        self.0.sample_points(n_points, rng) // delegate
    }
}

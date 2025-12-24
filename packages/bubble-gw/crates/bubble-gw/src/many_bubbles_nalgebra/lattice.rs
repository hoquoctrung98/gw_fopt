use std::borrow::Borrow;

use crate::many_bubbles_nalgebra::bubbles::Bubbles;
use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Isometry3, Point3, Vector3};
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

pub trait TransformationIsometry3: Clone + Sync {
    /// Transforms self by an isometry, returning a new instance.
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self;

    /// Transforms self in-place by an isometry.
    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I);
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
    /// - `CartesianLattice`: `[ox, oy, oz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z]`
    /// - `SphericalLattice`: `[cx, cy, cz, r]`
    fn parameters(&self) -> Vec<f64>;

    /// Batch containment check: returns `Vec<bool>` where `out[i] = self.contains(points[i])`.
    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmptyLattice {}

impl LatticeGeometry for EmptyLattice {
    fn volume(&self) -> f64 {
        0.0
    }

    fn reference_point(&self) -> Point3<f64> {
        Point3::origin()
    }

    fn parameters(&self) -> Vec<f64> {
        Vec::new()
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        vec![true; points.len()]
    }
}

impl TransformationIsometry3 for EmptyLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, _iso: I) -> Self {
        EmptyLattice {}
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, _iso: I) {}
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParallelepipedLattice {
    pub origin: Point3<f64>,
    pub basis: [Vector3<f64>; 3],
}

impl AbsDiffEq for ParallelepipedLattice {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.origin.abs_diff_eq(&other.origin, epsilon)
            && self.basis[0].abs_diff_eq(&other.basis[0], epsilon)
            && self.basis[1].abs_diff_eq(&other.basis[1], epsilon)
            && self.basis[2].abs_diff_eq(&other.basis[2], epsilon)
    }
}

impl RelativeEq for ParallelepipedLattice {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.origin
            .relative_eq(&other.origin, epsilon, max_relative)
            && self.basis[0].relative_eq(&other.basis[0], epsilon, max_relative)
            && self.basis[1].relative_eq(&other.basis[1], epsilon, max_relative)
            && self.basis[2].relative_eq(&other.basis[2], epsilon, max_relative)
    }
}

impl ParallelepipedLattice {
    /// Creates a parallelepiped from origin and basis vectors.
    ///
    /// Checks that the three basis vectors are linearly independent (non-zero volume).
    /// Panics if invalid (in debug builds; release may skip for performance).
    ///
    /// For fallible version, use `try_new`.
    pub fn new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        match Self::try_new(origin, basis) {
            Ok(l) => l,
            Err(e) => panic!("Invalid ParallelepipedLattice: {}", e),
        }
    }

    /// Fallible constructor: returns `Err` if basis vectors are linearly dependent.
    pub fn try_new(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Result<Self, LatticeError> {
        let mat = nalgebra::Matrix3::from_columns(&basis);
        if mat.determinant().abs() <= f64::EPSILON {
            Err(LatticeError::DegenerateBasis)
        } else {
            Ok(Self { origin, basis })
        }
    }

    /// Creates without validation — for performance-critical code after verification.
    pub fn new_unchecked(origin: Point3<f64>, basis: [Vector3<f64>; 3]) -> Self {
        Self { origin, basis }
    }

    /// Creates an axis-aligned box `[ox, ox+lx] × [oy, oy+ly] × [oz, oz+lz]`.
    pub fn axis_aligned(origin: Point3<f64>, lx: f64, ly: f64, lz: f64) -> Self {
        Self::new(origin, [Vector3::x() * lx, Vector3::y() * ly, Vector3::z() * lz])
    }

    /// Creates a cube centered at `origin`.
    pub fn cube_centered(origin: Point3<f64>, side: f64) -> Self {
        let h = side / 2.0;
        Self::new(
            origin - Vector3::repeat(h),
            [
                Vector3::x() * side,
                Vector3::y() * side,
                Vector3::z() * side,
            ],
        )
    }

    /// Returns normalized basis vectors directions.
    pub fn basis_vectors(&self) -> [Vector3<f64>; 3] {
        self.basis.map(|e| {
            let n2 = e.norm_squared();
            if n2 > f64::EPSILON {
                e / n2.sqrt()
            } else {
                Vector3::zeros()
            }
        })
    }
}

impl TransformationIsometry3 for ParallelepipedLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        let iso = iso.into();
        Self {
            origin: iso * self.origin,
            basis: [
                iso.rotation * self.basis[0],
                iso.rotation * self.basis[1],
                iso.rotation * self.basis[2],
            ],
        }
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        let iso = iso.into();
        self.origin = iso * self.origin;
        for v in &mut self.basis {
            *v = iso.rotation * *v;
        }
    }
}

impl LatticeGeometry for ParallelepipedLattice {
    fn volume(&self) -> f64 {
        let mat = nalgebra::Matrix3::from_columns(&self.basis);
        mat.determinant().abs()
    }

    fn reference_point(&self) -> Point3<f64> {
        self.origin
    }

    fn parameters(&self) -> Vec<f64> {
        let o = self.origin.coords;
        let [e1, e2, e3] = self.basis;
        vec![
            o.x, o.y, o.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z,
        ]
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        let mat = nalgebra::Matrix3::from_columns(&self.basis);
        let inv = match mat.try_inverse() {
            Some(m) => m,
            None => return vec![false; points.len()], // degenerate box
        };

        points
            .iter()
            .map(|p| {
                // d = p - origin → Vector3
                let d = *p - self.origin; // Vector3

                // uvw = inv * d → Vector3
                let uvw = inv * d; // Vector3

                // Use indexing to avoid field name issues
                uvw[0] >= 0.0
                    && uvw[0] <= 1.0
                    && uvw[1] >= 0.0
                    && uvw[1] <= 1.0
                    && uvw[2] >= 0.0
                    && uvw[2] <= 1.0
            })
            .collect()
    }
}

/// An oriented Cartesian box (i.e., a rectangular parallelepiped with orthogonal basis).
///
/// While stored as a `ParallelepipedLattice`, this type implies orthogonality of basis,
/// enabling optimizations (e.g., faster containment, volume = |e1|*|e2|*|e3|).
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

        let mut edges = [Vector3::x() * lx, Vector3::y() * ly, Vector3::z() * lz];

        if let Some(rot) = rotation {
            for e in &mut edges {
                *e = *rot * *e;
            }
        }

        // Since rotation preserves orthogonality, we can use `new_unchecked` safely
        Self::new_unchecked(Point3::from(origin), edges)
    }

    /// Returns normalized basis vectors (should be orthonormal).
    pub fn basis_vectors(&self) -> [Vector3<f64>; 3] {
        self.0.basis_vectors()
    }

    /// Returns side lengths (edge norms).
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

#[derive(Clone, Debug, PartialEq)]
pub struct SphericalLattice {
    pub center: Point3<f64>,
    pub radius: f64,
}

impl SphericalLattice {
    pub fn new(center: Point3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl TransformationIsometry3 for SphericalLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        Self {
            center: iso.into() * self.center,
            radius: self.radius,
        }
    }
    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        self.center = iso.into() * self.center;
    }
}

impl LatticeGeometry for SphericalLattice {
    fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    fn reference_point(&self) -> Point3<f64> {
        self.center
    }

    fn parameters(&self) -> Vec<f64> {
        let c = self.center.coords;
        vec![c.x, c.y, c.z, self.radius]
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        let r2 = self.radius * self.radius;
        points
            .iter()
            .map(|p| (*p - self.center).norm_squared() <= r2)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub enum BuiltInLattice {
    Parallelepiped(ParallelepipedLattice),
    Cartesian(CartesianLattice),
    Spherical(SphericalLattice),
    Empty(EmptyLattice),
}

impl LatticeGeometry for BuiltInLattice {
    fn volume(&self) -> f64 {
        match self {
            Self::Parallelepiped(l) => l.volume(),
            Self::Cartesian(l) => l.volume(),
            Self::Spherical(l) => l.volume(),
            Self::Empty(l) => l.volume(),
        }
    }

    fn reference_point(&self) -> Point3<f64> {
        match self {
            Self::Parallelepiped(l) => l.reference_point(),
            Self::Cartesian(l) => l.reference_point(),
            Self::Spherical(l) => l.reference_point(),
            Self::Empty(l) => l.reference_point(),
        }
    }

    fn parameters(&self) -> Vec<f64> {
        match self {
            Self::Parallelepiped(l) => l.parameters(),
            Self::Cartesian(l) => l.parameters(),
            Self::Spherical(l) => l.parameters(),
            Self::Empty(l) => l.parameters(),
        }
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        match self {
            Self::Parallelepiped(l) => l.contains(points),
            Self::Cartesian(l) => l.contains(points),
            Self::Spherical(l) => l.contains(points),
            Self::Empty(l) => l.contains(points),
        }
    }
}

impl TransformationIsometry3 for BuiltInLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        match self {
            Self::Parallelepiped(l) => Self::Parallelepiped(l.transform(iso)),
            Self::Cartesian(l) => Self::Cartesian(l.transform(iso)),
            Self::Spherical(l) => Self::Spherical(l.transform(iso)),
            Self::Empty(l) => Self::Empty(l.transform(iso)),
        }
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        match self {
            Self::Parallelepiped(l) => l.transform_mut(iso),
            Self::Cartesian(l) => l.transform_mut(iso),
            Self::Spherical(l) => l.transform_mut(iso),
            Self::Empty(l) => l.transform_mut(iso),
        }
    }
}

/// Enum representing boundary conditions for the simulation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    Periodic,
    Reflection,
}

pub trait GenerateBubblesExterior: Clone + Sync {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles;
}

impl GenerateBubblesExterior for ParallelepipedLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        let interior = bubbles_interior.borrow();
        match boundary_condition {
            BoundaryConditions::Periodic => {
                let mut exterior_spacetime = Vec::new();
                let [e1, e2, e3] = self.basis;

                // Only 6 directions: ±e1, ±e2, ±e3
                let shifts = [e1, -e1, e2, -e2, e3, -e3];

                for &shift in &shifts {
                    for &event in &interior.spacetime {
                        let t = event[0];
                        let x = event[1] + shift.x;
                        let y = event[2] + shift.y;
                        let z = event[3] + shift.z;
                        exterior_spacetime.push(nalgebra::Vector4::new(t, x, y, z));
                    }
                }

                Bubbles::new(exterior_spacetime)
            }

            BoundaryConditions::Reflection => {
                let mut exterior_spacetime = Vec::new();
                let [e1, e2, e3] = self.basis;

                // Compute face centers and outward normals (unit)
                let norm1 = e1.norm();
                let norm2 = e2.norm();
                let norm3 = e3.norm();

                // Unit normals
                let n1 = if norm1 > f64::EPSILON {
                    e1 / norm1
                } else {
                    nalgebra::Vector3::x()
                };
                let n2 = if norm2 > f64::EPSILON {
                    e2 / norm2
                } else {
                    nalgebra::Vector3::y()
                };
                let n3 = if norm3 > f64::EPSILON {
                    e3 / norm3
                } else {
                    nalgebra::Vector3::z()
                };

                // Face points (min and max along each axis)
                let origin = self.origin;
                let p1_max = nalgebra::Point3::from(origin.coords + e1);
                let p2_max = nalgebra::Point3::from(origin.coords + e2);
                let p3_max = nalgebra::Point3::from(origin.coords + e3);

                for &event in &interior.spacetime {
                    let t = event[0];
                    let p = nalgebra::Point3::new(event[1], event[2], event[3]);

                    // Reflect across 6 faces: min/max for each axis
                    // For face with point q and outward normal n: p' = p - 2*((p - q)·n)*n
                    let reflections = [
                        // -e1 face (at origin, outward = -n1)
                        {
                            let d = (p - origin).dot(&(-n1));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n1))
                        },
                        // +e1 face (at p1_max, outward = +n1)
                        {
                            let d = (p - p1_max).dot(&n1);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n1)
                        },
                        // -e2 face
                        {
                            let d = (p - origin).dot(&(-n2));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n2))
                        },
                        // +e2 face
                        {
                            let d = (p - p2_max).dot(&n2);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n2)
                        },
                        // -e3 face
                        {
                            let d = (p - origin).dot(&(-n3));
                            nalgebra::Point3::from(p.coords - 2.0 * d * (-n3))
                        },
                        // +e3 face
                        {
                            let d = (p - p3_max).dot(&n3);
                            nalgebra::Point3::from(p.coords - 2.0 * d * n3)
                        },
                    ];

                    for rp in reflections {
                        exterior_spacetime.push(nalgebra::Vector4::new(t, rp.x, rp.y, rp.z));
                    }
                }

                Bubbles::new(exterior_spacetime)
            }
        }
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

impl GenerateBubblesExterior for SphericalLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        match boundary_condition {
            BoundaryConditions::Periodic => {
                // Return empty as requested
                Bubbles::new(Vec::new())
            }

            BoundaryConditions::Reflection => {
                let interior = bubbles_interior.borrow();
                let mut exterior_spacetime = Vec::with_capacity(interior.spacetime.len());

                let center = self.center;
                let radius = self.radius;

                for &event in &interior.spacetime {
                    let t = event[0];
                    let p = nalgebra::Point3::new(event[1], event[2], event[3]);
                    let v = p - center; // Vector3
                    let d = v.norm();

                    let p_exterior = if d < f64::EPSILON {
                        // At center: reflect along x-axis
                        nalgebra::Point3::from(
                            center.coords + nalgebra::Vector3::x() * radius * 2.0,
                        )
                    } else {
                        // Mirror across surface: p' = c + (2*r/d - 1) * (p - c)
                        let scale = 2.0 * radius / d - 1.0;
                        nalgebra::Point3::from(center.coords + v * scale)
                    };

                    exterior_spacetime.push(nalgebra::Vector4::new(
                        t,
                        p_exterior.x,
                        p_exterior.y,
                        p_exterior.z,
                    ));
                }

                Bubbles::new(exterior_spacetime)
            }
        }
    }
}

// Generate no exterior bubbles as we have no information about the lattice
impl GenerateBubblesExterior for EmptyLattice {
    fn generate_bubbles_exterior(
        &self,
        _bubbles_interior: impl Borrow<Bubbles>,
        _boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        Bubbles::new(Vec::new())
    }
}

impl GenerateBubblesExterior for BuiltInLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        let bubbles = match self {
            BuiltInLattice::Parallelepiped(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            }
            BuiltInLattice::Cartesian(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            }
            BuiltInLattice::Spherical(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            }
            BuiltInLattice::Empty(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            }
        };
        bubbles
    }
}

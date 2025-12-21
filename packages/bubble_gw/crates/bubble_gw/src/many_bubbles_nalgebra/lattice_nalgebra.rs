use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Isometry3, Point3, Vector3};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LatticeError {
    #[error("edges are linearly dependent (zero volume)")]
    DegenerateEdges,

    #[error("edges are not pairwise orthogonal")]
    NonOrthogonalEdges,
}

pub trait LatticeTransform: Clone + Sync {
    /// Applies an isometry, returning a new lattice.
    fn transform(&self, iso: &Isometry3<f64>) -> Self;

    /// Applies an isometry in-place.
    fn transform_mut(&mut self, iso: &Isometry3<f64>);
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

impl LatticeTransform for EmptyLattice {
    fn transform(&self, _iso: &Isometry3<f64>) -> Self {
        EmptyLattice {}
    }
    fn transform_mut(&mut self, _iso: &Isometry3<f64>) {}
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParallelepipedLattice {
    pub origin: Point3<f64>,
    pub edges: [Vector3<f64>; 3],
}

impl AbsDiffEq for ParallelepipedLattice {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.origin.abs_diff_eq(&other.origin, epsilon)
            && self.edges[0].abs_diff_eq(&other.edges[0], epsilon)
            && self.edges[1].abs_diff_eq(&other.edges[1], epsilon)
            && self.edges[2].abs_diff_eq(&other.edges[2], epsilon)
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
            && self.edges[0].relative_eq(&other.edges[0], epsilon, max_relative)
            && self.edges[1].relative_eq(&other.edges[1], epsilon, max_relative)
            && self.edges[2].relative_eq(&other.edges[2], epsilon, max_relative)
    }
}

impl ParallelepipedLattice {
    /// Creates a parallelepiped from origin and edge vectors.
    ///
    /// Checks that the three edge vectors are linearly independent (non-zero volume).
    /// Panics if invalid (in debug builds; release may skip for performance).
    ///
    /// For fallible version, use `try_new`.
    pub fn new(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Self {
        match Self::try_new(origin, edges) {
            Ok(l) => l,
            Err(e) => panic!("Invalid ParallelepipedLattice: {}", e),
        }
    }

    /// Fallible constructor: returns `Err` if edges are linearly dependent.
    pub fn try_new(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Result<Self, LatticeError> {
        let mat = nalgebra::Matrix3::from_columns(&edges);
        if mat.determinant().abs() <= f64::EPSILON {
            Err(LatticeError::DegenerateEdges)
        } else {
            Ok(Self { origin, edges })
        }
    }

    /// Creates without validation — for performance-critical code after verification.
    pub fn new_unchecked(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Self {
        Self { origin, edges }
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

    /// Returns normalized edge directions.
    pub fn basis_vectors(&self) -> [Vector3<f64>; 3] {
        self.edges.map(|e| {
            let n2 = e.norm_squared();
            if n2 > f64::EPSILON {
                e / n2.sqrt()
            } else {
                Vector3::zeros()
            }
        })
    }
}

impl LatticeTransform for ParallelepipedLattice {
    fn transform(&self, iso: &Isometry3<f64>) -> Self {
        Self {
            origin: iso * self.origin,
            edges: [
                iso.rotation * self.edges[0],
                iso.rotation * self.edges[1],
                iso.rotation * self.edges[2],
            ],
        }
    }

    fn transform_mut(&mut self, iso: &Isometry3<f64>) {
        self.origin = iso * self.origin;
        for edge in &mut self.edges {
            *edge = iso.rotation * *edge;
        }
    }
}

impl LatticeGeometry for ParallelepipedLattice {
    fn volume(&self) -> f64 {
        let mat = nalgebra::Matrix3::from_columns(&self.edges);
        mat.determinant().abs()
    }

    fn reference_point(&self) -> Point3<f64> {
        self.origin
    }

    fn parameters(&self) -> Vec<f64> {
        let o = self.origin.coords;
        let [e1, e2, e3] = self.edges;
        vec![
            o.x, o.y, o.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z,
        ]
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        let mat = nalgebra::Matrix3::from_columns(&self.edges);
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

/// An oriented Cartesian box (i.e., a rectangular parallelepiped with orthogonal edges).
///
/// While stored as a `ParallelepipedLattice`, this type implies orthogonality of edges,
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
    /// Creates a Cartesian lattice (orthogonal edges) from origin and edges.
    ///
    /// Checks orthogonality and non-degeneracy.
    /// Panics on failure (debug-friendly).
    pub fn new(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Self {
        match Self::try_new(origin, edges) {
            Ok(l) => l,
            Err(e) => panic!("Invalid CartesianLattice: {}", e),
        }
    }

    /// Fallible constructor.
    pub fn try_new(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Result<Self, LatticeError> {
        // First check linear independence (via parallelepiped)
        ParallelepipedLattice::try_new(origin, edges)?;

        // Then check orthogonality: e_i · e_j ≈ 0 for i ≠ j
        let [e1, e2, e3] = edges;
        let eps = 1e-10; // tolerance for floating-point orthogonality

        if (e1.dot(&e2)).abs() > eps || (e1.dot(&e3)).abs() > eps || (e2.dot(&e3)).abs() > eps {
            return Err(LatticeError::NonOrthogonalEdges);
        }

        Ok(Self(ParallelepipedLattice { origin, edges }))
    }

    /// Fast constructor without checks.
    pub fn new_unchecked(origin: Point3<f64>, edges: [Vector3<f64>; 3]) -> Self {
        Self(ParallelepipedLattice::new_unchecked(origin, edges))
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
        self.0.edges.map(|e| e.norm())
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

impl LatticeTransform for CartesianLattice {
    fn transform(&self, iso: &Isometry3<f64>) -> Self {
        Self(self.0.transform(iso))
    }

    fn transform_mut(&mut self, iso: &Isometry3<f64>) {
        self.0.transform_mut(iso)
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

impl LatticeTransform for SphericalLattice {
    fn transform(&self, iso: &Isometry3<f64>) -> Self {
        Self {
            center: iso * self.center,
            radius: self.radius,
        }
    }

    fn transform_mut(&mut self, iso: &Isometry3<f64>) {
        self.center = iso * self.center;
        // radius unchanged
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

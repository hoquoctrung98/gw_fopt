use ndarray::{Array2, ArrayRef2};
use std::collections::HashSet;

/// A helper struct for hashing quantized 3D coordinates to ensure uniqueness.
///
/// This struct converts floating-point coordinates to integers by scaling and rounding,
/// enabling efficient storage and comparison in a `HashSet`. It is used primarily for
/// handling periodic boundary conditions in Cartesian lattices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedPoint(i64, i64, i64);

impl QuantizedPoint {
    /// Creates a new `QuantizedPoint` from floating-point coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - A tuple of three `f64` values representing the (x, y, z) coordinates.
    ///
    /// # Returns
    ///
    /// A `QuantizedPoint` with coordinates scaled by \(10^{10}\) and rounded to integers.
    fn new(coords: (f64, f64, f64)) -> Self {
        let quantize = |x: f64| (x * 1e10).round() as i64;
        QuantizedPoint(quantize(coords.0), quantize(coords.1), quantize(coords.2))
    }
}

impl std::hash::Hash for QuantizedPoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
        self.2.hash(state);
    }
}

/// Enum representing the type of lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LatticeType {
    Cartesian { sizes: [f64; 3] },
    Sphere { radius: f64 },
}

/// Represents the simulation domain, either a Cartesian box or a sphere.
///
/// The lattice defines the spatial boundaries and grid resolution for the bubble
/// formation simulation. It supports volume calculations and grid point generation.
#[derive(Debug, Clone, Copy)]
pub struct Lattice {
    pub lattice_type: LatticeType,
    pub n_grid: usize,
}

impl Lattice {
    /// Creates a new lattice with the specified type, sizes, and grid resolution.
    ///
    /// # Arguments
    ///
    /// * `lattice_type` - ("Cartesian" or "Sphere").
    /// * `sizes` - A vector of dimensions: `[lx, ly, lz]` for Cartesian, `[r]` for Sphere.
    /// * `n` - The number of grid points along each dimension.
    ///
    /// # Returns
    ///
    /// * `Ok(Lattice)` - A new `Lattice` instance.
    /// * `Err(String)` - An error message if the lattice type is invalid or sizes are incorrect.
    pub fn new(lattice_type: LatticeType, n_grid: usize) -> Result<Self, String> {
        Ok(Lattice {
            lattice_type,
            n_grid,
        })
    }

    /// Computes the total volume of the lattice.
    ///
    /// # Returns
    ///
    /// * For Cartesian: \( lx \times ly \times lz \).
    /// * For Sphere: \( \frac{4}{3} \pi r^3 \).
    pub fn volume(&self) -> f64 {
        match self.lattice_type {
            LatticeType::Cartesian { sizes } => sizes[0] * sizes[1] * sizes[2],
            LatticeType::Sphere { radius } => (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3),
        }
    }

    /// Generates a grid of points within the lattice.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` with shape `(N, 3)`, where each row is a point `[x, y, z]`.
    /// * For Cartesian: A uniform \( n \times n \times n \) grid spanning `[0, lx] × [0, ly] × [0, lz]`.
    /// * For Sphere: Points within a cube \([-r, r]^3\), filtered to lie within the sphere (\( x^2 + y^2 + z^2 \leq r^2 \)).
    pub fn generate_grid(&self) -> Array2<f64> {
        match self.lattice_type {
            LatticeType::Cartesian { sizes } => {
                let lx = sizes[0];
                let ly = sizes[1];
                let lz = sizes[2];
                let n = self.n_grid;
                let x: Vec<f64> = (0..n).map(|i| i as f64 * lx / (n - 1) as f64).collect();
                let y: Vec<f64> = (0..n).map(|i| i as f64 * ly / (n - 1) as f64).collect();
                let z: Vec<f64> = (0..n).map(|i| i as f64 * lz / (n - 1) as f64).collect();
                let mut grid_points = Vec::with_capacity(n * n * n * 3);
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            grid_points.extend_from_slice(&[x[i], y[j], z[k]]);
                        }
                    }
                }
                Array2::from_shape_vec((n * n * n, 3), grid_points).unwrap()
            }
            LatticeType::Sphere { radius } => {
                let n_grid = self.n_grid;
                let x: Vec<f64> = (0..n_grid)
                    .map(|i| -radius + 2.0 * i as f64 * radius / (n_grid - 1) as f64)
                    .collect();
                let mut grid_points = Vec::new();
                for i in 0..n_grid {
                    for j in 0..n_grid {
                        for k in 0..n_grid {
                            let point = [x[i], x[j], x[k]];
                            if point.iter().map(|&v| v * v).sum::<f64>() <= radius * radius {
                                grid_points.extend_from_slice(&[x[i], x[j], x[k]]);
                            }
                        }
                    }
                }
                Array2::from_shape_vec((grid_points.len() / 3, 3), grid_points).unwrap()
            }
        }
    }

    /// Returns the bounds of the lattice for each dimension.
    ///
    /// # Returns
    ///
    /// A vector of tuples `(min, max)` for each dimension:
    /// * For Cartesian: `[(0, lx), (0, ly), (0, lz)]`.
    /// * For Sphere: `[(-r, r), (-r, r), (-r, r)]`.
    pub fn lattice_bounds(&self) -> Vec<(f64, f64)> {
        match self.lattice_type {
            LatticeType::Cartesian { sizes } => {
                vec![(0.0, sizes[0]), (0.0, sizes[1]), (0.0, sizes[2])]
            }
            LatticeType::Sphere { radius } => vec![(0.0, radius)],
        }
    }
}

/// Enum representing boundary conditions for the simulation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditions {
    Periodic,
    Reflection,
}

/// Generates exterior (image) bubbles for handling boundary conditions.
///
/// For **Periodic**: creates 6 translated copies (±lx, ±ly, ±lz).
/// For **Reflective**: creates 6 mirrored copies across each face via refletions
///
/// Only works for `LatticeType::Cartesian`. Returns empty array otherwise.
///
/// # Arguments
///
/// * `lattice` - The simulation lattice.
/// * `bubbles_interior` - Interior bubbles as `[time, x, y, z]` rows.
/// * `boundary_condition` - Type of boundary condition to apply.
///
/// # Returns
///
/// `Array2<f64>` of shape `(M, 4)`: `[time, x, y, z]` for exterior bubbles.
pub fn generate_bubbles_exterior(
    lattice: &Lattice,
    bubbles_interior: &ArrayRef2<f64>,
    boundary_condition: BoundaryConditions,
) -> Array2<f64> {
    if bubbles_interior.is_empty() {
        return Array2::zeros((0, 4));
    }

    match lattice.lattice_type {
        LatticeType::Cartesian { sizes } => {
            let (lx, ly, lz) = (sizes[0], sizes[1], sizes[2]);
            let mut exterior = Vec::new();
            let mut seen = HashSet::new();

            match boundary_condition {
                BoundaryConditions::Periodic => {
                    let shifts = [
                        [lx, 0.0, 0.0],
                        [-lx, 0.0, 0.0],
                        [0.0, ly, 0.0],
                        [0.0, -ly, 0.0],
                        [0.0, 0.0, lz],
                        [0.0, 0.0, -lz],
                    ];

                    for bubble in bubbles_interior.outer_iter() {
                        let t = bubble[0];
                        let [x, y, z] = [bubble[1], bubble[2], bubble[3]];

                        for &[dx, dy, dz] in &shifts {
                            let p = QuantizedPoint::new((x + dx, y + dy, z + dz));
                            if seen.insert(p) {
                                exterior.extend_from_slice(&[t, x + dx, y + dy, z + dz]);
                            }
                        }
                    }
                }

                BoundaryConditions::Reflection => {
                    // 6 faces → 6 independent reflections
                    //  face translation flipped coordinate
                    let faces: [([f64; 3], usize); 6] = [
                        ([0.0, 0.0, 0.0], 0),      // x = 0  →  -x
                        ([2.0 * lx, 0.0, 0.0], 0), // x = lx → 2*lx-x
                        ([0.0, 0.0, 0.0], 1),      // y = 0  →  -y
                        ([0.0, 2.0 * ly, 0.0], 1), // y = ly → 2*ly-y
                        ([0.0, 0.0, 0.0], 2),      // z = 0  →  -z
                        ([0.0, 0.0, 2.0 * lz], 2), // z = lz → 2*lz-z
                    ];

                    for bubble in bubbles_interior.outer_iter() {
                        let t = bubble[0];
                        let mut center = [bubble[1], bubble[2], bubble[3]];

                        for &(trans, axis) in &faces {
                            // flip only the chosen axis
                            center[axis] = -center[axis];
                            let rx = center[0] + trans[0];
                            let ry = center[1] + trans[1];
                            let rz = center[2] + trans[2];

                            let p = QuantizedPoint::new((rx, ry, rz));
                            if seen.insert(p) {
                                exterior.extend_from_slice(&[t, rx, ry, rz]);
                            }

                            // restore original coordinate for the next face
                            center[axis] = -center[axis];
                        }
                    }
                }
            }

            if exterior.is_empty() {
                Array2::zeros((0, 4))
            } else {
                Array2::from_shape_vec((exterior.len() / 4, 4), exterior).unwrap()
            }
        }

        //TODO: implement exterior bubbles generation for the Sphere lattice
        LatticeType::Sphere { .. } => return Array2::zeros((0, 4)),
    }
}

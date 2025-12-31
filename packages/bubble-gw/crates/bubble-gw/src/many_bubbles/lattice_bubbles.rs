use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;
use crate::many_bubbles::lattice::{
    GenerateBubblesExterior, LatticeGeometry, TransformationIsometry3,
};
use csv::{ReaderBuilder, Writer};
use nalgebra::Vector3;
use nalgebra::{DMatrix, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::prelude::*;
use ndarray_csv::{Array2Reader, Array2Writer, ReadError};
use std::path::Path;
use thiserror::Error;

/// Represents a bubble index, distinguishing between an interior index, exterior index, and no collision.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BubbleIndex {
    Interior(usize),
    Exterior(usize),
    None,
}

/// Custom error type for `BulkFlow` operations.
#[derive(Error, Debug)]
pub enum LatticeBubblesError {
    #[error("Array shape mismatch: {0}")]
    ArrayShapeMismatch(String),
    #[error("Bubble {a} is formed inside bubble {b} at initial time (overlapping light cones)")]
    BubbleFormedInsideBubble { a: BubbleIndex, b: BubbleIndex },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("deserialize_array2 error")]
    DeserializeArray2(#[from] ReadError),

    #[error("serialize_array2 error")]
    SerializeArray2(#[from] csv::Error),

    #[error("Parse float error at {path}:{line}: '{value}'")]
    ParseFloat {
        path: String,
        line: usize,
        value: String,
    },

    #[error("Invalid number of columns")]
    InvalidNCols,

    #[error("Empty bubble file: {0}")]
    EmptyFile(String),

    #[error("Interior bubbles formed outside lattice: {indices:?}")]
    InteriorBubblesOutsideLattice { indices: Vec<BubbleIndex> },

    #[error("Exterior bubbles formed inside lattice: {indices:?}")]
    ExteriorBubblesInsideLattice { indices: Vec<BubbleIndex> },
}

// TODO: convert the input arguments to type Bubbles
// Checks if any bubble is contained within another at the initial time.
pub fn check_bubble_formed_inside_bubble(
    delta_squared: &DMatrix<f64>,
) -> Result<(), LatticeBubblesError> {
    let (n_interior, n_total) = delta_squared.shape();
    let n_exterior = n_total - n_interior;

    // Interior-Interior
    for a_idx in 0..n_interior {
        for b_idx in a_idx + 1..n_interior {
            if delta_squared[(a_idx, b_idx)] < 0.0 {
                return Err(LatticeBubblesError::BubbleFormedInsideBubble {
                    a: BubbleIndex::Interior(a_idx),
                    b: BubbleIndex::Interior(b_idx),
                });
            }
        }
    }

    // Interior-Exterior
    for a_idx in 0..n_interior {
        for b_ex in 0..n_exterior {
            let b_total = n_interior + b_ex;
            if delta_squared[(a_idx, b_total)] < 0.0 {
                return Err(LatticeBubblesError::BubbleFormedInsideBubble {
                    a: BubbleIndex::Interior(a_idx),
                    b: BubbleIndex::Exterior(b_ex),
                });
            }
        }
    }

    // FIXME: temporarily ignore so that the periodic boundary condition works
    // // Exterior-Exterior
    // for a_ex in 0..n_exterior {
    //     for b_ex in (a_ex + 1)..n_exterior {
    //         let delta_ba = bubbles_exterior.slice(s![b_ex, ..]).to_owned()
    //             - bubbles_exterior.slice(s![a_ex, ..]).to_owned();
    //         let delta_ba_squared = dot_minkowski_vec(delta_ba.view(), delta_ba.view());
    //         if delta_ba_squared < 0.0 {
    //             return Err(BubblesError::BubbleFormedInsideBubble {
    //                 a: BubbleIndex::Exterior(a_ex),
    //                 b: BubbleIndex::Exterior(b_ex),
    //             });
    //         }
    //     }
    // }

    Ok(())
}

// Helper Display implementation for BubbleIndex used in the error message
impl std::fmt::Display for BubbleIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BubbleIndex::Interior(i) => write!(f, "Interior({i})"),
            BubbleIndex::Exterior(i) => write!(f, "Exterior({i})"),
            BubbleIndex::None => write!(f, "None"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LatticeBubbles<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    pub interior: Bubbles,
    pub exterior: Bubbles,
    pub lattice: L,
    pub delta: DMatrix<Vector4<f64>>,
    pub delta_squared: DMatrix<f64>,
}

impl<L> TransformationIsometry3 for LatticeBubbles<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    fn transform<I: Into<nalgebra::Isometry3<f64>>>(&self, iso: I) -> Self {
        let iso = iso.into();
        let lattice = self.lattice.transform(iso);
        let interior = self.interior.transform(iso);
        let exterior = self.exterior.transform(iso);

        // Transform delta using rotation only
        let mut delta = self.delta.clone();
        let rot = iso.rotation;
        for da in delta.iter_mut() {
            let spatial = Vector3::new(da[1], da[2], da[3]);
            let rotated = rot * spatial;
            da[1] = rotated.x;
            da[2] = rotated.y;
            da[3] = rotated.z;
        }

        Self {
            interior,
            exterior,
            lattice,
            delta,
            delta_squared: self.delta_squared.clone(),
        }
    }

    fn transform_mut<I: Into<nalgebra::Isometry3<f64>>>(&mut self, iso: I) {
        let iso = iso.into();
        // Transform lattice
        self.lattice.transform_mut(iso);

        // Transform bubbles
        self.interior.transform_mut(iso);
        self.exterior.transform_mut(iso);

        // Update delta: only spatial rotation affects differences
        // (translation cancels, time unchanged)
        let rot = iso.rotation;
        for da in self.delta.iter_mut() {
            let spatial = Vector3::new(da[1], da[2], da[3]);
            let rotated = rot * spatial;
            da[1] = rotated.x;
            da[2] = rotated.y;
            da[3] = rotated.z;
            // da[0] unchanged
        }

        // delta_squared remains invariant under spatial isometries → no change
    }
}

// FIXME: the implementation of sort_by_time=true might be incorrect
impl<L> LatticeBubbles<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    /// Creates a new `LatticeBubbles` with an empty set of interior and exterior bubbles.
    /// `delta` and `delta_squared` are 0×0 matrices.
    /// Use `set_bubbles` or `with_bubbles` to populate data later.
    pub fn new(lattice: L) -> Self {
        let empty_spacetime = Vec::new();
        LatticeBubbles {
            interior: Bubbles::new(empty_spacetime.clone()),
            exterior: Bubbles::new(empty_spacetime),
            lattice,
            delta: DMatrix::from_element(0, 0, Vector4::zeros()),
            delta_squared: DMatrix::zeros(0, 0),
        }
    }

    /// Constructs a new `LatticeBubbles` by validating and processing given interior and exterior bubbles.
    /// Checks:
    /// - Shape: both arrays must be `(n, 4)` → `[t, x, y, z]`
    /// - Lattice containment: interior ⊆ lattice, exterior ∩ lattice = ∅
    /// - Causality: no bubble formed inside another’s past lightcone
    /// - (Optionally) sorts bubbles by nucleation time `t` (column 0)
    /// Precomputes pairwise spacetime intervals `delta` and Minkowski norms `delta_squared`.
    pub fn with_bubbles(
        mut bubbles_interior: Array2<f64>,
        mut bubbles_exterior: Array2<f64>,
        lattice: L,
        sort_by_time: bool,
    ) -> Result<LatticeBubbles<L>, LatticeBubblesError> {
        if bubbles_interior.ncols() != 4 || bubbles_exterior.ncols() != 4 {
            return Err(LatticeBubblesError::ArrayShapeMismatch(format!(
                "Expected 4 columns, got {} for interior, {} for exterior",
                bubbles_interior.ncols(),
                bubbles_exterior.ncols()
            )));
        }

        // Lattice containment
        let interior_points: Vec<nalgebra::Point3<f64>> = (0..bubbles_interior.nrows())
            .map(|i| {
                let row = bubbles_interior.row(i);
                nalgebra::Point3::new(row[1], row[2], row[3])
            })
            .collect();
        let exterior_points: Vec<nalgebra::Point3<f64>> = (0..bubbles_exterior.nrows())
            .map(|i| {
                let row = bubbles_exterior.row(i);
                nalgebra::Point3::new(row[1], row[2], row[3])
            })
            .collect();

        let interior_contained = lattice.contains(&interior_points);
        let exterior_contained = lattice.contains(&exterior_points);

        let outside_interior_indices: Vec<BubbleIndex> = interior_contained
            .into_iter()
            .enumerate()
            .filter_map(|(i, inside)| (!inside).then_some(BubbleIndex::Interior(i)))
            .collect();

        let inside_exterior_indices: Vec<BubbleIndex> = exterior_contained
            .into_iter()
            .enumerate()
            .filter_map(|(i, inside)| inside.then_some(BubbleIndex::Exterior(i)))
            .collect();

        if !outside_interior_indices.is_empty() {
            return Err(LatticeBubblesError::InteriorBubblesOutsideLattice {
                indices: outside_interior_indices,
            });
        }
        if !inside_exterior_indices.is_empty() {
            return Err(LatticeBubblesError::ExteriorBubblesInsideLattice {
                indices: inside_exterior_indices,
            });
        }

        // Sort by formation time (column 0: t)
        if sort_by_time {
            if !bubbles_interior.is_empty() {
                let mut rows: Vec<Array1<f64>> = bubbles_interior
                    .rows()
                    .into_iter()
                    .map(|r| r.to_owned())
                    .collect();
                rows.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
                for (i, row) in rows.into_iter().enumerate() {
                    bubbles_interior.row_mut(i).assign(&row);
                }
            }
            if !bubbles_exterior.is_empty() {
                let mut rows: Vec<Array1<f64>> = bubbles_exterior
                    .rows()
                    .into_iter()
                    .map(|r| r.to_owned())
                    .collect();
                rows.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
                for (i, row) in rows.into_iter().enumerate() {
                    bubbles_exterior.row_mut(i).assign(&row);
                }
            }
        }

        let n_interior = bubbles_interior.nrows();
        let n_exterior = bubbles_exterior.nrows();
        let n_total = n_interior + n_exterior;

        // Convert via Bubbles::from_array2 (safe, asserts ncols==4)
        let interior = Bubbles::from_array2(bubbles_interior);
        let exterior = Bubbles::from_array2(bubbles_exterior);

        // Precompute delta matrices
        let mut delta = DMatrix::from_element(n_interior, n_total, Vector4::zeros());
        let mut delta_squared = DMatrix::zeros(n_interior, n_total);

        let int_vecs = &interior.spacetime;
        let ext_vecs = &exterior.spacetime;

        for a_idx in 0..n_interior {
            // Interior–Interior (symmetric)
            for b_idx in a_idx..n_interior {
                let da = &int_vecs[b_idx] - &int_vecs[a_idx];
                let dsq = da.scalar(&da);
                delta[(a_idx, b_idx)] = da;
                delta_squared[(a_idx, b_idx)] = dsq;
                delta[(b_idx, a_idx)] = -da;
                delta_squared[(b_idx, a_idx)] = dsq;
            }
            // Interior–Exterior
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                let da = &ext_vecs[b_ex] - &int_vecs[a_idx];
                delta[(a_idx, b_total)] = da;
                delta_squared[(a_idx, b_total)] = da.scalar(&da);
            }
        }

        check_bubble_formed_inside_bubble(&delta_squared)?;

        Ok(LatticeBubbles {
            interior,
            exterior,
            lattice,
            delta,
            delta_squared,
        })
    }

    /// Replaces the current interior and exterior bubbles in-place.
    /// Performs the same validation and precomputation as `with_bubbles`.
    /// Returns error if validation fails; leaves self unchanged on error.
    pub fn set_bubbles(
        &mut self,
        bubbles_interior: Array2<f64>,
        bubbles_exterior: Array2<f64>,
        sort_by_time: bool,
    ) -> Result<(), LatticeBubblesError> {
        let new_self = Self::with_bubbles(
            bubbles_interior,
            bubbles_exterior,
            self.lattice.clone(),
            sort_by_time,
        )?;
        *self = new_self;
        Ok(())
    }

    pub fn with_boundary_condition(&mut self, boundary_condition: BoundaryConditions) {
        self.exterior = self
            .lattice
            .generate_bubbles_exterior(&self.interior, boundary_condition);

        // Update delta and delta_squared for the new exterior bubbles
        let n_interior = self.interior.n_bubbles();
        let n_exterior = self.exterior.n_bubbles();
        let n_total_new = n_interior + n_exterior;

        // Create new delta and delta_squared with updated column count
        let mut delta = DMatrix::from_element(n_interior, n_total_new, Vector4::zeros());
        let mut delta_squared = DMatrix::zeros(n_interior, n_total_new);

        // // Copy old interior–interior block (upper-left n_interior × n_interior)
        // // Since interior didn't change, this block is still valid
        // for a in 0..n_interior {
        //     for b in 0..n_interior {
        //         delta[(a, b)] = self.delta[(a, b)];
        //         delta_squared[(a, b)] = self.delta_squared[(a, b)];
        //     }
        // }
        //
        // // Recompute interior–exterior block: for each interior a, exterior e
        // let int_vecs = &self.interior.spacetime;
        // let ext_vecs = &self.exterior.spacetime;
        //
        // for a in 0..n_interior {
        //     for e in 0..n_exterior {
        //         let b_total = n_interior + e;
        //         let da = &ext_vecs[e] - &int_vecs[a]; // Vector4
        //         delta[(a, b_total)] = da;
        //         delta_squared[(a, b_total)] = da.scalar(&da);
        //     }
        // }
        // Pre-extract bubble positions as slices of Vector4 for fast access
        let int_vecs = &self.interior.spacetime;
        let ext_vecs = &self.exterior.spacetime;

        // Compute delta and delta_squared
        for a_idx in 0..n_interior {
            // Interior → Interior (a_idx to b_idx)
            for b_idx in a_idx..n_interior {
                let da = &int_vecs[b_idx] - &int_vecs[a_idx]; // Vector4 - Vector4 → Vector4
                delta[(a_idx, b_idx)] = da;
                let dsq = da.scalar(&da);
                delta_squared[(a_idx, b_idx)] = dsq;
                // Symmetry
                delta[(b_idx, a_idx)] = -da;
                delta_squared[(b_idx, a_idx)] = dsq;
            }

            // Interior → Exterior
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                let da = &ext_vecs[b_ex] - &int_vecs[a_idx];
                delta[(a_idx, b_total)] = da;
                delta_squared[(a_idx, b_total)] = da.scalar(&da);
            }
        }

        // Update fields
        self.delta = delta;
        self.delta_squared = delta_squared;
    }

    /// Get the distances between the bubble centers and the origin of the lattice
    pub fn get_distance_to_origin(&self) -> Vec<f64> {
        todo!()
    }

    /// Create a new `Bubbles` instance by reading interior and exterior bubbles from CSV files.
    ///
    /// Each CSV file must have exactly 4 columns: `t,x,y,z` (formation time first).
    /// No header row is required or expected.
    ///
    /// # Arguments
    /// * `interior_path` - Path to CSV with real (interior) bubbles
    /// * `exterior_path` - Path to CSV with periodic image (exterior) bubbles
    /// * `sort_by_time`  - Whether to sort bubbles by formation time after loading
    ///
    /// # Returns
    /// `Ok(Bubbles)` on success, or a descriptive error
    pub fn from_csv_files<P: AsRef<Path>>(
        interior_path: P,
        exterior_path: P,
        lattice: L,
        sort_by_time: bool,
        has_headers: bool,
    ) -> Result<Self, LatticeBubblesError> {
        let interior = Self::load_bubbles_from_csv(interior_path, has_headers)?;
        let exterior = Self::load_bubbles_from_csv(exterior_path, has_headers)?;

        Ok(Self::with_bubbles(interior, exterior, lattice, sort_by_time)?)
    }

    /// Load bubbles from a CSV file.
    ///
    /// # Arguments
    /// - `path`: Path to CSV file
    /// - `has_headers`: Set to `true` if the file has a header row (will be skipped)
    ///
    /// # Returns
    /// `Array2<f64>` with shape `(n_bubbles, 4)` → [t, x, y, z]
    fn load_bubbles_from_csv<P: AsRef<Path>>(
        path: P,
        has_headers: bool,
    ) -> Result<Array2<f64>, LatticeBubblesError> {
        let path = path.as_ref();
        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(false)
            .from_path(path)?;

        let array: Array2<f64> = reader
            .deserialize_array2_dynamic()
            .map_err(LatticeBubblesError::DeserializeArray2)?;

        if array.ncols() != 4 {
            return Err(LatticeBubblesError::InvalidNCols);
        }
        Ok(array)
    }

    /// Write bubbles to CSV with scientific notation e.8 and optional header
    fn write_bubbles_to_csv<P: AsRef<Path>>(
        bubbles: &Array2<f64>,
        path: P,
        has_headers: bool,
    ) -> Result<(), LatticeBubblesError> {
        let mut writer = Writer::from_path(path)?;
        if has_headers {
            writer.write_record(&["t", "x", "y", "z"])?;
        }

        writer.serialize_array2(&bubbles)?;
        Ok(())
    }

    /// Save interior bubbles to CSV
    pub fn write_bubbles_interior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), LatticeBubblesError> {
        Self::write_bubbles_to_csv(&self.interior.to_array2(), path, has_headers)
    }

    /// Save exterior bubbles to CSV
    pub fn save_bubbles_exterior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), LatticeBubblesError> {
        Self::write_bubbles_to_csv(&self.exterior.to_array2(), path, has_headers)
    }

    /// Save both interior and exterior with same header setting
    pub fn save_to_csv_files<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        interior_path: P,
        exterior_path: Q,
        has_headers: bool,
    ) -> Result<(), LatticeBubblesError> {
        self.write_bubbles_interior_to_csv(&interior_path, has_headers)?;
        self.save_bubbles_exterior_to_csv(&exterior_path, has_headers)?;
        Ok(())
    }
}

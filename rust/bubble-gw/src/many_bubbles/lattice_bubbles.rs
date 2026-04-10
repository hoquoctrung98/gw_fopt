use std::path::Path;

use csv::{ReaderBuilder, Writer};
use nalgebra::{DMatrix, Vector3, Vector4};
use ndarray::prelude::*;
use ndarray_csv::{Array2Reader, Array2Writer, ReadError};
use thiserror::Error;

use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    BoundaryConditions,
    GeneralLatticeProperties,
    TransformationIsometry3,
};
use crate::many_bubbles::spacetime::Lorentzian;

/// Represents a bubble index, distinguishing between an interior index,
/// exterior index, and no collision.
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

    #[error("Nucleation error: {0}")]
    NucleationError(String),

    #[error("Other error: {0}")]
    Other(String),
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
    //         let delta_ba_squared = dot_minkowski_vec(delta_ba.view(),
    // delta_ba.view());         if delta_ba_squared < 0.0 {
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
    L: GeneralLatticeProperties,
{
    pub interior: Bubbles,
    pub exterior: Bubbles,
    pub lattice: L,
    pub delta: DMatrix<Vector4<f64>>,
    pub delta_squared: DMatrix<f64>,
}

impl<L> TransformationIsometry3 for LatticeBubbles<L>
where
    L: GeneralLatticeProperties + Clone,
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

impl<L> LatticeBubbles<L>
where
    L: GeneralLatticeProperties + Clone,
{
    /// Constructs a new `LatticeBubbles` by validating and processing given
    /// interior and exterior bubbles. Checks:
    /// - Shape: both arrays must be `(n, 4)` → `[t, x, y, z]`
    /// - Lattice containment: interior ⊆ lattice, exterior ∩ lattice = ∅
    /// - Causality: no bubble formed inside another’s past lightcone
    /// - (Optionally) sorts bubbles by nucleation time `t` (column 0)
    /// Precomputes pairwise spacetime intervals `delta` and Minkowski norms
    /// `delta_squared`.
    pub fn new(
        bubbles_interior: Array2<f64>,
        bubbles_exterior: Option<Array2<f64>>,
        lattice: L,
    ) -> Result<LatticeBubbles<L>, LatticeBubblesError> {
        let bubbles_exterior = bubbles_exterior.unwrap_or_else(|| Array2::zeros((0, 4)));
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
                let delta_ab = &int_vecs[b_idx] - &int_vecs[a_idx];
                let delta_squared_ab = delta_ab.scalar(&delta_ab);
                delta[(a_idx, b_idx)] = delta_ab;
                delta_squared[(a_idx, b_idx)] = delta_squared_ab;
                delta[(b_idx, a_idx)] = -delta_ab;
                delta_squared[(b_idx, a_idx)] = delta_squared_ab;
            }
            // Interior–Exterior
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                let delta_ab = &ext_vecs[b_ex] - &int_vecs[a_idx];
                delta[(a_idx, b_total)] = delta_ab;
                delta_squared[(a_idx, b_total)] = delta_ab.scalar(&delta_ab);
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
    /// Performs the same validation and precomputation as `new`.
    /// Returns error if validation fails; leaves self unchanged on error.
    pub fn set_bubbles(
        &mut self,
        bubbles_interior: Array2<f64>,
        bubbles_exterior: Option<Array2<f64>>,
    ) -> Result<(), LatticeBubblesError> {
        let new_self = Self::new(bubbles_interior, bubbles_exterior, self.lattice.clone())?;
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

        let interior_spacetime = &self.interior.spacetime;
        let exterior_spacetime = &self.exterior.spacetime;

        // Compute delta and delta_squared
        for a_idx in 0..n_interior {
            // Interior → Interior (a_idx to b_idx)
            for b_idx in a_idx..n_interior {
                let da = &interior_spacetime[b_idx] - &interior_spacetime[a_idx];
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
                let da = &exterior_spacetime[b_ex] - &interior_spacetime[a_idx];
                delta[(a_idx, b_total)] = da;
                delta_squared[(a_idx, b_total)] = da.scalar(&da);
            }
        }

        // Update fields
        self.delta = delta;
        self.delta_squared = delta_squared;
    }

    // =========================================================================
    // Explicit Time Translation (by specified amount)
    // =========================================================================

    /// Performs in-place time translation on all bubbles by a specified amount.
    ///
    /// Adds `t_shift` to the nucleation time (component 0) of every bubble
    /// in both `interior` and `exterior`.
    ///
    /// # Arguments
    /// * `t_shift` - Amount to add to all bubble times (can be negative)
    ///
    /// # Effects
    /// - Modifies the time component of all `Vector4<f64>` entries
    /// - `delta` and `delta_squared` remain **unchanged** (invariant under
    ///   uniform time translation)
    ///
    /// # Example
    /// ```ignore
    /// // Shift all bubbles forward by 10 time units
    /// lattice_bubbles.translate_time_mut(10.0);
    ///
    /// // Shift backward by 2.5 time units
    /// lattice_bubbles.translate_time_mut(-2.5);
    /// ```
    pub fn translate_time_mut(&mut self, t_shift: f64) {
        // Apply shift to interior bubbles
        for bubble in self.interior.spacetime.iter_mut() {
            bubble[0] += t_shift;
        }
        // Apply shift to exterior bubbles
        for bubble in self.exterior.spacetime.iter_mut() {
            bubble[0] += t_shift;
        }
        // delta/delta_squared invariant under uniform time translation
    }

    /// Returns a new `LatticeBubbles` with explicit time translation applied.
    ///
    /// Functional (non-mutating) variant of `translate_time_mut`.
    ///
    /// # Arguments
    /// * `t_shift` - Amount to add to all bubble times
    ///
    /// # Returns
    /// A new instance with translated bubble times; precomputed matrices cloned.
    pub fn translate_time(&self, t_shift: f64) -> Self {
        let mut result = self.clone();
        result.translate_time_mut(t_shift);
        result
    }

    // =========================================================================
    // Auto-Normalization (shift so earliest bubble nucleates at t=0)
    // =========================================================================

    /// Performs in-place time normalization: shifts all bubbles so the earliest
    /// nucleation time becomes 0.
    ///
    /// Computes `shift = -min(interior_times ∪ exterior_times)` and applies
    /// it via `translate_time_mut`.
    ///
    /// # Effects
    /// - After this call, `min(bubble_times) == 0.0` (within floating-point precision)
    /// - If both bubble collections are empty, no change is made
    /// - `delta` and `delta_squared` remain unchanged (invariant)
    ///
    /// # Example
    /// ```ignore
    /// // Normalize so earliest bubble nucleates at t=0
    /// lattice_bubbles.normalize_time_mut();
    /// ```
    pub fn normalize_time_mut(&mut self) {
        // Find minimum time across all bubbles
        let min_interior = self
            .interior
            .spacetime
            .iter()
            .map(|v| v[0])
            .fold(f64::INFINITY, f64::min);
        let min_exterior = self
            .exterior
            .spacetime
            .iter()
            .map(|v| v[0])
            .fold(f64::INFINITY, f64::min);
        let min_time = min_interior.min(min_exterior);

        // Only apply shift if we found valid bubbles
        if min_time.is_finite() {
            self.translate_time_mut(-min_time);
        }
        // If empty: no-op, nothing to normalize
    }

    /// Returns a new `LatticeBubbles` with time normalization applied.
    ///
    /// Functional variant: shifts copy so earliest bubble nucleates at t=0.
    ///
    /// # Returns
    /// A new instance with normalized bubble times.
    pub fn normalize_time(&self) -> Self {
        let mut result = self.clone();
        result.normalize_time_mut();
        result
    }

    /// Create a new `Bubbles` instance by reading interior and exterior bubbles
    /// from CSV files.
    ///
    /// Each CSV file must have exactly 4 columns: `t,x,y,z` (formation time
    /// first). No header row is required or expected.
    ///
    /// # Arguments
    /// * `interior_path` - Path to CSV with real (interior) bubbles
    /// * `exterior_path` - Path to CSV with periodic image (exterior) bubbles
    /// * `sort_by_time`  - Whether to sort bubbles by formation time after
    ///   loading
    ///
    /// # Returns
    /// `Ok(Bubbles)` on success, or a descriptive error
    pub fn from_csv_files<P: AsRef<Path>>(
        interior_path: P,
        exterior_path: P,
        lattice: L,
        has_headers: bool,
    ) -> Result<Self, LatticeBubblesError> {
        let interior = Self::load_bubbles_from_csv(interior_path, has_headers)?;
        let exterior = Self::load_bubbles_from_csv(exterior_path, has_headers)?;

        Ok(Self::new(interior, Some(exterior), lattice)?)
    }

    /// Load bubbles from a CSV file.
    ///
    /// # Arguments
    /// - `path`: Path to CSV file
    /// - `has_headers`: Set to `true` if the file has a header row (will be
    ///   skipped)
    ///
    /// # Returns
    /// `Array2<f64>` with shape `(n_bubbles, 4)` → [t, x, y, z]
    pub fn load_bubbles_from_csv<P: AsRef<Path>>(
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

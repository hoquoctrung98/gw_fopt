use crate::many_bubbles_nalgebra::bubbles::Bubbles;
use crate::many_bubbles_nalgebra::lattice::BoundaryConditions;
use crate::many_bubbles_nalgebra::lattice::{
    GenerateBubblesExterior, LatticeGeometry, TransformationIsometry3,
};
use csv::{ReaderBuilder, Writer};
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
pub enum BubblesError {
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
}

// TODO: convert the input arguments to type Bubbles
// Checks if any bubble is contained within another at the initial time.
pub fn check_bubble_formed_inside_bubble(delta_squared: &DMatrix<f64>) -> Result<(), BubblesError> {
    let (n_interior, n_total) = delta_squared.shape();
    let n_exterior = n_total - n_interior;

    // Interior-Interior
    for a_idx in 0..n_interior {
        for b_idx in a_idx + 1..n_interior {
            if delta_squared[(a_idx, b_idx)] < 0.0 {
                return Err(BubblesError::BubbleFormedInsideBubble {
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
                return Err(BubblesError::BubbleFormedInsideBubble {
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
    // TODO: We need to perform the transformation on both the lattice and the bubbles
    fn transform_mut(&mut self, iso: &nalgebra::Isometry3<f64>) {
        self.lattice.transform_mut(iso);
        todo!()
    }

    // TODO: Avoid unwrap here
    fn transform(&self, iso: &nalgebra::Isometry3<f64>) -> Self {
        let lattice = self.lattice.transform(iso);
        Self::new(self.interior.to_array2(), self.exterior.to_array2(), lattice, false).unwrap()
    }
}

impl<L> LatticeBubbles<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    pub fn new(
        mut bubbles_interior: Array2<f64>,
        mut bubbles_exterior: Array2<f64>,
        lattice: L,
        sort_by_time: bool,
    ) -> Result<LatticeBubbles<L>, BubblesError> {
        // shape validation
        if bubbles_interior.ncols() != 4 || bubbles_exterior.ncols() != 4 {
            return Err(BubblesError::ArrayShapeMismatch(format!(
                "Expected 4 columns, got {} for interior, {} for exterior",
                bubbles_interior.ncols(),
                bubbles_exterior.ncols()
            )));
        }

        // optional sorting by formation time (column 3)
        if sort_by_time {
            // sort interior bubbles
            let mut rows: Vec<(usize, Array1<f64>)> = bubbles_interior
                .rows()
                .into_iter()
                .map(|r| r.to_owned()) // clone each row
                .enumerate()
                .collect();

            rows.sort_by(|a, b| {
                a.1[3]
                    .partial_cmp(&b.1[3])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (i, (_, row)) in rows.into_iter().enumerate() {
                bubbles_interior.row_mut(i).assign(&row);
            }

            // sort exterior bubbles
            if !bubbles_exterior.is_empty() {
                let mut rows: Vec<(usize, Array1<f64>)> = bubbles_exterior
                    .rows()
                    .into_iter()
                    .map(|r| r.to_owned())
                    .enumerate()
                    .collect();

                rows.sort_by(|a, b| {
                    a.1[3]
                        .partial_cmp(&b.1[3])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for (i, (_, row)) in rows.into_iter().enumerate() {
                    bubbles_exterior.row_mut(i).assign(&row);
                }
            }
        }

        let n_interior = bubbles_interior.nrows();
        let n_exterior = bubbles_exterior.nrows();
        let n_total = n_interior + n_exterior;

        // Initialize `delta` and `delta_squared`
        let mut delta = DMatrix::from_element(n_interior, n_total, Vector4::zeros());
        let mut delta_squared = DMatrix::zeros(n_interior, n_total);

        // Convert to Bubbles (still using Array2 as intermediate)
        let bubbles_interior = Bubbles::from_array2(bubbles_interior.clone());
        let bubbles_exterior = Bubbles::from_array2(bubbles_exterior.clone());
        // Pre-extract bubble positions as slices of Vector4 for fast access
        let int_vecs = &bubbles_interior.spacetime;
        let ext_vecs = &bubbles_exterior.spacetime;

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

        // Check for bubble containment
        check_bubble_formed_inside_bubble(&delta_squared)?;

        Ok(LatticeBubbles {
            interior: bubbles_interior,
            exterior: bubbles_exterior,
            lattice,
            delta,
            delta_squared,
        })
    }

    pub fn with_boundary_condition(&mut self, boundary_condition: BoundaryConditions) {
        self.exterior = self
            .lattice
            .generate_bubbles_exterior(&self.interior, boundary_condition);
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
    ) -> Result<Self, BubblesError> {
        let interior = Self::load_bubbles_from_csv(interior_path, has_headers)?;
        let exterior = Self::load_bubbles_from_csv(exterior_path, has_headers)?;

        Ok(Self::new(interior, exterior, lattice, sort_by_time)?)
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
    ) -> Result<Array2<f64>, BubblesError> {
        let path = path.as_ref();
        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(false)
            .from_path(path)?;

        let array: Array2<f64> = reader
            .deserialize_array2_dynamic()
            .map_err(BubblesError::DeserializeArray2)?;

        if array.ncols() != 4 {
            return Err(BubblesError::InvalidNCols);
        }
        Ok(array)
    }

    /// Write bubbles to CSV with scientific notation e.8 and optional header
    fn write_bubbles_to_csv<P: AsRef<Path>>(
        bubbles: &Array2<f64>,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
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
    ) -> Result<(), BubblesError> {
        Self::write_bubbles_to_csv(&self.interior.to_array2(), path, has_headers)
    }

    /// Save exterior bubbles to CSV
    pub fn save_bubbles_exterior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        Self::write_bubbles_to_csv(&self.exterior.to_array2(), path, has_headers)
    }

    /// Save both interior and exterior with same header setting
    pub fn save_to_csv_files<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        interior_path: P,
        exterior_path: Q,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        self.write_bubbles_interior_to_csv(&interior_path, has_headers)?;
        self.save_bubbles_exterior_to_csv(&exterior_path, has_headers)?;
        Ok(())
    }
}

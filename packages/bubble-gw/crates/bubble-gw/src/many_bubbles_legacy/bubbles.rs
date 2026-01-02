use std::path::Path;

use csv::{ReaderBuilder, Writer};
use ndarray::prelude::*;
use ndarray_csv::{Array2Reader, Array2Writer, ReadError};
use thiserror::Error;

#[inline]
pub fn dot_minkowski_vec(v1: &ArrayRef1<f64>, v2: &ArrayRef1<f64>) -> f64 {
    assert!(
        v1.len() == 4 && v2.len() == 4,
        "Error using dot_minkowski_vec: 4-vectors required"
    );
    let mut sum = 0.0;
    unsafe {
        for i in 0..4 {
            let t1 = *v1.uget(i);
            let t2 = *v2.uget(i);
            sum += if i == 0 { -t1 * t2 } else { t1 * t2 };
        }
    }
    sum
}

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

// Checks if any bubble is contained within another at the initial time.
pub fn check_bubble_formed_inside_bubble(
    bubbles_interior: &ArrayRef2<f64>,
    bubbles_exterior: &ArrayRef2<f64>,
    delta_squared: &ArrayRef2<f64>,
) -> Result<(), BubblesError> {
    let n_interior = bubbles_interior.nrows();
    let n_exterior = bubbles_exterior.nrows();

    // Interior-Interior
    for a_idx in 0..n_interior {
        for b_idx in a_idx + 1..n_interior {
            if delta_squared[[a_idx, b_idx]] < 0.0 {
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
            if delta_squared[[a_idx, b_total]] < 0.0 {
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

type Bubbles = Array2<f64>;

#[derive(Clone, Debug, PartialEq)]
pub struct LatticeBubbles {
    pub interior: Bubbles,
    pub exterior: Bubbles,
    pub delta: Array3<f64>,
    pub delta_squared: Array2<f64>,
}

impl LatticeBubbles {
    pub fn new(
        mut bubbles_interior: Bubbles,
        mut bubbles_exterior: Bubbles,
        sort_by_time: bool,
    ) -> Result<LatticeBubbles, BubblesError> {
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
        let mut delta = Array3::zeros((n_interior, n_total, 4));
        let mut delta_squared = Array2::zeros((n_interior, n_total));

        // Compute delta and delta_squared using symmetry for interior-interior
        for a_idx in 0..n_interior {
            // Interior to interior (upper triangular and diagonal)
            for b_idx in a_idx..n_interior {
                let delta_ba = bubbles_interior.slice(s![b_idx, ..]).to_owned()
                    - bubbles_interior.slice(s![a_idx, ..]).to_owned();
                delta.slice_mut(s![a_idx, b_idx, ..]).assign(&delta_ba);
                delta_squared[[a_idx, b_idx]] = dot_minkowski_vec(&delta_ba, &delta_ba);
                // Symmetry: delta[b_idx, a_idx, ..] = -delta[a_idx, b_idx, ..]
                delta.slice_mut(s![b_idx, a_idx, ..]).assign(&(-&delta_ba));
                delta_squared[[b_idx, a_idx]] = delta_squared[[a_idx, b_idx]];
            }
            // Interior to exterior
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                let delta_ba = bubbles_exterior.slice(s![b_ex, ..]).to_owned()
                    - bubbles_interior.slice(s![a_idx, ..]).to_owned();
                delta.slice_mut(s![a_idx, b_total, ..]).assign(&delta_ba);
                delta_squared[[a_idx, b_total]] = dot_minkowski_vec(&delta_ba, &delta_ba);
            }
        }

        // Check for bubble containment
        check_bubble_formed_inside_bubble(&bubbles_interior, &bubbles_exterior, &delta_squared)?;

        Ok(LatticeBubbles {
            interior: bubbles_interior,
            exterior: bubbles_exterior,
            delta,
            delta_squared,
        })
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
        sort_by_time: bool,
        has_headers: bool,
    ) -> Result<Self, BubblesError> {
        let interior = Self::load_bubbles_from_csv(interior_path, has_headers)?;
        let exterior = Self::load_bubbles_from_csv(exterior_path, has_headers)?;

        Ok(Self::new(interior, exterior, sort_by_time)?)
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
    fn load_bubbles_from_csv<P: AsRef<Path>>(
        path: P,
        has_headers: bool,
    ) -> Result<Bubbles, BubblesError> {
        let path = path.as_ref();
        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(false)
            .from_path(path)?;

        let array: Bubbles = reader
            .deserialize_array2_dynamic()
            .map_err(BubblesError::DeserializeArray2)?;

        if array.ncols() != 4 {
            return Err(BubblesError::InvalidNCols);
        }
        Ok(array)
    }

    /// Write bubbles to CSV with scientific notation e.8 and optional header
    fn write_bubbles_to_csv<P: AsRef<Path>>(
        bubbles: &Bubbles,
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
        Self::write_bubbles_to_csv(&self.interior, path, has_headers)
    }

    /// Save exterior bubbles to CSV
    pub fn save_bubbles_exterior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        Self::write_bubbles_to_csv(&self.exterior, path, has_headers)
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

    /// Add new interior bubbles, optionally checking causality only if
    /// requested. This is much faster when adding many safe bubbles.
    pub fn add_interior_bubbles(
        &mut self,
        new_bubbles: Bubbles,
        check_bubbles_inside_bubbles: bool,
    ) -> Result<(), BubblesError> {
        if new_bubbles.ncols() != 4 {
            return Err(BubblesError::ArrayShapeMismatch(format!(
                "New interior bubbles must have 4 columns, got {}",
                new_bubbles.ncols()
            )));
        }
        if new_bubbles.is_empty() {
            return Ok(());
        }

        let n_old = self.interior.nrows();
        let n_new = new_bubbles.nrows();
        let n_ext = self.exterior.nrows();
        let n_total_old = n_old + n_ext;
        let n_total_new = n_total_old + n_new;

        // Extend interior
        let mut extended_interior = Array2::zeros((n_old + n_new, 4));
        extended_interior
            .slice_mut(s![..n_old, ..])
            .assign(&self.interior);
        extended_interior
            .slice_mut(s![n_old.., ..])
            .assign(&new_bubbles);
        self.interior = extended_interior;

        // Extend delta and delta_squared
        let mut new_delta = Array3::zeros((n_old + n_new, n_total_new, 4));
        new_delta
            .slice_mut(s![..n_old, ..n_total_old, ..])
            .assign(&self.delta);

        let mut new_delta_sq = Array2::zeros((n_old + n_new, n_total_new));
        new_delta_sq
            .slice_mut(s![..n_old, ..n_total_old])
            .assign(&self.delta_squared);

        // === Fill new entries ===
        for i_new in 0..n_new {
            let a_global = n_old + i_new;
            let a_pos = self.interior.row(a_global);

            // Against all old bubbles (interior + exterior)
            for b in 0..n_total_old {
                let b_pos = if b < n_old {
                    self.interior.row(b)
                } else {
                    self.exterior.row(b - n_old)
                };
                let delta_ba = &b_pos - &a_pos;
                let dsq = dot_minkowski_vec(&delta_ba, &delta_ba);

                new_delta.slice_mut(s![a_global, b, ..]).assign(&delta_ba);
                new_delta_sq[[a_global, b]] = dsq;

                if b < n_old {
                    new_delta.slice_mut(s![b, a_global, ..]).assign(&-&delta_ba);
                    new_delta_sq[[b, a_global]] = dsq;
                }

                if check_bubbles_inside_bubbles && dsq < 0.0 {
                    let b_index = if b < n_old {
                        BubbleIndex::Interior(b)
                    } else {
                        BubbleIndex::Exterior(b - n_old)
                    };
                    return Err(BubblesError::BubbleFormedInsideBubble {
                        a: BubbleIndex::Interior(a_global),
                        b: b_index,
                    });
                }
            }

            // New interior vs later new interior (upper triangle)
            for j_new in (i_new + 1)..n_new {
                let b_global = n_old + j_new;
                let delta_ba = &self.interior.row(b_global) - &a_pos;
                let dsq = dot_minkowski_vec(&delta_ba, &delta_ba);

                new_delta
                    .slice_mut(s![a_global, b_global, ..])
                    .assign(&delta_ba);
                new_delta_sq[[a_global, b_global]] = dsq;
                new_delta
                    .slice_mut(s![b_global, a_global, ..])
                    .assign(&-&delta_ba);
                new_delta_sq[[b_global, a_global]] = dsq;

                if check_bubbles_inside_bubbles && dsq < 0.0 {
                    return Err(BubblesError::BubbleFormedInsideBubble {
                        a: BubbleIndex::Interior(a_global),
                        b: BubbleIndex::Interior(b_global),
                    });
                }
            }
        }

        self.delta = new_delta;
        self.delta_squared = new_delta_sq;
        Ok(())
    }

    /// Add new exterior (periodic image) bubbles, optionally skipping causality
    /// checks.
    pub fn add_exterior_bubbles(
        &mut self,
        new_bubbles: Bubbles,
        check_bubbles_inside_bubbles: bool,
    ) -> Result<(), BubblesError> {
        if new_bubbles.ncols() != 4 {
            return Err(BubblesError::ArrayShapeMismatch(format!(
                "New exterior bubbles must have 4 columns, got {}",
                new_bubbles.ncols()
            )));
        }
        if new_bubbles.is_empty() {
            return Ok(());
        }

        let n_int = self.interior.nrows();
        let n_ext_old = self.exterior.nrows();
        let n_new = new_bubbles.nrows();

        // Extend exterior array
        let mut extended_exterior = Array2::zeros((n_ext_old + n_new, 4));
        extended_exterior
            .slice_mut(s![..n_ext_old, ..])
            .assign(&self.exterior);
        extended_exterior
            .slice_mut(s![n_ext_old.., ..])
            .assign(&new_bubbles);
        self.exterior = extended_exterior;

        let n_total_new = n_int + n_ext_old + n_new;

        // Extend delta and delta_squared (only columns grow)
        let mut new_delta = Array3::zeros((n_int, n_total_new, 4));
        new_delta
            .slice_mut(s![.., ..n_int + n_ext_old, ..])
            .assign(&self.delta);

        let mut new_delta_sq = Array2::zeros((n_int, n_total_new));
        new_delta_sq
            .slice_mut(s![.., ..n_int + n_ext_old])
            .assign(&self.delta_squared);

        // Fill new columns: all interior → new exterior
        for i in 0..n_int {
            let a_pos = self.interior.row(i);
            for j_new in 0..n_new {
                let b_global = n_int + n_ext_old + j_new;
                let b_pos = self.exterior.row(n_ext_old + j_new);
                let delta_ba = &b_pos - &a_pos;
                let dsq = dot_minkowski_vec(&delta_ba, &delta_ba);

                new_delta.slice_mut(s![i, b_global, ..]).assign(&delta_ba);
                new_delta_sq[[i, b_global]] = dsq;

                if check_bubbles_inside_bubbles && dsq < 0.0 {
                    return Err(BubblesError::BubbleFormedInsideBubble {
                        a: BubbleIndex::Interior(i),
                        b: BubbleIndex::Exterior(n_ext_old + j_new),
                    });
                }
            }
        }

        self.delta = new_delta;
        self.delta_squared = new_delta_sq;
        Ok(())
    }

    // sort the bubbles by reconstruct the whole instance of Bubbles again
    pub fn sort_by_time(&mut self) -> Result<(), BubblesError> {
        let interior = self.interior.clone();
        let exterior = self.exterior.clone();
        let sorted = LatticeBubbles::new(interior, exterior, true)?;
        *self = sorted;
        Ok(())
    }
}

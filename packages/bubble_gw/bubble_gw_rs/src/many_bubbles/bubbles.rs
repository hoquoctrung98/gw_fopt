use csv::{ReaderBuilder, Writer};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, s};
use std::path::Path;
use thiserror::Error;

pub fn dot_minkowski_vec(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
    assert!(v1.len() == 4 && v2.len() == 4, "4-vectors required");
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

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse float error at {path}:{line}: '{value}'")]
    ParseFloat {
        path: String,
        line: usize,
        value: String,
    },

    #[error("Invalid row in {path}:{line}: expected 4 columns, got {got}")]
    InvalidColumnCount {
        path: String,
        line: usize,
        got: usize,
    },

    #[error("Empty bubble file: {0}")]
    EmptyFile(String),
}

// Checks if any bubble is contained within another at the initial time.
pub fn check_bubble_formed_inside_bubble(
    bubbles_interior: &Array2<f64>,
    bubbles_exterior: &Array2<f64>,
    delta_squared: &Array2<f64>,
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
pub struct Bubbles {
    pub interior: Array2<f64>,
    pub exterior: Array2<f64>,
    pub delta: Array3<f64>,
    pub delta_squared: Array2<f64>,
}

impl Bubbles {
    pub fn new(
        mut bubbles_interior: Array2<f64>,
        mut bubbles_exterior: Array2<f64>,
        sort_by_time: bool,
    ) -> Result<Bubbles, BubblesError> {
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

        // Initialize delta and delta_squared
        let mut delta = Array3::zeros((n_interior, n_total, 4));
        let mut delta_squared = Array2::zeros((n_interior, n_total));

        // Compute delta and delta_squared using symmetry for interior-interior
        for a_idx in 0..n_interior {
            // Interior to interior (upper triangular and diagonal)
            for b_idx in a_idx..n_interior {
                let delta_ba = bubbles_interior.slice(s![b_idx, ..]).to_owned()
                    - bubbles_interior.slice(s![a_idx, ..]).to_owned();
                delta.slice_mut(s![a_idx, b_idx, ..]).assign(&delta_ba);
                delta_squared[[a_idx, b_idx]] = dot_minkowski_vec(delta_ba.view(), delta_ba.view());
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
                delta_squared[[a_idx, b_total]] =
                    dot_minkowski_vec(delta_ba.view(), delta_ba.view());
            }
        }

        // Check for bubble containment
        check_bubble_formed_inside_bubble(&bubbles_interior, &bubbles_exterior, &delta_squared)?;

        Ok(Bubbles {
            interior: bubbles_interior,
            exterior: bubbles_exterior,
            delta,
            delta_squared,
        })
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
        sort_by_time: bool,
        has_headers: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let interior = Self::load_bubbles_from_csv(interior_path, has_headers)?;
        let exterior = Self::load_bubbles_from_csv(exterior_path, has_headers)?;

        Self::new(interior, exterior, sort_by_time).map_err(Into::into)
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
        let mut rdr = ReaderBuilder::new()
            .has_headers(has_headers)
            .flexible(false)
            .from_path(path)?;

        let mut records = Vec::new();
        for (line_num, result) in rdr.records().enumerate() {
            let record = result.map_err(csv::Error::from)?;
            let line = line_num + if has_headers { 2 } else { 1 }; // +1 for 1-based, +1 more if header

            if record.len() != 4 {
                return Err(BubblesError::InvalidColumnCount {
                    path: path.display().to_string(),
                    line,
                    got: record.len(),
                });
            }

            let row: Result<Array1<f64>, _> = record
                .iter()
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| BubblesError::ParseFloat {
                            path: path.display().to_string(),
                            line,
                            value: s.to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Array1::from);

            records.push(row?);
        }

        if records.is_empty() {
            // Allow empty exterior files
            return Ok(Array2::zeros((0, 4)));
        }

        let n = records.len();
        let mut array = Array2::zeros((n, 4));
        for (i, row) in records.into_iter().enumerate() {
            array.row_mut(i).assign(&row);
        }
        Ok(array)
    }

    /// Write bubbles to CSV with scientific notation e.8 and optional header
    fn write_bubbles_to_csv<P: AsRef<Path>>(
        bubbles: ArrayView2<f64>,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        let mut wtr = Writer::from_path(path)?;
        if has_headers {
            wtr.write_record(&["t", "x", "y", "z"])?;
        }
        for row in bubbles.rows() {
            let fields: Vec<String> = row.iter().map(|&v| format!("{:.8e}", v)).collect();
            wtr.write_record(&fields)?;
        }
        wtr.flush()?;
        Ok(())
    }

    /// Save interior bubbles to CSV
    pub fn save_interior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        Self::write_bubbles_to_csv(self.interior.view(), path, has_headers)
    }

    /// Save exterior bubbles to CSV
    pub fn save_exterior_to_csv<P: AsRef<Path>>(
        &self,
        path: P,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        Self::write_bubbles_to_csv(self.exterior.view(), path, has_headers)
    }

    /// Save both interior and exterior with same header setting
    pub fn save_to_csv_files<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        interior_path: P,
        exterior_path: Q,
        has_headers: bool,
    ) -> Result<(), BubblesError> {
        self.save_interior_to_csv(&interior_path, has_headers)?;
        self.save_exterior_to_csv(&exterior_path, has_headers)?;
        Ok(())
    }
}

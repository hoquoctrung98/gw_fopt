use crate::utils::integrate::Integrate;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, Zip, azip, s, stack};
use num_complex::Complex64;
use rayon::ThreadPool;
use rayon::prelude::*;

/// Represents a bubble index, distinguishing between an interior index, exterior index, and no collision.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BubbleIndex {
    Interior(usize),
    Exterior(usize),
    None,
}

/// Stores precomputed data to optimize collision and boundary calculations.
/// Contains 4-vector differences (`delta`), their squared Minkowski norms (`delta_squared`),
/// and an optional cache of first colliding bubble data (`first_colliding_bubbles`).
pub struct CachedData {
    delta: Array3<f64>,
    delta_squared: Array2<f64>,
    first_colliding_bubbles: Option<Array3<BubbleIndex>>,
}

/// Represents the collision status of a direction relative to a reference bubble.
/// - `NeverCollided` (0): No collision occurs.
/// - `AlreadyCollided` (1): Collision has occurred.
/// - `NotYetCollided` (2): Collision will occur in the future.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CollisionStatus {
    NeverCollided = 0,
    AlreadyCollided = 1,
    NotYetCollided = 2,
}

/// Represents a segment of the angular grid with collision information.
/// Contains the cosine of the polar angle (`cos_thetax`), azimuthal angle bounds (`phi_lower`, `phi_upper`),
/// the index of the colliding bubble (`bubble_index`), and the collision status (`collision_status`).
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub cos_thetax: f64,
    pub phi_lower: f64,
    pub phi_upper: f64,
    pub bubble_index: BubbleIndex,
    pub collision_status: CollisionStatus,
}

/// Manages bulk flow calculations for a set of bubbles, including their 4-vectors, collision data,
/// and angular grid computations. Utilizes a thread pool for parallel processing and caches
/// precomputed data to improve performance. Angular resolutions (`n_cos_thetax`, `n_phix`) must
/// be set explicitly using `set_resolution` before computations.
pub struct BulkFlow {
    bubbles_interior: Array2<f64>,
    bubbles_exterior: Array2<f64>,
    cached_data: CachedData,
    coefficients_sets: Array2<f64>,
    powers_sets: Array2<f64>,
    damping_width: Option<f64>,
    active_bubbles: Array1<bool>,
    thread_pool: ThreadPool,
    n_cos_thetax: Option<usize>,
    n_phix: Option<usize>,
    cos_thetax: Option<Array1<f64>>,
    phix: Option<Array1<f64>>,
    direction_vectors: Option<Array3<f64>>,
}

impl BulkFlow {
    /// Creates a new `BulkFlow` instance with the given bubble 4-vectors and indices for interior and exterior sets.
    /// Initializes precomputed `delta` and `delta_squared` arrays (from interior to all bubbles), sets up a thread pool,
    /// and configures default coefficient and power sets. Angular resolutions are set to None,
    /// requiring a call to `set_resolution` before computations.
    pub fn new(bubbles_interior: Array2<f64>, bubbles_exterior: Array2<f64>) -> Self {
        let n_interior = bubbles_interior.shape()[0];
        let n_exterior = bubbles_exterior.shape()[0];
        let n_total = n_interior + n_exterior;
        let mut delta = Array3::zeros((n_interior, n_total, 4));
        let mut delta_squared = Array2::zeros((n_interior, n_total));
        for a_idx in 0..n_interior {
            for b_idx in 0..n_interior {
                if a_idx != b_idx {
                    delta.slice_mut(s![a_idx, b_idx, ..]).assign(
                        &(bubbles_interior.slice(s![b_idx, ..]).to_owned()
                            - bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                    );
                    let delta_ba = delta.slice(s![a_idx, b_idx, ..]);
                    delta_squared[[a_idx, b_idx]] = dot_minkowski_vec(delta_ba, delta_ba);
                }
            }
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                delta.slice_mut(s![a_idx, b_total, ..]).assign(
                    &(bubbles_exterior.slice(s![b_ex, ..]).to_owned()
                        - bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                );
                let delta_ba = delta.slice(s![a_idx, b_total, ..]);
                delta_squared[[a_idx, b_total]] = dot_minkowski_vec(delta_ba, delta_ba);
            }
        }
        let cached_data = CachedData {
            delta,
            delta_squared,
            first_colliding_bubbles: None,
        };
        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(default_threads)
            .build()
            .unwrap();

        BulkFlow {
            bubbles_interior,
            bubbles_exterior,
            cached_data,
            coefficients_sets: Array2::from_elem((1, 1), 1.0),
            powers_sets: Array2::from_elem((1, 1), 3.0),
            damping_width: None,
            active_bubbles: Array1::from_elem(1, true),
            thread_pool,
            n_cos_thetax: None,
            n_phix: None,
            cos_thetax: None,
            phix: None,
            direction_vectors: None,
        }
    }

    /// Computes the first colliding bubble index for each direction in the angular grid,
    /// returning a matrix where `BubbleIndex::None` indicates no collision.
    /// Panics if resolution is not set.
    pub fn compute_first_colliding_bubble(&self, a_idx: usize) -> Array2<BubbleIndex> {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_phix = self.n_phix.unwrap();
        let n_interior = self.bubbles_interior.shape()[0];
        let n_exterior = self.bubbles_exterior.shape()[0];
        let n_total = n_interior + n_exterior;
        let mut first_bubble = Array2::from_elem((n_cos_thetax, n_phix), BubbleIndex::None);
        let tolerance = 1e-10;

        for i in 0..n_cos_thetax {
            for j in 0..n_phix {
                let x_vec = self.direction_vectors.as_ref().unwrap().slice(s![i, j, ..]);
                let mut earliest_bubble_idx = BubbleIndex::None;
                let mut earliest_delta_tab = f64::INFINITY;

                for b_total in 0..n_total {
                    let skip_self = b_total < n_interior && b_total == a_idx;
                    if skip_self {
                        continue;
                    }
                    let delta_ba = self.cached_data.delta.slice(s![a_idx, b_total, ..]);
                    let delta_ba_squared = self.cached_data.delta_squared[[a_idx, b_total]];
                    let dot_ba_x = dot_minkowski_vec(delta_ba, x_vec);
                    if dot_ba_x <= tolerance {
                        continue; // No future collision
                    }
                    let delta_tab = delta_ba_squared / (2.0 * dot_ba_x);
                    if delta_tab <= 0.0 || delta_tab >= earliest_delta_tab {
                        continue; // Not the earliest collision
                    }
                    let mut is_first = true;
                    for c_total in 0..n_total {
                        let skip_self_c = c_total < n_interior && c_total == a_idx;
                        if skip_self_c || c_total == b_total {
                            continue;
                        }
                        let delta_ca = self.cached_data.delta.slice(s![a_idx, c_total, ..]);
                        if !check_collision_point(delta_ba, delta_ca, x_vec) {
                            is_first = false;
                            break;
                        }
                    }
                    if is_first {
                        earliest_bubble_idx = if b_total < n_interior {
                            BubbleIndex::Interior(b_total)
                        } else {
                            BubbleIndex::Exterior(b_total - n_interior)
                        };
                        earliest_delta_tab = delta_tab;
                    }
                }
                first_bubble[[i, j]] = earliest_bubble_idx;
            }
        }
        first_bubble
    }

    /// Sets the angular resolutions and updates the associated grids (`cos_thetax`, `phix`, `direction_vectors`).
    /// Optionally precomputes the `first_colliding_bubbles` cache if `precompute_first_bubbles` is true.
    /// Panics if resolutions are zero.
    pub fn set_resolution(
        &mut self,
        n_cos_thetax: usize,
        n_phix: usize,
        precompute_first_bubbles: bool,
    ) {
        if n_cos_thetax == 0 || n_phix == 0 {
            panic!("Angular resolutions must be greater than zero");
        }
        self.n_cos_thetax = Some(n_cos_thetax);
        self.n_phix = Some(n_phix);
        self.cos_thetax = Some(Array1::linspace(-1.0, 1.0, n_cos_thetax));
        self.phix = Some(Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n_phix));
        let mut direction_vectors = Array3::zeros((n_cos_thetax, n_phix, 4));
        Zip::indexed(&mut direction_vectors).for_each(|(i, j, k), val| {
            let cos_thetax = self.cos_thetax.as_ref().unwrap()[i];
            let sin_thetax = f64::sqrt(1.0 - cos_thetax * cos_thetax).abs();
            let phix = self.phix.as_ref().unwrap()[j];
            *val = match k {
                0 => 1.0,
                1 => sin_thetax * phix.cos(),
                2 => sin_thetax * phix.sin(),
                3 => cos_thetax,
                _ => unreachable!(),
            };
        });
        self.direction_vectors = Some(direction_vectors);

        if precompute_first_bubbles {
            // Compute and cache first_colliding_bubble
            let n_interior = self.bubbles_interior.shape()[0];
            let first_bubble_arrays: Vec<Array2<BubbleIndex>> = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .map(|a_idx| self.compute_first_colliding_bubble(a_idx))
                    .collect()
            });
            let first_bubble_cache = stack(
                Axis(0),
                &first_bubble_arrays
                    .iter()
                    .map(|arr| arr.view())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            self.cached_data.first_colliding_bubbles = Some(first_bubble_cache);
        } else {
            self.cached_data.first_colliding_bubbles = None;
        }
    }

    /// Sets the active bubble sets array.
    pub fn set_active_sets(&mut self, active_sets: Array1<bool>) {
        self.active_bubbles = active_sets;
    }

    /// Returns a reference to the 4-vector array of interior bubbles.
    pub fn bubbles_interior(&self) -> &Array2<f64> {
        &self.bubbles_interior
    }

    /// Returns a reference to the 4-vector array of exterior bubbles.
    pub fn bubbles_exterior(&self) -> &Array2<f64> {
        &self.bubbles_exterior
    }

    /// Updates the interior 4-vector array and invalidates the first colliding bubble cache,
    /// recomputing `delta` and `delta_squared`.
    pub fn set_interior_four_vectors(&mut self, four_vectors: Array2<f64>) {
        self.bubbles_interior = four_vectors;
        self.cached_data.first_colliding_bubbles = None;
        let n_interior = self.bubbles_interior.shape()[0];
        let n_exterior = self.bubbles_exterior.shape()[0];
        let n_total = n_interior + n_exterior;
        self.cached_data.delta = Array3::zeros((n_interior, n_total, 4));
        self.cached_data.delta_squared = Array2::zeros((n_interior, n_total));
        for a_idx in 0..n_interior {
            for b_idx in 0..n_interior {
                if a_idx != b_idx {
                    self.cached_data
                        .delta
                        .slice_mut(s![a_idx, b_idx, ..])
                        .assign(
                            &(self.bubbles_interior.slice(s![b_idx, ..]).to_owned()
                                - self.bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                        );
                    let delta_ba = self.cached_data.delta.slice(s![a_idx, b_idx, ..]);
                    self.cached_data.delta_squared[[a_idx, b_idx]] =
                        dot_minkowski_vec(delta_ba, delta_ba);
                }
            }
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                self.cached_data
                    .delta
                    .slice_mut(s![a_idx, b_total, ..])
                    .assign(
                        &(self.bubbles_exterior.slice(s![b_ex, ..]).to_owned()
                            - self.bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                    );
                let delta_ba = self.cached_data.delta.slice(s![a_idx, b_total, ..]);
                self.cached_data.delta_squared[[a_idx, b_total]] =
                    dot_minkowski_vec(delta_ba, delta_ba);
            }
        }
    }

    /// Updates the exterior 4-vector array and invalidates the first colliding bubble cache,
    /// recomputing `delta` and `delta_squared`.
    pub fn set_exterior_four_vectors(&mut self, four_vectors: Array2<f64>) {
        self.bubbles_exterior = four_vectors;
        self.cached_data.first_colliding_bubbles = None;
        let n_interior = self.bubbles_interior.shape()[0];
        let n_exterior = self.bubbles_exterior.shape()[0];
        let n_total = n_interior + n_exterior;
        self.cached_data.delta = Array3::zeros((n_interior, n_total, 4));
        self.cached_data.delta_squared = Array2::zeros((n_interior, n_total));
        for a_idx in 0..n_interior {
            for b_idx in 0..n_interior {
                if a_idx != b_idx {
                    self.cached_data
                        .delta
                        .slice_mut(s![a_idx, b_idx, ..])
                        .assign(
                            &(self.bubbles_interior.slice(s![b_idx, ..]).to_owned()
                                - self.bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                        );
                    let delta_ba = self.cached_data.delta.slice(s![a_idx, b_idx, ..]);
                    self.cached_data.delta_squared[[a_idx, b_idx]] =
                        dot_minkowski_vec(delta_ba, delta_ba);
                }
            }
            for b_ex in 0..n_exterior {
                let b_total = n_interior + b_ex;
                self.cached_data
                    .delta
                    .slice_mut(s![a_idx, b_total, ..])
                    .assign(
                        &(self.bubbles_exterior.slice(s![b_ex, ..]).to_owned()
                            - self.bubbles_interior.slice(s![a_idx, ..]).to_owned()),
                    );
                let delta_ba = self.cached_data.delta.slice(s![a_idx, b_total, ..]);
                self.cached_data.delta_squared[[a_idx, b_total]] =
                    dot_minkowski_vec(delta_ba, delta_ba);
            }
        }
    }

    /// Returns a reference to the precomputed 4-vector difference array.
    pub fn delta(&self) -> &Array3<f64> {
        &self.cached_data.delta
    }

    /// Updates the precomputed 4-vector difference array.
    pub fn set_delta(&mut self, delta: Array3<f64>) {
        self.cached_data.delta = delta;
    }

    /// Returns a reference to the coefficient sets array.
    pub fn coefficients_sets(&self) -> &Array2<f64> {
        &self.coefficients_sets
    }

    /// Updates the coefficient sets array.
    pub fn set_coefficients_sets(&mut self, coefficients_sets: Array2<f64>) {
        self.coefficients_sets = coefficients_sets;
    }

    /// Returns a reference to the power sets array.
    pub fn powers_sets(&self) -> &Array2<f64> {
        &self.powers_sets
    }

    /// Updates the power sets array.
    pub fn set_powers_sets(&mut self, powers_sets: Array2<f64>) {
        self.powers_sets = powers_sets;
    }

    /// Returns a reference to the array indicating active bubble sets.
    pub fn active_sets(&self) -> &Array1<bool> {
        &self.active_bubbles
    }

    /// Updates the active sets array and related parameters based on coefficient and power sets.
    /// Returns an error if the input sets are inconsistent or coefficients do not sum to 0 or 1.
    pub fn set_gradient_scaling_params(
        &mut self,
        coefficients_sets: Vec<Vec<f64>>,
        powers_sets: Vec<Vec<f64>>,
        damping_width: Option<f64>,
    ) -> Result<(), String> {
        let n_sets = coefficients_sets.len();
        let n_coeffs = coefficients_sets.first().map_or(0, |v| v.len());
        if coefficients_sets.iter().any(|v| v.len() != n_coeffs) {
            return Err("All coefficient sets must have the same length".to_string());
        }
        let coefficients_sets = Array2::from_shape_vec(
            (n_sets, n_coeffs),
            coefficients_sets.into_iter().flatten().collect(),
        )
        .map_err(|_| "Invalid shape for coefficients_sets".to_string())?;

        let n_powers = powers_sets.len();
        let n_pows = powers_sets.first().map_or(0, |v| v.len());
        if powers_sets.iter().any(|v| v.len() != n_pows) {
            return Err("All power sets must have the same length".to_string());
        }
        let powers_sets = Array2::from_shape_vec(
            (n_powers, n_pows),
            powers_sets.into_iter().flatten().collect(),
        )
        .map_err(|_| "Invalid shape for powers_sets".to_string())?;

        if n_sets != n_powers {
            return Err("Number of coefficient sets and power sets must be equal".to_string());
        }

        let mut active_bubbles = Array1::from_elem(n_sets, false);
        for s in 0..n_sets {
            let coeff_sum = coefficients_sets.slice(s![s, ..]).sum();
            if coeff_sum.abs() >= 1e-10 && (coeff_sum - 1.0).abs() > 1e-10 {
                return Err(format!("Set {}: coefficients must sum to 0 or 1", s));
            }
            active_bubbles[s] = coeff_sum.abs() > 1e-10;
        }

        self.coefficients_sets = coefficients_sets;
        self.powers_sets = powers_sets;
        self.damping_width = damping_width;
        self.active_bubbles = active_bubbles;
        Ok(())
    }

    /// Computes the collision status for each direction relative to a reference bubble at a given time.
    /// Panics if resolution is not set.
    pub fn compute_collision_status(
        &self,
        a_idx: usize,
        t: f64,
        first_bubble: ArrayView2<BubbleIndex>,
        delta_tab_grid: ArrayView2<f64>,
    ) -> Array2<CollisionStatus> {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_phix = self.n_phix.unwrap();
        let mut collision_status =
            Array2::from_elem((n_cos_thetax, n_phix), CollisionStatus::NeverCollided);
        let ta = self.bubbles_interior[[a_idx, 0]];
        let delta_ta = t - ta;

        for i in 0..n_cos_thetax {
            for j in 0..n_phix {
                match first_bubble[[i, j]] {
                    BubbleIndex::None => {
                        collision_status[[i, j]] = CollisionStatus::NeverCollided;
                        continue;
                    }
                    _ => (),
                }
                let delta_tab_val = delta_tab_grid[[i, j]];
                if delta_tab_val > 0.0 && delta_ta >= delta_tab_val {
                    collision_status[[i, j]] = CollisionStatus::AlreadyCollided;
                } else {
                    collision_status[[i, j]] = CollisionStatus::NotYetCollided;
                }
            }
        }

        collision_status
    }

    /// Generates a vector of `Segment` objects by grouping contiguous regions with the same
    /// bubble index and collision status across the angular grid.
    /// Panics if resolution is not set.
    pub fn generate_segments(
        &self,
        first_bubble: ArrayView2<BubbleIndex>,
        collision_status: ArrayView2<i32>,
    ) -> Vec<Segment> {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_phix = self.n_phix.unwrap();
        if first_bubble.shape() != [n_cos_thetax, n_phix]
            || collision_status.shape() != [n_cos_thetax, n_phix]
        {
            panic!("Input arrays must match the set resolution");
        }
        let mut segments = Vec::with_capacity(n_cos_thetax * 10);

        for i in 0..n_cos_thetax {
            let mut phi_left = self.phix.as_ref().unwrap()[0];
            let mut current_bubble = first_bubble[[i, 0]];
            let mut current_status = collision_status[[i, 0]];

            for j in 1..n_phix {
                let bubble = first_bubble[[i, j]];
                let status = collision_status[[i, j]];

                if bubble != current_bubble || status != current_status || j == n_phix - 1 {
                    let phi_right = if j == n_phix - 1
                        && bubble == current_bubble
                        && status == current_status
                    {
                        self.phix.as_ref().unwrap()[j]
                    } else {
                        self.phix.as_ref().unwrap()[j]
                    };
                    if phi_right != phi_left {
                        segments.push(Segment {
                            cos_thetax: self.cos_thetax.as_ref().unwrap()[i],
                            phi_lower: phi_left,
                            phi_upper: phi_right,
                            bubble_index: current_bubble,
                            collision_status: match current_status {
                                0 => CollisionStatus::NeverCollided,
                                1 => CollisionStatus::AlreadyCollided,
                                2 => CollisionStatus::NotYetCollided,
                                _ => CollisionStatus::NeverCollided,
                            },
                        });
                    }
                    phi_left = self.phix.as_ref().unwrap()[j];
                    current_bubble = bubble;
                    current_status = status;
                }
            }
        }
        segments
    }

    /// Computes the time differences (`delta_tab`) for each direction
    /// relative to a reference bubble.
    /// Panics if resolution is not set.
    pub fn compute_delta_tab(
        &self,
        a_idx: usize,
        first_bubble: ArrayView2<BubbleIndex>,
    ) -> Array2<f64> {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_phix = self.n_phix.unwrap();
        let n_interior = self.bubbles_interior.shape()[0];
        let mut delta_tab_grid = Array2::zeros((n_cos_thetax, n_phix));

        for i in 0..n_cos_thetax {
            for j in 0..n_phix {
                let b_total = match first_bubble[[i, j]] {
                    BubbleIndex::None => {
                        continue;
                    }
                    BubbleIndex::Interior(b_idx) => b_idx,
                    BubbleIndex::Exterior(b_idx) => n_interior + b_idx,
                };
                let delta_ba = self.cached_data.delta.slice(s![a_idx, b_total, ..]);
                let x_vec = self.direction_vectors.as_ref().unwrap().slice(s![i, j, ..]);
                let delta_ba_squared = self.cached_data.delta_squared[[a_idx, b_total]];
                let dot_ba_x = dot_minkowski_vec(delta_ba, x_vec);
                if dot_ba_x.abs() < 1e-10 {
                    delta_tab_grid[[i, j]] = 0.0;
                    continue;
                }
                let delta_tab_val = delta_ba_squared / (2.0 * dot_ba_x);
                delta_tab_grid[[i, j]] = delta_tab_val;
            }
        }

        delta_tab_grid
    }

    /// Computes the B-matrix components (`b_plus` and `b_minus`) for a given polar angle and segments.
    /// Incorporates collision status and performs numerical integration where needed.
    /// Panics if resolution is not set.
    pub fn compute_b_matrix(
        &self,
        cos_thetax: f64,
        segments: &[Segment],
        delta_tab_grid: ArrayView2<f64>,
        idx: usize,
        delta_ta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_sets = self.coefficients_sets.shape()[0];
        if idx >= n_cos_thetax {
            panic!("Index out of bounds for cos_thetax grid");
        }
        let sin_squared_thetax = 1.0 - cos_thetax.powi(2);
        let mut b_plus_arr = Array1::zeros(n_sets);
        let mut b_minus_arr = Array1::zeros(n_sets);
        let dphi_grid = 2.0 * std::f64::consts::PI / self.n_phix.unwrap() as f64;
        let phi_to_idx = self.n_phix.unwrap() as f64 / (2.0 * std::f64::consts::PI);
        let tolerance = 1e-10;

        // Only consider bubbles nucleated before given time
        let delta_ta_cubed = if delta_ta > 0.0 {
            delta_ta.powi(3)
        } else {
            0.0
        };

        for seg in segments.iter() {
            // Find the segment correspond to input cos_thetax
            if (seg.cos_thetax - cos_thetax).abs() >= tolerance {
                continue;
            }
            let phi_left = seg.phi_lower;
            let phi_right = seg.phi_upper;
            // ignore integrating over zero-width range
            if (phi_right - phi_left).abs() < 1e-10 {
                continue;
            }
            let status = seg.collision_status;
            // envelope contribution
            if status == CollisionStatus::NeverCollided || status == CollisionStatus::NotYetCollided
            {
                let sin_term = (2.0 * phi_right).sin() - (2.0 * phi_left).sin();
                let cos_term = (2.0 * phi_right).cos() - (2.0 * phi_left).cos();
                let scaling_factor = 0.25 * sin_squared_thetax * delta_ta_cubed;
                for s in 0..n_sets {
                    b_plus_arr[s] += scaling_factor * sin_term;
                    b_minus_arr[s] += -scaling_factor * cos_term;
                }
                continue;
            // bulkflow contribution
            } else if status == CollisionStatus::AlreadyCollided {
                let segment_width = phi_right - phi_left;
                let n_integration_points = (segment_width / dphi_grid).ceil().max(2.0) as usize;
                let dphi = segment_width / (n_integration_points - 1) as f64;

                let mut sin_terms = Array1::zeros(n_integration_points);
                let mut cos_terms = Array1::zeros(n_integration_points);
                let mut time_ratios = Array1::zeros(n_integration_points);
                let mut integration_weights = Array1::zeros(n_integration_points);

                for i in 0..n_integration_points {
                    let phi = phi_left + i as f64 * dphi;
                    sin_terms[i] = (2.0 * phi).sin();
                    cos_terms[i] = (2.0 * phi).cos();
                    let j_float = phi * phi_to_idx;
                    let j = (j_float.round() as usize)
                        .min(self.n_phix.unwrap() - 1)
                        .max(0);
                    let delta_tab = delta_tab_grid[[idx, j]];
                    time_ratios[i] = delta_tab / delta_ta;
                    integration_weights[i] = if i == 0 || i == n_integration_points - 1 {
                        1.0
                    } else {
                        2.0
                    };
                }

                let mut integral_cos = Array1::zeros(n_sets);
                let mut integral_sin = Array1::zeros(n_sets);
                // if bubble(a_idx) is nucleated before time step t
                if delta_ta > 0.0 {
                    let active_indices: Vec<usize> =
                        (0..n_sets).filter(|&s| self.active_bubbles[s]).collect();
                    for &s in &active_indices {
                        let mut scaling_factors = Array1::zeros(n_integration_points);
                        azip!((factor in &mut scaling_factors, &ratio in &time_ratios) {
                            let mut f = 0.0;
                            for k in 0..self.coefficients_sets.shape()[1] {
                                f += self.coefficients_sets[[s, k]] * ratio.powf(self.powers_sets[[s, k]]);
                            }
                            *factor = f * delta_ta_cubed;
                        });
                        if let Some(damping_width) = self.damping_width {
                            let damping_factor = (-delta_ta * (1.0 - time_ratios.clone())
                                / damping_width)
                                .mapv(|x| x.exp());
                            azip!((factor in &mut scaling_factors, &damp in &damping_factor) *factor *= damp);
                        }

                        let mut cos_sum = 0.0;
                        let mut sin_sum = 0.0;
                        azip!((factor in &scaling_factors, &cos_val in &cos_terms, &sin_val in &sin_terms, &weight in &integration_weights) {
                            cos_sum += weight * cos_val * factor;
                            sin_sum += weight * sin_val * factor;
                        });
                        integral_cos[s] = cos_sum;
                        integral_sin[s] = sin_sum;
                    }
                }

                integral_cos *= dphi / 2.0;
                integral_sin *= dphi / 2.0;
                for s in 0..n_sets {
                    b_plus_arr[s] += 0.5 * sin_squared_thetax * integral_cos[s];
                    b_minus_arr[s] += 0.5 * sin_squared_thetax * integral_sin[s];
                }
            }
        }

        (b_plus_arr, b_minus_arr)
    }

    // /// use simpson method, 5% slower
    // /// Computes the B-matrix components (`b_plus` and `b_minus`) for a given polar angle and segments.
    // /// Incorporates collision status and performs numerical integration using integrate.rs where needed.
    // /// Panics if resolution is not set.
    // pub fn compute_b_matrix(
    //     &self,
    //     cos_thetax: f64,
    //     segments: &[Segment],
    //     delta_tab_grid: ArrayView2<f64>,
    //     idx: usize,
    //     delta_ta: f64,
    // ) -> (Array1<f64>, Array1<f64>) {
    //     let n_cos_thetax = self.n_cos_thetax.unwrap();
    //     let n_sets = self.coefficients_sets.shape()[0];
    //     if idx >= n_cos_thetax {
    //         panic!("Index out of bounds for cos_thetax grid");
    //     }
    //     let sin_squared_thetax = 1.0 - cos_thetax.powi(2);
    //     let mut b_plus_arr = Array1::zeros(n_sets);
    //     let mut b_minus_arr = Array1::zeros(n_sets);
    //     let dphi_grid = 2.0 * std::f64::consts::PI / self.n_phix.unwrap() as f64;
    //     let phi_to_idx = self.n_phix.unwrap() as f64 / (2.0 * std::f64::consts::PI);
    //     let tolerance = 1e-10;
    //
    //     // Only consider bubbles nucleated before given time
    //     let delta_ta_cubed = if delta_ta > 0.0 {
    //         delta_ta.powi(3)
    //     } else {
    //         0.0
    //     };
    //
    //     for seg in segments.iter() {
    //         // Find the segment corresponding to input cos_thetax
    //         if (seg.cos_thetax - cos_thetax).abs() >= tolerance {
    //             continue;
    //         }
    //         let phi_left = seg.phi_lower;
    //         let phi_right = seg.phi_upper;
    //         // Ignore integrating over zero-width range
    //         if (phi_right - phi_left).abs() < 1e-10 {
    //             continue;
    //         }
    //         let status = seg.collision_status;
    //         if status == CollisionStatus::NeverCollided || status == CollisionStatus::NotYetCollided
    //         {
    //             let sin_term = (2.0 * phi_right).sin() - (2.0 * phi_left).sin();
    //             let cos_term = (2.0 * phi_right).cos() - (2.0 * phi_left).cos();
    //             let scaling_factor = 0.25 * sin_squared_thetax * delta_ta_cubed;
    //             for s in 0..n_sets {
    //                 b_plus_arr[s] += scaling_factor * sin_term;
    //                 b_minus_arr[s] += -scaling_factor * cos_term;
    //             }
    //             continue;
    //         } else if status == CollisionStatus::AlreadyCollided {
    //             let segment_width = phi_right - phi_left;
    //             let n_integration_points = (segment_width / dphi_grid).ceil().max(2.0) as usize;
    //             let phi: Vec<f64> = (0..n_integration_points)
    //                 .map(|i| {
    //                     phi_left + i as f64 * segment_width / (n_integration_points - 1) as f64
    //                 })
    //                 .collect();
    //             let sin_terms: Vec<f64> = phi.iter().map(|&p| (2.0 * p).sin()).collect();
    //             let cos_terms: Vec<f64> = phi.iter().map(|&p| (2.0 * p).cos()).collect();
    //             let time_ratios: Vec<f64> = phi
    //                 .iter()
    //                 .map(|&p| {
    //                     let j_float = p * phi_to_idx;
    //                     let j = (j_float.round() as usize)
    //                         .min(self.n_phix.unwrap() - 1)
    //                         .max(0);
    //                     delta_tab_grid[[idx, j]] / delta_ta
    //                 })
    //                 .collect();
    //
    //             let mut integral_cos = Array1::zeros(n_sets);
    //             let mut integral_sin = Array1::zeros(n_sets);
    //             if delta_ta > 0.0 {
    //                 let active_indices: Vec<usize> =
    //                     (0..n_sets).filter(|&s| self.active_bubbles[s]).collect();
    //                 for &s in &active_indices {
    //                     let scaling_factors: Vec<f64> = time_ratios
    //                         .iter()
    //                         .map(|&ratio| {
    //                             let mut f = 0.0;
    //                             for k in 0..self.coefficients_sets.shape()[1] {
    //                                 f += self.coefficients_sets[[s, k]]
    //                                     * ratio.powf(self.powers_sets[[s, k]]);
    //                             }
    //                             if let Some(damping_width) = self.damping_width {
    //                                 f *= (-delta_ta * (1.0 - ratio) / damping_width).exp();
    //                             }
    //                             f * delta_ta_cubed
    //                         })
    //                         .collect();
    //                     let sin_integrand: Vec<f64> = scaling_factors
    //                         .iter()
    //                         .zip(&sin_terms)
    //                         .map(|(&f, &s)| f * s)
    //                         .collect();
    //                     let cos_integrand: Vec<f64> = scaling_factors
    //                         .iter()
    //                         .zip(&cos_terms)
    //                         .map(|(&f, &c)| f * c)
    //                         .collect();
    //                     integral_sin[s] = sin_integrand
    //                         .as_slice()
    //                         .simpson(Some(&phi), None, None)
    //                         .unwrap();
    //                     integral_cos[s] = cos_integrand
    //                         .as_slice()
    //                         .simpson(Some(&phi), None, None)
    //                         .unwrap();
    //                 }
    //             }
    //             for s in 0..n_sets {
    //                 b_plus_arr[s] += 0.5 * sin_squared_thetax * integral_cos[s];
    //                 b_minus_arr[s] += 0.5 * sin_squared_thetax * integral_sin[s];
    //             }
    //         }
    //     }
    //
    //     (b_plus_arr, b_minus_arr)
    // }

    /// Computes the A-matrix components (`a_plus` and `a_minus`) for a reference bubble,
    /// using frequency array, time, and angular grid data.
    /// Panics if resolution is not set.
    pub fn compute_a_matrix(
        &self,
        a_idx: usize,
        w_arr: ArrayView1<f64>,
        t: f64,
        first_bubble: ArrayView2<BubbleIndex>,
        delta_tab_grid: ArrayView2<f64>,
    ) -> (Array2<Complex64>, Array2<Complex64>) {
        let n_cos_thetax = self.n_cos_thetax.unwrap();
        let n_w = w_arr.len();
        let n_sets = self.coefficients_sets.shape()[0];
        let ta = self.bubbles_interior[[a_idx, 0]];
        let delta_ta = t - ta;
        let collision_status =
            self.compute_collision_status(a_idx, t, first_bubble, delta_tab_grid);
        let segments =
            self.generate_segments(first_bubble, collision_status.mapv(|s| s as i32).view());

        let mut a_plus = Array2::zeros((n_sets, n_w));
        let mut a_minus = Array2::zeros((n_sets, n_w));
        let dcos_thetax = 2.0 / (n_cos_thetax - 1) as f64;

        for i in 0..n_cos_thetax {
            let cos_thetax_val = self.cos_thetax.as_ref().unwrap()[i];
            let (b_plus, b_minus) =
                self.compute_b_matrix(cos_thetax_val, &segments, delta_tab_grid, i, delta_ta);

            let phase_base = Complex64::new(0.0, -delta_ta * cos_thetax_val);
            let mut angular_phases = Array1::zeros(n_w);
            for w_idx in 0..n_w {
                angular_phases[w_idx] = (w_arr[w_idx] * phase_base).exp();
            }
            let phase_factors = angular_phases.mapv(|p| p * dcos_thetax);

            for s in 0..n_sets {
                let b_plus_s = Complex64::new(b_plus[s], 0.0);
                let b_minus_s = Complex64::new(b_minus[s], 0.0);
                azip!((a_plus_val in a_plus.slice_mut(s![s, ..]), a_minus_val in a_minus.slice_mut(s![s, ..]), &factor in &phase_factors) {
                    *a_plus_val += b_plus_s * factor;
                    *a_minus_val += b_minus_s * factor;
                });
            }
        }

        (a_plus, a_minus)
    }

    /// Returns an array of shape (2, n_sets, n_w, n_t) containing integrand_dt_plus and integrand_dt_minus
    /// stacked along the first axis (0 for plus, 1 for minus).
    /// Panics if resolution is not set.
    pub fn compute_c_integrand(
        &self,
        w_arr: ArrayView1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> Array4<Complex64> {
        let n_interior = self.bubbles_interior.shape()[0];
        let n_sets = self.coefficients_sets.shape()[0];
        let n_w = w_arr.len();
        let n_t = n_t;
        let t_arr = Array1::linspace(0.0, t_max, n_t);
        let dt = t_max / (n_t - 1) as f64;

        // Initialize the output array with shape (2, n_sets, n_w, n_t)
        let mut integrand = Array4::zeros((2, n_sets, n_w, n_t));

        // Precompute z_a for each a_idx
        let z_a_vec: Vec<f64> = (0..n_interior)
            .map(|a_idx| self.bubbles_interior[[a_idx, 3]])
            .collect();

        // Check if first_colliding_bubbles is precomputed
        let first_colliding_bubbles = self.cached_data.first_colliding_bubbles.as_ref();

        if let Some(first_bubble_cache) = first_colliding_bubbles {
            // Original approach: use precomputed first_colliding_bubbles
            // Precompute delta_tab_grid for each a_idx in parallel
            let delta_tab_grids: Vec<Array2<f64>> = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .map(|a_idx| {
                        let first_bubble = first_bubble_cache.slice(s![a_idx, .., ..]);
                        self.compute_delta_tab(a_idx, first_bubble)
                    })
                    .collect()
            });

            // Compute integrand slices in parallel
            let results: Vec<(usize, Array2<Complex64>, Array2<Complex64>)> =
                self.thread_pool.install(|| {
                    t_arr
                        .iter()
                        .enumerate()
                        .par_bridge()
                        .map(|(t_idx, &t)| {
                            let mut integrand_dt_plus: Array2<Complex64> =
                                Array2::zeros((n_sets, n_w));
                            let mut integrand_dt_minus: Array2<Complex64> =
                                Array2::zeros((n_sets, n_w));

                            // Sum over interior bubbles only
                            for a_idx in 0..n_interior {
                                // Skip bubbles that nucleate after time t
                                let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                                if t_nucleation >= t {
                                    continue;
                                }

                                let first_bubble = first_bubble_cache.slice(s![a_idx, .., ..]);
                                let delta_tab = delta_tab_grids[a_idx].view();
                                let z_a = z_a_vec[a_idx];
                                let (a_plus, a_minus) =
                                    self.compute_a_matrix(a_idx, w_arr, t, first_bubble, delta_tab);
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integrand_dt_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight;
                                        integrand_dt_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight;
                                    }
                                }
                            }

                            // Scale by dt
                            let dt_complex = Complex64::new(dt, 0.0);
                            integrand_dt_plus *= dt_complex;
                            integrand_dt_minus *= dt_complex;

                            (t_idx, integrand_dt_plus, integrand_dt_minus)
                        })
                        .collect()
                });

            // Assign results to integrand array
            for (t_idx, integrand_dt_plus, integrand_dt_minus) in results {
                for s in 0..n_sets {
                    for w_idx in 0..n_w {
                        integrand[[0, s, w_idx, t_idx]] = integrand_dt_plus[[s, w_idx]];
                        integrand[[1, s, w_idx, t_idx]] = integrand_dt_minus[[s, w_idx]];
                    }
                }
            }
        } else {
            // On-the-fly approach: compute first_bubble for each a_idx
            let results: Vec<(usize, Array2<Complex64>, Array2<Complex64>)> =
                self.thread_pool.install(|| {
                    t_arr
                        .iter()
                        .enumerate()
                        .par_bridge()
                        .map(|(t_idx, &t)| {
                            let mut integrand_dt_plus: Array2<Complex64> =
                                Array2::zeros((n_sets, n_w));
                            let mut integrand_dt_minus: Array2<Complex64> =
                                Array2::zeros((n_sets, n_w));

                            // Sum over interior bubbles only
                            for a_idx in 0..n_interior {
                                // Skip bubbles that nucleate after time t
                                let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                                if t_nucleation >= t {
                                    continue;
                                }

                                // Compute first_bubble on-the-fly
                                let first_bubble = self.compute_first_colliding_bubble(a_idx);
                                let delta_tab = self.compute_delta_tab(a_idx, first_bubble.view());
                                let z_a = z_a_vec[a_idx];
                                let (a_plus, a_minus) = self.compute_a_matrix(
                                    a_idx,
                                    w_arr,
                                    t,
                                    first_bubble.view(),
                                    delta_tab.view(),
                                );
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integrand_dt_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight;
                                        integrand_dt_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight;
                                    }
                                }
                            }

                            // Scale by dt
                            let dt_complex = Complex64::new(dt, 0.0);
                            integrand_dt_plus *= dt_complex;
                            integrand_dt_minus *= dt_complex;

                            (t_idx, integrand_dt_plus, integrand_dt_minus)
                        })
                        .collect()
                });

            // Assign results to integrand array
            for (t_idx, integrand_dt_plus, integrand_dt_minus) in results {
                for s in 0..n_sets {
                    for w_idx in 0..n_w {
                        integrand[[0, s, w_idx, t_idx]] = integrand_dt_plus[[s, w_idx]];
                        integrand[[1, s, w_idx, t_idx]] = integrand_dt_minus[[s, w_idx]];
                    }
                }
            }
        }

        let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
        integrand *= factor;

        integrand
    }

    /// Computes the C-matrix by integrating over time and bubble indices,
    /// returning a stacked array of `c_plus` and `c_minus` components.
    /// Panics if resolution is not set.
    pub fn compute_c_matrix(
        &mut self,
        w_arr: ArrayView1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> Array3<Complex64> {
        let n_interior = self.bubbles_interior.shape()[0];
        let n_sets = self.coefficients_sets.shape()[0];
        let n_w = w_arr.len();
        let t_arr = Array1::linspace(0.0, t_max, n_t);
        let dt = t_max / (n_t - 1) as f64;

        // Precompute z_a for each a_idx
        let z_a_vec: Vec<f64> = (0..n_interior)
            .map(|a_idx| self.bubbles_interior[[a_idx, 3]])
            .collect();

        let first_colliding_bubbles = self.cached_data.first_colliding_bubbles.as_ref();

        if let Some(first_colliding_bubbles) = first_colliding_bubbles {
            // Original approach: use precomputed first_colliding_bubbles
            // Precompute delta_tab_grid for each a_idx in parallel
            let delta_tab_grid: Vec<Array2<f64>> = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .map(|a_idx| {
                        self.compute_delta_tab(
                            a_idx,
                            first_colliding_bubbles.slice(s![a_idx, .., ..]),
                        )
                    })
                    .collect()
            });

            let (c_plus, c_minus) = self.thread_pool.install(|| {
                t_arr
                    .iter()
                    .enumerate()
                    .par_bridge()
                    .fold(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(mut integral_plus, mut integral_minus), (t_idx, &t)| {
                            let mut integrand_dt_plus = Array2::zeros((n_sets, n_w));
                            let mut integrand_dt_minus = Array2::zeros((n_sets, n_w));
                            // Sum over interior bubbles only
                            for a_idx in 0..n_interior {
                                // Skip bubbles that nucleate after time t
                                let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                                if t_nucleation >= t {
                                    continue;
                                }

                                let z_a = z_a_vec[a_idx];
                                let (a_plus, a_minus) = self.compute_a_matrix(
                                    a_idx,
                                    w_arr,
                                    t,
                                    first_colliding_bubbles.slice(s![a_idx, .., ..]),
                                    delta_tab_grid[a_idx].view(),
                                );
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integrand_dt_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight;
                                        integrand_dt_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight;
                                    }
                                }
                            }
                            let dt_complex = Complex64::new(dt, 0.0);
                            integrand_dt_plus *= dt_complex;
                            integrand_dt_minus *= dt_complex;
                            for s in 0..n_sets {
                                for w_idx in 0..n_w {
                                    integral_plus[[s, w_idx]] += integrand_dt_plus[[s, w_idx]];
                                    integral_minus[[s, w_idx]] += integrand_dt_minus[[s, w_idx]];
                                }
                            }
                            (integral_plus, integral_minus)
                        },
                    )
                    .reduce(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(plus1, minus1), (plus2, minus2)| (plus1 + plus2, minus1 + minus2),
                    )
            });
            let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
            let c_plus = c_plus * factor;
            let c_minus = c_minus * factor;

            stack(Axis(0), &[c_plus.view(), c_minus.view()]).unwrap()
        } else {
            // On-the-fly approach: compute first_colliding_bubbles_a for each a_idx
            let (c_plus, c_minus) = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .fold(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(mut integral_plus, mut integral_minus), a_idx| {
                            // Compute first_colliding_bubbles_a and delta_tab once per a_idx
                            let first_colliding_bubbles_a =
                                self.compute_first_colliding_bubble(a_idx);
                            let delta_tab =
                                self.compute_delta_tab(a_idx, first_colliding_bubbles_a.view());

                            // Iterate over time points
                            for (t_idx, &t) in t_arr.iter().enumerate() {
                                // Skip bubbles that nucleate after time t
                                let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                                if t_nucleation >= t {
                                    continue;
                                }

                                let z_a = z_a_vec[a_idx];
                                let (a_plus, a_minus) = self.compute_a_matrix(
                                    a_idx,
                                    w_arr,
                                    t,
                                    first_colliding_bubbles_a.view(),
                                    delta_tab.view(),
                                );
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integral_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight * dt;
                                        integral_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight * dt;
                                    }
                                }
                            }
                            (integral_plus, integral_minus)
                        },
                    )
                    .reduce(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(plus1, minus1), (plus2, minus2)| (plus1 + plus2, minus1 + minus2),
                    )
            });

            let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
            let c_plus = c_plus * factor;
            let c_minus = c_minus * factor;

            stack(Axis(0), &[c_plus.view(), c_minus.view()]).unwrap()
        }
    }

    /// Computes the C-matrix by integrating over time and bubble indices,
    /// returning a stacked array of `c_plus` and `c_minus` components.
    /// Panics if resolution is not set.
    pub fn compute_c_matrix_fixed_bubble(
        &mut self,
        a_idx: usize,
        w_arr: ArrayView1<f64>,
        t_max: f64,
        n_t: usize,
    ) -> Array3<Complex64> {
        let n_interior = self.bubbles_interior.shape()[0];
        let n_sets = self.coefficients_sets.shape()[0];
        let n_w = w_arr.len();
        let t_arr = Array1::linspace(0.0, t_max, n_t);
        let dt = t_max / (n_t - 1) as f64;

        // Precompute z_a for each a_idx
        let z_a = self.bubbles_interior[[a_idx, 3]];

        let first_colliding_bubbles = self.cached_data.first_colliding_bubbles.as_ref();

        if let Some(first_colliding_bubbles) = first_colliding_bubbles {
            // Original approach: use precomputed first_colliding_bubbles
            // Precompute delta_tab_grid for each a_idx in parallel
            let delta_tab_grid: Vec<Array2<f64>> = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .map(|a_idx| {
                        self.compute_delta_tab(
                            a_idx,
                            first_colliding_bubbles.slice(s![a_idx, .., ..]),
                        )
                    })
                    .collect()
            });

            let (c_plus, c_minus) = self.thread_pool.install(|| {
                t_arr
                    .iter()
                    .enumerate()
                    .par_bridge()
                    .fold(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(mut integral_plus, mut integral_minus), (t_idx, &t)| {
                            let mut integrand_dt_plus = Array2::zeros((n_sets, n_w));
                            let mut integrand_dt_minus = Array2::zeros((n_sets, n_w));
                            // Skip bubbles that nucleate after time t
                            let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                            if t_nucleation <= t {
                                let (a_plus, a_minus) = self.compute_a_matrix(
                                    a_idx,
                                    w_arr,
                                    t,
                                    first_colliding_bubbles.slice(s![a_idx, .., ..]),
                                    delta_tab_grid[a_idx].view(),
                                );
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integrand_dt_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight;
                                        integrand_dt_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight;
                                    }
                                }
                                let dt_complex = Complex64::new(dt, 0.0);
                                integrand_dt_plus *= dt_complex;
                                integrand_dt_minus *= dt_complex;
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        integral_plus[[s, w_idx]] += integrand_dt_plus[[s, w_idx]];
                                        integral_minus[[s, w_idx]] +=
                                            integrand_dt_minus[[s, w_idx]];
                                    }
                                }
                            }
                            (integral_plus, integral_minus)
                        },
                    )
                    .reduce(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(plus1, minus1), (plus2, minus2)| (plus1 + plus2, minus1 + minus2),
                    )
            });
            let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
            let c_plus = c_plus * factor;
            let c_minus = c_minus * factor;

            stack(Axis(0), &[c_plus.view(), c_minus.view()]).unwrap()
        } else {
            // On-the-fly approach: compute first_colliding_bubbles_a for each a_idx
            let (c_plus, c_minus) = self.thread_pool.install(|| {
                (0..n_interior)
                    .into_par_iter()
                    .fold(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(mut integral_plus, mut integral_minus), a_idx| {
                            // Compute first_colliding_bubbles_a and delta_tab once per a_idx
                            let first_colliding_bubbles_a =
                                self.compute_first_colliding_bubble(a_idx);
                            let delta_tab =
                                self.compute_delta_tab(a_idx, first_colliding_bubbles_a.view());

                            // Iterate over time points
                            for (t_idx, &t) in t_arr.iter().enumerate() {
                                // Skip bubbles that nucleate after time t
                                let t_nucleation = self.bubbles_interior[[a_idx, 0]];
                                if t_nucleation >= t {
                                    continue;
                                }

                                let (a_plus, a_minus) = self.compute_a_matrix(
                                    a_idx,
                                    w_arr,
                                    t,
                                    first_colliding_bubbles_a.view(),
                                    delta_tab.view(),
                                );
                                for s in 0..n_sets {
                                    for w_idx in 0..n_w {
                                        let w = w_arr[w_idx];
                                        let complex_phase =
                                            Complex64::new(0.0, w * (t - z_a)).exp();
                                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                                            0.5
                                        } else {
                                            1.0
                                        };
                                        integral_plus[[s, w_idx]] +=
                                            a_plus[[s, w_idx]] * complex_phase * weight * dt;
                                        integral_minus[[s, w_idx]] +=
                                            a_minus[[s, w_idx]] * complex_phase * weight * dt;
                                    }
                                }
                            }
                            (integral_plus, integral_minus)
                        },
                    )
                    .reduce(
                        || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                        |(plus1, minus1), (plus2, minus2)| (plus1 + plus2, minus1 + minus2),
                    )
            });

            let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
            let c_plus = c_plus * factor;
            let c_minus = c_minus * factor;

            stack(Axis(0), &[c_plus.view(), c_minus.view()]).unwrap()
        }
    }
}

/// Computes the Minkowski dot product of two 4-vectors using the (-, +, +, +) metric.
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

/// Determines if bubble `b` is the first to collide with a direction `x` relative to bubble `a`,
/// considering potential earlier collisions with other bubbles `c`.
pub fn check_collision_point(
    delta_ba: ArrayView1<f64>,
    delta_ca: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> bool {
    let delta_ba_dot_x = dot_minkowski_vec(delta_ba, x);
    if delta_ba_dot_x <= 0.0 {
        return false;
    }

    // Compute Minkowski norms
    let delta_ba_norm = dot_minkowski_vec(delta_ba, delta_ba);
    let delta_ca_norm = dot_minkowski_vec(delta_ca, delta_ca);

    // Compute collision vector element-wise
    let collision_vec = {
        let delta_ba_scaled = delta_ba.mapv(|v| v * delta_ca_norm);
        let delta_ca_scaled = delta_ca.mapv(|v| v * delta_ba_norm);
        delta_ba_scaled - delta_ca_scaled
    };

    // Check the dot product of the collision vector with x
    let dot_collision_x = dot_minkowski_vec(collision_vec.view(), x);
    dot_collision_x > 0.0
}

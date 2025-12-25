use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    GenerateBubblesExterior, LatticeGeometry, TransformationIsometry3,
};
use crate::many_bubbles::lattice_bubbles::{BubbleIndex, LatticeBubbles, LatticeBubblesError};
use nalgebra::{DMatrix, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::prelude::*;
use ndarray::stack;
use num_complex::Complex64;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};
use thiserror::Error;

/// Represents the collision status of a direction relative to a reference bubble.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CollisionStatus {
    NeverCollided = 0,
    AlreadyCollided = 1,
    NotYetCollided = 2,
}

/// Custom error type for `BulkFlow` operations.
#[derive(Error, Debug)]
pub enum BulkFlowError {
    #[error("Field '{0}' is not initialized")]
    UninitializedField(String),

    #[error("Index {index} out of bounds for max {max}")]
    InvalidIndex { index: usize, max: usize },

    #[error("Invalid resolution: {0}")]
    InvalidResolution(String),

    #[error("Invalid time range: t_begin={begin} > t_end={end}")]
    InvalidTimeRange { begin: f64, end: f64 },

    #[error("Array shape mismatch: {0}")]
    ArrayShapeMismatch(String),

    #[error("Failed to build thread pool")]
    ThreadPoolBuildError(
        #[from]
        #[source]
        ThreadPoolBuildError,
    ),

    #[error("Causality Error: Bubble {a} is formed inside bubble {b}")]
    BubbleFormedInsideBubble { a: BubbleIndex, b: BubbleIndex },

    #[error("Bubbles Error")]
    BubblesError(#[from] LatticeBubblesError),
}

#[derive(Debug)]
pub struct BulkFlow<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    pub bubbles: LatticeBubbles<L>,
    pub first_colliding_bubbles: Option<Array3<BubbleIndex>>,
    pub coefficients_sets: Array2<f64>,
    pub powers_sets: Array2<f64>,
    pub damping_width: Option<f64>,
    pub active_bubbles: Array1<bool>,
    pub thread_pool: ThreadPool,
    pub n_cos_thetax: Option<usize>,
    pub n_phix: Option<usize>,
    pub cos_thetax: Option<Array1<f64>>,
    pub phix: Option<Array1<f64>>,
    pub direction_vectors: Option<DMatrix<Vector4<f64>>>,
}

impl<L> BulkFlow<L>
where
    L: LatticeGeometry + TransformationIsometry3 + GenerateBubblesExterior,
{
    /// Create a new `BulkFlow`.
    ///
    /// * `bubbles` – spacetime coordinates of nucleated bubbles inside and outside the lattice
    /// * `sort_by_time`    – if `true` the two bubble lists are sorted by formation time
    pub fn new(bubbles: LatticeBubbles<L>) -> Result<Self, BulkFlowError> {
        let default_num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(default_num_threads)
            .build()
            .map_err(BulkFlowError::ThreadPoolBuildError)?;

        Ok(BulkFlow {
            bubbles,
            first_colliding_bubbles: None,
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
        })
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> Result<(), BulkFlowError> {
        self.thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(BulkFlowError::ThreadPoolBuildError)?;
        Ok(())
    }

    pub fn first_colliding_bubbles(&self) -> Option<&Array3<BubbleIndex>> {
        self.first_colliding_bubbles.as_ref()
    }

    pub fn compute_first_colliding_bubble(
        &self,
        a_idx: usize,
    ) -> Result<Array2<BubbleIndex>, BulkFlowError> {
        let n_cos_thetax = self
            .n_cos_thetax
            .ok_or_else(|| BulkFlowError::UninitializedField("n_cos_thetax".to_string()))?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| BulkFlowError::UninitializedField("n_phix".to_string()))?;
        let direction_vectors = self
            .direction_vectors
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("direction_vectors".to_string()))?;
        let n_interior = self.bubbles.interior.n_bubbles();
        let n_exterior = self.bubbles.exterior.n_bubbles();
        let n_total = n_interior + n_exterior;
        let tolerance = 1e-10;

        // Pre-extract shared read-only data
        let delta = &self.bubbles.delta;
        let delta_squared = &self.bubbles.delta_squared;

        // Allocate output
        let mut first_bubble = Array2::from_elem((n_cos_thetax, n_phix), BubbleIndex::None);

        // Parallelize over (i,j) using Rayon
        self.thread_pool.install(|| {
            first_bubble
                .indexed_iter_mut()
                .par_bridge()
                .for_each(|((i, j), bubble_index)| {
                    let x_vec = direction_vectors[(i, j)];
                    let mut earliest_bubble_idx = BubbleIndex::None;
                    let mut earliest_delta_tab = f64::INFINITY;

                    for b_total in 0..n_total {
                        let skip_self = b_total < n_interior && b_total == a_idx;
                        if skip_self {
                            continue;
                        }
                        let delta_ba = delta[(a_idx, b_total)];
                        let delta_ba_squared = delta_squared[(a_idx, b_total)];
                        let dot_ba_x = delta_ba.scalar(&x_vec);
                        if dot_ba_x <= tolerance {
                            continue;
                        }
                        let delta_tab = delta_ba_squared / (2.0 * dot_ba_x);
                        if delta_tab <= 0.0 || delta_tab >= earliest_delta_tab {
                            continue;
                        }

                        let mut is_first = true;
                        for c_total in 0..n_total {
                            let skip_self_c = c_total < n_interior && c_total == a_idx;
                            if skip_self_c || c_total == b_total {
                                continue;
                            }
                            let delta_ca = delta[(a_idx, c_total)];
                            if !check_collision_point(&delta_ba, &delta_ca, &x_vec) {
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

                    *bubble_index = earliest_bubble_idx;
                });
        });

        Ok(first_bubble)
    }

    pub fn set_resolution(
        &mut self,
        n_cos_thetax: usize,
        n_phix: usize,
        precompute_first_bubbles: bool,
    ) -> Result<(), BulkFlowError> {
        if n_cos_thetax < 2 || n_phix < 2 {
            return Err(BulkFlowError::InvalidResolution(
                "Angular resolutions must be greater than one".to_string(),
            ));
        }
        self.n_cos_thetax = Some(n_cos_thetax);
        self.n_phix = Some(n_phix);
        let cos_thetax = Array1::linspace(-1.0, 1.0, n_cos_thetax);
        let phix = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n_phix);
        let direction_vectors: DMatrix<Vector4<f64>> =
            DMatrix::from_fn(n_cos_thetax, n_phix, |i, j| {
                let cos_theta = cos_thetax[i];
                let sin_theta = f64::sqrt(1.0 - cos_theta * cos_theta).abs();
                let phi = phix[j];

                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                Vector4::new(
                    1.0,                 // t
                    sin_theta * cos_phi, // x
                    sin_theta * sin_phi, // y
                    cos_theta,           // z
                )
            });

        self.direction_vectors = Some(direction_vectors);

        if precompute_first_bubbles {
            let first_colliding_bubbles: Vec<Array2<BubbleIndex>> =
                (0..self.bubbles.interior.n_bubbles())
                    .into_par_iter()
                    .map(|a_idx| self.compute_first_colliding_bubble(a_idx).unwrap())
                    .collect();
            self.first_colliding_bubbles = Some(
                stack(
                    Axis(0),
                    &first_colliding_bubbles
                        .iter()
                        .map(|x| x.view())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| {
                    BulkFlowError::ArrayShapeMismatch(format!("Failed to stack arrays: {}", e))
                })?,
            );
        } else {
            self.first_colliding_bubbles = None;
        }

        self.cos_thetax = Some(cos_thetax);
        self.phix = Some(phix);
        Ok(())
    }

    pub fn set_active_sets(&mut self, active_sets: Array1<bool>) {
        self.active_bubbles = active_sets;
    }

    // pub fn bubbles_interior(&self) -> &Array2<f64> {
    //     &self.bubbles.interior
    // }
    //
    // pub fn bubbles_exterior(&self) -> &Array2<f64> {
    //     &self.bubbles.exterior
    // }

    pub fn coefficients_sets(&self) -> &Array2<f64> {
        &self.coefficients_sets
    }

    pub fn set_coefficients_sets(&mut self, coefficients_sets: Array2<f64>) {
        self.coefficients_sets = coefficients_sets;
    }

    pub fn powers_sets(&self) -> &Array2<f64> {
        &self.powers_sets
    }

    pub fn set_powers_sets(&mut self, powers_sets: Array2<f64>) {
        self.powers_sets = powers_sets;
    }

    pub fn active_sets(&self) -> &Array1<bool> {
        &self.active_bubbles
    }

    pub fn cos_thetax(&self) -> Result<&Array1<f64>, BulkFlowError> {
        self.cos_thetax
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("cos_thetax".to_string()))
    }

    pub fn phix(&self) -> Result<&Array1<f64>, BulkFlowError> {
        self.phix
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("phix".to_string()))
    }

    pub fn set_gradient_scaling_params(
        &mut self,
        coefficients_sets: Vec<Vec<f64>>,
        powers_sets: Vec<Vec<f64>>,
        damping_width: Option<f64>,
    ) -> Result<(), BulkFlowError> {
        let n_sets = coefficients_sets.len();
        let n_coeffs = coefficients_sets.first().map_or(0, |v| v.len());
        if coefficients_sets.iter().any(|v| v.len() != n_coeffs) {
            return Err(BulkFlowError::ArrayShapeMismatch(
                "All coefficient sets must have the same length".to_string(),
            ));
        }
        let coefficients_sets = Array2::from_shape_vec(
            (n_sets, n_coeffs),
            coefficients_sets.into_iter().flatten().collect(),
        )
        .map_err(|_| {
            BulkFlowError::ArrayShapeMismatch("Invalid shape for coefficients_sets".to_string())
        })?;

        let n_powers = powers_sets.len();
        let n_pows = powers_sets.first().map_or(0, |v| v.len());
        if powers_sets.iter().any(|v| v.len() != n_pows) {
            return Err(BulkFlowError::ArrayShapeMismatch(
                "All power sets must have the same length".to_string(),
            ));
        }
        let powers_sets =
            Array2::from_shape_vec((n_powers, n_pows), powers_sets.into_iter().flatten().collect())
                .map_err(|_| {
                    BulkFlowError::ArrayShapeMismatch("Invalid shape for powers_sets".to_string())
                })?;

        if n_sets != n_powers {
            return Err(BulkFlowError::ArrayShapeMismatch(
                "Number of coefficient sets and power sets must be equal".to_string(),
            ));
        }

        let mut active_bubbles = Array1::from_elem(n_sets, false);
        for s in 0..n_sets {
            let coeff_sum = coefficients_sets.slice(s![s, ..]).sum();
            if coeff_sum.abs() >= 1e-10 && (coeff_sum - 1.0).abs() > 1e-10 {
                return Err(BulkFlowError::ArrayShapeMismatch(format!(
                    "Set {}: coefficients must sum to 0 or 1",
                    s
                )));
            }
            active_bubbles[s] = coeff_sum.abs() > 1e-10;
        }

        self.coefficients_sets = coefficients_sets;
        self.powers_sets = powers_sets;
        self.damping_width = damping_width;
        self.active_bubbles = active_bubbles;
        Ok(())
    }

    pub fn compute_collision_status(
        &self,
        a_idx: usize,
        t: f64,
        first_bubble: &ArrayRef2<BubbleIndex>,
        delta_tab_grid: &ArrayRef2<f64>,
    ) -> Result<Array2<CollisionStatus>, BulkFlowError> {
        let n_cos_thetax = self
            .n_cos_thetax
            .ok_or_else(|| BulkFlowError::UninitializedField("n_cos_thetax".to_string()))?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| BulkFlowError::UninitializedField("n_phix".to_string()))?;
        if first_bubble.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(BulkFlowError::ArrayShapeMismatch(format!(
                "Input arrays must match resolution: expected [{}, {}], got first_bubble: {:?}, delta_tab_grid: {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape(),
                delta_tab_grid.shape()
            )));
        }
        let mut collision_status =
            Array2::from_elem((n_cos_thetax, n_phix), CollisionStatus::NeverCollided);
        let ta = self.bubbles.interior.spacetime[a_idx][0];
        let delta_ta = t - ta;

        for i in 0..n_cos_thetax {
            for j in 0..n_phix {
                if first_bubble[[i, j]] == BubbleIndex::None {
                    collision_status[[i, j]] = CollisionStatus::NeverCollided;
                    continue;
                }
                let delta_tab_val = delta_tab_grid[[i, j]];
                if delta_tab_val > 0.0 && delta_ta >= delta_tab_val {
                    collision_status[[i, j]] = CollisionStatus::AlreadyCollided;
                } else {
                    collision_status[[i, j]] = CollisionStatus::NotYetCollided;
                }
            }
        }

        Ok(collision_status)
    }

    pub fn compute_delta_tab(
        &self,
        a_idx: usize,
        first_bubble: &ArrayRef2<BubbleIndex>,
    ) -> Result<Array2<f64>, BulkFlowError> {
        let n_cos_thetax = self
            .n_cos_thetax
            .ok_or_else(|| BulkFlowError::UninitializedField("n_cos_thetax".to_string()))?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| BulkFlowError::UninitializedField("n_phix".to_string()))?;
        let direction_vectors = self
            .direction_vectors
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("direction_vectors".to_string()))?;
        if first_bubble.shape() != [n_cos_thetax, n_phix] {
            return Err(BulkFlowError::ArrayShapeMismatch(format!(
                "First bubble array must match resolution: expected [{}, {}], got {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape()
            )));
        }
        let n_interior = self.bubbles.interior.n_bubbles();
        let mut delta_tab_grid = Array2::zeros((n_cos_thetax, n_phix));

        for i in 0..n_cos_thetax {
            // TODO: This loop can be replaced by iterating over segment with const BubbleIndex
            for j in 0..n_phix {
                let b_total = match first_bubble[[i, j]] {
                    BubbleIndex::None => {
                        continue;
                    }
                    BubbleIndex::Interior(b_idx) => b_idx,
                    BubbleIndex::Exterior(b_idx) => n_interior + b_idx,
                };
                let delta_ba = self.bubbles.delta[(a_idx, b_total)];
                let x_vec = direction_vectors[(i, j)];
                let delta_ba_squared = self.bubbles.delta_squared[(a_idx, b_total)];
                let dot_ba_x = delta_ba.scalar(&x_vec);
                if dot_ba_x.abs() < 1e-10 {
                    delta_tab_grid[[i, j]] = 0.0;
                    continue;
                }
                let delta_tab_val = delta_ba_squared / (2.0 * dot_ba_x);
                delta_tab_grid[[i, j]] = delta_tab_val;
            }
        }

        Ok(delta_tab_grid)
    }

    pub fn compute_b_integral(
        &self,
        cos_thetax_idx: usize,
        collision_status_grid: &ArrayRef2<CollisionStatus>,
        delta_tab_grid: &ArrayRef2<f64>,
        delta_ta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), BulkFlowError> {
        let n_cos_thetax = self
            .n_cos_thetax
            .ok_or_else(|| BulkFlowError::UninitializedField("n_cos_thetax".to_string()))?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| BulkFlowError::UninitializedField("n_phix".to_string()))?;
        let cos_thetax_grid = self
            .cos_thetax
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("cos_thetax".to_string()))?;
        let phix = self
            .phix
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("phix".to_string()))?;
        let n_sets = self.coefficients_sets.nrows();

        if cos_thetax_idx >= n_cos_thetax {
            return Err(BulkFlowError::InvalidIndex {
                index: cos_thetax_idx,
                max: n_cos_thetax,
            });
        }
        if collision_status_grid.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(BulkFlowError::ArrayShapeMismatch(format!(
                "Grids must be [{}, {}], got status: {:?}, delta_tab: {:?}",
                n_cos_thetax,
                n_phix,
                collision_status_grid.shape(),
                delta_tab_grid.shape()
            )));
        }

        let cos_thetax = cos_thetax_grid[cos_thetax_idx];
        let sin_squared_thetax = 1.0 - cos_thetax.powi(2);
        let dphi = 2.0 * std::f64::consts::PI / n_phix as f64;

        let mut b_plus_arr = Array1::zeros(n_sets);
        let mut b_minus_arr = Array1::zeros(n_sets);

        let mut sin_2phi_row = Array1::zeros(n_phix);
        let mut cos_2phi_row = Array1::zeros(n_phix);
        azip!((sin_2phi in &mut sin_2phi_row, p in phix) *sin_2phi = (2.0 * p).sin());
        azip!((cos_2phi in &mut cos_2phi_row, p in phix) *cos_2phi = (2.0 * p).cos());

        let status_row = collision_status_grid.slice(s![cos_thetax_idx, ..]);
        let delta_tab_row = delta_tab_grid.slice(s![cos_thetax_idx, ..]);

        // Trapezoidal weights: 0.5 at endpoints, 1.0 in middle
        let mut weights = Array1::from_elem(n_phix, 1.0);
        weights[0] = 0.5;
        weights[n_phix - 1] = 0.5;

        if delta_ta <= 0.0 {
            return Ok((b_plus_arr, b_minus_arr));
        }
        let delta_ta_cubed = delta_ta.powi(3);

        for s in 0..n_sets {
            let mut factors = Array1::zeros(n_phix);

            azip!((factor in &mut factors, &status in &status_row, &delta_tab in &delta_tab_row) {
                match status {
                    CollisionStatus::NeverCollided | CollisionStatus::NotYetCollided => {
                        *factor = 1.0;
                    }
                    CollisionStatus::AlreadyCollided => {
                        let ratio = delta_tab / delta_ta;
                        let mut f = 0.0;
                        for k in 0..self.coefficients_sets.shape()[1] {
                            f += self.coefficients_sets[[s, k]] * ratio.powf(self.powers_sets[[s, k]]);
                        }
                        if let Some(damping_width) = self.damping_width {
                            let damp = (-delta_ta * (1.0 - ratio) / damping_width).exp();
                            f *= damp;
                        }
                        *factor = f;
                    }
                }
            });

            let mut contrib_sin = Array1::zeros(n_phix);
            let mut contrib_cos = Array1::zeros(n_phix);
            azip!((sin_val in &mut contrib_sin, &sin2 in &sin_2phi_row, &factor in &factors, &weight in &weights) {
                *sin_val = sin2 * factor * weight;
            });
            azip!((cos_val in &mut contrib_cos, &cos2 in &cos_2phi_row, &scale in &factors, &weight in &weights) {
                *cos_val = cos2 * scale * weight;
            });

            let integral_minus = contrib_sin.sum() * dphi * delta_ta_cubed;
            let integral_plus = contrib_cos.sum() * dphi * delta_ta_cubed;

            b_plus_arr[s] += 0.5 * sin_squared_thetax * integral_plus;
            b_minus_arr[s] += 0.5 * sin_squared_thetax * integral_minus;
        }

        Ok((b_plus_arr, b_minus_arr))
    }

    pub fn compute_a_integral<W>(
        &self,
        a_idx: usize,
        w_arr: W,
        t: f64,
        first_bubble: &ArrayRef2<BubbleIndex>,
        delta_tab_grid: &ArrayRef2<f64>,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>), BulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_cos_thetax = self
            .n_cos_thetax
            .ok_or_else(|| BulkFlowError::UninitializedField("n_cos_thetax".to_string()))?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| BulkFlowError::UninitializedField("n_phix".to_string()))?;
        let cos_thetax_grid = self
            .cos_thetax
            .as_ref()
            .ok_or_else(|| BulkFlowError::UninitializedField("cos_thetax".to_string()))?;

        if first_bubble.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(BulkFlowError::ArrayShapeMismatch(format!(
                "Input arrays must be [{}, {}], got first_bubble: {:?}, delta_tab: {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape(),
                delta_tab_grid.shape()
            )));
        }

        let n_w = w_arr.len();
        let n_sets = self.coefficients_sets.nrows();
        let ta = self.bubbles.interior.spacetime[a_idx][0];
        let delta_ta = t - ta;

        // Compute collision status grid
        let collision_status =
            self.compute_collision_status(a_idx, t, &first_bubble, &delta_tab_grid)?;

        let mut a_plus = Array2::<Complex64>::zeros((n_sets, n_w));
        let mut a_minus = Array2::<Complex64>::zeros((n_sets, n_w));
        let dcos_thetax = 2.0 / (n_cos_thetax - 1) as f64;

        for i in 0..n_cos_thetax {
            let (b_plus, b_minus) =
                self.compute_b_integral(i, &collision_status, &delta_tab_grid, delta_ta)?;

            let cos_thetax_val = cos_thetax_grid[i];
            let phase_base = Complex64::new(0.0, -delta_ta * cos_thetax_val);
            let mut angular_phases = Array1::zeros(n_w);
            for w_idx in 0..n_w {
                angular_phases[w_idx] = (w_arr[w_idx] * phase_base).exp();
            }

            let weight = if i == 0 || i == n_cos_thetax - 1 {
                0.5
            } else {
                1.0
            };
            let phase_factors = angular_phases.mapv(|p| p * dcos_thetax * weight);

            for s in 0..n_sets {
                let b_plus_s = Complex64::new(b_plus[s], 0.0);
                let b_minus_s = Complex64::new(b_minus[s], 0.0);
                azip!((a_plus_val in a_plus.slice_mut(s![s, ..]), a_minus_val in a_minus.slice_mut(s![s, ..]), &factor in &phase_factors) {
                    *a_plus_val += b_plus_s * factor;
                    *a_minus_val += b_minus_s * factor;
                });
            }
        }

        Ok((a_plus, a_minus))
    }

    pub fn compute_c_integrand_fixed_bubble<W>(
        &self,
        a_idx: usize,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> Result<Array4<Complex64>, BulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        let t_nucleation = self.bubbles.interior.spacetime[a_idx][0];
        if t_nucleation >= t_end {
            return Ok(Array4::zeros((2, n_sets, n_w, n_t)));
        }

        let t_begin = t_begin.unwrap_or(0.0);
        if t_begin > t_end {
            return Err(BulkFlowError::InvalidTimeRange {
                begin: t_begin,
                end: t_end,
            });
        }
        let t_arr = Array1::linspace(t_begin, t_end, n_t).to_vec();
        let dt = if n_t > 1 { t_arr[1] - t_arr[0] } else { 0.0 };
        let z_a = self.bubbles.interior.spacetime[a_idx][3];

        let first_colliding_bubbles_with_a: Array2<BubbleIndex> =
            if let Some(cache) = self.first_colliding_bubbles.as_ref() {
                cache.slice(s![a_idx, .., ..]).to_owned()
            } else {
                self.compute_first_colliding_bubble(a_idx)?
            };
        let delta_tab = self.compute_delta_tab(a_idx, &first_colliding_bubbles_with_a)?;

        // Final result
        let mut c_integrand = Array4::<Complex64>::zeros((2, n_sets, n_w, n_t));

        let time_and_integrand_results: Vec<
            Result<(usize, Array2<Complex64>, Array2<Complex64>), BulkFlowError>,
        > = self.thread_pool.install(|| {
            t_arr
                .into_par_iter()
                .enumerate()
                .map(|(t_idx, t)| {
                    if t <= t_nucleation {
                        let zero_p = Array2::zeros((n_sets, n_w));
                        let zero_m = Array2::zeros((n_sets, n_w));
                        return Ok((t_idx, zero_p, zero_m));
                    }

                    let (a_plus, a_minus) = self.compute_a_integral(
                        a_idx,
                        w_arr,
                        t,
                        &first_colliding_bubbles_with_a,
                        &delta_tab,
                    )?;

                    let weight = if t_idx == 0 || t_idx == n_t - 1 {
                        0.5
                    } else {
                        1.0
                    };
                    let dt_weight = Complex64::new(dt * weight, 0.0);

                    let mut c_integrand_plus = Array2::zeros((n_sets, n_w));
                    let mut c_integrand_minus = Array2::zeros((n_sets, n_w));

                    for s in 0..n_sets {
                        for w_idx in 0..n_w {
                            let w = w_arr[w_idx];
                            let phase = Complex64::new(0.0, w * (t - z_a)).exp();
                            c_integrand_plus[[s, w_idx]] = a_plus[[s, w_idx]] * phase * dt_weight;
                            c_integrand_minus[[s, w_idx]] = a_minus[[s, w_idx]] * phase * dt_weight;
                        }
                    }

                    Ok((t_idx, c_integrand_plus, c_integrand_minus))
                })
                .collect()
        });

        for result in time_and_integrand_results {
            let (t_idx, c_integrand_plus, c_integrand_minus) = result?;
            for s in 0..n_sets {
                for w_idx in 0..n_w {
                    c_integrand[[0, s, w_idx, t_idx]] += c_integrand_plus[[s, w_idx]];
                    c_integrand[[1, s, w_idx, t_idx]] += c_integrand_minus[[s, w_idx]];
                }
            }
        }

        let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
        c_integrand *= factor;

        Ok(c_integrand)
    }

    pub fn compute_c_integral_fixed_bubble<W>(
        &mut self,
        a_idx: usize,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> Result<Array3<Complex64>, BulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();
        let t_begin = t_begin.unwrap_or(0.0);
        if t_begin > t_end {
            return Err(BulkFlowError::InvalidTimeRange {
                begin: t_begin,
                end: t_end,
            });
        }
        let t_arr = Array1::linspace(t_begin, t_end, n_t).to_vec();
        let dt = if n_t > 1 { t_arr[1] - t_arr[0] } else { 0.0 };

        let z_a = self.bubbles.interior.spacetime[a_idx][3];

        let first_colliding_bubbles_with_a: Array2<BubbleIndex> =
            if let Some(cache) = self.first_colliding_bubbles.as_ref() {
                cache.slice(s![a_idx, .., ..]).to_owned()
            } else {
                self.compute_first_colliding_bubble(a_idx)?
            };
        let delta_tab = self.compute_delta_tab(a_idx, &first_colliding_bubbles_with_a)?;

        let (c_plus, c_minus) = self.thread_pool.install(|| {
            t_arr
                .into_par_iter()
                .enumerate()
                .fold(
                    || (Array2::zeros((n_sets, n_w)), Array2::zeros((n_sets, n_w))),
                    |(mut integral_plus, mut integral_minus), (t_idx, t)| {
                        let mut integrand_dt_plus = Array2::zeros((n_sets, n_w));
                        let mut integrand_dt_minus = Array2::zeros((n_sets, n_w));
                        let t_nucleation = self.bubbles.interior.spacetime[a_idx][0];
                        if t_nucleation <= t {
                            let (a_plus, a_minus) = self
                                .compute_a_integral(
                                    a_idx,
                                    w_arr,
                                    t,
                                    &first_colliding_bubbles_with_a,
                                    &delta_tab,
                                )
                                .unwrap();
                            for s in 0..n_sets {
                                for w_idx in 0..n_w {
                                    let w = w_arr[w_idx];
                                    let complex_phase = Complex64::new(0.0, w * (t - z_a)).exp();
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
                                    integral_minus[[s, w_idx]] += integrand_dt_minus[[s, w_idx]];
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
        stack(Axis(0), &[c_plus.view(), c_minus.view()]).map_err(|e| {
            BulkFlowError::ArrayShapeMismatch(format!("Failed to stack arrays: {}", e))
        })
    }

    pub fn compute_c_integrand<W>(
        &self,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<&[usize]>,
    ) -> Result<Array4<Complex64>, BulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_interior = self.bubbles.interior.n_bubbles();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        if n_t < 2 {
            return Err(BulkFlowError::InvalidResolution("n_t must be >= 2".into()));
        }

        // Validate and collect bubble indices
        let bubble_ids: Vec<usize> = match selected_bubbles {
            Some(ids) => {
                if ids.is_empty() {
                    return Ok(Array4::zeros((2, n_sets, n_w, n_t)));
                }
                for &a in ids {
                    if a >= n_interior {
                        return Err(BulkFlowError::InvalidIndex {
                            index: a,
                            max: n_interior,
                        });
                    }
                }
                ids.to_vec()
            }
            None => (0..n_interior).collect(),
        };

        let mut total = Array4::zeros((2, n_sets, n_w, n_t));
        for &a_idx in &bubble_ids {
            total += &self.compute_c_integrand_fixed_bubble(a_idx, w_arr, t_begin, t_end, n_t)?;
        }

        Ok(total)
    }

    pub fn compute_c_integral<W>(
        &mut self,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<&[usize]>,
    ) -> Result<Array3<Complex64>, BulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_interior = self.bubbles.interior.n_bubbles();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        if n_t < 2 {
            return Err(BulkFlowError::InvalidResolution(
                "n_t must be >= 2 for integration".to_string(),
            ));
        }

        // Validate and collect bubble indices
        let bubble_ids: Vec<usize> = match selected_bubbles {
            Some(ids) => {
                if ids.is_empty() {
                    return Ok(Array3::zeros((2, n_sets, n_w)));
                }
                for &a in ids {
                    if a >= n_interior {
                        return Err(BulkFlowError::InvalidIndex {
                            index: a,
                            max: n_interior,
                        });
                    }
                }
                ids.to_vec()
            }
            None => (0..n_interior).collect(),
        };

        let mut c_total = Array3::<Complex64>::zeros((2, n_sets, n_w));
        for &a_idx in &bubble_ids {
            c_total += &self.compute_c_integral_fixed_bubble(a_idx, w_arr, t_begin, t_end, n_t)?;
        }

        Ok(c_total)
    }

    pub fn delta_squared(&self) -> &DMatrix<f64> {
        &self.bubbles.delta_squared
    }

    pub fn bubbles_interior(&self) -> &Bubbles {
        &self.bubbles.interior
    }

    pub fn bubbles_exterior(&self) -> &Bubbles {
        &self.bubbles.exterior
    }
}

pub fn check_collision_point(
    delta_ba: &Vector4<f64>,
    delta_ca: &Vector4<f64>,
    x: &Vector4<f64>,
) -> bool {
    let delta_ba_dot_x = delta_ba.scalar(&x);
    if delta_ba_dot_x <= 0.0 {
        return false;
    }

    let delta_ba_norm = delta_ba.scalar(&delta_ba);
    let delta_ca_norm = delta_ca.scalar(&delta_ca);

    let collision_vec = delta_ba * delta_ca_norm - delta_ca * delta_ba_norm;
    collision_vec.scalar(&x) > 0.0
}

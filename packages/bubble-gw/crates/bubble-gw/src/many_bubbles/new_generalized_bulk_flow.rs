use std::fmt::Debug;

use nalgebra::{DMatrix, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::prelude::*;
use ndarray::stack;
use num::Zero;
use num_complex::Complex64;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};
use thiserror::Error;

use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::GeneralLatticeProperties;
use crate::many_bubbles::lattice_bubbles::{BubbleIndex, LatticeBubbles, LatticeBubblesError};

/// A 3×3 Hermitian tensor stored in upper-triangular order:
/// [xx, xy, xz, yy, yz, zz]
///
/// For real `T`, this reduces to a symmetric tensor.
/// For complex `T`, Hermiticity implies:
///   xx, yy, zz ∈ ℝ (though not enforced — caller ensures),
///   yx = conj(xy), zx = conj(xz), zy = conj(yz).
///
/// This storage layout is optimal for memory and computation.
#[derive(Debug, Clone, PartialEq)]
#[repr(transparent)]
pub struct HermitianTensor3<T>(pub [T; 6]);

impl<T> HermitianTensor3<T> {
    /// Construct from components in upper-triangular order.
    pub const fn new(xx: T, xy: T, xz: T, yy: T, yz: T, zz: T) -> Self {
        Self([xx, xy, xz, yy, yz, zz])
    }

    /// Accessors (read-only)
    pub fn xx(&self) -> &T {
        &self.0[0]
    }
    pub fn xy(&self) -> &T {
        &self.0[1]
    }
    pub fn xz(&self) -> &T {
        &self.0[2]
    }
    pub fn yy(&self) -> &T {
        &self.0[3]
    }
    pub fn yz(&self) -> &T {
        &self.0[4]
    }
    pub fn zz(&self) -> &T {
        &self.0[5]
    }

    /// Borrow underlying array
    pub fn as_array(&self) -> &[T; 6] {
        &self.0
    }
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Mutable access (for accumulation patterns)
    pub fn as_array_mut(&mut self) -> &mut [T; 6] {
        &mut self.0
    }
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> std::ops::Index<usize> for HermitianTensor3<T> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl<T> std::ops::IndexMut<usize> for HermitianTensor3<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl<T> From<[T; 6]> for HermitianTensor3<T> {
    fn from(arr: [T; 6]) -> Self {
        Self(arr)
    }
}

impl<T> From<HermitianTensor3<T>> for [T; 6] {
    fn from(tensor: HermitianTensor3<T>) -> Self {
        tensor.0
    }
}

impl<T: Clone + PartialEq + Debug + Zero + 'static> From<HermitianTensor3<T>>
    for nalgebra::SMatrix<T, 3, 3>
{
    fn from(t: HermitianTensor3<T>) -> Self {
        let mut m = nalgebra::SMatrix::<T, 3, 3>::zeros();
        m[(0, 0)] = t.xx().clone();
        m[(0, 1)] = t.xy().clone();
        m[(1, 0)] = t.xy().clone();
        m[(0, 2)] = t.xz().clone();
        m[(2, 0)] = t.xz().clone();
        m[(1, 1)] = t.yy().clone();
        m[(1, 2)] = t.yz().clone();
        m[(2, 1)] = t.yz().clone();
        m[(2, 2)] = t.zz().clone();
        m
    }
}

/// Represents the collision status of a direction relative to a reference
/// bubble.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CollisionStatus {
    NeverCollided = 0,
    AlreadyCollided = 1,
    NotYetCollided = 2,
}

/// Custom error type for `BulkFlow` operations.
#[derive(Error, Debug)]
pub enum GeneralizedBulkFlowError {
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
/// Computes the frequency-domain bulk-flow integrals $C_{+}(\omega)$ and
/// $C_{\times}(\omega)$ for gravitational-wave emission from colliding vacuum
/// lattice_bubbles in a first-order phase transition. Implements the triple
/// integral: $$
/// C_{+,\times}(\omega) = \frac{1}{6\pi} \sum_n \int dt\ e^{i\omega(t - z_n)}
/// A_{n,\pm}(\omega, t), \\ A_{n,\pm}(\omega, t) = \int_{-1}^{1} d\zeta\
/// e^{-i\omega(t-t_n)\zeta} B_{n,\pm}(\zeta, t), \\ B_{n,\pm}(\zeta, t) =
/// \frac{1-\zeta^2}{2} \int_0^{2\pi} d\phi\ g_{\pm}(\phi)\ (t-t_n)^3 f(t, t_n,
/// t_{n,c}), $$
/// where $g_{+} = \cos 2\phi$, $g_{\times} = \sin 2\phi$, $\zeta = \cos\theta$,
/// and $f$ is the collision-aware scaling function.
#[derive(Debug)]
pub struct GeneralizedBulkFlow<L>
where
    L: GeneralLatticeProperties,
{
    pub lattice_bubbles: LatticeBubbles<L>,
    pub first_colliding_bubbles: Option<Array3<BubbleIndex>>,
    pub coefficients_sets: Array2<f64>,
    pub powers_sets: Array2<f64>,
    pub damping_width: Option<f64>,
    pub thread_pool: ThreadPool,
    pub n_cos_thetax: Option<usize>,
    pub n_phix: Option<usize>,
    pub cos_thetax: Option<Array1<f64>>,
    pub phix: Option<Array1<f64>>,
    pub direction_vectors: Option<DMatrix<Vector4<f64>>>,
}

impl<L> GeneralizedBulkFlow<L>
where
    L: GeneralLatticeProperties,
{
    /// Create a new `BulkFlow`.
    ///
    /// * `bubbles` – spacetime coordinates of nucleated bubbles inside and
    ///   outside the lattice
    /// * `sort_by_time`    – if `true` the two bubble lists are sorted by
    ///   formation time
    pub fn new(bubbles: LatticeBubbles<L>) -> Result<Self, GeneralizedBulkFlowError> {
        let default_num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(default_num_threads)
            .build()
            .map_err(GeneralizedBulkFlowError::ThreadPoolBuildError)?;

        Ok(GeneralizedBulkFlow {
            lattice_bubbles: bubbles,
            first_colliding_bubbles: None,
            coefficients_sets: Array2::from_elem((1, 1), 1.0),
            powers_sets: Array2::from_elem((1, 1), 3.0),
            damping_width: None,
            thread_pool,
            n_cos_thetax: None,
            n_phix: None,
            cos_thetax: None,
            phix: None,
            direction_vectors: None,
        })
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> Result<(), GeneralizedBulkFlowError> {
        self.thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(GeneralizedBulkFlowError::ThreadPoolBuildError)?;
        Ok(())
    }

    pub fn first_colliding_bubbles(&self) -> Option<&Array3<BubbleIndex>> {
        self.first_colliding_bubbles.as_ref()
    }

    /// For reference bubble $n$, computes $t_{n,c}(\theta,\phi)$,
    /// the index of the first bubble that collides with bubble $n$ along each
    /// direction $(\cos\theta, \phi)$. This defines the collision time
    /// function $t_{n,c}(\cos\theta, \phi)$ required for the scaling function
    /// $f$.
    pub fn compute_first_colliding_bubble(
        &self,
        a_idx: usize,
    ) -> Result<Array2<BubbleIndex>, GeneralizedBulkFlowError> {
        let n_cos_thetax = self.n_cos_thetax.ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("n_cos_thetax".to_string())
        })?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("n_phix".to_string()))?;
        let direction_vectors = self.direction_vectors.as_ref().ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("direction_vectors".to_string())
        })?;
        let n_interior = self.lattice_bubbles.interior.n_bubbles();
        let n_exterior = self.lattice_bubbles.exterior.n_bubbles();
        let n_total = n_interior + n_exterior;
        let tolerance = 1e-10;

        // Pre-extract shared read-only data
        let delta = &self.lattice_bubbles.delta;
        let delta_squared = &self.lattice_bubbles.delta_squared;

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
    ) -> Result<(), GeneralizedBulkFlowError> {
        if n_cos_thetax < 2 || n_phix < 2 {
            return Err(GeneralizedBulkFlowError::InvalidResolution(
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
                (0..self.lattice_bubbles.interior.n_bubbles())
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
                    GeneralizedBulkFlowError::ArrayShapeMismatch(format!(
                        "Failed to stack arrays: {}",
                        e
                    ))
                })?,
            );
        } else {
            self.first_colliding_bubbles = None;
        }

        self.cos_thetax = Some(cos_thetax);
        self.phix = Some(phix);
        Ok(())
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

    pub fn cos_thetax(&self) -> Result<&Array1<f64>, GeneralizedBulkFlowError> {
        self.cos_thetax
            .as_ref()
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("cos_thetax".to_string()))
    }

    pub fn phix(&self) -> Result<&Array1<f64>, GeneralizedBulkFlowError> {
        self.phix
            .as_ref()
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("phix".to_string()))
    }

    /// Configures the post-collision scaling function:
    /// $$
    /// f = \sum_\xi a_\xi \left(\frac{t_{n,c} - t_n}{t - t_n}\right)^\xi,
    /// $$
    /// where `coefficients_sets[s][k] = a_\xi`, `powers_sets[s][k] = \xi` for
    /// model set `s`. Enforces physical constraints: coefficients=0 (no
    /// flow) or  or sum to 1 (full bulk-flow).
    pub fn set_gradient_scaling_params(
        &mut self,
        coefficients_sets: Vec<Vec<f64>>,
        powers_sets: Vec<Vec<f64>>,
        damping_width: Option<f64>,
    ) -> Result<(), GeneralizedBulkFlowError> {
        let n_sets = coefficients_sets.len();
        let n_coeffs = coefficients_sets.first().map_or(0, |v| v.len());
        if coefficients_sets.iter().any(|v| v.len() != n_coeffs) {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(
                "All coefficient sets must have the same length".to_string(),
            ));
        }
        let coefficients_sets = Array2::from_shape_vec(
            (n_sets, n_coeffs),
            coefficients_sets.into_iter().flatten().collect(),
        )
        .map_err(|_| {
            GeneralizedBulkFlowError::ArrayShapeMismatch(
                "Invalid shape for coefficients_sets".to_string(),
            )
        })?;

        let n_powers = powers_sets.len();
        let n_pows = powers_sets.first().map_or(0, |v| v.len());
        if powers_sets.iter().any(|v| v.len() != n_pows) {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(
                "All power sets must have the same length".to_string(),
            ));
        }
        let powers_sets =
            Array2::from_shape_vec((n_powers, n_pows), powers_sets.into_iter().flatten().collect())
                .map_err(|_| {
                    GeneralizedBulkFlowError::ArrayShapeMismatch(
                        "Invalid shape for powers_sets".to_string(),
                    )
                })?;

        if n_sets != n_powers {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(
                "Number of coefficient sets and power sets must be equal".to_string(),
            ));
        }

        self.coefficients_sets = coefficients_sets;
        self.powers_sets = powers_sets;
        self.damping_width = damping_width;
        Ok(())
    }

    /// Evaluates the Heaviside conditions $\Theta(t_{n,c} - t)$ and $\Theta(t -
    /// t_{n,c})$ by comparing current time $t$ to collision time $t_{n,c} =
    /// t_n + \Delta t_{n,c}$, returning `NeverCollided`, `NotYetCollided`,
    /// or `AlreadyCollided` per direction.
    pub fn compute_collision_status(
        &self,
        a_idx: usize,
        t: f64,
        first_bubble: &ArrayRef2<BubbleIndex>,
        delta_tab_grid: &ArrayRef2<f64>,
    ) -> Result<Array2<CollisionStatus>, GeneralizedBulkFlowError> {
        let n_cos_thetax = self.n_cos_thetax.ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("n_cos_thetax".to_string())
        })?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("n_phix".to_string()))?;
        if first_bubble.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(format!(
                "Input arrays must match resolution: expected [{}, {}], got first_bubble: {:?}, delta_tab_grid: {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape(),
                delta_tab_grid.shape()
            )));
        }
        let mut collision_status =
            Array2::from_elem((n_cos_thetax, n_phix), CollisionStatus::NeverCollided);
        let ta = self.lattice_bubbles.interior.spacetime[a_idx][0];
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

    /// Computes the collision time delay $\Delta t_{n,c} = t_{n,c} - t_n$ on a
    /// $(\cos\theta, \phi)$ grid, where $t_{n,c} - t_n = \frac{1}{2}
    /// \frac{(x_c - x_n)^2}{(x_c - x_n) \cdot \hat{x}}$ is the solution to
    /// the null-intersection condition for bubbles $n$ and $c$ along direction
    /// $\hat{x} = (1,\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$.
    pub fn compute_delta_tab(
        &self,
        a_idx: usize,
        first_bubble: &ArrayRef2<BubbleIndex>,
    ) -> Result<Array2<f64>, GeneralizedBulkFlowError> {
        let n_cos_thetax = self.n_cos_thetax.ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("n_cos_thetax".to_string())
        })?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("n_phix".to_string()))?;
        let direction_vectors = self.direction_vectors.as_ref().ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("direction_vectors".to_string())
        })?;
        if first_bubble.shape() != [n_cos_thetax, n_phix] {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(format!(
                "First bubble array must match resolution: expected [{}, {}], got {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape()
            )));
        }
        let n_interior = self.lattice_bubbles.interior.n_bubbles();
        let mut delta_tab_grid = Array2::zeros((n_cos_thetax, n_phix));

        for i in 0..n_cos_thetax {
            // TODO: This loop can be replaced by iterating over segment with const
            // BubbleIndex
            for j in 0..n_phix {
                let b_total = match first_bubble[[i, j]] {
                    BubbleIndex::None => {
                        continue;
                    },
                    BubbleIndex::Interior(b_idx) => b_idx,
                    BubbleIndex::Exterior(b_idx) => n_interior + b_idx,
                };
                let delta_ba = self.lattice_bubbles.delta[(a_idx, b_total)];
                let x_vec = direction_vectors[(i, j)];
                let delta_ba_squared = self.lattice_bubbles.delta_squared[(a_idx, b_total)];
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

    /// Computes $B_{n,\pm}(\zeta, t)$ for fixed $\zeta = \cos\theta$ by
    /// numerically evaluating: $$
    /// B_{n,\pm} = \frac{1-\zeta^2}{2} \int_0^{2\pi} g_{\pm}(\phi)\ (t - t_n)^3
    /// f(t, t_n, t_{n,c})\, d\phi, $$
    /// where $f$ switches between pre-collision ($f=1$) and post-collision
    /// profiles using user-defined coefficients $a_\xi$ and powers $\xi$
    /// (stored in `coefficients_sets`, `powers_sets`), with optional
    /// exponential damping.
    pub fn compute_b_tensor(
        &self,
        cos_thetax_idx: usize,
        collision_status_grid: &ArrayRef2<CollisionStatus>,
        delta_tab_grid: &ArrayRef2<f64>,
        delta_ta: f64,
    ) -> Result<HermitianTensor3<Array1<f64>>, GeneralizedBulkFlowError> {
        let n_cos_thetax = self.n_cos_thetax.ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("n_cos_thetax".to_string())
        })?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("n_phix".to_string()))?;
        let cos_thetax_grid = self.cos_thetax.as_ref().ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("cos_thetax".to_string())
        })?;
        let phix = self
            .phix
            .as_ref()
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("phix".to_string()))?;
        let n_sets = self.coefficients_sets.nrows();

        if cos_thetax_idx >= n_cos_thetax {
            return Err(GeneralizedBulkFlowError::InvalidIndex {
                index: cos_thetax_idx,
                max: n_cos_thetax,
            });
        }
        if collision_status_grid.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(format!(
                "Grids must be [{}, {}], got status: {:?}, delta_tab: {:?}",
                n_cos_thetax,
                n_phix,
                collision_status_grid.shape(),
                delta_tab_grid.shape()
            )));
        }

        let cos_thetax = cos_thetax_grid[cos_thetax_idx];
        let sin_thetax = f64::sqrt((1.0 - cos_thetax * cos_thetax).max(0.0));
        let sin_squared_thetax = sin_thetax * sin_thetax;
        let cos_squared_thetax = cos_thetax * cos_thetax;
        let sin_cos_thetax = sin_thetax * cos_thetax;

        let dphi = 2.0 * std::f64::consts::PI / n_phix as f64;

        // Precompute trig over ϕ
        let mut cos_phi = Array1::zeros(n_phix);
        let mut sin_phi = Array1::zeros(n_phix);
        azip!((c in &mut cos_phi, s in &mut sin_phi, &p in phix) {
            *c = p.cos();
            *s = p.sin();
        });

        let cos_phi_sq = &cos_phi * &cos_phi;
        let sin_phi_sq = &sin_phi * &sin_phi;
        let cos_sin_phi = &cos_phi * &sin_phi;

        let basis0 = &cos_phi_sq * sin_squared_thetax;
        let basis1 = &cos_sin_phi * sin_squared_thetax;
        let basis2 = &cos_phi * sin_cos_thetax;
        let basis3 = &sin_phi_sq * sin_squared_thetax;
        let basis4 = &sin_phi * sin_cos_thetax;
        let basis5 = Array1::from_elem(n_phix, cos_squared_thetax);

        let status_row = collision_status_grid.slice(s![cos_thetax_idx, ..]);
        let delta_tab_row = delta_tab_grid.slice(s![cos_thetax_idx, ..]);

        let mut weights = Array1::from_elem(n_phix, 1.0);
        if n_phix > 1 {
            weights[0] = 0.5;
            weights[n_phix - 1] = 0.5;
        }

        let delta_ta_cubed = if delta_ta > 0.0 {
            delta_ta.powi(3)
        } else {
            0.0
        };

        // Initialize 6 arrays (n_sets each)
        let mut b0 = Array1::zeros(n_sets);
        let mut b1 = Array1::zeros(n_sets);
        let mut b2 = Array1::zeros(n_sets);
        let mut b3 = Array1::zeros(n_sets);
        let mut b4 = Array1::zeros(n_sets);
        let mut b5 = Array1::zeros(n_sets);

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

            let weighted0 = &basis0 * &factors * &weights;
            let weighted1 = &basis1 * &factors * &weights;
            let weighted2 = &basis2 * &factors * &weights;
            let weighted3 = &basis3 * &factors * &weights;
            let weighted4 = &basis4 * &factors * &weights;
            let weighted5 = &basis5 * &factors * &weights;

            let integral0 = weighted0.sum() * dphi * delta_ta_cubed;
            let integral1 = weighted1.sum() * dphi * delta_ta_cubed;
            let integral2 = weighted2.sum() * dphi * delta_ta_cubed;
            let integral3 = weighted3.sum() * dphi * delta_ta_cubed;
            let integral4 = weighted4.sum() * dphi * delta_ta_cubed;
            let integral5 = weighted5.sum() * dphi * delta_ta_cubed;

            b0[s] = integral0;
            b1[s] = integral1;
            b2[s] = integral2;
            b3[s] = integral3;
            b4[s] = integral4;
            b5[s] = integral5;
        }

        Ok(HermitianTensor3::new(b0, b1, b2, b3, b4, b5))
    }

    pub fn compute_a_tensor<W>(
        &self,
        a_idx: usize,
        w_arr: W,
        t: f64,
        first_bubble: &ArrayRef2<BubbleIndex>,
        delta_tab_grid: &ArrayRef2<f64>,
    ) -> Result<HermitianTensor3<Array2<Complex64>>, GeneralizedBulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_cos_thetax = self.n_cos_thetax.ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("n_cos_thetax".to_string())
        })?;
        let n_phix = self
            .n_phix
            .ok_or_else(|| GeneralizedBulkFlowError::UninitializedField("n_phix".to_string()))?;
        let cos_thetax_grid = self.cos_thetax.as_ref().ok_or_else(|| {
            GeneralizedBulkFlowError::UninitializedField("cos_thetax".to_string())
        })?;

        if first_bubble.shape() != [n_cos_thetax, n_phix]
            || delta_tab_grid.shape() != [n_cos_thetax, n_phix]
        {
            return Err(GeneralizedBulkFlowError::ArrayShapeMismatch(format!(
                "Input arrays must be [{}, {}], got first_bubble: {:?}, delta_tab: {:?}",
                n_cos_thetax,
                n_phix,
                first_bubble.shape(),
                delta_tab_grid.shape()
            )));
        }

        let n_w = w_arr.len();
        let n_sets = self.coefficients_sets.nrows();
        let ta = self.lattice_bubbles.interior.spacetime[a_idx][0];
        let delta_ta = t - ta;

        let collision_status =
            self.compute_collision_status(a_idx, t, &first_bubble, &delta_tab_grid)?;

        let mut a0 = Array2::zeros((n_sets, n_w));
        let mut a1 = Array2::zeros((n_sets, n_w));
        let mut a2 = Array2::zeros((n_sets, n_w));
        let mut a3 = Array2::zeros((n_sets, n_w));
        let mut a4 = Array2::zeros((n_sets, n_w));
        let mut a5 = Array2::zeros((n_sets, n_w));

        let dcos_thetax = if n_cos_thetax > 1 {
            2.0 / (n_cos_thetax - 1) as f64
        } else {
            1.0
        };

        for i in 0..n_cos_thetax {
            let b_tensor =
                self.compute_b_tensor(i, &collision_status, &delta_tab_grid, delta_ta)?;

            let cos_thetax_val = cos_thetax_grid[i];
            let phase_base = Complex64::new(0.0, -delta_ta * cos_thetax_val);
            let angular_phases: Vec<Complex64> =
                w_arr.iter().map(|&w| (w * phase_base).exp()).collect();

            let weight_theta = if n_cos_thetax == 1 {
                1.0
            } else if i == 0 || i == n_cos_thetax - 1 {
                0.5
            } else {
                1.0
            };
            let phase_factor_scalar = dcos_thetax * weight_theta;

            let b0 = b_tensor.xx(); // &Array1<f64>
            let b1 = b_tensor.xy();
            let b2 = b_tensor.xz();
            let b3 = b_tensor.yy();
            let b4 = b_tensor.yz();
            let b5 = b_tensor.zz();

            for s in 0..n_sets {
                let b0_val = Complex64::new(b0[s], 0.0);
                let b1_val = Complex64::new(b1[s], 0.0);
                let b2_val = Complex64::new(b2[s], 0.0);
                let b3_val = Complex64::new(b3[s], 0.0);
                let b4_val = Complex64::new(b4[s], 0.0);
                let b5_val = Complex64::new(b5[s], 0.0);

                for w_idx in 0..n_w {
                    let phase = angular_phases[w_idx] * phase_factor_scalar;
                    a0[[s, w_idx]] += b0_val * phase;
                    a1[[s, w_idx]] += b1_val * phase;
                    a2[[s, w_idx]] += b2_val * phase;
                    a3[[s, w_idx]] += b3_val * phase;
                    a4[[s, w_idx]] += b4_val * phase;
                    a5[[s, w_idx]] += b5_val * phase;
                }
            }
        }

        Ok(HermitianTensor3::new(a0, a1, a2, a3, a4, a5))
    }

    pub fn compute_c_tensor_integrand_fixed_bubble<W>(
        &self,
        a_idx: usize,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> Result<HermitianTensor3<Array3<Complex64>>, GeneralizedBulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        let t_nucleation = self.lattice_bubbles.interior.spacetime[a_idx][0];
        if t_nucleation >= t_end {
            let zeros = Array3::zeros((n_sets, n_w, n_t));
            return Ok(HermitianTensor3::new(
                zeros.clone(),
                zeros.clone(),
                zeros.clone(),
                zeros.clone(),
                zeros.clone(),
                zeros,
            ));
        }

        let t_begin = t_begin.unwrap_or(0.0);
        if t_begin > t_end {
            return Err(GeneralizedBulkFlowError::InvalidTimeRange {
                begin: t_begin,
                end: t_end,
            });
        }

        let t_arr = Array1::linspace(t_begin, t_end, n_t).to_vec();
        let dt = if n_t > 1 { t_arr[1] - t_arr[0] } else { 0.0 };
        let z_a = self.lattice_bubbles.interior.spacetime[a_idx][3];

        let first_colliding_bubbles_with_a: Array2<BubbleIndex> =
            if let Some(cache) = self.first_colliding_bubbles.as_ref() {
                cache.slice(s![a_idx, .., ..]).to_owned()
            } else {
                self.compute_first_colliding_bubble(a_idx)?
            };
        let delta_tab = self.compute_delta_tab(a_idx, &first_colliding_bubbles_with_a)?;

        let mut c0 = Array3::zeros((n_sets, n_w, n_t));
        let mut c1 = Array3::zeros((n_sets, n_w, n_t));
        let mut c2 = Array3::zeros((n_sets, n_w, n_t));
        let mut c3 = Array3::zeros((n_sets, n_w, n_t));
        let mut c4 = Array3::zeros((n_sets, n_w, n_t));
        let mut c5 = Array3::zeros((n_sets, n_w, n_t));

        let time_and_integrand_results: Vec<
            Result<(usize, [Array2<Complex64>; 6]), GeneralizedBulkFlowError>,
        > = self.thread_pool.install(|| {
            t_arr
                .into_par_iter()
                .enumerate()
                .map(|(t_idx, t)| {
                    if t <= t_nucleation {
                        let zeros = Array2::zeros((n_sets, n_w));
                        return Ok((
                            t_idx,
                            [
                                zeros.clone(),
                                zeros.clone(),
                                zeros.clone(),
                                zeros.clone(),
                                zeros.clone(),
                                zeros,
                            ],
                        ));
                    }

                    let a_tensor = self.compute_a_tensor(
                        a_idx,
                        w_arr,
                        t,
                        &first_colliding_bubbles_with_a,
                        &delta_tab,
                    )?;

                    let a0 = a_tensor.xx();
                    let a1 = a_tensor.xy();
                    let a2 = a_tensor.xz();
                    let a3 = a_tensor.yy();
                    let a4 = a_tensor.yz();
                    let a5 = a_tensor.zz();

                    let weight = if t_idx == 0 || t_idx == n_t - 1 {
                        0.5
                    } else {
                        1.0
                    };
                    let dt_weight = Complex64::new(dt * weight, 0.0);

                    let mut c_integrand0 = Array2::zeros((n_sets, n_w));
                    let mut c_integrand1 = Array2::zeros((n_sets, n_w));
                    let mut c_integrand2 = Array2::zeros((n_sets, n_w));
                    let mut c_integrand3 = Array2::zeros((n_sets, n_w));
                    let mut c_integrand4 = Array2::zeros((n_sets, n_w));
                    let mut c_integrand5 = Array2::zeros((n_sets, n_w));

                    for s in 0..n_sets {
                        for w_idx in 0..n_w {
                            let w = w_arr[w_idx];
                            let phase = Complex64::new(0.0, w * (t - z_a)).exp();
                            let factor = phase * dt_weight;

                            c_integrand0[[s, w_idx]] = a0[[s, w_idx]] * factor;
                            c_integrand1[[s, w_idx]] = a1[[s, w_idx]] * factor;
                            c_integrand2[[s, w_idx]] = a2[[s, w_idx]] * factor;
                            c_integrand3[[s, w_idx]] = a3[[s, w_idx]] * factor;
                            c_integrand4[[s, w_idx]] = a4[[s, w_idx]] * factor;
                            c_integrand5[[s, w_idx]] = a5[[s, w_idx]] * factor;
                        }
                    }

                    Ok((
                        t_idx,
                        [
                            c_integrand0,
                            c_integrand1,
                            c_integrand2,
                            c_integrand3,
                            c_integrand4,
                            c_integrand5,
                        ],
                    ))
                })
                .collect()
        });

        for result in time_and_integrand_results {
            let (t_idx, [arr0, arr1, arr2, arr3, arr4, arr5]) = result?;
            for s in 0..n_sets {
                for w_idx in 0..n_w {
                    c0[[s, w_idx, t_idx]] += arr0[[s, w_idx]];
                    c1[[s, w_idx, t_idx]] += arr1[[s, w_idx]];
                    c2[[s, w_idx, t_idx]] += arr2[[s, w_idx]];
                    c3[[s, w_idx, t_idx]] += arr3[[s, w_idx]];
                    c4[[s, w_idx, t_idx]] += arr4[[s, w_idx]];
                    c5[[s, w_idx, t_idx]] += arr5[[s, w_idx]];
                }
            }
        }

        let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
        c0 *= factor;
        c1 *= factor;
        c2 *= factor;
        c3 *= factor;
        c4 *= factor;
        c5 *= factor;

        Ok(HermitianTensor3::new(c0, c1, c2, c3, c4, c5))
    }

    pub fn compute_c_tensor_integrand<W>(
        &self,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<&[usize]>,
    ) -> Result<HermitianTensor3<Array3<Complex64>>, GeneralizedBulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_interior = self.lattice_bubbles.interior.n_bubbles();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        if n_t < 2 {
            return Err(GeneralizedBulkFlowError::InvalidResolution(
                "n_t must be >= 2 for integration".to_string(),
            ));
        }

        // Validate and collect bubble indices
        let bubble_ids: Vec<usize> = match selected_bubbles {
            Some(ids) => {
                if ids.is_empty() {
                    let zeros = Array3::zeros((n_sets, n_w, n_t));
                    return Ok(HermitianTensor3::new(
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros,
                    ));
                }
                for &a in ids {
                    if a >= n_interior {
                        return Err(GeneralizedBulkFlowError::InvalidIndex {
                            index: a,
                            max: n_interior,
                        });
                    }
                }
                ids.to_vec()
            },
            None => (0..n_interior).collect(),
        };

        // Initialize total accumulators: (n_sets, n_w, n_t)
        let mut total0 = Array3::zeros((n_sets, n_w, n_t));
        let mut total1 = Array3::zeros((n_sets, n_w, n_t));
        let mut total2 = Array3::zeros((n_sets, n_w, n_t));
        let mut total3 = Array3::zeros((n_sets, n_w, n_t));
        let mut total4 = Array3::zeros((n_sets, n_w, n_t));
        let mut total5 = Array3::zeros((n_sets, n_w, n_t));

        // Sum integrands over selected bubbles
        for &a_idx in &bubble_ids {
            let integrand =
                self.compute_c_tensor_integrand_fixed_bubble(a_idx, w_arr, t_begin, t_end, n_t)?;

            // Borrow components and accumulate
            total0 += &integrand[0];
            total1 += &integrand[1];
            total2 += &integrand[2];
            total3 += &integrand[3];
            total4 += &integrand[4];
            total5 += &integrand[5];
        }

        Ok(HermitianTensor3::new(total0, total1, total2, total3, total4, total5))
    }

    pub fn compute_c_tensor_fixed_bubble<W>(
        &mut self,
        a_idx: usize,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
    ) -> Result<HermitianTensor3<Array2<Complex64>>, GeneralizedBulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();
        let t_begin = t_begin.unwrap_or(0.0);
        if t_begin > t_end {
            return Err(GeneralizedBulkFlowError::InvalidTimeRange {
                begin: t_begin,
                end: t_end,
            });
        }

        let t_nucleation = self.lattice_bubbles.interior.spacetime[a_idx][0];
        // // FIXME: see if this is necessary
        // if t_nucleation >= t_end {
        //     let zeros = Array2::zeros((n_sets, n_w));
        //     return Ok(HermitianTensor3::new(
        //         zeros.clone(),
        //         zeros.clone(),
        //         zeros.clone(),
        //         zeros.clone(),
        //         zeros.clone(),
        //         zeros,
        //     ));
        // }

        let t_arr = Array1::linspace(t_begin, t_end, n_t).to_vec();
        let dt = if n_t > 1 { t_arr[1] - t_arr[0] } else { 0.0 };
        let z_a = self.lattice_bubbles.interior.spacetime[a_idx][3];

        let first_colliding_bubbles_with_a: Array2<BubbleIndex> =
            if let Some(cache) = self.first_colliding_bubbles.as_ref() {
                cache.slice(s![a_idx, .., ..]).to_owned()
            } else {
                self.compute_first_colliding_bubble(a_idx)?
            };
        let delta_tab = self.compute_delta_tab(a_idx, &first_colliding_bubbles_with_a)?;

        let (c0, c1, c2, c3, c4, c5) = self.thread_pool.install(|| {
            t_arr
                .into_par_iter()
                .enumerate()
                .fold(
                    || {
                        (
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                        )
                    },
                    |(mut i0, mut i1, mut i2, mut i3, mut i4, mut i5), (t_idx, t)| {
                        if t <= t_nucleation {
                            return (i0, i1, i2, i3, i4, i5);
                        }

                        let a_tensor = self
                            .compute_a_tensor(
                                a_idx,
                                w_arr,
                                t,
                                &first_colliding_bubbles_with_a,
                                &delta_tab,
                            )
                            .unwrap();

                        let a0 = a_tensor.xx();
                        let a1 = a_tensor.xy();
                        let a2 = a_tensor.xz();
                        let a3 = a_tensor.yy();
                        let a4 = a_tensor.yz();
                        let a5 = a_tensor.zz();

                        let weight = if t_idx == 0 || t_idx == n_t - 1 {
                            0.5
                        } else {
                            1.0
                        };
                        let dt_complex = Complex64::new(dt * weight, 0.0);

                        for s in 0..n_sets {
                            for w_idx in 0..n_w {
                                let w = w_arr[w_idx];
                                let phase = Complex64::new(0.0, w * (t - z_a)).exp();
                                let factor = phase * dt_complex;

                                i0[[s, w_idx]] += a0[[s, w_idx]] * factor;
                                i1[[s, w_idx]] += a1[[s, w_idx]] * factor;
                                i2[[s, w_idx]] += a2[[s, w_idx]] * factor;
                                i3[[s, w_idx]] += a3[[s, w_idx]] * factor;
                                i4[[s, w_idx]] += a4[[s, w_idx]] * factor;
                                i5[[s, w_idx]] += a5[[s, w_idx]] * factor;
                            }
                        }

                        (i0, i1, i2, i3, i4, i5)
                    },
                )
                .reduce(
                    || {
                        (
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                            Array2::zeros((n_sets, n_w)),
                        )
                    },
                    |(i0a, i1a, i2a, i3a, i4a, i5a), (i0b, i1b, i2b, i3b, i4b, i5b)| {
                        (i0a + i0b, i1a + i1b, i2a + i2b, i3a + i3b, i4a + i4b, i5a + i5b)
                    },
                )
        });

        let factor = Complex64::new(1.0 / (6.0 * std::f64::consts::PI), 0.0);
        Ok(HermitianTensor3::new(
            c0 * factor,
            c1 * factor,
            c2 * factor,
            c3 * factor,
            c4 * factor,
            c5 * factor,
        ))
    }

    pub fn compute_c_tensor<W>(
        &mut self,
        w_arr: W,
        t_begin: Option<f64>,
        t_end: f64,
        n_t: usize,
        selected_bubbles: Option<&[usize]>,
    ) -> Result<HermitianTensor3<Array2<Complex64>>, GeneralizedBulkFlowError>
    where
        W: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let n_interior = self.lattice_bubbles.interior.n_bubbles();
        let n_sets = self.coefficients_sets.nrows();
        let n_w = w_arr.len();

        if n_t < 2 {
            return Err(GeneralizedBulkFlowError::InvalidResolution(
                "n_t must be >= 2 for integration".to_string(),
            ));
        }

        // Validate and collect bubble indices
        let bubble_ids: Vec<usize> = match selected_bubbles {
            Some(ids) => {
                if ids.is_empty() {
                    let zeros = Array2::zeros((n_sets, n_w));
                    return Ok(HermitianTensor3::new(
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros.clone(),
                        zeros,
                    ));
                }
                for &a in ids {
                    if a >= n_interior {
                        return Err(GeneralizedBulkFlowError::InvalidIndex {
                            index: a,
                            max: n_interior,
                        });
                    }
                }
                ids.to_vec()
            },
            None => (0..n_interior).collect(),
        };

        // Initialize accumulator tensor (time-integrated: (n_sets, n_w))
        let mut total0 = Array2::zeros((n_sets, n_w));
        let mut total1 = Array2::zeros((n_sets, n_w));
        let mut total2 = Array2::zeros((n_sets, n_w));
        let mut total3 = Array2::zeros((n_sets, n_w));
        let mut total4 = Array2::zeros((n_sets, n_w));
        let mut total5 = Array2::zeros((n_sets, n_w));

        // Sum over selected bubbles
        for &a_idx in &bubble_ids {
            let c_tensor = self.compute_c_tensor_fixed_bubble(a_idx, w_arr, t_begin, t_end, n_t)?;

            total0 += c_tensor.xx();
            total1 += c_tensor.xy();
            total2 += c_tensor.xz();
            total3 += c_tensor.yy();
            total4 += c_tensor.yz();
            total5 += c_tensor.zz();
        }

        Ok(HermitianTensor3::new(total0, total1, total2, total3, total4, total5))
    }

    pub fn delta_squared(&self) -> &DMatrix<f64> {
        &self.lattice_bubbles.delta_squared
    }

    pub fn bubbles_interior(&self) -> &Bubbles {
        &self.lattice_bubbles.interior
    }

    pub fn bubbles_exterior(&self) -> &Bubbles {
        &self.lattice_bubbles.exterior
    }
}

/// Determines if bubble $c$ blocks the collision between bubbles $a$ and $b$
/// along direction $\hat{x}$, by verifying whether the collision event of
/// $a$–$b$ lies outside the lightcone of $a$–$c$. Returns `true` iff $b$ is the
/// *first* colliding bubble along $\hat{x}$.
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

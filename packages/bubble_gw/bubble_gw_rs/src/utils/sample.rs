use num::traits::{Float, FromPrimitive, ToPrimitive};
use std::{f64, fmt::Debug};
use thiserror::Error;

/// Errors that can occur during sampling parameter validation or numeric conversion.
#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Invalid range: start ({start}) >= stop ({stop})")]
    InvalidRange { start: f64, stop: f64 },

    #[error("Invalid base for {sample_type} sampling: base = {base} (must be > 0)")]
    InvalidBase {
        base: f64,
        sample_type: &'static str,
    },

    #[error("Invalid parameter '{param}': {value}")]
    InvalidParameter { param: &'static str, value: f64 },

    #[error("Failed to convert integer to float type (overflow or unsupported)")]
    ConversionError,
}

/// Type of sampling distribution to use when generating points between `start` and `stop`.
///
/// This enum defines how points are distributed in the output space:
/// - `Uniform` and `Linear`: identical behavior (linear spacing in value space)
/// - `Logarithmic`: logarithmic spacing (base > 0 required)
/// - `Exponential`: exponential spacing (base > 0 required)
/// - `Distribution`: arbitrary monotonic mapping via closure pair
#[derive(Clone, Debug)]
pub enum SampleType<T> {
    Uniform,
    Linear,
    Logarithmic {
        base: T,
    },
    Exponential {
        base: T,
    },
    Distribution {
        /// Monotonic increasing function: `value → transformed`
        dist: fn(T) -> T,
        /// Inverse of `dist`: `transformed → value`
        dist_inv: fn(T) -> T,
    },
}

impl<T: Float + PartialEq> PartialEq for SampleType<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SampleType::Uniform, SampleType::Uniform)
            | (SampleType::Linear, SampleType::Linear) => true,
            (SampleType::Logarithmic { base: b1 }, SampleType::Logarithmic { base: b2 })
            | (SampleType::Exponential { base: b1 }, SampleType::Exponential { base: b2 }) => {
                b1 == b2
            }
            (SampleType::Distribution { .. }, SampleType::Distribution { .. }) => false,
            _ => false,
        }
    }
}

/// Parameters defining a 1D sampling strategy.
///
/// This struct holds the full specification for generating samples
/// from `start` to `stop` using a given `sample_type`.
///
/// # Constraints
/// - `start < stop`
/// - For `Logarithmic` and `Exponential`: `base > 0` and `start > 0`, `stop > 0`
/// - `start` and `stop` must not be NaN
#[derive(Clone, Debug, PartialEq)]
pub struct SampleParams<T: Float>
where
    T: FromPrimitive + ToPrimitive,
{
    start: T,
    stop: T,
    sample_type: SampleType<T>,
}

impl<T> SampleParams<T>
where
    T: Float + Debug + FromPrimitive + ToPrimitive + 'static,
{
    /// Create a new `SampleParams` instance with validation.
    ///
    /// # Errors
    /// Returns `SampleError` if:
    /// - `start >= stop`
    /// - `start` or `stop` is NaN
    /// - Logarithmic/Exponential used with invalid base or non-positive range
    ///
    /// # Example
    /// ```rust
    /// use bubble_gw_rs::utils::sample::{SampleParams, SampleType};
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let params = SampleParams::new(1.0, 100.0, SampleType::Logarithmic { base: 10.0 })?;
    /// Ok(())
    /// }
    /// ```
    pub fn new(start: T, stop: T, sample_type: SampleType<T>) -> Result<Self, SampleError> {
        let start_f64 = start.to_f64().unwrap_or(f64::NAN);
        let stop_f64 = stop.to_f64().unwrap_or(f64::NAN);

        if start >= stop {
            return Err(SampleError::InvalidRange {
                start: start_f64,
                stop: stop_f64,
            });
        }
        if start.is_nan() || stop.is_nan() {
            return Err(SampleError::InvalidParameter {
                param: "start or stop",
                value: f64::NAN,
            });
        }

        match &sample_type {
            SampleType::Logarithmic { base } | SampleType::Exponential { base } => {
                let base_f64 = base.to_f64().unwrap_or(f64::NAN);
                if *base <= T::zero() {
                    return Err(SampleError::InvalidBase {
                        base: base_f64,
                        sample_type: if matches!(sample_type, SampleType::Logarithmic { .. }) {
                            "Logarithmic"
                        } else {
                            "Exponential"
                        },
                    });
                }
                if start <= T::zero() || stop <= T::zero() {
                    return Err(SampleError::InvalidRange {
                        start: start_f64,
                        stop: stop_f64,
                    });
                }
            }
            _ => {}
        }

        Ok(Self {
            start,
            stop,
            sample_type,
        })
    }

    /// Generate `n_sample + 1` points (inclusive) using the configured sampling strategy.
    ///
    /// If `n_iter > 0`, uses adaptive grid refinement with `n_grid` subdivisions per iteration.
    ///
    /// # Parameters
    /// - `n_sample`: Number of intervals (result has `n_sample + 1` points)
    /// - `n_grid`: Number of sub-grid points per refinement level (must be ≥ 2 if `n_iter > 0`)
    /// - `n_iter`: Number of refinement iterations (0 = simple uniform sampling)
    ///
    /// # Returns
    /// A `Vec<T>` of sampled points, always including `start` and `stop`.
    ///
    /// # Errors
    /// - `ConversionError` if integer → float conversion fails
    /// - `InvalidParameter` if `n_grid < 2` when `n_iter > 0`
    pub fn sample(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
    ) -> Result<Vec<T>, SampleError> {
        if n_sample == 0 {
            return Ok(vec![self.start]);
        }

        if n_iter > 0 && n_grid < 2 {
            return Err(SampleError::InvalidParameter {
                param: "n_grid",
                value: n_grid as f64,
            });
        }

        match &self.sample_type {
            SampleType::Uniform | SampleType::Linear => {
                let dist = |x: T| x;
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist)
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist)
                }
            }
            SampleType::Logarithmic { base } => {
                let dist = |x: T| x.log(*base);
                let dist_inv = |x: T| base.powf(x);
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist_inv)
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist_inv)
                }
            }
            SampleType::Exponential { base } => {
                let dist = |x: T| x;
                let dist_inv = |x: T| base.powf(x);
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist_inv)
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist_inv)
                }
            }
            SampleType::Distribution { dist, dist_inv } => {
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, dist, dist_inv)
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, dist, dist_inv)
                }
            }
        }
    }

    /// Generate samples across multiple refinement iterations.
    ///
    /// Calls `sample()` for `iter = 0..=n_iter` and concatenates results.
    /// Useful for creating nested or multi-resolution datasets.
    pub fn sample_arr(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
    ) -> Result<Vec<T>, SampleError> {
        let mut out: Vec<T> = Vec::new();
        for iter in 0..=n_iter {
            let mut s = self.sample(n_sample, n_grid, iter)?;
            out.append(&mut s);
        }
        Ok(out)
    }

    /// Simple uniform sampling in transformed space (no refinement).
    fn distribution_sample_simple(
        &self,
        n_sample: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &impl Fn(T) -> T,
    ) -> Result<Vec<T>, SampleError> {
        let n_sample_t = T::from_usize(n_sample).ok_or(SampleError::ConversionError)?;
        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        let mut v = Vec::with_capacity(n_sample + 1);
        for i in 0..=n_sample {
            let i_t = T::from_usize(i).ok_or(SampleError::ConversionError)?;
            let t = i_t / n_sample_t;
            v.push(dist_inv(transformed_start + t * transformed_range));
        }
        Ok(v)
    }

    /// Adaptive grid sampling — returns **only the new interior points** at refinement level `n_iter`
    ///
    /// This produces points that fill in the gaps between the grid at iteration `n_iter - 1`.
    /// The full nested grid is obtained by concatenating results from `iter = 0..=n_iter` via `sample_arr`.
    fn distribution_sample_grid(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &impl Fn(T) -> T,
    ) -> Result<Vec<T>, SampleError> {
        if n_iter == 0 {
            return self.distribution_sample_simple(n_sample, dist, dist_inv);
        }

        let n_grid_t = T::from_usize(n_grid).ok_or(SampleError::ConversionError)?;
        let factor = n_grid_t.powi(n_iter as i32); // grid spacing at current level
        let sub_factor = n_grid_t.powi((n_iter - 1) as i32); // spacing of parent level

        let n_parent_cells = n_sample
            .checked_mul(sub_factor.to_usize().ok_or(SampleError::ConversionError)?)
            .ok_or(SampleError::ConversionError)?; // # of parent intervals

        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        // Number of new points per parent interval: (n_grid - 1)
        // We skip the first point in each subinterval (it belongs to coarser grid)
        let points_per_cell = n_grid - 1;
        let total_points = n_parent_cells * points_per_cell;

        let mut out = Vec::with_capacity(total_points);

        for i2 in 0..n_parent_cells {
            for i1 in 1..n_grid {
                // Start from 1 → skip left endpoint (already exists)
                let i1_t = T::from_usize(i1).ok_or(SampleError::ConversionError)?;
                let i2_t = T::from_usize(i2).ok_or(SampleError::ConversionError)?;

                // Position: i2 * (coarse_step) + i1 * (fine_step)
                let t = (i1_t / factor + i2_t / sub_factor)
                    / T::from_usize(n_sample).ok_or(SampleError::ConversionError)?;
                let value = dist_inv(transformed_start + t * transformed_range);
                out.push(value);
            }
        }

        Ok(out)
    }
}

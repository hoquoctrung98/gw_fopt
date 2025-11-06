use num::Float;
use num::traits::{FromPrimitive, ToPrimitive};
use std::{f64, fmt::Debug};

#[derive(Debug)]
pub enum SampleError {
    InvalidRange {
        start: f64,
        stop: f64,
    },
    InvalidBase {
        base: f64,
        sample_type: &'static str,
    },
    InvalidParameter {
        param: &'static str,
        value: f64,
    },
    ConversionError,
}

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
        dist: fn(T) -> T,
        dist_inv: fn(T) -> T,
    },
}

impl<T: Float + PartialEq> PartialEq for SampleType<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SampleType::Uniform, SampleType::Uniform) => true,
            (SampleType::Linear, SampleType::Linear) => true,
            (SampleType::Logarithmic { base: b1 }, SampleType::Logarithmic { base: b2 }) => {
                b1 == b2
            }
            (SampleType::Exponential { base: b1 }, SampleType::Exponential { base: b2 }) => {
                b1 == b2
            }
            (SampleType::Distribution { .. }, SampleType::Distribution { .. }) => false,
            _ => false,
        }
    }
}

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
    pub fn new(start: T, stop: T, sample_type: SampleType<T>) -> Result<Self, SampleError> {
        if start >= stop {
            return Err(SampleError::InvalidRange {
                start: start.to_f64().unwrap_or(f64::NAN),
                stop: stop.to_f64().unwrap_or(f64::NAN),
            });
        }
        if start.is_nan() || stop.is_nan() {
            return Err(SampleError::InvalidParameter {
                param: "start or stop",
                value: f64::NAN,
            });
        }
        match &sample_type {
            SampleType::Logarithmic { base } => {
                if *base <= T::zero() {
                    return Err(SampleError::InvalidBase {
                        base: base.to_f64().unwrap_or(f64::NAN),
                        sample_type: "Logarithmic",
                    });
                }
                if start <= T::zero() || stop <= T::zero() {
                    return Err(SampleError::InvalidRange {
                        start: start.to_f64().unwrap_or(f64::NAN),
                        stop: stop.to_f64().unwrap_or(f64::NAN),
                    });
                }
            }
            SampleType::Exponential { base } => {
                if *base <= T::zero() {
                    return Err(SampleError::InvalidBase {
                        base: base.to_f64().unwrap_or(f64::NAN),
                        sample_type: "Exponential",
                    });
                }
                if start <= T::zero() || stop <= T::zero() {
                    return Err(SampleError::InvalidRange {
                        start: start.to_f64().unwrap_or(f64::NAN),
                        stop: stop.to_f64().unwrap_or(f64::NAN),
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
            SampleType::Uniform => {
                let dist = |x: T| x;
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist)
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist)
                }
            }
            SampleType::Linear => {
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

    fn distribution_sample_simple(
        &self,
        n_sample: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &impl Fn(T) -> T,
    ) -> Result<Vec<T>, SampleError> {
        // precompute conversions and validate
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

    fn distribution_sample_grid(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &impl Fn(T) -> T,
    ) -> Result<Vec<T>, SampleError> {
        // Validate and precompute numeric conversions
        let n_grid_t = T::from_usize(n_grid).ok_or(SampleError::ConversionError)?;
        let factor = n_grid_t.powi(n_iter as i32);
        let sub_factor = if n_iter > 0 {
            n_grid_t.powi((n_iter - 1) as i32)
        } else {
            T::one()
        };

        let sub_factor_usize = sub_factor.to_usize().ok_or(SampleError::ConversionError)?;

        // guard against overflow when computing total iterations
        let total_outer = n_sample
            .checked_mul(sub_factor_usize)
            .ok_or(SampleError::ConversionError)?;

        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        let n_sample_t = T::from_usize(n_sample).ok_or(SampleError::ConversionError)?;

        let mut out: Vec<T> = Vec::with_capacity(total_outer * n_grid);

        for i2 in 0..total_outer {
            for i1 in 0..n_grid {
                let i1_t = T::from_usize(i1).ok_or(SampleError::ConversionError)?;
                let i2_t = T::from_usize(i2).ok_or(SampleError::ConversionError)?;

                let t = (i1_t / factor + i2_t / sub_factor) / n_sample_t;
                out.push(dist_inv(transformed_start + t * transformed_range));
            }
        }

        Ok(out)
    }
}

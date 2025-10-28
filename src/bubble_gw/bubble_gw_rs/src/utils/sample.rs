use num::Float;
use std::fmt::Debug;

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
pub struct SampleParams<T: Float> {
    start: T,
    stop: T,
    sample_type: SampleType<T>,
}

impl<T: Float + Debug + 'static> SampleParams<T> {
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
            return Err(SampleError::InvalidParameter {
                param: "n_sample",
                value: 0.0,
            });
        }
        if n_iter > 0 && n_grid < 2 {
            return Err(SampleError::InvalidParameter {
                param: "n_grid",
                value: n_grid as f64,
            });
        }

        Ok(match &self.sample_type {
            SampleType::Uniform => {
                let dist = |x: T| x;
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist)
                        .collect()
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist)
                        .collect()
                }
            }
            SampleType::Linear => {
                let dist = |x: T| x;
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist)
                        .collect()
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist)
                        .collect()
                }
            }
            SampleType::Logarithmic { base } => {
                let dist = |x: T| x.log(*base);
                let dist_inv = |x: T| base.powf(x);
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist_inv)
                        .collect()
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist_inv)
                        .collect()
                }
            }
            SampleType::Exponential { base } => {
                let dist = |x: T| x;
                let dist_inv = |x: T| base.powf(x);
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, &dist, &dist_inv)
                        .collect()
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, &dist, &dist_inv)
                        .collect()
                }
            }
            SampleType::Distribution { dist, dist_inv } => {
                if n_iter == 0 {
                    self.distribution_sample_simple(n_sample, dist, dist_inv)
                        .collect()
                } else {
                    self.distribution_sample_grid(n_sample, n_grid, n_iter, dist, dist_inv)
                        .collect()
                }
            }
        })
    }

    pub fn sample_arr(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
    ) -> Result<Vec<T>, SampleError> {
        let samples: Result<Vec<Vec<T>>, SampleError> = (0..=n_iter)
            .map(|iter| self.sample(n_sample, n_grid, iter))
            .collect();
        samples.map(|v| v.into_iter().flatten().collect())
    }

    fn distribution_sample_simple<'a>(
        &self,
        n_sample: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &'a impl Fn(T) -> T,
    ) -> impl Iterator<Item = T> + 'a {
        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        (0..=n_sample).map(move |i| {
            let t = T::from(i).unwrap() / T::from(n_sample).unwrap();
            dist_inv(transformed_start + t * transformed_range)
        })
    }

    fn distribution_sample_grid<'a>(
        &self,
        n_sample: usize,
        n_grid: usize,
        n_iter: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &'a impl Fn(T) -> T,
    ) -> impl Iterator<Item = T> + 'a {
        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        let factor = T::from(n_grid).unwrap().powi(n_iter as i32);
        let sub_factor = T::from(n_grid).unwrap().powi((n_iter - 1) as i32);

        (0..n_sample * sub_factor.to_usize().unwrap()).flat_map(move |i2| {
            (0..n_grid).map(move |i1| {
                let t = (T::from(i1).unwrap() / factor + T::from(i2).unwrap() / sub_factor)
                    / T::from(n_sample).unwrap();
                dist_inv(transformed_start + t * transformed_range)
            })
        })
    }
}

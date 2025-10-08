use num::Float;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
/// Enum to specify the sampling type.
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

#[derive(Clone, Debug, PartialEq)]
/// Struct for sampling parameters.
pub struct SampleParams<T> {
    start: T,
    stop: T,
    sample_type: SampleType<T>,
}

impl<T: Float + Debug> SampleParams<T> {
    /// Create a new `SampleParams` instance.
    pub fn new(start: T, stop: T, sample_type: SampleType<T>) -> Result<Self, &'static str> {
        if start >= stop {
            return Err("`start` must be less than `stop`.");
        }
        if let SampleType::Logarithmic { base } = &sample_type {
            if *base <= T::zero() {
                return Err("`base` must be greater than 0 for logarithmic sampling.");
            }
        }
        if let SampleType::Exponential { base } = &sample_type {
            if *base <= T::zero() {
                return Err("`base` must be greater than 0 for exponential sampling.");
            }
        }
        Ok(Self {
            start,
            stop,
            sample_type,
        })
    }

    /// Perform distribution-based sampling.
    fn distribution_sample(
        &self,
        nsample: usize,
        ngrid: usize,
        niter: usize,
        dist: &impl Fn(T) -> T,
        dist_inv: &impl Fn(T) -> T,
    ) -> Vec<T> {
        let transformed_start = dist(self.start);
        let transformed_stop = dist(self.stop);
        let transformed_range = transformed_stop - transformed_start;

        if niter == 0 {
            (0..=nsample)
                .map(|i| {
                    let t = T::from(i).unwrap() / T::from(nsample).unwrap();
                    dist_inv(transformed_start + t * transformed_range)
                })
                .collect()
        } else {
            let factor = T::from(ngrid).unwrap().powi(niter as i32);
            let sub_factor = T::from(ngrid).unwrap().powi((niter - 1) as i32);

            (0..nsample * sub_factor.to_usize().unwrap())
                .flat_map(|i2| {
                    (1..ngrid).map(move |i1| {
                        let t = (T::from(i1).unwrap() / factor + T::from(i2).unwrap() / sub_factor)
                            / T::from(nsample).unwrap();
                        dist_inv(transformed_start + t * transformed_range)
                    })
                })
                .collect()
        }
    }

    /// Perform sampling based on the specified `SampleType`.
    pub fn sample(&self, nsample: usize, ngrid: usize, niter: usize) -> Vec<T> {
        match &self.sample_type {
            SampleType::Uniform => {
                let dist = |x: T| x;
                self.distribution_sample(nsample, ngrid, niter, &dist, &dist)
            }
            SampleType::Linear => {
                let step = (self.stop - self.start) / T::from(nsample).unwrap();
                (0..=nsample)
                    .map(|i| self.start + T::from(i).unwrap() * step)
                    .collect()
            }
            SampleType::Logarithmic { base } => {
                let dist = |x: T| x.log(*base);
                let dist_inv = |x: T| base.powf(x);
                self.distribution_sample(nsample, ngrid, niter, &dist, &dist_inv)
            }
            SampleType::Exponential { base } => {
                let dist = |x: T| base.powf(x);
                let dist_inv = |x: T| x.log(*base);
                self.distribution_sample(nsample, ngrid, niter, &dist, &dist_inv)
            }
            SampleType::Distribution { dist, dist_inv } => {
                self.distribution_sample(nsample, ngrid, niter, dist, dist_inv)
            }
        }
    }

    /// Perform sampling over multiple iterations.
    pub fn sample_arr(&self, nsample: usize, ngrid: usize, niter: usize) -> Vec<T> {
        (0..=niter)
            .flat_map(|iter| self.sample(nsample, ngrid, iter))
            .collect()
    }
}

// fn main() {
//     let dist = |x: f64| x.log10(); // Example distribution function
//     let dist_inv = |x: f64| (10.).powf(x); // Inverse of the distribution function
//     let params1 = SampleParams::new(1.0, 10.0, SampleType::Logarithmic { base: 10.0 }).unwrap();
//
//     let result1 = params1.sample_arr(2, 3, 2);
//     println!("{:?}", result1);
//
//     let params2 =
//         SampleParams::new(1.0, 10.0, SampleType::Distribution { dist, dist_inv }).unwrap();
//
//     let result2 = params2.sample_arr(2, 3, 2);
//     println!("{:?}", result2 == result1);
//
//     let dist = |x: f64| x; // Example distribution function
//     let dist_inv = |x: f64| x; // Inverse of the distribution function
//     let params1 = SampleParams::new(1.0, 10.0, SampleType::Uniform).unwrap();
//
//     let result1 = params1.sample_arr(2, 3, 2);
//     println!("{:?}", result1);
//
//     let params2 =
//         SampleParams::new(1.0, 10.0, SampleType::Distribution { dist, dist_inv }).unwrap();
//
//     let result2 = params2.sample_arr(2, 3, 2);
//     println!("{:?}", result2 == result1);
// }

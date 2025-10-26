use num::Float;
use std::fmt::Debug;

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
            (SampleType::Distribution { .. }, SampleType::Distribution { .. }) => false, // Avoid comparing function pointers
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleParams<T: num::Float> {
    start: T,
    stop: T,
    sample_type: SampleType<T>,
}

impl<T: Float + Debug> SampleParams<T> {
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

    pub fn sample_arr(&self, nsample: usize, ngrid: usize, niter: usize) -> Vec<T> {
        (0..=niter)
            .flat_map(|iter| self.sample(nsample, ngrid, iter))
            .collect()
    }
}

pub trait TimeCutoff: Clone + Send + Sync {
    fn evaluate(&self, _t: f64) -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UnitTimeCutoff;

impl TimeCutoff for UnitTimeCutoff {
}

/// Configuration for cutoff parameters in the gravitational wave calculator.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialTimeCutoff {
    pub t_cut: f64,
    pub t_0: f64,
    inv_t_0_sq: f64,
}

impl ExponentialTimeCutoff {
    pub fn new(smax: f64, ratio_t_cut: Option<f64>, ratio_t_0: Option<f64>) -> Self {
        let t_cut = ratio_t_cut.unwrap_or(0.999999999) * smax;
        let t_0 = ratio_t_0.unwrap_or(0.25) * (smax - t_cut);
        Self {
            t_cut,
            t_0,
            inv_t_0_sq: 1.0 / t_0.powi(2),
        }
    }
}

impl TimeCutoff for ExponentialTimeCutoff {
    #[inline]
    fn evaluate(&self, t: f64) -> f64 {
        if t < self.t_cut {
            1.0
        } else {
            let exponent = -((t - self.t_cut).powi(2)) * self.inv_t_0_sq;
            exponent.exp()
        }
    }
}

use ndarray::prelude::*;
use num_complex::Complex64;
use peroxide::numerical::integral::{Integral::G30K61, gauss_kronrod_quadrature};
use puruspe::Jn;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

pub trait TimeCutoff: Clone + Send + Sync {
    fn cutoff(&self, t: f64) -> f64;
    fn u_max(&self, s: f64) -> f64;
}

/// Configuration for cutoff parameters in the gravitational wave calculator.
#[derive(Debug, Clone)]
pub struct ExponentialTimeCutoff {
    pub t_cut: f64,
    pub t_0: f64,
}

impl ExponentialTimeCutoff {
    pub fn new(smax: f64, ratio_t_cut: Option<f64>, ratio_t_0: Option<f64>) -> Self {
        let t_cut = ratio_t_cut.unwrap_or(0.999999999) * smax;
        let t_0 = ratio_t_0.unwrap_or(0.25) * (smax - t_cut);
        Self { t_cut, t_0 }
    }
}

impl TimeCutoff for ExponentialTimeCutoff {
    #[inline]
    fn cutoff(&self, t: f64) -> f64 {
        if t < self.t_cut {
            1.0
        } else {
            let exponent = -((t - self.t_cut).powi(2)) / self.t_0.powi(2);
            exponent.exp()
        }
    }

    /// Cut-off for integral over u
    #[inline]
    fn u_max(&self, s: f64) -> f64 {
        1.0 + (self.t_cut + 7.0 * self.t_0) / s
    }
}

/// Enum to represent the type of integrand to compute.
#[derive(Debug, Clone, Copy)]
pub enum IntegrandType {
    XX,
    YY,
    ZZ,
    XZ,
}

/// Struct to hold integrand parameters and precompute shared terms.
#[derive(Debug, Clone)]
pub struct IntegrandCalculator<T>
where
    T: TimeCutoff,
{
    time_cutoff: T,
}

impl<T> IntegrandCalculator<T>
where
    T: TimeCutoff,
{
    pub fn new(time_cutoff: T) -> Self {
        Self { time_cutoff }
    }

    /// Computes both the real and imaginary parts of the integrand at point u.
    #[inline]
    pub fn compute(
        &self,
        w: f64,
        cos_thetak: f64,
        s: f64,
        sign: f64,
        u: f64,
        int_type: IntegrandType,
    ) -> Complex64 {
        let sin_thetak = (1.0 - cos_thetak * cos_thetak).sqrt();
        // Precompute common terms
        let u_squared_plus_sign = u * u + sign;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = w * sin_thetak * s * sqrt_term;
        let wsu = w * s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let cutoff_val = self.time_cutoff.cutoff(u * s);

        // Precompute Bessel functions
        let (bessel_0, bessel_1, bessel_2) = match int_type {
            IntegrandType::XX | IntegrandType::YY => {
                let b0 = Jn(0, bessel_arg);
                let b2 = Jn(2, bessel_arg);
                (b0, 0.0, b2)
            }
            IntegrandType::ZZ => (Jn(0, bessel_arg), 0.0, 0.0),
            IntegrandType::XZ => (0.0, Jn(1, bessel_arg), 0.0),
        };

        let (real, imag) = match int_type {
            IntegrandType::XX => {
                let factor = u_squared_plus_sign;
                let bessel_diff = bessel_0 - bessel_2;
                (
                    factor * exp_term_real * bessel_diff * cutoff_val,
                    factor * exp_term_imag * bessel_diff * cutoff_val,
                )
            }
            IntegrandType::YY => {
                let factor = u_squared_plus_sign;
                let bessel_sum = bessel_0 + bessel_2;
                (
                    factor * exp_term_real * bessel_sum * cutoff_val,
                    factor * exp_term_imag * bessel_sum * cutoff_val,
                )
            }
            IntegrandType::ZZ => {
                (exp_term_real * bessel_0 * cutoff_val, exp_term_imag * bessel_0 * cutoff_val)
            }
            IntegrandType::XZ => {
                let factor = sign * sqrt_term;
                (
                    factor * exp_term_real * bessel_1 * cutoff_val,
                    factor * exp_term_imag * bessel_1 * cutoff_val,
                )
            }
        };
        Complex64::new(real, imag)
    }
}

#[derive(Error, Debug)]
pub enum GWCalcError {
    #[error("phi1 and phi2 must have the same shape: got phi1={phi1:?}, phi2={phi2:?}")]
    ShapeMismatch { phi1: Vec<usize>, phi2: Vec<usize> },

    #[error(
        "Input arrays must have the same length: w_arr.len()={w_len}, cos_thetak_arr.len()={k_len}"
    )]
    LengthMismatch { w_len: usize, k_len: usize },

    #[error("Tolerance must be positive: got {0}")]
    InvalidTolerance(f64),

    #[error("Maximum iterations must be > 0: got {0}")]
    InvalidMaxIter(u32),

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),

    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("Invalid cutoff configuration: t_cut={t_cut}, t_0={t_0}, smax={smax}")]
    InvalidCutoff { t_cut: f64, t_0: f64, smax: f64 },
}

/// Struct to hold precomputed field arrays and weights.
#[derive(Debug, Clone)]
pub struct PrecomputedFieldArrays {
    pub dphi1_dz_sq: Array3<f64>,
    pub dphi1_ds_sq: Array3<f64>,
    pub dphi2_dz_sq: Array3<f64>,
    pub dphi2_ds_sq: Array3<f64>,
    pub xz_deriv_dphi1_dz: Array3<f64>,
    pub xz_deriv_dphi1_ds: Array3<f64>,
    pub xz_deriv_dphi2_dz: Array3<f64>,
    pub xz_deriv_dphi2_ds: Array3<f64>,
    pub zz_weights: Array1<f64>,
    pub rr_weights: Array1<f64>,
    pub xz_weights: Array1<f64>,
}

// Enum to hold the number of bubbles in the simulation.
// Depending on whether we have OneBubble or TwoBubbles, the computation of GW spectrum should be
// scaled differently.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialFieldStatus {
    OneBubble,
    TwoBubbles,
}

/// Configuration for the space-time lattice.
#[derive(Debug, Clone)]
pub struct LatticeConfig {
    pub z_grid: Array1<f64>,
    pub s_grid: Array1<f64>,
    pub ds: f64,
    pub dz: f64,
    pub n_s: usize,
    pub n_z: usize,
    pub initial_field_status: InitialFieldStatus,
}

/// Gravitational wave calculator for computing the spectrum.
/// phi1 is the solution for t >= r, phi2 is the solution for t < r.
pub struct GravitationalWaveCalculator<T>
where
    T: TimeCutoff,
{
    pub phi1: Array3<f64>,
    pub phi2: Array3<f64>,
    pub precomputed: PrecomputedFieldArrays,
    pub lattice: LatticeConfig,
    pub integrand: IntegrandCalculator<T>,
    pub tol: f64,
    pub max_iter: u32,
    pub s_offset: Array1<f64>,
    pub n_fields: usize,
    thread_pool: ThreadPool,
}

impl<T> GravitationalWaveCalculator<T>
where
    T: TimeCutoff,
{
    pub fn new(
        initial_field_status: InitialFieldStatus,
        phi1: Array3<f64>,
        phi2: Array3<f64>,
        z_grid: Array1<f64>,
        ds: f64,
        integrand: IntegrandCalculator<T>,
    ) -> Result<Self, GWCalcError> {
        let n_fields = phi1.shape()[0];
        let n_s = phi1.shape()[1];
        let n_z = phi1.shape()[2];
        let dz = (z_grid[1] - z_grid[0]).abs();
        let s_grid: Array1<f64> = (0..n_s).map(|i| i as f64 * ds).collect();

        if phi2.shape() != phi1.shape() {
            return Err(GWCalcError::ShapeMismatch {
                phi1: phi1.shape().to_vec(),
                phi2: phi2.shape().to_vec(),
            });
        }

        let lattice = LatticeConfig {
            z_grid,
            s_grid,
            ds,
            dz,
            n_s,
            n_z,
            initial_field_status,
        };
        let precomputed = Self::compute_precomputed_arrays(&phi1, &phi2, n_fields, &lattice);

        let s_offset: Array1<f64> = lattice.s_grid.slice(s![1..]).mapv(|s| s - 0.5 * ds);

        let default_num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(default_num_threads)
            .build()
            .map_err(GWCalcError::ThreadPoolBuildError)?;

        Ok(GravitationalWaveCalculator {
            phi1,
            phi2,
            precomputed,
            lattice,
            integrand,
            tol: 1e-5,
            max_iter: 20,
            s_offset,
            n_fields,
            thread_pool,
        })
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> Result<(), GWCalcError> {
        self.thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(GWCalcError::ThreadPoolBuildError)?;
        Ok(())
    }

    pub fn set_integral_params(&mut self, tol: f64, max_iter: u32) -> Result<(), GWCalcError> {
        if tol <= 0.0 {
            return Err(GWCalcError::InvalidTolerance(tol));
        }
        if max_iter == 0 {
            return Err(GWCalcError::InvalidMaxIter(max_iter));
        }
        self.tol = tol;
        self.max_iter = max_iter;
        Ok(())
    }

    pub fn compute_precomputed_arrays(
        phi1: &Array3<f64>,
        phi2: &Array3<f64>,
        n_fields: usize,
        lattice: &LatticeConfig,
    ) -> PrecomputedFieldArrays {
        let mut dphi1_dz_sq = Array3::zeros((lattice.n_s, n_fields, lattice.n_z - 1));
        let mut dphi1_ds_sq = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z));
        let mut dphi2_dz_sq = Array3::zeros((lattice.n_s, n_fields, lattice.n_z - 1));
        let mut dphi2_ds_sq = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z));
        let mut xz_deriv_dphi1_dz = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z - 1));
        let mut xz_deriv_dphi1_ds = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z - 1));
        let mut xz_deriv_dphi2_dz = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z - 1));
        let mut xz_deriv_dphi2_ds = Array3::zeros((lattice.n_s - 1, n_fields, lattice.n_z - 1));

        for n in 0..n_fields {
            for i_s in 0..lattice.n_s {
                for i_z in 0..(lattice.n_z - 1) {
                    let diff1 = (phi1[[n, i_s, i_z + 1]] - phi1[[n, i_s, i_z]]) / lattice.dz;
                    let diff2 = (phi2[[n, i_s, i_z + 1]] - phi2[[n, i_s, i_z]]) / lattice.dz;
                    dphi1_dz_sq[[i_s, n, i_z]] = diff1 * diff1;
                    dphi2_dz_sq[[i_s, n, i_z]] = diff2 * diff2;
                }
            }
        }

        for n in 0..n_fields {
            for i_s in 0..(lattice.n_s - 1) {
                for i_z in 0..lattice.n_z {
                    let diff1 = (phi1[[n, i_s + 1, i_z]] - phi1[[n, i_s, i_z]]) / lattice.ds;
                    let diff2 = (phi2[[n, i_s + 1, i_z]] - phi2[[n, i_s, i_z]]) / lattice.ds;
                    dphi1_ds_sq[[i_s, n, i_z]] = diff1 * diff1;
                    dphi2_ds_sq[[i_s, n, i_z]] = diff2 * diff2;
                }
            }
        }

        for n in 0..n_fields {
            for i_s in 1..lattice.n_s {
                for i_z in 1..lattice.n_z {
                    let i_s_m1 = i_s - 1;
                    let i_z_m1 = i_z - 1;
                    xz_deriv_dphi1_dz[[i_s_m1, n, i_z_m1]] = (1.0 / (2.0 * lattice.dz))
                        * (phi1[[n, i_s, i_z]] - phi1[[n, i_s, i_z_m1]] + phi1[[n, i_s_m1, i_z]]
                            - phi1[[n, i_s_m1, i_z_m1]]);
                    xz_deriv_dphi1_ds[[i_s_m1, n, i_z_m1]] = (1.0 / (2.0 * lattice.ds))
                        * (phi1[[n, i_s, i_z]] - phi1[[n, i_s_m1, i_z]] + phi1[[n, i_s, i_z_m1]]
                            - phi1[[n, i_s_m1, i_z_m1]]);
                    xz_deriv_dphi2_dz[[i_s_m1, n, i_z_m1]] = (1.0 / (2.0 * lattice.dz))
                        * (phi2[[n, i_s, i_z]] - phi2[[n, i_s, i_z_m1]] + phi2[[n, i_s_m1, i_z]]
                            - phi2[[n, i_s_m1, i_z_m1]]);
                    xz_deriv_dphi2_ds[[i_s_m1, n, i_z_m1]] = (1.0 / (2.0 * lattice.ds))
                        * (phi2[[n, i_s, i_z]] - phi2[[n, i_s_m1, i_z]] + phi2[[n, i_s, i_z_m1]]
                            - phi2[[n, i_s_m1, i_z_m1]]);
                }
            }
        }

        let zz_weights: Array1<f64> = (1..lattice.n_s)
            .map(|i| {
                let factor = if i == lattice.n_s - 1 { 0.5 } else { 1.0 };
                (i as f64).powi(2) * lattice.ds.powi(3) * factor
            })
            .collect();
        let rr_weights: Array1<f64> = lattice
            .s_grid
            .slice(s![1..])
            .mapv(|s| 0.5 * (s - 0.5 * lattice.ds).powi(2) * lattice.ds);
        let xz_weights: Array1<f64> = lattice
            .s_grid
            .slice(s![1..])
            .mapv(|s| (s - 0.5 * lattice.ds).powi(2) * lattice.ds);

        PrecomputedFieldArrays {
            dphi1_dz_sq,
            dphi1_ds_sq,
            dphi2_dz_sq,
            dphi2_ds_sq,
            xz_deriv_dphi1_dz,
            xz_deriv_dphi1_ds,
            xz_deriv_dphi2_dz,
            xz_deriv_dphi2_ds,
            zz_weights,
            rr_weights,
            xz_weights,
        }
    }

    /// The zz component of energy momentum tensor after Fourier transform
    #[inline]
    pub fn compute_t_zz(
        &self,
        w: f64,
        cos_thetak: f64,
        exp_wkz_shifted: &Array1<Complex64>,
    ) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_zz_r1: Array1<Complex64> = self
            .lattice
            .s_grid
            .slice(s![1..])
            .mapv(|s| self.integral_u_quad(IntegrandType::ZZ, s, cos_thetak, w, -1.0));
        let u_zz_r2: Array1<Complex64> = self
            .lattice
            .s_grid
            .slice(s![1..])
            .mapv(|s| self.integral_u_quad(IntegrandType::ZZ, s, cos_thetak, w, 1.0));

        let zz_result: Array1<Complex64> = u_zz_r1
            .iter()
            .zip(u_zz_r2.iter())
            .enumerate()
            .map(|(i_s, (&u_r1, &u_r2))| {
                let mut sum_r1 = Complex64::new(0.0, 0.0);
                let mut sum_r2 = Complex64::new(0.0, 0.0);
                for i_z in 0..(self.lattice.n_z - 1) {
                    let mut dz_sum1 = 0.0;
                    let mut dz_sum2 = 0.0;
                    for n in 0..self.n_fields {
                        dz_sum1 += self.precomputed.dphi1_dz_sq[[i_s + 1, n, i_z]];
                        dz_sum2 += self.precomputed.dphi2_dz_sq[[i_s + 1, n, i_z]];
                    }
                    sum_r1 += dz_sum1 * exp_wkz_shifted[i_z];
                    sum_r2 += dz_sum2 * exp_wkz_shifted[i_z];
                }
                (u_r1 * sum_r1 + u_r2 * sum_r2) * dz_factor
            })
            .collect();

        let mut t_zz = Complex64::new(0.0, 0.0);
        for i_s in 0..(self.lattice.n_s - 1) {
            t_zz += zz_result[i_s] * self.precomputed.zz_weights[i_s];
        }
        t_zz
    }

    /// Computes the xx component of the energy-momentum tensor after Fourier transform.
    #[inline]
    pub fn compute_t_xx(&self, w: f64, cos_thetak: f64, exp_wkz: &Array1<Complex64>) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_xx_r1: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::XX, s, cos_thetak, w, -1.0));
        let u_xx_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::XX, s, cos_thetak, w, 1.0));

        let xx_result: Array1<Complex64> = u_xx_r1
            .iter()
            .zip(u_xx_r2.iter())
            .enumerate()
            .map(|(i_s, (&u_xx1, &u_xx2))| {
                let mut sum_r1 = Complex64::new(0.0, 0.0);
                let mut sum_r2 = Complex64::new(0.0, 0.0);
                for i_z in 0..self.lattice.n_z {
                    let mut ds_sum1 = 0.0;
                    let mut ds_sum2 = 0.0;
                    for n in 0..self.n_fields {
                        ds_sum1 += self.precomputed.dphi1_ds_sq[[i_s, n, i_z]];
                        ds_sum2 += self.precomputed.dphi2_ds_sq[[i_s, n, i_z]];
                    }
                    let z_weight = if i_z == 0 || i_z == self.lattice.n_z - 1 {
                        0.5
                    } else {
                        1.0
                    };
                    let weighted_exp = exp_wkz[i_z] * z_weight;
                    sum_r1 += ds_sum1 * weighted_exp;
                    sum_r2 += ds_sum2 * weighted_exp;
                }
                (u_xx1 * sum_r1 + u_xx2 * sum_r2) * dz_factor
            })
            .collect();

        let mut t_xx = Complex64::new(0.0, 0.0);
        for i_s in 0..(self.lattice.n_s - 1) {
            t_xx += xx_result[i_s] * self.precomputed.rr_weights[i_s];
        }
        t_xx
    }

    /// Computes the yy component of the energy-momentum tensor after Fourier transform.
    #[inline]
    pub fn compute_t_yy(&self, w: f64, cos_thetak: f64, exp_wkz: &Array1<Complex64>) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_yy_r1: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::YY, s, cos_thetak, w, -1.0));
        let u_yy_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::YY, s, cos_thetak, w, 1.0));

        let yy_result: Array1<Complex64> = u_yy_r1
            .iter()
            .zip(u_yy_r2.iter())
            .enumerate()
            .map(|(i_s, (&u_yy1, &u_yy2))| {
                let mut sum_r1 = Complex64::new(0.0, 0.0);
                let mut sum_r2 = Complex64::new(0.0, 0.0);
                for i_z in 0..self.lattice.n_z {
                    let mut ds_sum1 = 0.0;
                    let mut ds_sum2 = 0.0;
                    for n in 0..self.n_fields {
                        ds_sum1 += self.precomputed.dphi1_ds_sq[[i_s, n, i_z]];
                        ds_sum2 += self.precomputed.dphi2_ds_sq[[i_s, n, i_z]];
                    }
                    let z_weight = if i_z == 0 || i_z == self.lattice.n_z - 1 {
                        0.5
                    } else {
                        1.0
                    };
                    let weighted_exp = exp_wkz[i_z] * z_weight;
                    sum_r1 += ds_sum1 * weighted_exp;
                    sum_r2 += ds_sum2 * weighted_exp;
                }
                (u_yy1 * sum_r1 + u_yy2 * sum_r2) * dz_factor
            })
            .collect();

        let mut t_yy = Complex64::new(0.0, 0.0);
        for i_s in 0..(self.lattice.n_s - 1) {
            t_yy += yy_result[i_s] * self.precomputed.rr_weights[i_s];
        }
        t_yy
    }

    /// The xz component of energy momentum tensor after Fourier transform
    #[inline]
    pub fn compute_t_xz(
        &self,
        w: f64,
        cos_thetak: f64,
        exp_wkz_shifted: &Array1<Complex64>,
    ) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_xz_r1: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::XZ, s, cos_thetak, w, -1.0));
        let u_xz_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad(IntegrandType::XZ, s, cos_thetak, w, 1.0));

        let xz_result: Array1<Complex64> = u_xz_r1
            .iter()
            .zip(u_xz_r2.iter())
            .enumerate()
            .map(|(i_s, (&u_r1, &u_r2))| {
                let mut sum_r1 = Complex64::new(0.0, 0.0);
                let mut sum_r2 = Complex64::new(0.0, 0.0);
                for i_z in 0..(self.lattice.n_z - 1) {
                    let mut xz_sum1 = 0.0;
                    let mut xz_sum2 = 0.0;
                    for n in 0..self.n_fields {
                        xz_sum1 += self.precomputed.xz_deriv_dphi1_ds[[i_s, n, i_z]]
                            * self.precomputed.xz_deriv_dphi1_dz[[i_s, n, i_z]];
                        xz_sum2 += self.precomputed.xz_deriv_dphi2_ds[[i_s, n, i_z]]
                            * self.precomputed.xz_deriv_dphi2_dz[[i_s, n, i_z]];
                    }
                    sum_r1 += exp_wkz_shifted[i_z] * xz_sum1;
                    sum_r2 += exp_wkz_shifted[i_z] * xz_sum2;
                }
                -(u_r1 * sum_r1 + u_r2 * sum_r2) * dz_factor
            })
            .collect();

        let mut t_xz = Complex64::new(0.0, 0.0);
        for i_s in 0..(self.lattice.n_s - 1) {
            t_xz += xz_result[i_s] * self.precomputed.xz_weights[i_s];
        }
        t_xz *= Complex64::new(0., 1.);
        t_xz
    }

    pub fn integral_u_quad(
        &self,
        int_type: IntegrandType,
        s: f64,
        cos_thetak: f64,
        w: f64,
        sign: f64,
    ) -> Complex64 {
        let u_min = if sign == -1.0 { 1.0 } else { 0.0 };
        let u_max = self.integrand.time_cutoff.u_max(s);

        let integrand =
            |u: f64| -> Complex64 { self.integrand.compute(w, cos_thetak, s, sign, u, int_type) };

        gauss_kronrod_quadrature(integrand, (u_min, u_max), G30K61(self.tol, self.max_iter))
    }

    /// Computes the energy-momentum tensor components (t_xx, t_yy, t_zz, t_xz)
    /// on the full Cartesian product of `w_arr` × `cos_thetak_arr`.
    ///
    /// Returns `Array3<Complex64>` of shape `(n_k, n_w, 4)`:
    /// - axis 0 → cosθ_k index
    /// - axis 1 → ω index
    /// - axis 2 → component: [0]=t_xx, [1]=t_yy, [2]=t_zz, [3]=t_xz
    ///
    /// Points with |cosθ_k| = 1.0 are left as zero.
    #[inline]
    pub fn compute_t_tensor<W, C>(
        &self,
        w_arr: W,
        cos_thetak_arr: C,
    ) -> Result<Array3<Complex64>, GWCalcError>
    where
        W: AsRef<[f64]>,
        C: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let cos_thetak_arr = cos_thetak_arr.as_ref();

        let n_w = w_arr.len();
        let n_k = cos_thetak_arr.len();

        if n_w == 0 || n_k == 0 {
            return Err(GWCalcError::LengthMismatch {
                w_len: n_w,
                k_len: n_k,
            });
        }

        // Build task list of valid (w, cosθ_k) pairs + output position mapping
        let mut tasks = Vec::with_capacity(n_w * n_k);
        let mut index_map = Vec::with_capacity(n_w * n_k);

        for (i_w, &w) in w_arr.iter().enumerate() {
            for (i_k, &cos_thetak) in cos_thetak_arr.iter().enumerate() {
                if cos_thetak == 1.0 || cos_thetak == -1.0 {
                    continue; // skip poles → remain zero
                }
                tasks.push((w, cos_thetak));
                index_map.push((i_k, i_w));
            }
        }

        // Use the owned thread_pool stored in self
        let results: Vec<(Complex64, Complex64, Complex64, Complex64)> =
            self.thread_pool.install(|| {
                tasks
                    .par_iter()
                    .copied()
                    .map(|(w, cos_thetak)| self.compute_t_tensor_single_point(w, cos_thetak))
                    .collect()
            });

        // Fill output tensor
        let mut t_tensor = Array3::<Complex64>::zeros((n_k, n_w, 4));

        for (&(i_k, i_w), (t_xx, t_yy, t_zz, t_xz)) in index_map.iter().zip(results.iter()) {
            t_tensor[[i_k, i_w, 0]] = *t_xx;
            t_tensor[[i_k, i_w, 1]] = *t_yy;
            t_tensor[[i_k, i_w, 2]] = *t_zz;
            t_tensor[[i_k, i_w, 3]] = *t_xz;
        }

        Ok(t_tensor)
    }

    #[inline]
    fn compute_t_tensor_single_point(
        &self,
        w: f64,
        cos_thetak: f64,
    ) -> (Complex64, Complex64, Complex64, Complex64) {
        let exp_wkz: Array1<Complex64> = self
            .lattice
            .z_grid
            .mapv(|z| Complex64::new(0.0, -w * cos_thetak * z).exp());

        let wkz_shifted: Array1<f64> = self
            .lattice
            .z_grid
            .slice(s![1..])
            .mapv(|z| w * cos_thetak * (z - 0.5 * self.lattice.dz));
        let exp_wkz_shifted: Array1<Complex64> =
            wkz_shifted.mapv(|x| Complex64::new(0.0, -x).exp());

        let t_zz = self.compute_t_zz(w, cos_thetak, &exp_wkz_shifted);
        let t_xx = self.compute_t_xx(w, cos_thetak, &exp_wkz);
        let t_yy = self.compute_t_yy(w, cos_thetak, &exp_wkz);
        let t_xz = self.compute_t_xz(w, cos_thetak, &exp_wkz_shifted);

        (t_xx, t_yy, t_zz, t_xz)
    }

    /// Computes the gravitational wave spectrum on a full (w × cosθ_k) grid.
    /// Returns an Array2<f64> of shape (n_cos_thetak, n_w) where:
    ///   result[i_k, i_w] = dE/(dlogω dcosθ_k) at cos_thetak_arr[i_k] and w_arr[i_w]
    #[inline]
    pub fn compute_angular_gw_spectrum<W, C>(
        &self,
        w_arr: W,
        cos_thetak_arr: C,
    ) -> Result<Array2<f64>, GWCalcError>
    where
        W: AsRef<[f64]>,
        C: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let cos_thetak_arr = cos_thetak_arr.as_ref();
        let n_w = w_arr.len();
        let n_k = cos_thetak_arr.len();

        // Build full mesh of valid (w, cosθ_k) pairs
        let mut wk_pairs = Vec::with_capacity(n_w * n_k);
        let mut row_indices = Vec::with_capacity(n_w * n_k); // (i_k, i_w) → flat index mapping

        for (i_w, &w) in w_arr.iter().enumerate() {
            for (i_k, &cos_thetak) in cos_thetak_arr.iter().enumerate() {
                if cos_thetak == 1.0 || cos_thetak == -1.0 {
                    continue; // skip poles
                }
                wk_pairs.push((w, cos_thetak));
                row_indices.push((i_k, i_w));
            }
        }

        // Parallel computation over all valid pairs
        let results: Vec<f64> = self.thread_pool.install(|| {
            wk_pairs
                .par_iter()
                .map(|&(w, cos_thetak)| self.compute_single_angular_point(w, cos_thetak))
                .collect()
        });

        // Fill output Array2, zeros where cosθ_k = ±1
        let mut spectrum = Array2::<f64>::zeros((n_k, n_w));

        for (&(i_k, i_w), val) in row_indices.iter().zip(results.iter()) {
            spectrum[[i_k, i_w]] = *val;
        }

        Ok(spectrum)
    }

    // Helper: extracted computation for a single (w, cosθ_k) point
    // (avoids duplicating the long body in two branches)
    #[inline]
    fn compute_single_angular_point(&self, w: f64, cos_thetak: f64) -> f64 {
        let cos_thetak_sq = cos_thetak.powi(2);
        let sin_thetak_sq = 1.0 - cos_thetak_sq;

        let exp_wkz: Array1<Complex64> = self
            .lattice
            .z_grid
            .mapv(|z| Complex64::new(0.0, -w * cos_thetak * z).exp());

        let wkz_shifted: Array1<f64> = self
            .lattice
            .z_grid
            .slice(s![1..])
            .mapv(|z| w * cos_thetak * (z - 0.5 * self.lattice.dz));
        let exp_wkz_shifted: Array1<Complex64> =
            wkz_shifted.mapv(|x| Complex64::new(0.0, -x).exp());

        let t_zz = self.compute_t_zz(w, cos_thetak, &exp_wkz_shifted);
        let t_xx = self.compute_t_xx(w, cos_thetak, &exp_wkz);
        let t_yy = self.compute_t_yy(w, cos_thetak, &exp_wkz);
        let t_xz = self.compute_t_xz(w, cos_thetak, &exp_wkz_shifted);

        let t_rr = t_xx * cos_thetak_sq - t_yy;
        let w_cubed = w.powi(3);
        let two_pi = 2.0 * std::f64::consts::PI;

        let t_total_squared = (t_rr + sin_thetak_sq * t_zz
            - 2.0 * cos_thetak * sin_thetak_sq.sqrt() * t_xz)
            .norm_sqr();

        t_total_squared * w_cubed * two_pi
    }

    /// Computes the direction-averaged GW spectrum dE/dlogω
    pub fn compute_averaged_gw_spectrum<W, C>(
        &self,
        w_arr: W,
        cos_thetak_grid: C,
    ) -> Result<Vec<f64>, GWCalcError>
    where
        W: AsRef<[f64]>,
        C: AsRef<[f64]>,
    {
        let w_arr = w_arr.as_ref();
        let cos_thetak_grid = cos_thetak_grid.as_ref();
        let angular_spectrum = self.compute_angular_gw_spectrum(w_arr, cos_thetak_grid)?;

        let dcos = if cos_thetak_grid.len() > 1 {
            cos_thetak_grid[1] - cos_thetak_grid[0]
        } else {
            return Err(GWCalcError::LengthMismatch {
                w_len: w_arr.len(),
                k_len: cos_thetak_grid.len(),
            });
        };

        let n_w = w_arr.len();
        let mut de_dlogw_arr = vec![0.0; n_w];

        for (i_w, de_dlogw) in de_dlogw_arr.iter_mut().enumerate().take(n_w) {
            let column = angular_spectrum.column(i_w);
            let mut integral = 0.0;
            for (i_k, &val) in column.iter().enumerate() {
                let weight = if i_k == 0 || i_k == column.len() - 1 {
                    1.0
                } else {
                    2.0
                };
                integral += weight * val;
            }
            *de_dlogw = integral * dcos; // trapezoidal rule (Simpson-like)
        }

        Ok(de_dlogw_arr)
    }
}

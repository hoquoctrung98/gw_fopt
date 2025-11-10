use ndarray::prelude::*;
use num_complex::Complex64;
use peroxide::numerical::integral::{Integral::G30K61, gauss_kronrod_quadrature};
use rayon::prelude::*;

use super::gw_integrand::{IntegrandCalculator, IntegrandType};
use thiserror::Error;

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
    ThreadPoolBuild(#[from] rayon::ThreadPoolBuildError),

    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("Invalid cutoff configuration: t_cut={t_cut}, t_0={t_0}, smax={smax}")]
    InvalidCutoff { t_cut: f64, t_0: f64, smax: f64 },
}

/// Configuration for cutoff parameters in the gravitational wave calculator.
#[derive(Debug, Clone)]
pub struct CutoffConfig {
    pub t_cut: f64,
    pub t_0: f64,
}

impl CutoffConfig {
    pub fn new(smax: f64, ratio_t_cut: Option<f64>, ratio_t_0: Option<f64>) -> Self {
        let t_cut = ratio_t_cut.unwrap_or(0.9) * smax;
        let t_0 = ratio_t_0.unwrap_or(0.25) * (smax - t_cut);
        Self { t_cut, t_0 }
    }
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
pub struct GravitationalWaveCalculator {
    pub phi1: Array3<f64>,
    pub phi2: Array3<f64>,
    pub precomputed: PrecomputedFieldArrays,
    pub lattice: LatticeConfig,
    pub config: CutoffConfig,
    pub tol: f64,
    pub max_iter: u32,
    pub s_offset: Array1<f64>,
    pub n_fields: usize,
}

impl GravitationalWaveCalculator {
    pub fn new(
        initial_field_status: InitialFieldStatus,
        phi1: Array3<f64>,
        phi2: Array3<f64>,
        z_grid: Array1<f64>,
        ds: f64,
        ratio_t_cut: Option<f64>,
        ratio_t_0: Option<f64>,
    ) -> Result<Self, GWCalcError> {
        let n_fields = phi1.shape()[0];
        let n_s = phi1.shape()[1];
        let n_z = phi1.shape()[2];
        let dz = (z_grid[1] - z_grid[0]).abs();
        let s_grid: Array1<f64> = (0..n_s).map(|i| i as f64 * ds).collect();
        let smax = s_grid[n_s - 1];

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
        let config = CutoffConfig::new(smax, ratio_t_cut, ratio_t_0);
        let precomputed = Self::compute_precomputed_arrays(&phi1, &phi2, n_fields, &lattice);

        let s_offset: Array1<f64> = lattice.s_grid.slice(s![1..]).mapv(|s| s - 0.5 * ds);

        Ok(GravitationalWaveCalculator {
            phi1,
            phi2,
            precomputed,
            lattice,
            config,
            tol: 1e-5,
            max_iter: 20,
            s_offset,
            n_fields,
        })
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

    /// Cut-off for integral over u
    #[inline]
    pub fn u_max(&self, s: f64) -> f64 {
        1.0 + (self.config.t_cut + 7.0 * self.config.t_0) / s
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
        let u_max = self.u_max(s);

        let integrand_calc =
            IntegrandCalculator::new(w, cos_thetak, s, sign, self.config.t_cut, self.config.t_0);
        let integrand = |u: f64| -> Complex64 { integrand_calc.compute(u, int_type).unwrap() };

        gauss_kronrod_quadrature(integrand, (u_min, u_max), G30K61(self.tol, self.max_iter))
    }

    /// Computes the energy-momentum tensor components (t_xx, t_yy, t_zz, t_xz) for given arrays of
    /// frequencies (w) and wavenumbers (cos_thetak) in parallel.
    #[inline]
    pub fn compute_t_tensor(
        &self,
        w_arr: &[f64],
        cos_thetak_arr: &[f64],
        num_threads: Option<usize>,
    ) -> Result<Vec<(Complex64, Complex64, Complex64, Complex64)>, GWCalcError> {
        // Check that w_arr and cos_thetak_arr have the same length
        if w_arr.len() != cos_thetak_arr.len() {
            return Err(GWCalcError::LengthMismatch {
                w_len: w_arr.len(),
                k_len: cos_thetak_arr.len(),
            });
        }

        // Filter out invalid cos_thetak values (cos_thetak = ±1)
        let valid_wk_pairs: Vec<(f64, f64)> = w_arr
            .iter()
            .zip(cos_thetak_arr.iter())
            .filter(|&(_w, &cos_thetak)| cos_thetak != 1.0 && cos_thetak != -1.0)
            .map(|(&w, &cos_thetak)| (w, cos_thetak))
            .collect();

        // Compute the tensor components for each valid (w, cos_thetak) pair in parallel
        let results: Vec<Result<(Complex64, Complex64, Complex64, Complex64), String>> =
            match num_threads {
                Some(n) if n > 0 => rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()?
                    .install(|| {
                        valid_wk_pairs
                            .par_iter()
                            .map(|&(w, cos_thetak)| {
                                let exp_wkz: Array1<Complex64> = self
                                    .lattice
                                    .z_grid
                                    .clone()
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

                                Ok((t_xx, t_yy, t_zz, t_xz))
                            })
                            .collect::<Vec<_>>()
                    }),
                _ => valid_wk_pairs
                    .par_iter()
                    .map(|&(w, cos_thetak)| {
                        let exp_wkz: Array1<Complex64> = self
                            .lattice
                            .z_grid
                            .clone()
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

                        Ok((t_xx, t_yy, t_zz, t_xz))
                    })
                    .collect::<Vec<_>>(),
            };

        let results: Vec<(Complex64, Complex64, Complex64, Complex64)> = results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| GWCalcError::IntegrationFailed(e.to_string()))?;

        // Create an output vector matching the input w_arr/cos_thetak_arr size, with zeros for invalid cos_thetak
        let mut t_tensor = vec![
            (
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            );
            w_arr.len()
        ];
        let mut valid_idx = 0;
        for (i, &cos_thetak) in cos_thetak_arr.iter().enumerate() {
            if cos_thetak != 1.0 && cos_thetak != -1.0 {
                t_tensor[i] = results[valid_idx];
                valid_idx += 1;
            }
        }

        Ok(t_tensor)
    }

    /// Computes the gravitational wave spectrum for given arrays of frequencies (w) and wavenumbers (k) in parallel.
    #[inline]
    pub fn compute_angular_gw_spectrum(
        &self,
        w_arr: &[f64],
        cos_thetak_arr: &[f64],
        num_threads: Option<usize>,
    ) -> Result<Vec<f64>, GWCalcError> {
        // Check that w_arr and k_arr have the same length
        if w_arr.len() != cos_thetak_arr.len() {
            return Err(GWCalcError::LengthMismatch {
                w_len: w_arr.len(),
                k_len: cos_thetak_arr.len(),
            });
        }

        // Filter out invalid k values (k = ±1)
        let valid_wk_pairs: Vec<(f64, f64)> = w_arr
            .iter()
            .zip(cos_thetak_arr.iter())
            .filter(|&(_w, &cos_thetak)| cos_thetak != 1.0 && cos_thetak != -1.0)
            .map(|(&w, &cos_thetak)| (w, cos_thetak))
            .collect();

        // Compute the spectrum for each valid (w, k) pair in parallel
        let results: Vec<Result<f64, GWCalcError>> = match num_threads {
            Some(n) if n > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()?
                .install(|| {
                    valid_wk_pairs
                        .par_iter()
                        .map(|&(w, cos_thetak)| {
                            let cos_thetak_sq = cos_thetak.powi(2);
                            let sin_thetak_sq = 1.0 - cos_thetak_sq;

                            let exp_wkz: Array1<Complex64> = self
                                .lattice
                                .z_grid
                                .clone()
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
                            let total_integrand_value = t_total_squared * w_cubed * two_pi;

                            Ok(total_integrand_value)
                        })
                        .collect::<Vec<_>>()
                }),
            _ => valid_wk_pairs
                .par_iter()
                .map(|&(w, cos_thetak)| {
                    let cos_thetak_sq = cos_thetak.powi(2);
                    let sin_thetak_sq = 1.0 - cos_thetak_sq;

                    let exp_wkz: Array1<Complex64> = self
                        .lattice
                        .z_grid
                        .clone()
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
                    let total_integrand_value = t_total_squared * w_cubed * two_pi;

                    Ok(total_integrand_value)
                })
                .collect::<Vec<_>>(),
        };

        let results: Vec<f64> = results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| GWCalcError::IntegrationFailed(e.to_string()))?;

        // Create an output vector matching the input w_arr/k_arr size, with 0.0 for invalid cos_thetak
        let mut de_dlogw_dcos_thetak = vec![0.0; w_arr.len()];
        let mut valid_idx = 0;
        for (i, &cos_thetak) in cos_thetak_arr.iter().enumerate() {
            if cos_thetak != 1.0 && cos_thetak != -1.0 {
                de_dlogw_dcos_thetak[i] = results[valid_idx];
                valid_idx += 1;
            }
        }

        Ok(de_dlogw_dcos_thetak)
    }

    /// Computes the gravitational wave spectrum averaged over all directions, returning only dE/dlogw.
    pub fn compute_averaged_gw_spectrum(
        &self,
        w_arr: &[f64],
        n_k: usize,
        num_threads: Option<usize>,
    ) -> Result<Vec<f64>, GWCalcError> {
        let cos_thetak_arr: Vec<f64> = (0..n_k).map(|i| i as f64 / (n_k - 1) as f64).collect();
        let dcos_thetak = cos_thetak_arr[1] - cos_thetak_arr[0];

        // Create (w, k) pairs
        let wk_pairs: Vec<(f64, f64)> = w_arr
            .iter()
            .flat_map(|&w| {
                cos_thetak_arr
                    .iter()
                    .map(move |&cos_thetak| (w, cos_thetak))
            })
            .collect();

        // Compute spectrum for all (w, k) pairs
        let results = self.compute_angular_gw_spectrum(
            &wk_pairs.iter().map(|&(w, _)| w).collect::<Vec<_>>(),
            &wk_pairs.iter().map(|&(_, k)| k).collect::<Vec<_>>(),
            num_threads,
        )?;

        // Organize results into integrand_dict
        let mut integrand_dict: Vec<Vec<f64>> = vec![vec![0.0; n_k]; w_arr.len()];
        for (i, &(w, _)) in wk_pairs.iter().enumerate() {
            let i_w = w_arr.iter().position(|&x| x == w).unwrap();
            let i_k = i % n_k;
            integrand_dict[i_w][i_k] = results[i];
        }

        // Compute dE/dlogw for each w by integrating over k
        let de_dlogw: Vec<f64> = w_arr
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let integrand_values = &integrand_dict[i];
                integrand_values
                    .iter()
                    .enumerate()
                    .fold(0.0, |acc, (i_k, &val)| {
                        let weight = if i_k == 0 || i_k == n_k - 1 { 1.0 } else { 2.0 };
                        acc + weight * val * dcos_thetak
                    })
            })
            .collect();

        Ok(de_dlogw)
    }
}

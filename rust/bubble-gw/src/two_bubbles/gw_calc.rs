// generalize gw_calc.rs in terms of cut-off functions
use ndarray::prelude::*;
use num_complex::Complex64;
use peroxide::numerical::integral::{Integral, gauss_kronrod_quadrature, integrate};
use puruspe::Jn;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

pub use crate::time_cutoff::{ExponentialTimeCutoff, TimeCutoff, UnitTimeCutoff};

pub trait IntegrationDomain: Clone + Send + Sync {
    fn u_bounds(&self, s: f64, sign: f64) -> (f64, f64);

    #[inline]
    fn u_max(&self, s: f64) -> f64 {
        self.u_bounds(s, 1.0).1
    }
}

impl IntegrationDomain for ExponentialTimeCutoff {
    /// Cut-off for integral over u
    #[inline]
    fn u_bounds(&self, s: f64, sign: f64) -> (f64, f64) {
        let u_min = if sign == -1.0 { 1.0 } else { 0.0 };
        let u_max = 1.0 + (self.t_cut + 7.0 * self.t_0) / s;
        (u_min, u_max)
    }

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

pub trait SignRegion: Copy + Send + Sync {
    const SIGN: f64;
    const U_MIN: f64;
}

#[derive(Debug, Clone, Copy)]
pub struct PositiveRegion;

impl SignRegion for PositiveRegion {
    const SIGN: f64 = 1.0;
    const U_MIN: f64 = 0.0;
}

#[derive(Debug, Clone, Copy)]
pub struct NegativeRegion;

impl SignRegion for NegativeRegion {
    const SIGN: f64 = -1.0;
    const U_MIN: f64 = 1.0;
}

pub trait TensorComponent: Copy + Send + Sync {
    fn compute<T, R>(integrand: &GwIntegrand<T, R, Self>, u: f64) -> Complex64
    where
        T: TimeCutoff,
        R: SignRegion;
}

#[derive(Debug, Clone, Copy)]
pub struct XXComponent;

impl TensorComponent for XXComponent {
    #[inline(always)]
    fn compute<T, R>(integrand: &GwIntegrand<T, R, Self>, u: f64) -> Complex64
    where
        T: TimeCutoff,
        R: SignRegion,
    {
        let u_squared_plus_sign = u * u + R::SIGN;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = integrand.w * integrand.sin_thetak * integrand.s * sqrt_term;
        let wsu = integrand.w * integrand.s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let cutoff_val = integrand.time_cutoff.evaluate(u * integrand.s);
        let bessel_0 = Jn(0, bessel_arg);
        let bessel_2 = Jn(2, bessel_arg);
        let bessel_diff = bessel_0 - bessel_2;
        Complex64::new(
            u_squared_plus_sign * exp_term_real * bessel_diff * cutoff_val,
            u_squared_plus_sign * exp_term_imag * bessel_diff * cutoff_val,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct YYComponent;

impl TensorComponent for YYComponent {
    #[inline(always)]
    fn compute<T, R>(integrand: &GwIntegrand<T, R, Self>, u: f64) -> Complex64
    where
        T: TimeCutoff,
        R: SignRegion,
    {
        let u_squared_plus_sign = u * u + R::SIGN;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = integrand.w * integrand.sin_thetak * integrand.s * sqrt_term;
        let wsu = integrand.w * integrand.s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let cutoff_val = integrand.time_cutoff.evaluate(u * integrand.s);
        let bessel_0 = Jn(0, bessel_arg);
        let bessel_2 = Jn(2, bessel_arg);
        let bessel_sum = bessel_0 + bessel_2;
        Complex64::new(
            u_squared_plus_sign * exp_term_real * bessel_sum * cutoff_val,
            u_squared_plus_sign * exp_term_imag * bessel_sum * cutoff_val,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ZZComponent;

impl TensorComponent for ZZComponent {
    #[inline(always)]
    fn compute<T, R>(integrand: &GwIntegrand<T, R, Self>, u: f64) -> Complex64
    where
        T: TimeCutoff,
        R: SignRegion,
    {
        let u_squared_plus_sign = u * u + R::SIGN;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = integrand.w * integrand.sin_thetak * integrand.s * sqrt_term;
        let wsu = integrand.w * integrand.s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let cutoff_val = integrand.time_cutoff.evaluate(u * integrand.s);
        let bessel_0 = Jn(0, bessel_arg);
        Complex64::new(exp_term_real * bessel_0 * cutoff_val, exp_term_imag * bessel_0 * cutoff_val)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct XZComponent;

impl TensorComponent for XZComponent {
    #[inline(always)]
    fn compute<T, R>(integrand: &GwIntegrand<T, R, Self>, u: f64) -> Complex64
    where
        T: TimeCutoff,
        R: SignRegion,
    {
        let u_squared_plus_sign = u * u + R::SIGN;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = integrand.w * integrand.sin_thetak * integrand.s * sqrt_term;
        let wsu = integrand.w * integrand.s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let cutoff_val = integrand.time_cutoff.evaluate(u * integrand.s);
        let bessel_1 = Jn(1, bessel_arg);
        let factor = R::SIGN * sqrt_term;
        Complex64::new(
            factor * exp_term_real * bessel_1 * cutoff_val,
            factor * exp_term_imag * bessel_1 * cutoff_val,
        )
    }
}

pub struct GwIntegrand<T, R, C>
where
    T: TimeCutoff,
    R: SignRegion,
    C: TensorComponent,
{
    time_cutoff: T,
    w: f64,
    sin_thetak: f64,
    s: f64,
    _region: std::marker::PhantomData<R>,
    _component: std::marker::PhantomData<C>,
}

impl<T, R, C> GwIntegrand<T, R, C>
where
    T: TimeCutoff,
    R: SignRegion,
    C: TensorComponent,
{
    #[inline]
    pub fn new(time_cutoff: T, w: f64, cos_thetak: f64, s: f64) -> Self {
        Self {
            time_cutoff,
            w,
            sin_thetak: (1.0 - cos_thetak * cos_thetak).sqrt(),
            s,
            _region: std::marker::PhantomData,
            _component: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn compute(&self, u: f64) -> Complex64 {
        C::compute(self, u)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuadratureConfig {
    pub method: Integral,
}

impl Default for QuadratureConfig {
    fn default() -> Self {
        Self {
            method: Integral::G30K61(1e-5, 20),
        }
    }
}

// there's an issue of using method `integrate` of peroxide directly with gauss-kronrod methods,
// this is temporary solution
#[inline]
fn integrate_with_method<F>(f: F, interval: (f64, f64), method: Integral) -> Complex64
where
    F: Fn(f64) -> Complex64 + Copy,
{
    match method {
        Integral::GaussLegendre(_) | Integral::NewtonCotes(_) => integrate(f, interval, method),
        Integral::G7K15(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G7K15(tol, max_iter))
        },
        Integral::G10K21(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G10K21(tol, max_iter))
        },
        Integral::G15K31(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G15K31(tol, max_iter))
        },
        Integral::G20K41(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G20K41(tol, max_iter))
        },
        Integral::G25K51(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G25K51(tol, max_iter))
        },
        Integral::G30K61(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G30K61(tol, max_iter))
        },
        Integral::G7K15R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G7K15R(tol, max_iter))
        },
        Integral::G10K21R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G10K21R(tol, max_iter))
        },
        Integral::G15K31R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G15K31R(tol, max_iter))
        },
        Integral::G20K41R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G20K41R(tol, max_iter))
        },
        Integral::G25K51R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G25K51R(tol, max_iter))
        },
        Integral::G30K61R(tol, max_iter) => {
            gauss_kronrod_quadrature(f, interval, Integral::G30K61R(tol, max_iter))
        },
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

    #[error("Gauss-Legendre integration order must be in 2..=30: got {0}")]
    InvalidGaussLegendreOrder(usize),

    #[error("Newton-Cotes integration requires at least 1 subinterval: got {0}")]
    InvalidNewtonCotesOrder(usize),

    #[error("The selected integration method does not accept tolerance/max_iter parameters")]
    IntegralParamsUnsupported,

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),

    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("Invalid cutoff configuration: t_cut={t_cut}, t_0={t_0}, smax={smax}")]
    InvalidCutoff { t_cut: f64, t_0: f64, smax: f64 },
}

/// Precomputed finite-difference derivatives and quadrature weights used in the
/// discrete approximation of the Fourier-transformed stress tensor.
///
/// The stored arrays correspond to lattice approximations of
/// $\left(\partial_s \Phi_\pm\right)^2$, $\left(\partial_z \Phi_\pm\right)^2$,
/// and $\partial_s \Phi_\pm \partial_z \Phi_\pm$, together with the
/// one-dimensional weights appearing in the outer $s$ integration.
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

/// Indicates whether the input field data represents a half-domain
/// single-bubble-symmetry setup or a full two-bubble domain.
///
/// This affects the normalization of the discrete $z$ integral:
/// `OneBubble` uses the mirror-symmetry factor $2\,dz$, while `TwoBubbles`
/// uses the full-domain factor $dz$.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialFieldStatus {
    OneBubble,
    TwoBubbles,
}

/// Geometry of the discrete $(s,z)$ lattice used for the two-bubble field
/// evolution.
///
/// The fields $\phi_+(s,z)$ and $\phi_-(s,z)$ are sampled on this grid, where
/// $\phi_+$ denotes the region $t^2 > x^2 + y^2$ and $\phi_-$ denotes the
/// region $t^2 < x^2 + y^2$.
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

/// Computes the exact two-bubble gravitational-wave spectrum from lattice field
/// data.
///
/// Here `phi1` stores the plus-region field $\phi_+(s, z)$ for $t^2 > x^2 + y^2$,
/// while `phi2` stores the minus-region field $\phi_-(s, z)$ for
/// $t^2 < x^2 + y^2$.
///
/// The calculator evaluates the Fourier-space stress tensor components and
/// combines them into the angular GW
/// spectrum. Following `docs/two_bubbles.md`, the basic building blocks are
/// <div>\[ \widetilde{T}_{ij}(\omega,\theta_k) = \int ds \int dz\ \mathcal{K}_{ij}(s,z;\omega,\theta_k). \]</div>
/// The kernel <span>\(\mathcal{K}_{ij}\)</span> contains the
/// field-derivative bilinears, the phase
/// <span>\(e^{-i \omega z \cos \theta_k}\)</span>, Bessel kernels from the
/// transverse coordinates, and the time cutoff
/// <span>\(C(t) = C(su)\)</span>.
pub struct GravitationalWaveCalculator<T>
where
    T: TimeCutoff + IntegrationDomain,
{
    pub phi1: Array3<f64>,
    pub phi2: Array3<f64>,
    pub precomputed: PrecomputedFieldArrays,
    pub lattice: LatticeConfig,
    pub time_cutoff: T,
    pub quadrature: QuadratureConfig,
    pub s_offset: Array1<f64>,
    pub n_fields: usize,
    thread_pool: ThreadPool,
}

impl<T> GravitationalWaveCalculator<T>
where
    T: TimeCutoff + IntegrationDomain,
{
    /// Builds a calculator from the sampled fields $\phi_+(s,z)$ (identified with `phi1`) and
    /// $\phi_-(s,z)$ (identified with `phi2`).
    ///
    /// `phi1` and `phi2` must have shape `(n_fields, n_s, n_z)` and share the
    /// same `z_grid` and spacing `ds`.
    pub fn new(
        initial_field_status: InitialFieldStatus,
        phi1: Array3<f64>,
        phi2: Array3<f64>,
        z_grid: Array1<f64>,
        ds: f64,
        time_cutoff: T,
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
            time_cutoff,
            quadrature: QuadratureConfig::default(),
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
        self.set_integration_params(Integral::G30K61(tol, max_iter))
    }

    pub fn set_integration_params(&mut self, method: Integral) -> Result<(), GWCalcError> {
        Self::validate_integration_method(method)?;
        self.quadrature.method = method;
        Ok(())
    }

    fn validate_integration_method(method: Integral) -> Result<(), GWCalcError> {
        match method {
            Integral::GaussLegendre(n) if !(2..=30).contains(&n) => {
                Err(GWCalcError::InvalidGaussLegendreOrder(n))
            },
            Integral::NewtonCotes(0) => Err(GWCalcError::InvalidNewtonCotesOrder(0)),
            Integral::G7K15(tol, _)
            | Integral::G10K21(tol, _)
            | Integral::G15K31(tol, _)
            | Integral::G20K41(tol, _)
            | Integral::G25K51(tol, _)
            | Integral::G30K61(tol, _)
            | Integral::G7K15R(tol, _)
            | Integral::G10K21R(tol, _)
            | Integral::G15K31R(tol, _)
            | Integral::G20K41R(tol, _)
            | Integral::G25K51R(tol, _)
            | Integral::G30K61R(tol, _)
                if tol <= 0.0 =>
            {
                Err(GWCalcError::InvalidTolerance(tol))
            },
            Integral::G7K15(_, max_iter)
            | Integral::G10K21(_, max_iter)
            | Integral::G15K31(_, max_iter)
            | Integral::G20K41(_, max_iter)
            | Integral::G25K51(_, max_iter)
            | Integral::G30K61(_, max_iter)
            | Integral::G7K15R(_, max_iter)
            | Integral::G10K21R(_, max_iter)
            | Integral::G15K31R(_, max_iter)
            | Integral::G20K41R(_, max_iter)
            | Integral::G25K51R(_, max_iter)
            | Integral::G30K61R(_, max_iter)
                if max_iter == 0 =>
            {
                Err(GWCalcError::InvalidMaxIter(max_iter))
            },
            _ => Ok(()),
        }
    }

    /// Precomputes the discrete derivative bilinears used in the tensor
    /// integrals.
    ///
    /// In continuum form, the tensor components depend on
    /// $\left(\partial_s \Phi_\pm\right)^2$,
    /// $\left(\partial_z \Phi_\pm\right)^2$, and
    /// $\partial_s \Phi_\pm \partial_z \Phi_\pm$.
    /// This method replaces those by centered or one-sided lattice finite
    /// differences and stores the corresponding $s$-integration weights.
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

    /// Computes the Fourier-space tensor component $\widetilde{T}_{zz}$.
    ///
    /// The continuum expression is
    /// <div>$ \widetilde{T}_{zz} = \int ds\ s^2 \int dz\ e^{-i \omega z \cos\theta_k} \left[ \left(\partial_z \Phi_+\right)^2 I^{(+)}_{zz}(s) + \left(\partial_z \Phi_-\right)^2 I^{(-)}_{zz}(s) \right]. $</div>
    /// where
    /// <div>\[ I^{(\pm)}_{zz}(s) = \int du\ C(su)\ e^{i \omega s u} J_0\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right). \]</div>
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
            .mapv(|s| self.integral_u_quad::<NegativeRegion, ZZComponent>(s, cos_thetak, w));
        let u_zz_r2: Array1<Complex64> = self
            .lattice
            .s_grid
            .slice(s![1..])
            .mapv(|s| self.integral_u_quad::<PositiveRegion, ZZComponent>(s, cos_thetak, w));

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

    /// Computes the Fourier-space tensor component $\widetilde{T}_{xx}$.
    ///
    /// The continuum expression is
    /// <div>\[ \widetilde{T}_{xx} = \frac{1}{2} \int ds\ s^2 \int dz\ e^{-i \omega z \cos\theta_k} \left[ \left(\partial_s \Phi_+\right)^2 I^{(+)}_{xx}(s) + \left(\partial_s \Phi_-\right)^2 I^{(-)}_{xx}(s) \right]. \]</div>
    /// with
    /// <div>\[ I^{(\pm)}_{xx}(s) = \int du\ C(su)\ (u^2 \pm 1)\ e^{i \omega s u} \left[ J_0\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right) - J_2\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right) \right]. \]</div>
    #[inline]
    pub fn compute_t_xx(&self, w: f64, cos_thetak: f64, exp_wkz: &Array1<Complex64>) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_xx_r1: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad::<NegativeRegion, XXComponent>(s, cos_thetak, w));
        let u_xx_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad::<PositiveRegion, XXComponent>(s, cos_thetak, w));

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

    /// Computes the Fourier-space tensor component $\widetilde{T}_{yy}$.
    ///
    /// This differs from <span>\(\widetilde{T}_{xx}\)</span> only in the
    /// Bessel-kernel combination:
    /// <div>\[ I^{(\pm)}_{yy}(s) = \int du\ C(su)\ (u^2 \pm 1)\ e^{i \omega s u} \left[ J_0\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right) + J_2\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right) \right]. \]</div>
    #[inline]
    pub fn compute_t_yy(&self, w: f64, cos_thetak: f64, exp_wkz: &Array1<Complex64>) -> Complex64 {
        let dz_factor = match self.lattice.initial_field_status {
            InitialFieldStatus::OneBubble => self.lattice.dz * 2.0,
            InitialFieldStatus::TwoBubbles => self.lattice.dz,
        };
        let u_yy_r1: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad::<NegativeRegion, YYComponent>(s, cos_thetak, w));
        let u_yy_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad::<PositiveRegion, YYComponent>(s, cos_thetak, w));

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

    /// Computes the Fourier-space tensor component $\widetilde{T}_{xz}$.
    ///
    /// The continuum expression is
    /// <div>\[ \widetilde{T}_{xz} = i \int ds\ s^2 \int dz\ e^{-i \omega z \cos\theta_k} \left[ \left(\partial_s \Phi_+ \partial_z \Phi_+\right) I^{(+)}_{xz}(s) - \left(\partial_s \Phi_- \partial_z \Phi_-\right) I^{(-)}_{xz}(s) \right]. \]</div>
    /// with
    /// <div>\[ I^{(\pm)}_{xz}(s) = \int du\ C(su)\ \sqrt{u^2 \pm 1}\ e^{i \omega s u} J_1\!\left(\omega s \sin\theta_k \sqrt{u^2 \pm 1}\right). \]</div>
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
            .mapv(|s| self.integral_u_quad::<NegativeRegion, XZComponent>(s, cos_thetak, w));
        let u_xz_r2: Array1<Complex64> = self
            .s_offset
            .mapv(|s| self.integral_u_quad::<PositiveRegion, XZComponent>(s, cos_thetak, w));

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

    #[inline]
    fn integral_u_quad<R, C>(&self, s: f64, cos_thetak: f64, w: f64) -> Complex64
    where
        R: SignRegion,
        C: TensorComponent,
    {
        let u_min = R::U_MIN;
        let u_max = self.time_cutoff.u_max(s);
        let integrand_kernel =
            GwIntegrand::<_, R, C>::new(self.time_cutoff.clone(), w, cos_thetak, s);
        let integrand = |u: f64| -> Complex64 { integrand_kernel.compute(u) };

        integrate_with_method(integrand, (u_min, u_max), self.quadrature.method)
    }

    /// Computes the tensor components
    /// <div>\[ (\widetilde{T}_{xx}, \widetilde{T}_{yy}, \widetilde{T}_{zz}, \widetilde{T}_{xz}) \]</div>
    /// on the full Cartesian product of `w_arr × cos_thetak_arr`.
    ///
    /// Returns `Array3<Complex64>` of shape `(n_k, n_w, 4)`:
    /// - axis 0: `cos(theta_k)` index
    /// - axis 1: `omega` index
    /// - axis 2: tensor component, namely
    ///   `result[..., 0] = T_xx`, `result[..., 1] = T_yy`,
    ///   `result[..., 2] = T_zz`, `result[..., 3] = T_xz`
    ///
    /// Points with `|cos(theta_k)| = 1` are left as zero because the transverse
    /// direction degenerates at the poles.
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

    /// Computes the angular spectrum
    /// $\dfrac{dE_{\mathrm{GW}}}{d\log\omega\ d\cos\theta_k}$ on the full
    /// $(n_\omega \times n_{\cos \theta_k})$ grid.
    ///
    /// With $\hat{k} = (\sin\theta_k, 0, \cos\theta_k)$, the projection formula
    /// reduces to
    /// <div>\[ \frac{dE_{\mathrm{GW}}}{d\omega\, d\Omega} = G \omega^2 \left| \cos^2\theta_k\, \widetilde{T}_{xx} - \widetilde{T}_{yy} + \sin^2\theta_k\, \widetilde{T}_{zz} - \sin(2\theta_k)\, \widetilde{T}_{xz} \right|^2. \]</div>
    /// Since this method returns the spectrum per logarithmic frequency
    /// interval and per `dcos(theta_k)`, the implementation uses the equivalent
    /// normalization
    /// <div>\[ \frac{dE_{\mathrm{GW}}}{d\log\omega\, d\cos\theta_k} = 2\pi\, \omega^3 \left| \cos^2\theta_k\, \widetilde{T}_{xx} - \widetilde{T}_{yy} + \sin^2\theta_k\, \widetilde{T}_{zz} - 2 \sin\theta_k \cos\theta_k\, \widetilde{T}_{xz} \right|^2. \]</div>
    ///
    /// Returns `Array2<f64>` of shape $(n_{\cos \theta_k}, n_\omega)$ where
    /// `result[i_k, i_w]` is the spectrum evaluated at
    /// `cos_thetak_arr[i_k]` and `w_arr[i_w]`.
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

    /// Computes the direction-averaged spectrum
    /// $\dfrac{dE_{\mathrm{GW}}}{d\log\omega}$ by integrating over
    /// $\cos\theta_k$.
    ///
    /// Numerically this applies a trapezoidal rule to
    /// `compute_angular_gw_spectrum`, i.e.
    /// <div>\[ \frac{dE_{\mathrm{GW}}}{d\log\omega} = \int_{-1}^{1} d \cos\theta_k\, \frac{dE_{\mathrm{GW}}}{d\log\omega\, d\cos\theta_k}. \]</div>
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

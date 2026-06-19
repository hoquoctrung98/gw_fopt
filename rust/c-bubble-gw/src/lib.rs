use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow;
use bubble_gw::many_bubbles::lattice::EmptyLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use bubble_gw::time_cutoff::{ExponentialTimeCutoff, UnitTimeCutoff};
use bubble_gw::two_bubbles::gw_calc::{GravitationalWaveCalculator, InitialFieldStatus};
use ndarray::{Array1, Array2, Array3};
use peroxide::numerical::integral::Integral;

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgwStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidArgument = 2,
    CalculationError = 3,
    Panic = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgwInitialFieldStatus {
    OneBubble = 0,
    TwoBubbles = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgwIntegrationKind {
    GaussLegendre = 0,
    NewtonCotes = 1,
    G7K15 = 2,
    G10K21 = 3,
    G15K31 = 4,
    G20K41 = 5,
    G25K51 = 6,
    G30K61 = 7,
    G7K15R = 8,
    G10K21R = 9,
    G15K31R = 10,
    G20K41R = 11,
    G25K51R = 12,
    G30K61R = 13,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BgwIntegrationConfig {
    pub kind: BgwIntegrationKind,
    pub order: usize,
    pub tol: f64,
    pub max_iter: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgwTimeCutoffKind {
    Unit = 0,
    Exponential = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BgwTimeCutoffConfig {
    pub kind: BgwTimeCutoffKind,
    pub smax: f64,
    pub ratio_t_cut: f64,
    pub ratio_t_0: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BgwComplex64 {
    pub re: f64,
    pub im: f64,
}

#[allow(non_camel_case_types)]
pub struct bgw_two_bubbles_calculator {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
pub struct bgw_generalized_bulk_flow {
    _private: [u8; 0],
}

struct TwoBubblesCalculatorHandle {
    inner: GravitationalWaveCalculator<ExponentialTimeCutoff>,
}

struct GeneralizedBulkFlowHandle {
    inner: GeneralizedBulkFlow<EmptyLattice>,
}

fn set_last_error(message: impl Into<String>) {
    let mut bytes = message.into().into_bytes();
    for byte in &mut bytes {
        if *byte == 0 {
            *byte = b' ';
        }
    }
    bytes.push(0);
    LAST_ERROR.with(|slot| *slot.borrow_mut() = bytes);
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = vec![0]);
}

fn run_ffi(f: impl FnOnce() -> Result<(), String>) -> BgwStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            clear_last_error();
            BgwStatus::Ok
        },
        Ok(Err(message)) => {
            set_last_error(message);
            BgwStatus::CalculationError
        },
        Err(_) => {
            set_last_error("Rust panic crossed the C ABI boundary");
            BgwStatus::Panic
        },
    }
}

fn null_error(name: &str) -> String {
    format!("{name} must not be null")
}

unsafe fn slice_from_raw<'a, T>(ptr: *const T, len: usize, name: &str) -> Result<&'a [T], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(null_error(name));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn slice_from_raw_mut<'a, T>(
    ptr: *mut T,
    len: usize,
    name: &str,
) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(null_error(name));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

fn checked_product(values: &[usize], name: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| format!("{name} length overflow"))
    })
}

fn option_from_nan(value: f64) -> Option<f64> {
    if value.is_nan() { None } else { Some(value) }
}

fn initial_field_status(status: BgwInitialFieldStatus) -> InitialFieldStatus {
    match status {
        BgwInitialFieldStatus::OneBubble => InitialFieldStatus::OneBubble,
        BgwInitialFieldStatus::TwoBubbles => InitialFieldStatus::TwoBubbles,
    }
}

fn integration_method(config: BgwIntegrationConfig) -> Integral {
    match config.kind {
        BgwIntegrationKind::GaussLegendre => Integral::GaussLegendre(config.order),
        BgwIntegrationKind::NewtonCotes => Integral::NewtonCotes(config.order),
        BgwIntegrationKind::G7K15 => Integral::G7K15(config.tol, config.max_iter),
        BgwIntegrationKind::G10K21 => Integral::G10K21(config.tol, config.max_iter),
        BgwIntegrationKind::G15K31 => Integral::G15K31(config.tol, config.max_iter),
        BgwIntegrationKind::G20K41 => Integral::G20K41(config.tol, config.max_iter),
        BgwIntegrationKind::G25K51 => Integral::G25K51(config.tol, config.max_iter),
        BgwIntegrationKind::G30K61 => Integral::G30K61(config.tol, config.max_iter),
        BgwIntegrationKind::G7K15R => Integral::G7K15R(config.tol, config.max_iter),
        BgwIntegrationKind::G10K21R => Integral::G10K21R(config.tol, config.max_iter),
        BgwIntegrationKind::G15K31R => Integral::G15K31R(config.tol, config.max_iter),
        BgwIntegrationKind::G20K41R => Integral::G20K41R(config.tol, config.max_iter),
        BgwIntegrationKind::G25K51R => Integral::G25K51R(config.tol, config.max_iter),
        BgwIntegrationKind::G30K61R => Integral::G30K61R(config.tol, config.max_iter),
    }
}

fn two_bubbles_handle<'a>(
    calculator: *mut bgw_two_bubbles_calculator,
) -> Result<&'a mut TwoBubblesCalculatorHandle, String> {
    if calculator.is_null() {
        return Err(null_error("calculator"));
    }
    Ok(unsafe { &mut *(calculator as *mut TwoBubblesCalculatorHandle) })
}

fn bulk_flow_handle<'a>(
    bulk_flow: *mut bgw_generalized_bulk_flow,
) -> Result<&'a mut GeneralizedBulkFlowHandle, String> {
    if bulk_flow.is_null() {
        return Err(null_error("bulk_flow"));
    }
    Ok(unsafe { &mut *(bulk_flow as *mut GeneralizedBulkFlowHandle) })
}

/// Returns the crate version string as a null-terminated static C string.
#[unsafe(no_mangle)]
pub extern "C" fn bgw_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Returns the thread-local error string for the most recent failing C API call.
#[unsafe(no_mangle)]
pub extern "C" fn bgw_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr().cast())
}

/// Creates a two-bubble gravitational-wave calculator.
///
/// `phi1` and `phi2` must be C-contiguous arrays with shape
/// `(n_fields, n_s, n_z)` and index order `((field * n_s + i_s) * n_z + i_z)`.
/// Pass `NAN` for either cutoff ratio to use the Rust default.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_new(
    out: *mut *mut bgw_two_bubbles_calculator,
    field_status: BgwInitialFieldStatus,
    phi1: *const f64,
    phi2: *const f64,
    n_fields: usize,
    n_s: usize,
    n_z: usize,
    z_grid: *const f64,
    ds: f64,
    ratio_t_cut: f64,
    ratio_t_0: f64,
) -> BgwStatus {
    if out.is_null() {
        set_last_error(null_error("out"));
        return BgwStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };

    run_ffi(|| {
        if n_fields == 0 || n_s < 2 || n_z < 2 {
            return Err("n_fields must be > 0 and n_s/n_z must be >= 2".to_string());
        }
        if ds <= 0.0 {
            return Err("ds must be positive".to_string());
        }
        let field_len = checked_product(&[n_fields, n_s, n_z], "field")?;
        let phi1 = unsafe { slice_from_raw(phi1, field_len, "phi1")? };
        let phi2 = unsafe { slice_from_raw(phi2, field_len, "phi2")? };
        let z_grid = unsafe { slice_from_raw(z_grid, n_z, "z_grid")? };

        let phi1 = Array3::from_shape_vec((n_fields, n_s, n_z), phi1.to_vec())
            .map_err(|e| e.to_string())?;
        let phi2 = Array3::from_shape_vec((n_fields, n_s, n_z), phi2.to_vec())
            .map_err(|e| e.to_string())?;
        let z_grid = Array1::from_vec(z_grid.to_vec());
        let smax = (n_s - 1) as f64 * ds;
        let time_cutoff = ExponentialTimeCutoff::new(
            smax,
            option_from_nan(ratio_t_cut),
            option_from_nan(ratio_t_0),
        );
        let inner = GravitationalWaveCalculator::new(
            initial_field_status(field_status),
            phi1,
            phi2,
            z_grid,
            ds,
            time_cutoff,
        )
        .map_err(|e| e.to_string())?;
        let handle = Box::new(TwoBubblesCalculatorHandle { inner });
        unsafe { *out = Box::into_raw(handle) as *mut bgw_two_bubbles_calculator };
        Ok(())
    })
}

/// Frees a two-bubble calculator. Passing null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_free(
    calculator: *mut bgw_two_bubbles_calculator,
) {
    if !calculator.is_null() {
        drop(unsafe { Box::from_raw(calculator as *mut TwoBubblesCalculatorHandle) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_set_num_threads(
    calculator: *mut bgw_two_bubbles_calculator,
    num_threads: usize,
) -> BgwStatus {
    run_ffi(|| {
        if num_threads == 0 {
            return Err("num_threads must be > 0".to_string());
        }
        two_bubbles_handle(calculator)?
            .inner
            .set_num_threads(num_threads)
            .map_err(|e| e.to_string())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_set_integration_params(
    calculator: *mut bgw_two_bubbles_calculator,
    config: BgwIntegrationConfig,
) -> BgwStatus {
    run_ffi(|| {
        two_bubbles_handle(calculator)?
            .inner
            .set_integration_params(integration_method(config))
            .map_err(|e| e.to_string())
    })
}

/// Computes angular spectrum into `out`, shape `(n_cos_thetak, n_w)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_compute_angular_gw_spectrum(
    calculator: *mut bgw_two_bubbles_calculator,
    w_arr: *const f64,
    n_w: usize,
    cos_thetak_arr: *const f64,
    n_cos_thetak: usize,
    out: *mut f64,
) -> BgwStatus {
    run_ffi(|| {
        let out_len = checked_product(&[n_cos_thetak, n_w], "angular spectrum output")?;
        let w_arr = unsafe { slice_from_raw(w_arr, n_w, "w_arr")? };
        let cos_thetak_arr =
            unsafe { slice_from_raw(cos_thetak_arr, n_cos_thetak, "cos_thetak_arr")? };
        let out = unsafe { slice_from_raw_mut(out, out_len, "out")? };
        let result = two_bubbles_handle(calculator)?
            .inner
            .compute_angular_gw_spectrum(w_arr, cos_thetak_arr)
            .map_err(|e| e.to_string())?;
        out.copy_from_slice(
            result
                .as_slice()
                .ok_or_else(|| "angular spectrum result is not contiguous".to_string())?,
        );
        Ok(())
    })
}

/// Computes averaged spectrum into `out`, shape `(n_w)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_two_bubbles_calculator_compute_averaged_gw_spectrum(
    calculator: *mut bgw_two_bubbles_calculator,
    w_arr: *const f64,
    n_w: usize,
    cos_thetak_arr: *const f64,
    n_cos_thetak: usize,
    out: *mut f64,
) -> BgwStatus {
    run_ffi(|| {
        let w_arr = unsafe { slice_from_raw(w_arr, n_w, "w_arr")? };
        let cos_thetak_arr =
            unsafe { slice_from_raw(cos_thetak_arr, n_cos_thetak, "cos_thetak_arr")? };
        let out = unsafe { slice_from_raw_mut(out, n_w, "out")? };
        let result = two_bubbles_handle(calculator)?
            .inner
            .compute_averaged_gw_spectrum(w_arr, cos_thetak_arr)
            .map_err(|e| e.to_string())?;
        out.copy_from_slice(result.as_slice());
        Ok(())
    })
}

/// Creates a generalized bulk-flow calculator using `EmptyLattice`.
///
/// Bubble arrays are C-contiguous with shape `(n_bubbles, 4)` and columns
/// `(t_c, x_c, y_c, z_c)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_empty_lattice_new(
    out: *mut *mut bgw_generalized_bulk_flow,
    bubbles_interior: *const f64,
    n_interior: usize,
    bubbles_exterior: *const f64,
    n_exterior: usize,
) -> BgwStatus {
    if out.is_null() {
        set_last_error(null_error("out"));
        return BgwStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };

    run_ffi(|| {
        let interior_len = checked_product(&[n_interior, 4], "interior bubbles")?;
        let exterior_len = checked_product(&[n_exterior, 4], "exterior bubbles")?;
        let interior =
            unsafe { slice_from_raw(bubbles_interior, interior_len, "bubbles_interior")? };
        let exterior =
            unsafe { slice_from_raw(bubbles_exterior, exterior_len, "bubbles_exterior")? };
        let interior = Array2::from_shape_vec((n_interior, 4), interior.to_vec())
            .map_err(|e| e.to_string())?;
        let exterior = Array2::from_shape_vec((n_exterior, 4), exterior.to_vec())
            .map_err(|e| e.to_string())?;
        let lattice_bubbles = LatticeBubbles::new(interior, Some(exterior), EmptyLattice {})
            .map_err(|e| e.to_string())?;
        let inner = GeneralizedBulkFlow::new(lattice_bubbles).map_err(|e| e.to_string())?;
        let handle = Box::new(GeneralizedBulkFlowHandle { inner });
        unsafe { *out = Box::into_raw(handle) as *mut bgw_generalized_bulk_flow };
        Ok(())
    })
}

/// Frees a generalized bulk-flow handle. Passing null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_free(bulk_flow: *mut bgw_generalized_bulk_flow) {
    if !bulk_flow.is_null() {
        drop(unsafe { Box::from_raw(bulk_flow as *mut GeneralizedBulkFlowHandle) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_set_num_threads(
    bulk_flow: *mut bgw_generalized_bulk_flow,
    num_threads: usize,
) -> BgwStatus {
    run_ffi(|| {
        if num_threads == 0 {
            return Err("num_threads must be > 0".to_string());
        }
        bulk_flow_handle(bulk_flow)?
            .inner
            .set_num_threads(num_threads)
            .map_err(|e| e.to_string())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_set_resolution(
    bulk_flow: *mut bgw_generalized_bulk_flow,
    n_cos_thetax: usize,
    n_phix: usize,
    precompute_first_bubbles: bool,
) -> BgwStatus {
    run_ffi(|| {
        bulk_flow_handle(bulk_flow)?
            .inner
            .set_resolution(n_cos_thetax, n_phix, precompute_first_bubbles)
            .map_err(|e| e.to_string())
    })
}

/// Sets scaling parameters from row-major arrays with shape `(n_sets, n_terms)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_set_gradient_scaling_params(
    bulk_flow: *mut bgw_generalized_bulk_flow,
    coefficients: *const f64,
    powers: *const f64,
    n_sets: usize,
    n_terms: usize,
    damping_width: f64,
) -> BgwStatus {
    run_ffi(|| {
        let len = checked_product(&[n_sets, n_terms], "gradient scaling")?;
        let coefficients = unsafe { slice_from_raw(coefficients, len, "coefficients")? };
        let powers = unsafe { slice_from_raw(powers, len, "powers")? };
        let coefficients_sets = coefficients
            .chunks(n_terms)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let powers_sets = powers
            .chunks(n_terms)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        bulk_flow_handle(bulk_flow)?
            .inner
            .set_gradient_scaling_params(
                coefficients_sets,
                powers_sets,
                option_from_nan(damping_width),
            )
            .map_err(|e| e.to_string())
    })
}

fn compute_c_integral_with_cutoff(
    handle: &mut GeneralizedBulkFlowHandle,
    w_arr: &[f64],
    has_t_begin: bool,
    t_begin: f64,
    t_end: f64,
    n_t: usize,
    selected_bubbles: Option<&[usize]>,
    time_cutoff: BgwTimeCutoffConfig,
) -> Result<Array3<num_complex::Complex64>, String> {
    let t_begin = if has_t_begin { Some(t_begin) } else { None };
    match time_cutoff.kind {
        BgwTimeCutoffKind::Unit => handle
            .inner
            .compute_c_integral(w_arr, t_begin, t_end, n_t, selected_bubbles, UnitTimeCutoff)
            .map_err(|e| e.to_string()),
        BgwTimeCutoffKind::Exponential => {
            let cutoff = ExponentialTimeCutoff::new(
                time_cutoff.smax,
                option_from_nan(time_cutoff.ratio_t_cut),
                option_from_nan(time_cutoff.ratio_t_0),
            );
            handle
                .inner
                .compute_c_integral(w_arr, t_begin, t_end, n_t, selected_bubbles, cutoff)
                .map_err(|e| e.to_string())
        },
    }
}

/// Computes C integral into `out`, shape `(2, n_sets, n_w)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bgw_generalized_bulk_flow_compute_c_integral(
    bulk_flow: *mut bgw_generalized_bulk_flow,
    w_arr: *const f64,
    n_w: usize,
    has_t_begin: bool,
    t_begin: f64,
    t_end: f64,
    n_t: usize,
    selected_bubbles: *const usize,
    n_selected_bubbles: usize,
    time_cutoff: BgwTimeCutoffConfig,
    out: *mut BgwComplex64,
    out_len: usize,
) -> BgwStatus {
    run_ffi(|| {
        let w_arr = unsafe { slice_from_raw(w_arr, n_w, "w_arr")? };
        let selected_bubbles = if selected_bubbles.is_null() {
            None
        } else {
            Some(unsafe {
                slice_from_raw(selected_bubbles, n_selected_bubbles, "selected_bubbles")?
            })
        };
        let result = compute_c_integral_with_cutoff(
            bulk_flow_handle(bulk_flow)?,
            w_arr,
            has_t_begin,
            t_begin,
            t_end,
            n_t,
            selected_bubbles,
            time_cutoff,
        )?;
        let expected_len = result.len();
        if out_len != expected_len {
            return Err(format!("out_len={out_len} does not match expected length {expected_len}"));
        }
        let out = unsafe { slice_from_raw_mut(out, out_len, "out")? };
        for (dst, src) in out.iter_mut().zip(result.iter()) {
            *dst = BgwComplex64 {
                re: src.re,
                im: src.im,
            };
        }
        Ok(())
    })
}

use std::ffi::CStr;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow;
use bubble_gw::many_bubbles::lattice::EmptyLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use bubble_gw::time_cutoff::{ExponentialTimeCutoff, UnitTimeCutoff};
use bubble_gw::two_bubbles::gw_calc::{GravitationalWaveCalculator, InitialFieldStatus};
use bubble_gw_c::*;
use ndarray::{Array1, Array2, Array3, arr2};

fn synthetic_fields() -> (Array3<f64>, Array3<f64>, Array1<f64>, f64) {
    let n_fields = 2;
    let n_s = 5;
    let n_z = 6;
    let ds = 0.35;
    let z_grid = Array1::linspace(-0.9, 1.1, n_z);

    let phi1 = Array3::from_shape_fn((n_fields, n_s, n_z), |(field, i_s, i_z)| {
        let s: f64 = i_s as f64 * ds;
        let z: f64 = z_grid[i_z];
        let field_shift: f64 = field as f64 + 1.0;
        (field_shift * s).sin() + 0.3 * z.powi(2) + 0.07 * field_shift * s * z
    });

    let phi2 = Array3::from_shape_fn((n_fields, n_s, n_z), |(field, i_s, i_z)| {
        let s: f64 = i_s as f64 * ds;
        let z: f64 = z_grid[i_z];
        let field_shift: f64 = field as f64 + 1.0;
        0.5 * (field_shift * z).cos() + 0.2 * s.powi(2) - 0.05 * field_shift * s * z
    });

    (phi1, phi2, z_grid, ds)
}

fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(rel_tol * scale),
        "actual={actual:?}, expected={expected:?}, diff={diff:?}"
    );
}

fn last_error() -> String {
    unsafe {
        CStr::from_ptr(bgw_last_error_message())
            .to_string_lossy()
            .into_owned()
    }
}

#[test]
fn two_bubbles_c_api_matches_direct_rust() {
    let (phi1, phi2, z_grid, ds) = synthetic_fields();
    let ratios = (0.8, 0.12);
    let smax = (phi1.shape()[1] - 1) as f64 * ds;
    let cutoff = ExponentialTimeCutoff::new(smax, Some(ratios.0), Some(ratios.1));
    let mut expected_calc = GravitationalWaveCalculator::new(
        InitialFieldStatus::TwoBubbles,
        phi1.clone(),
        phi2.clone(),
        z_grid.clone(),
        ds,
        cutoff,
    )
    .unwrap();
    expected_calc.set_num_threads(1).unwrap();
    expected_calc
        .set_integration_params(peroxide::numerical::integral::Integral::G15K31(1e-7, 30))
        .unwrap();

    let w_arr = [0.4, 0.9];
    let cos_thetak_arr = [-0.65, 0.45];
    let expected = expected_calc
        .compute_angular_gw_spectrum(w_arr, cos_thetak_arr)
        .unwrap();

    let mut handle = std::ptr::null_mut();
    let status = unsafe {
        bgw_two_bubbles_calculator_new(
            &mut handle,
            BgwInitialFieldStatus::TwoBubbles,
            phi1.as_slice().unwrap().as_ptr(),
            phi2.as_slice().unwrap().as_ptr(),
            phi1.shape()[0],
            phi1.shape()[1],
            phi1.shape()[2],
            z_grid.as_slice().unwrap().as_ptr(),
            ds,
            ratios.0,
            ratios.1,
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());
    assert!(!handle.is_null());

    let status = unsafe { bgw_two_bubbles_calculator_set_num_threads(handle, 1) };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());
    let status = unsafe {
        bgw_two_bubbles_calculator_set_integration_params(
            handle,
            BgwIntegrationConfig {
                kind: BgwIntegrationKind::G15K31,
                order: 0,
                tol: 1e-7,
                max_iter: 30,
            },
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());

    let mut actual = vec![0.0; w_arr.len() * cos_thetak_arr.len()];
    let status = unsafe {
        bgw_two_bubbles_calculator_compute_angular_gw_spectrum(
            handle,
            w_arr.as_ptr(),
            w_arr.len(),
            cos_thetak_arr.as_ptr(),
            cos_thetak_arr.len(),
            actual.as_mut_ptr(),
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected, 1e-10, 1e-10);
    }

    unsafe { bgw_two_bubbles_calculator_free(handle) };
}

#[test]
fn many_bubbles_c_api_matches_direct_bulk_flow() {
    let bubbles_interior = arr2(&[[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
    let bubbles_exterior = Array2::<f64>::zeros((0, 4));
    let coefficients = vec![0.0, 1.0];
    let powers = vec![3.0, 3.0];
    let w_arr = [0.5, 1.0];

    let mut expected_bulk = GeneralizedBulkFlow::new(
        LatticeBubbles::new(
            bubbles_interior.clone(),
            Some(bubbles_exterior.clone()),
            EmptyLattice {},
        )
        .unwrap(),
    )
    .unwrap();
    expected_bulk.set_num_threads(1).unwrap();
    expected_bulk.set_resolution(4, 5, true).unwrap();
    expected_bulk
        .set_gradient_scaling_params(vec![vec![0.0], vec![1.0]], vec![vec![3.0], vec![3.0]], None)
        .unwrap();
    let expected = expected_bulk
        .compute_c_integral(w_arr, Some(0.0), 8.0, 40, None, UnitTimeCutoff)
        .unwrap();

    let mut handle = std::ptr::null_mut();
    let status = unsafe {
        bgw_generalized_bulk_flow_empty_lattice_new(
            &mut handle,
            bubbles_interior.as_slice().unwrap().as_ptr(),
            bubbles_interior.nrows(),
            bubbles_exterior.as_slice().unwrap().as_ptr(),
            bubbles_exterior.nrows(),
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());
    let status = unsafe { bgw_generalized_bulk_flow_set_num_threads(handle, 1) };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());
    let status = unsafe { bgw_generalized_bulk_flow_set_resolution(handle, 4, 5, true) };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());
    let status = unsafe {
        bgw_generalized_bulk_flow_set_gradient_scaling_params(
            handle,
            coefficients.as_ptr(),
            powers.as_ptr(),
            2,
            1,
            f64::NAN,
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());

    let mut actual = vec![BgwComplex64 { re: 0.0, im: 0.0 }; expected.len()];
    let status = unsafe {
        bgw_generalized_bulk_flow_compute_c_integral(
            handle,
            w_arr.as_ptr(),
            w_arr.len(),
            true,
            0.0,
            8.0,
            40,
            std::ptr::null(),
            0,
            BgwTimeCutoffConfig {
                kind: BgwTimeCutoffKind::Unit,
                smax: 0.0,
                ratio_t_cut: f64::NAN,
                ratio_t_0: f64::NAN,
            },
            actual.as_mut_ptr(),
            actual.len(),
        )
    };
    assert_eq!(status, BgwStatus::Ok, "{}", last_error());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(actual.re, expected.re, 1e-10, 1e-10);
        assert_close(actual.im, expected.im, 1e-10, 1e-10);
    }

    unsafe { bgw_generalized_bulk_flow_free(handle) };
}

#[test]
fn header_is_accepted_by_c_and_cpp_compilers_when_available() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let header = manifest_dir.join("include/bubble_gw.h");
    let temp_dir = std::env::temp_dir().join(format!("bubble_gw_c_header_{}", std::process::id()));
    fs::create_dir_all(&temp_dir).unwrap();

    let c_file = temp_dir.join("smoke.c");
    fs::write(
        &c_file,
        "#include \"bubble_gw.h\"\nint main(void) { return bgw_version() == 0; }\n",
    )
    .unwrap();
    if let Some(cc) = find_command(&["cc", "gcc", "clang"]) {
        let status = Command::new(cc)
            .arg("-std=c11")
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(header.parent().unwrap())
            .arg(&c_file)
            .status()
            .unwrap();
        assert!(status.success());
    }

    let cpp_file = temp_dir.join("smoke.cpp");
    fs::write(
        &cpp_file,
        "#include \"bubble_gw.h\"\nint main() { return bgw_version() == nullptr; }\n",
    )
    .unwrap();
    if let Some(cxx) = find_command(&["c++", "g++", "clang++"]) {
        let status = Command::new(cxx)
            .arg("-std=c++17")
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(header.parent().unwrap())
            .arg(&cpp_file)
            .status()
            .unwrap();
        assert!(status.success());
    }
}

fn find_command<'a>(candidates: &'a [&'a str]) -> Option<&'a str> {
    candidates.iter().copied().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

use std::error::Error;

use bubble_gw::two_bubbles::gw_calc::{
    GravitationalWaveCalculator as OldGravitationalWaveCalculator,
    InitialFieldStatus as OldInitialFieldStatus,
};
use bubble_gw::two_bubbles::new_gw_calc::{
    ExponentialTimeCutoff,
    GravitationalWaveCalculator as NewGravitationalWaveCalculator,
    InitialFieldStatus as NewInitialFieldStatus,
};
use ndarray::{Array1, Array2, Array3};
use peroxide::numerical::integral::Integral;

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

fn assert_arrays_close(actual: &Array2<f64>, expected: &Array2<f64>, abs_tol: f64, rel_tol: f64) {
    assert_eq!(actual.shape(), expected.shape());

    for ((i, j), &actual_value) in actual.indexed_iter() {
        let expected_value = expected[[i, j]];
        let diff = (actual_value - expected_value).abs();
        let scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        assert!(
            diff <= abs_tol.max(rel_tol * scale),
            "mismatch at ({i}, {j}): actual={actual_value:?}, expected={expected_value:?}, diff={diff:?}"
        );
    }
}

#[test]
fn old_and_new_compute_angular_gw_spectrum_match() -> Result<(), Box<dyn Error>> {
    let (phi1, phi2, z_grid, ds) = synthetic_fields();
    let ratio_t_cut = Some(0.8);
    let ratio_t_0 = Some(0.12);
    let smax = (phi1.shape()[1] - 1) as f64 * ds;

    let mut old_calc = OldGravitationalWaveCalculator::new(
        OldInitialFieldStatus::TwoBubbles,
        phi1.clone(),
        phi2.clone(),
        z_grid.clone(),
        ds,
        ratio_t_cut,
        ratio_t_0,
    )?;
    old_calc.set_num_threads(1)?;
    old_calc.set_integral_params(1e-8, 40)?;

    let time_cutoff = ExponentialTimeCutoff::new(smax, ratio_t_cut, ratio_t_0);
    // let integrand = IntegrandCalculator::new(cutoff);
    let mut new_calc = NewGravitationalWaveCalculator::new(
        NewInitialFieldStatus::TwoBubbles,
        phi1,
        phi2,
        z_grid,
        ds,
        time_cutoff,
    )?;
    new_calc.set_num_threads(1)?;
    new_calc.set_integration_params(Integral::G30K61(1e-8, 40))?;
    // new_calc.set_integration_params("G15K31", 1e-8, 40)?;

    let w_arr = [0.4, 0.9, 1.7];
    let cos_thetak_arr = [-1.0, -0.65, -0.1, 0.45, 1.0];

    let old_spectrum = old_calc.compute_angular_gw_spectrum(w_arr, cos_thetak_arr)?;
    let new_spectrum = new_calc.compute_angular_gw_spectrum(w_arr, cos_thetak_arr)?;

    assert_arrays_close(&new_spectrum, &old_spectrum, 1e-10, 1e-10);
    Ok(())
}

#[test]
fn new_gw_calc_supports_peroxide_integration_methods() -> Result<(), Box<dyn Error>> {
    let (phi1, phi2, z_grid, ds) = synthetic_fields();
    let time_cutoff =
        ExponentialTimeCutoff::new((phi1.shape()[1] - 1) as f64 * ds, Some(0.8), Some(0.12));
    let w_arr = [0.4, 0.9];
    let cos_thetak_arr = [-0.65, 0.45];

    for method in [
        Integral::GaussLegendre(16),
        Integral::NewtonCotes(6),
        Integral::G15K31(1e-7, 30),
    ] {
        let mut calc = NewGravitationalWaveCalculator::new(
            NewInitialFieldStatus::TwoBubbles,
            phi1.clone(),
            phi2.clone(),
            z_grid.clone(),
            ds,
            time_cutoff.clone(),
        )?;
        calc.set_num_threads(1)?;
        calc.set_integration_params(method)?;
        let spectrum = calc.compute_angular_gw_spectrum(w_arr, cos_thetak_arr)?;
        assert_eq!(spectrum.shape(), &[cos_thetak_arr.len(), w_arr.len()]);
        assert!(spectrum.iter().all(|value| value.is_finite()));
    }

    Ok(())
}

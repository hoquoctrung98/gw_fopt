use bubble_gw_rs::utils::integrate_old::Integrate;
use bubble_gw_rs::utils::is_close::IsClose;
use ndarray::{Array, Array1, Array2};
use num::complex::Complex;

fn test_slice_integration<F>(n: usize, integrand: F, expected: &[f64], abs_tol: f64, rel_tol: f64)
where
    F: Fn(f64) -> f64,
{
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| integrand(v)).collect();

    println!("Testing slice integration (n={}):", n);
    let trap_result = (&y as &[f64]).trapezoid(Some(&x), None, None).unwrap();
    let simp_result = (&y as &[f64]).simpson(Some(&x), None, None).unwrap();
    println!(
        "Trapezoid result: {:.6}, Error: {:.6}",
        trap_result,
        (trap_result - expected[0]).abs()
    );
    println!(
        "Simpson result: {:.6}, Error: {:.6}",
        simp_result,
        (simp_result - expected[0]).abs()
    );

    assert!(
        trap_result.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Slice trapezoid result differs: {}",
        trap_result
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );
    assert!(
        simp_result.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Slice Simpson result differs: {}",
        simp_result
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );
}

fn test_array1_integration<F>(n: usize, integrand: F, expected: &[f64], abs_tol: f64, rel_tol: f64)
where
    F: Fn(f64) -> f64,
{
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| integrand(v)).collect();
    let y_arr = Array::from_vec(y);

    println!("\nTesting Array1 integration (n={}):", n);
    let trap_result = y_arr.trapezoid(Some(&x), None, None).unwrap();
    let trap_scalar = trap_result.into_scalar();
    let simp_result = y_arr.simpson(Some(&x), None, None).unwrap();
    let simp_scalar = simp_result.into_scalar();
    println!(
        "Trapezoid result: {:.6}, Error: {:.6}",
        trap_scalar,
        (trap_scalar - expected[0]).abs()
    );
    println!(
        "Simpson result: {:.6}, Error: {:.6}",
        simp_scalar,
        (simp_scalar - expected[0]).abs()
    );

    assert!(
        trap_scalar.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Array1 trapezoid result differs: {}",
        trap_scalar
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );
    assert!(
        simp_scalar.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Array1 Simpson result differs: {}",
        simp_scalar
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );
}

fn test_array2_integration<F>(n: usize, integrand: F, expected: &[f64], abs_tol: f64, rel_tol: f64)
where
    F: Fn(f64) -> f64,
{
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y2 = Array2::from_shape_fn((2, n), |(_, j)| integrand(x[j]));

    println!("\nTesting Array2 integration along axis 1 (n={}):", n);
    let trap_result2 = y2.trapezoid(Some(&x), None, Some(1)).unwrap();
    let simp_result2 = y2.simpson(Some(&x), None, Some(1)).unwrap();
    println!("Trapezoid results: {:?}", trap_result2);
    println!("Trapezoid errors: {:?}", trap_result2.mapv(|v| (v - expected[0]).abs()));
    println!("Simpson results: {:?}", simp_result2);
    println!("Simpson errors: {:?}", simp_result2.mapv(|v| (v - expected[0]).abs()));

    assert!(
        trap_result2.is_close(&expected, abs_tol, rel_tol).is_ok(),
        "Array2 trapezoid results differ: {}",
        trap_result2
            .is_close(&expected, abs_tol, rel_tol)
            .unwrap_err()
    );
    assert!(
        simp_result2.is_close(&expected, abs_tol, rel_tol).is_ok(),
        "Array2 Simpson results differ: {}",
        simp_result2
            .is_close(&expected, abs_tol, rel_tol)
            .unwrap_err()
    );
}

fn test_linear_integration<F>(integrand: F, expected: &[f64], abs_tol: f64, rel_tol: f64)
where
    F: Fn(f64) -> f64,
{
    let x = Array1::range(0., 11., 1.); // [0, 1, 2, ..., 10]
    let y = x.mapv(|v| integrand(v));
    let n = x.len(); // n = 11

    println!(
        "\nTesting slice integration with x = range(0., 11., 1.), y = integrand(x) (n={}):",
        n
    );
    let x_slice = x.as_slice().unwrap();
    let simp_result = y
        .as_slice()
        .unwrap()
        .simpson(Some(x_slice), None, None)
        .unwrap();
    println!(
        "Simpson result: {:.6}, Error: {:.6}",
        simp_result,
        (simp_result - expected[0]).abs()
    );
    assert!(
        simp_result.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Slice Simpson result differs: {}",
        simp_result
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );

    println!(
        "\nTesting Array1 integration with x = range(0., 11., 1.), y = integrand(x) (n={}):",
        n
    );
    let simp_result = y.simpson(Some(x_slice), None, None).unwrap();
    let simp_scalar = simp_result.into_scalar();
    println!(
        "Simpson result: {:.6}, Error: {:.6}",
        simp_scalar,
        (simp_scalar - expected[0]).abs()
    );
    assert!(
        simp_scalar.is_close(&expected[0], abs_tol, rel_tol).is_ok(),
        "Array1 Simpson result differs: {}",
        simp_scalar
            .is_close(&expected[0], abs_tol, rel_tol)
            .unwrap_err()
    );
}

// Test function for complex slice integration
fn test_complex_slice_integration<F>(
    n: usize,
    integrand: F,
    expected: &Complex<f64>,
    abs_tol: f64,
    rel_tol: f64,
) where
    F: Fn(f64) -> Complex<f64>,
{
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<Complex<f64>> = x.iter().map(|&v| integrand(v)).collect();

    println!("\nTesting complex slice integration (n={}):", n);
    let trap_result = (&y as &[Complex<f64>])
        .trapezoid(Some(&x), None, None)
        .unwrap();
    let simp_result = (&y as &[Complex<f64>])
        .simpson(Some(&x), None, None)
        .unwrap();
    println!("Trapezoid result: {} + {}i", trap_result.re, trap_result.im);
    println!("Simpson result: {} + {}i", simp_result.re, simp_result.im);

    assert!(
        trap_result.is_close(expected, abs_tol, rel_tol).is_ok(),
        "Complex slice trapezoid result differs: {}",
        trap_result
            .is_close(expected, abs_tol, rel_tol)
            .unwrap_err()
    );
    assert!(
        simp_result.is_close(expected, abs_tol, rel_tol).is_ok(),
        "Complex slice Simpson result differs: {}",
        simp_result
            .is_close(expected, abs_tol, rel_tol)
            .unwrap_err()
    );
}

// Test function for complex Array1 integration
fn test_complex_array1_integration<F>(
    n: usize,
    integrand: F,
    expected: &Complex<f64>,
    abs_tol: f64,
    rel_tol: f64,
) where
    F: Fn(f64) -> Complex<f64>,
{
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<Complex<f64>> = x.iter().map(|&v| integrand(v)).collect();
    let y_arr = Array::from_vec(y);

    println!("\nTesting complex Array1 integration (n={}):", n);
    let trap_result = y_arr.trapezoid(Some(&x), None, None).unwrap();
    let trap_scalar = trap_result.into_scalar();
    let simp_result = y_arr.simpson(Some(&x), None, None).unwrap();
    let simp_scalar = simp_result.into_scalar();
    println!("Trapezoid result: {} + {}i", trap_scalar.re, trap_scalar.im);
    println!("Simpson result: {} + {}i", simp_scalar.re, simp_scalar.im);

    assert!(
        trap_scalar.is_close(expected, abs_tol, rel_tol).is_ok(),
        "Complex Array1 trapezoid result differs: {}",
        trap_scalar
            .is_close(expected, abs_tol, rel_tol)
            .unwrap_err()
    );
    assert!(
        simp_scalar.is_close(expected, abs_tol, rel_tol).is_ok(),
        "Complex Array1 Simpson result differs: {}",
        simp_scalar
            .is_close(expected, abs_tol, rel_tol)
            .unwrap_err()
    );
}

#[test]
fn test_slice_odd_n() {
    test_slice_integration(1001, |x| x * x, &[1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_slice_even_n() {
    test_slice_integration(1000, |x| x * x, &[1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_array1_odd_n() {
    test_array1_integration(1001, |x| x * x, &[1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_array1_even_n() {
    test_array1_integration(1000, |x| x * x, &[1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_array2_odd_n() {
    test_array2_integration(1001, |x| x * x, &[1.0 / 3.0, 1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_array2_even_n() {
    test_array2_integration(1000, |x| x * x, &[1.0 / 3.0, 1.0 / 3.0], 1e-6, 1e-6);
}

#[test]
fn test_linear() {
    test_linear_integration(|x| x, &[50.0], 1e-6, 1e-6);
}

#[test]
fn test_linear_cubic() {
    test_linear_integration(|x| x * x * x, &[2500.0], 1e-6, 1e-6);
}

#[test]
fn test_complex_slice_odd_n() {
    test_complex_slice_integration(
        1001,
        |x| Complex::new(x, x * x),
        &Complex::new(0.5, 1.0 / 3.0),
        1e-6,
        1e-6,
    );
}

#[test]
fn test_complex_slice_even_n() {
    test_complex_slice_integration(
        1000,
        |x| Complex::new(x, x * x),
        &Complex::new(0.5, 1.0 / 3.0),
        1e-6,
        1e-6,
    );
}

#[test]
fn test_complex_array1_odd_n() {
    test_complex_array1_integration(
        1001,
        |x| Complex::new(x, x * x),
        &Complex::new(0.5, 1.0 / 3.0),
        1e-6,
        1e-6,
    );
}

#[test]
fn test_complex_array1_even_n() {
    test_complex_array1_integration(
        1000,
        |x| Complex::new(x, x * x),
        &Complex::new(0.5, 1.0 / 3.0),
        1e-6,
        1e-6,
    );
}

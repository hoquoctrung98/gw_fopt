use bubble_gw::utils::is_close::IsClose;
use bubble_gw::utils::sample::{SampleError, SampleParams, SampleType};
use ndarray::Array1;

#[test]
fn test_new_invalid_range() {
    let result = SampleParams::new(10.0_f64, 0.0, SampleType::Uniform);
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == 10.0 && stop == 0.0
    ));

    let result = SampleParams::new(5.0_f64, 5.0, SampleType::Linear);
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == 5.0 && stop == 5.0
    ));
}

#[test]
fn test_new_invalid_base_logarithmic() {
    let result = SampleParams::new(1.0_f64, 10.0, SampleType::Logarithmic { base: 0.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidBase { base, sample_type }) if base == 0.0 && sample_type == "Logarithmic"
    ));

    let result = SampleParams::new(1.0_f64, 10.0, SampleType::Logarithmic { base: -1.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidBase { base, sample_type }) if base == -1.0 && sample_type == "Logarithmic"
    ));
}

#[test]
fn test_new_invalid_base_exponential() {
    let result = SampleParams::new(1.0_f64, 10.0, SampleType::Exponential { base: 0.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidBase { base, sample_type }) if base == 0.0 && sample_type == "Exponential"
    ));
}

#[test]
fn test_new_invalid_range_logarithmic() {
    let result = SampleParams::new(0.0_f64, 10.0, SampleType::Logarithmic { base: 10.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == 0.0 && stop == 10.0
    ));

    let result = SampleParams::new(-1.0_f64, 10.0, SampleType::Logarithmic { base: 10.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == -1.0 && stop == 10.0
    ));
}

#[test]
fn test_new_invalid_range_exponential() {
    let result = SampleParams::new(0.0_f64, 2.0, SampleType::Exponential { base: 10.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == 0.0 && stop == 2.0
    ));

    let result = SampleParams::new(-1.0_f64, 2.0, SampleType::Exponential { base: 10.0 });
    assert!(matches!(
        result,
        Err(SampleError::InvalidRange { start, stop }) if start == -1.0 && stop == 2.0
    ));
}

#[test]
fn test_sample_linear() {
    let params = SampleParams::new(0.0_f64, 10.0, SampleType::Linear).unwrap();
    let samples = params.sample(4, 2, 0).unwrap();
    let samples_array = Array1::from_vec(samples);
    let expected = Array1::from_vec(vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    samples_array
        .is_close(&expected, 1e-10, 1e-10)
        .expect("Linear samples differ");
}

#[test]
fn test_sample_uniform() {
    let params = SampleParams::new(0.0_f64, 10.0, SampleType::Uniform).unwrap();
    let samples = params.sample(4, 2, 0).unwrap();
    let samples_array = Array1::from_vec(samples);
    let expected = Array1::from_vec(vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    samples_array
        .is_close(&expected, 1e-10, 1e-10)
        .expect("Uniform samples differ");
}

#[test]
fn test_sample_logarithmic() {
    let params = SampleParams::new(1.0_f64, 100.0, SampleType::Logarithmic { base: 10.0 }).unwrap();
    let samples = params.sample(4, 2, 0).unwrap();
    let samples_array = Array1::from_vec(samples);
    let expected = Array1::from_vec(vec![1.0, 3.16227766017, 10.0, 31.6227766017, 100.0]);
    samples_array
        .is_close(&expected, 1e-10, 1e-10)
        .expect("Logarithmic samples differ");
}

#[test]
fn test_sample_exponential() {
    let params = SampleParams::new(0.001_f64, 2.0, SampleType::Exponential { base: 10.0 }).unwrap();
    let samples = params.sample(4, 2, 0).unwrap();
    let samples_array = Array1::from_vec(samples);
    println!("Actual exponential samples: {:?}", samples_array); // Debug output
    let expected = Array1::from_vec(vec![
        1.0023052380778996,
        3.1677434383924457,
        10.011519555381687,
        31.640985375587913,
        100.0,
    ]);
    samples_array
        .is_close(&expected, 1e-8, 1e-8)
        .expect("Exponential samples differ");
}

#[test]
fn test_sample_distribution() {
    let dist = |x: f64| x * 2.0;
    let dist_inv = |x: f64| x / 2.0;
    let params =
        SampleParams::new(0.0_f64, 10.0, SampleType::Distribution { dist, dist_inv }).unwrap();
    let samples = params.sample(4, 2, 0).unwrap();
    let samples_array = Array1::from_vec(samples);
    let expected = Array1::from_vec(vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    samples_array
        .is_close(&expected, 1e-10, 1e-10)
        .expect("Distribution samples differ");
}

#[test]
fn test_sample_invalid_n_grid() {
    let params = SampleParams::new(0.0_f64, 10.0, SampleType::Uniform).unwrap();
    let result = params.sample(4, 1, 1);
    assert!(matches!(
        result,
        Err(SampleError::InvalidParameter { param, value }) if param == "n_grid" && value == 1.0
    ));
}

#[test]
fn test_sample_arr_linear() {
    let params = SampleParams::new(0.0_f64, 12.0, SampleType::Linear).unwrap();
    let samples_arr = params.sample_arr(3, 2, 2).unwrap();
    let samples_arr = Array1::from_vec(samples_arr);
    let expected = Array1::from_vec(vec![
        0.0, 4.0, 8.0, 12.0, // n_iter=0
        2.0, 6.0, 10.0, // n_iter=1
        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, // n_iter=2
    ]);
    samples_arr
        .is_close(&expected, 1e-10, 1e-10)
        .expect("Linear sample_arr samples differ");
}

#[test]
fn test_sample_arr_logarithmic() {
    let params = SampleParams::new(1.0, 100.0, SampleType::Logarithmic { base: 10.0 }).unwrap();
    let samples = params.sample_arr(4, 2, 2).unwrap();
    let samples_array = Array1::from_vec(samples);
    let expected = Array1::from_vec(vec![
        1.,
        3.16227766,
        10.,
        31.6227766,
        100., // n_iter=0
        1.77827941,
        5.62341325,
        17.7827941,
        56.23413252, // n_iter=1
        1.33352143,
        2.37137371,
        4.21696503,
        7.49894209,
        13.33521432,
        23.71373706,
        42.16965034,
        74.98942093, // n_iter=2
    ]);
    samples_array
        .is_close(&expected, 1e-8, 1e-8)
        .expect("Logarithmic sample_arr samples differ");
}

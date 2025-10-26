use bubble_gw_rs::utils::sample::{SampleParams, SampleType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*; // Import from utils.rs

/// Performs sampling based on the specified parameters.
///
/// Args:
///     start (float): The start of the sampling range.
///     stop (float): The end of the sampling range.
///     nsample (int): Number of samples to generate.
///     ngrid (int): Number of grid points for iterative sampling.
///     niter (int): Number of iterations for sampling.
///     sample_type (str): Type of sampling ("uniform", "linear", "log", "exp").
///     base (float, optional): Base for logarithmic or exponential sampling. Defaults to 10.0.
///
/// Returns:
///     List[float]: The generated samples for a single iteration.
#[pyfunction]
#[pyo3(signature = (start, stop, nsample, ngrid, niter, sample_type, base=10.0))]
pub fn sample(
    start: f64,
    stop: f64,
    nsample: usize,
    ngrid: usize,
    niter: usize,
    sample_type: String,
    base: f64,
) -> PyResult<Vec<f64>> {
    // Map Python sample_type string to Rust SampleType
    let rust_sample_type = match sample_type.to_lowercase().as_str() {
        "uniform" => SampleType::Uniform,
        "linear" => SampleType::Linear,
        "log" => {
            if base <= 0.0 {
                return Err(PyValueError::new_err(
                    "`base` must be greater than 0 for logarithmic sampling",
                ));
            }
            SampleType::Logarithmic { base }
        }
        "exp" => {
            if base <= 0.0 {
                return Err(PyValueError::new_err(
                    "`base` must be greater than 0 for exponential sampling",
                ));
            }
            SampleType::Exponential { base }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid sample_type: {}",
                sample_type
            )));
        }
    };

    // Create SampleParams
    let params =
        SampleParams::new(start, stop, rust_sample_type).map_err(|e| PyValueError::new_err(e))?;

    // Perform sampling
    let samples = params.sample(nsample, ngrid, niter);
    Ok(samples)
}

/// Performs sampling over multiple iterations based on the specified parameters.
///
/// Args:
///     start (float): The start of the sampling range.
///     stop (float): The end of the sampling range.
///     nsample (int): Number of samples to generate per iteration.
///     ngrid (int): Number of grid points for iterative sampling.
///     niter (int): Number of iterations for sampling.
///     sample_type (str): Type of sampling ("uniform", "linear", "log", "exp").
///     base (float, optional): Base for logarithmic or exponential sampling. Defaults to 10.0.
///
/// Returns:
///     List[float]: The generated samples across all iterations.
#[pyfunction]
#[pyo3(signature = (start, stop, nsample, ngrid, niter, sample_type, base=10.0))]
pub fn sample_arr(
    start: f64,
    stop: f64,
    nsample: usize,
    ngrid: usize,
    niter: usize,
    sample_type: String,
    base: f64,
) -> PyResult<Vec<f64>> {
    // Map Python sample_type string to Rust SampleType
    let rust_sample_type = match sample_type.to_lowercase().as_str() {
        "uniform" => SampleType::Uniform,
        "linear" => SampleType::Linear,
        "log" => {
            if base <= 0.0 {
                return Err(PyValueError::new_err(
                    "`base` must be greater than 0 for logarithmic sampling",
                ));
            }
            SampleType::Logarithmic { base }
        }
        "exp" => {
            if base <= 0.0 {
                return Err(PyValueError::new_err(
                    "`base` must be greater than 0 for exponential sampling",
                ));
            }
            SampleType::Exponential { base }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid sample_type: {}",
                sample_type
            )));
        }
    };

    // Create SampleParams
    let params =
        SampleParams::new(start, stop, rust_sample_type).map_err(|e| PyValueError::new_err(e))?;

    // Perform sampling over multiple iterations
    let samples = params.sample_arr(nsample, ngrid, niter);
    Ok(samples)
}

/// Python module definition.
#[pymodule]
fn utils_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_arr, m)?)?;
    Ok(())
}

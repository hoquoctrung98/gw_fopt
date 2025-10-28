// use bubble_gw_rs::utils::sample::{SampleParams, SampleType};
// use pyo3::exceptions::PyValueError;
// use pyo3::prelude::*; // Import from utils.rs
//
// /// Performs sampling based on the specified parameters.
// ///
// /// Args:
// ///     start (float): The start of the sampling range.
// ///     stop (float): The end of the sampling range.
// ///     nsample (int): Number of samples to generate.
// ///     ngrid (int): Number of grid points for iterative sampling.
// ///     niter (int): Number of iterations for sampling.
// ///     sample_type (str): Type of sampling ("uniform", "linear", "log", "exp").
// ///     base (float, optional): Base for logarithmic or exponential sampling. Defaults to 10.0.
// ///
// /// Returns:
// ///     List[float]: The generated samples for a single iteration.
// #[pyfunction]
// #[pyo3(signature = (start, stop, nsample, ngrid, niter, sample_type, base=10.0))]
// pub fn sample(
//     start: f64,
//     stop: f64,
//     nsample: usize,
//     ngrid: usize,
//     niter: usize,
//     sample_type: String,
//     base: f64,
// ) -> PyResult<Vec<f64>> {
//     // Map Python sample_type string to Rust SampleType
//     let rust_sample_type = match sample_type.to_lowercase().as_str() {
//         "uniform" => SampleType::Uniform,
//         "linear" => SampleType::Linear,
//         "log" => {
//             if base <= 0.0 {
//                 return Err(PyValueError::new_err(
//                     "`base` must be greater than 0 for logarithmic sampling",
//                 ));
//             }
//             SampleType::Logarithmic { base }
//         }
//         "exp" => {
//             if base <= 0.0 {
//                 return Err(PyValueError::new_err(
//                     "`base` must be greater than 0 for exponential sampling",
//                 ));
//             }
//             SampleType::Exponential { base }
//         }
//         _ => {
//             return Err(PyValueError::new_err(format!(
//                 "Invalid sample_type: {}",
//                 sample_type
//             )));
//         }
//     };
//
//     // Create SampleParams
//     let params =
//         SampleParams::new(start, stop, rust_sample_type).map_err(|e| PyValueError::new_err(e))?;
//
//     // Perform sampling
//     let samples = params.sample(nsample, ngrid, niter);
//     Ok(samples)
// }
//
// /// Performs sampling over multiple iterations based on the specified parameters.
// ///
// /// Args:
// ///     start (float): The start of the sampling range.
// ///     stop (float): The end of the sampling range.
// ///     nsample (int): Number of samples to generate per iteration.
// ///     ngrid (int): Number of grid points for iterative sampling.
// ///     niter (int): Number of iterations for sampling.
// ///     sample_type (str): Type of sampling ("uniform", "linear", "log", "exp").
// ///     base (float, optional): Base for logarithmic or exponential sampling. Defaults to 10.0.
// ///
// /// Returns:
// ///     List[float]: The generated samples across all iterations.
// #[pyfunction]
// #[pyo3(signature = (start, stop, nsample, ngrid, niter, sample_type, base=10.0))]
// pub fn sample_arr(
//     start: f64,
//     stop: f64,
//     nsample: usize,
//     ngrid: usize,
//     niter: usize,
//     sample_type: String,
//     base: f64,
// ) -> PyResult<Vec<f64>> {
//     // Map Python sample_type string to Rust SampleType
//     let rust_sample_type = match sample_type.to_lowercase().as_str() {
//         "uniform" => SampleType::Uniform,
//         "linear" => SampleType::Linear,
//         "log" => {
//             if base <= 0.0 {
//                 return Err(PyValueError::new_err(
//                     "`base` must be greater than 0 for logarithmic sampling",
//                 ));
//             }
//             SampleType::Logarithmic { base }
//         }
//         "exp" => {
//             if base <= 0.0 {
//                 return Err(PyValueError::new_err(
//                     "`base` must be greater than 0 for exponential sampling",
//                 ));
//             }
//             SampleType::Exponential { base }
//         }
//         _ => {
//             return Err(PyValueError::new_err(format!(
//                 "Invalid sample_type: {}",
//                 sample_type
//             )));
//         }
//     };
//
//     // Create SampleParams
//     let params =
//         SampleParams::new(start, stop, rust_sample_type).map_err(|e| PyValueError::new_err(e))?;
//
//     // Perform sampling over multiple iterations
//     let samples = params.sample_arr(nsample, ngrid, niter);
//     Ok(samples)
// }
//
// /// Python module definition.
// #[pymodule]
// fn utils_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sample, m)?)?;
//     m.add_function(wrap_pyfunction!(sample_arr, m)?)?;
//     Ok(())
// }

use bubble_gw_rs::utils::sample::{SampleError, SampleParams, SampleType};
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Performs sampling based on the specified parameters.
///
/// This function generates a sequence of samples between `start` and `stop` using the specified
/// sampling type. The samples are returned as a NumPy array for efficient numerical computations.
///
/// # Arguments
/// * `start` (float): The start of the sampling range (must be less than `stop`).
/// * `stop` (float): The end of the sampling range.
/// * `n_sample` (int): Number of intervals (produces `n_sample + 1` points for `n_iter = 0`).
/// * `n_grid` (int): Number of grid points for iterative sampling (must be >= 2 for `n_iter > 0`).
/// * `n_iter` (int): Number of iterations for refined sampling (non-negative).
/// * `sample_type` (str): Type of sampling ("uniform", "linear", "log", "exp").
/// * `base` (float, optional): Base for logarithmic or exponential sampling (must be positive). Defaults to 10.0.
///
/// # Returns
/// A NumPy array (`np.ndarray`) containing the generated samples for a single iteration.
///
/// # Raises
/// * `ValueError`: If `start >= stop`, `base <= 0` for logarithmic/exponential sampling,
///   `n_sample == 0`, `n_grid < 2` for `n_iter > 0`, or if `sample_type` is invalid.
///
/// # Examples
/// ```python
/// import utils_bindings
/// samples = utils_bindings.sample(1.0, 10.0, 4, 2, 0, "linear")
/// # Returns: np.array([1.0, 3.25, 5.5, 7.75, 10.0])
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, n_sample, n_grid, n_iter, sample_type, base=10.0))]
pub fn sample(
    py: Python,
    start: f64,
    stop: f64,
    n_sample: usize,
    n_grid: usize,
    n_iter: usize,
    sample_type: String,
    base: f64,
) -> PyResult<Py<PyArray1<f64>>> {
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
                "Invalid sample_type: {}. Expected 'uniform', 'linear', 'log', or 'exp'.",
                sample_type
            )));
        }
    };

    // Create SampleParams and map errors
    let params = SampleParams::new(start, stop, rust_sample_type).map_err(|e| match e {
        SampleError::InvalidRange { start, stop } => PyValueError::new_err(format!(
            "`start` ({}) must be less than `stop` ({})",
            start, stop
        )),
        SampleError::InvalidBase { base, sample_type } => PyValueError::new_err(format!(
            "`base` ({}) must be greater than 0 for {} sampling",
            base, sample_type
        )),
        SampleError::InvalidParameter { param, value } => {
            PyValueError::new_err(format!("Invalid parameter {}: {}", param, value))
        }
        SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
    })?;

    // Perform sampling and convert to NumPy array
    let samples = params
        .sample(n_sample, n_grid, n_iter)
        .map_err(|e| match e {
            SampleError::InvalidParameter { param, value } => {
                PyValueError::new_err(format!("Invalid parameter {}: {}", param, value))
            }
            SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
            _ => PyValueError::new_err("Unexpected error during sampling"),
        })?;
    Ok(samples.to_pyarray(py).into())
}

/// Performs sampling over multiple iterations based on the specified parameters.
///
/// This function generates samples for iterations from 0 to `n_iter` and concatenates them
/// into a single NumPy array. Useful for generating samples with increasing granularity.
///
/// # Arguments
/// * `start` (float): The start of the sampling range (must be less than `stop`).
/// * `stop` (float): The end of the sampling range.
/// * `n_sample` (int): Number of intervals per iteration.
/// * `n_grid` (int): Number of grid points for iterative sampling (must be >= 2 for `n_iter > 0`).
/// * `n_iter` (int): Maximum number of iterations (non-negative).
/// * `sample_type` (str): Type of sampling ("uniform", "linear", "log", "exp").
/// * `base` (float, optional): Base for logarithmic or exponential sampling (must be positive). Defaults to 10.0.
///
/// # Returns
/// A NumPy array (`np.ndarray`) containing the concatenated samples across all iterations.
///
/// # Raises
/// * `ValueError`: If `start >= stop`, `base <= 0` for logarithmic/exponential sampling,
///   `n_sample == 0`, `n_grid < 2` for `n_iter > 0`, or if `sample_type` is invalid.
///
/// # Examples
/// ```python
/// import utils_bindings
/// samples = utils_bindings.sample_arr(1.0, 10.0, 4, 2, 1, "uniform")
/// # Returns: np.array([...]) # Concatenated samples for n_iter=0 and n_iter=1
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, n_sample, n_grid, n_iter, sample_type, base=10.0))]
pub fn sample_arr(
    py: Python,
    start: f64,
    stop: f64,
    n_sample: usize,
    n_grid: usize,
    n_iter: usize,
    sample_type: String,
    base: f64,
) -> PyResult<Py<PyArray1<f64>>> {
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
                "Invalid sample_type: {}. Expected 'uniform', 'linear', 'log', or 'exp'.",
                sample_type
            )));
        }
    };

    // Create SampleParams and map errors
    let params = SampleParams::new(start, stop, rust_sample_type).map_err(|e| match e {
        SampleError::InvalidRange { start, stop } => PyValueError::new_err(format!(
            "`start` ({}) must be less than `stop` ({})",
            start, stop
        )),
        SampleError::InvalidBase { base, sample_type } => PyValueError::new_err(format!(
            "`base` ({}) must be greater than 0 for {} sampling",
            base, sample_type
        )),
        SampleError::InvalidParameter { param, value } => {
            PyValueError::new_err(format!("Invalid parameter {}: {}", param, value))
        }
        SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
    })?;

    // Perform sampling over multiple iterations and convert to NumPy array
    let samples = params
        .sample_arr(n_sample, n_grid, n_iter)
        .map_err(|e| match e {
            SampleError::InvalidParameter { param, value } => {
                PyValueError::new_err(format!("Invalid parameter {}: {}", param, value))
            }
            SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
            _ => PyValueError::new_err("Unexpected error during sampling"),
        })?;
    Ok(samples.to_pyarray(py).into())
}

/// Python module definition.
#[pymodule]
fn utils_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_arr, m)?)?;
    Ok(())
}

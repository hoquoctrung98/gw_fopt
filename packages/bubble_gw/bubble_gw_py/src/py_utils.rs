use bubble_gw_rs::utils::sample::{SampleError, SampleParams, SampleType};
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Performs sampling based on the specified parameters.
///
/// This function generates a sequence of samples between `start` and `stop` using the specified
/// sampling type. The samples are returned as a `NumPy` array for efficient numerical computations.
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
/// A `NumPy` array (`np.ndarray`) containing the generated samples.
///
/// For `n_iter == 0`: includes endpoints, linearly spaced in transformed space.
/// For `n_iter > 0`: returns **only the new interior points** introduced at this refinement level
/// (i.e., excludes points from coarser grids).
///
/// Use `sample_arr` to get the full concatenated hierarchical grid.
///
/// # Examples
/// ```python
/// sample(0, 5, 5, 2, 0, "uniform")   # → [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
/// sample(0, 5, 5, 2, 1, "uniform")   # → [0.5, 1.5, 2.5, 3.5, 4.5]           # only new points
/// sample(0, 5, 5, 2, 2, "uniform")   # → [0.25, 0.75, ..., 4.75]           # only newest level
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
                "Invalid sample_type: {sample_type}. Expected 'uniform', 'linear', 'log', or 'exp'."
            )));
        }
    };

    // Create SampleParams and map errors
    let params = SampleParams::new(start, stop, rust_sample_type).map_err(|e| match e {
        SampleError::InvalidRange { start, stop } => {
            PyValueError::new_err(format!("`start` ({start}) must be less than `stop` ({stop})"))
        }
        SampleError::InvalidBase { base, sample_type } => PyValueError::new_err(format!(
            "`base` ({base}) must be greater than 0 for {sample_type} sampling"
        )),
        SampleError::InvalidParameter { param, value } => {
            PyValueError::new_err(format!("Invalid parameter {param}: {value}"))
        }
        SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
    })?;

    // Perform sampling and convert to NumPy array
    let samples = params
        .sample(n_sample, n_grid, n_iter)
        .map_err(|e| match e {
            SampleError::InvalidParameter { param, value } => {
                PyValueError::new_err(format!("Invalid parameter {param}: {value}"))
            }
            SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
            _ => PyValueError::new_err("Unexpected error during sampling"),
        })?;
    Ok(samples.to_pyarray(py).into())
}

/// Performs sampling over multiple iterations based on the specified parameters.
///
/// This function generates samples for iterations from 0 to `n_iter` and concatenates them
/// into a single `NumPy` array. Useful for generating samples with increasing granularity.
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
/// Concatenates samples from `n_iter = 0..=n_iter`, producing a full nested dyadic grid
/// with no duplicate points. Ideal for hierarchical training or multi-resolution analysis.
/// # Raises
/// * `ValueError`: If `start >= stop`, `base <= 0` for logarithmic/exponential sampling,
///   `n_sample == 0`, `n_grid < 2` for `n_iter > 0`, or if `sample_type` is invalid.
///
/// # Examples
/// ```python
/// from bubble_gw.utils import sample_arr
/// samples = sample_arr(1.0, 10.0, 4, 2, 1, "uniform")
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
    sample_type: &str,
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
        SampleError::InvalidRange { start, stop } => {
            PyValueError::new_err(format!("`start` ({start}) must be less than `stop` ({stop})"))
        }
        SampleError::InvalidBase { base, sample_type } => PyValueError::new_err(format!(
            "`base` ({base}) must be greater than 0 for {sample_type} sampling",
        )),
        SampleError::InvalidParameter { param, value } => {
            PyValueError::new_err(format!("Invalid parameter {param}: {value}"))
        }
        SampleError::ConversionError => PyValueError::new_err("Type conversion error"),
    })?;

    // Perform sampling over multiple iterations and convert to NumPy array
    let samples = params
        .sample_arr(n_sample, n_grid, n_iter)
        .map_err(|e| match e {
            SampleError::InvalidParameter { param, value } => {
                PyValueError::new_err(format!("Invalid parameter {param}: {value}"))
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

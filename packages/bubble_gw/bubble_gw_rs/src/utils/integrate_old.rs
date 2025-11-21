use ndarray::{ArrayBase, Axis, Data, Dimension, OwnedRepr, RemoveAxis};
use num_traits::Num;
use std::fmt;

/// Errors that can occur during numerical integration.
#[derive(Debug)]
pub enum IntegrationError {
    /// Input array has fewer than 2 points, which is insufficient for integration.
    InsufficientPoints { len: usize },
    /// The length of the x-coordinates does not match the length of the y-values.
    MismatchedLengths { x_len: usize, y_len: usize },
    /// The specified axis is out of bounds for the array's dimensions.
    InvalidAxis { axis: usize, ndim: usize },
    /// Failed to obtain a contiguous slice from the array.
    NonContiguousSlice,
    /// Failed to create the output array with the specified shape.
    OutputArrayCreationFailed,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::InsufficientPoints { len } => {
                write!(f, "At least 2 points required for integration, got {}", len)
            }
            IntegrationError::MismatchedLengths { x_len, y_len } => {
                write!(
                    f,
                    "x and y must have the same length, got x: {}, y: {}",
                    x_len, y_len
                )
            }
            IntegrationError::InvalidAxis { axis, ndim } => {
                write!(
                    f,
                    "Axis {} out of bounds for array with {} dimensions",
                    axis, ndim
                )
            }
            IntegrationError::NonContiguousSlice => {
                write!(f, "Cannot get contiguous slice from array")
            }
            IntegrationError::OutputArrayCreationFailed => {
                write!(f, "Failed to create output array")
            }
        }
    }
}

impl std::error::Error for IntegrationError {}

/// A trait for performing numerical integration on data structures.
///
/// This trait provides methods for numerical integration using the trapezoid rule and
/// Simpson's rules. It is implemented for slices (`&[T]`) and `ndarray::ArrayBase` types,
/// supporting both evenly spaced (`dx`) and unevenly spaced (`x`) data points.
///
/// # Type Parameters
/// - `X`: The type of the x-coordinates, which must support numeric operations and ordering.
///
/// # Associated Types
/// - `Output`: The type of the integration result, which depends on the implementing type
///   (e.g., a scalar for slices, an array for `ArrayBase`).
pub trait Integrate<X: PartialOrd> {
    /// The output type of the integration methods.
    type Output;

    /// Performs integration using the trapezoid rule.
    ///
    /// The trapezoid rule approximates the integral by summing the areas of trapezoids
    /// formed by connecting consecutive data points:
    /// \[
    /// \int_a^b f(x) \, dx \approx \sum_{i=0}^{n-2} \frac{f(x_i) + f(x_{i+1})}{2} \cdot (x_{i+1} - x_i)
    /// \]
    /// For evenly spaced points with step size `dx`, the formula simplifies to:
    /// \[
    /// \int_a^b f(x) \, dx \approx \sum_{i=0}^{n-2} \frac{f(x_i) + f(x_{i+1})}{2} \cdot dx
    /// \]
    ///
    /// # Arguments
    /// - `x`: Optional slice of x-coordinates for unevenly spaced data. Must have the same length as the input data.
    /// - `dx`: Optional step size for evenly spaced data. Defaults to 1.0 if `x` is not provided.
    /// - `axis`: Optional axis along which to integrate (for `ndarray`). Defaults to the last axis.
    ///
    /// # Errors
    /// Returns an `IntegrationError` if:
    /// - The input has fewer than 2 points.
    /// - The lengths of `x` and the input data do not match.
    /// - The specified axis is out of bounds (for `ndarray`).
    /// - A contiguous slice cannot be obtained (for `ndarray`).
    /// - The output array cannot be created (for `ndarray`).
    fn trapezoid<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;

    /// Performs integration using Simpson's rules.
    ///
    /// The implementation uses:
    /// - Trapezoid rule for `n = 2` points (insufficient for Simpson's rules).
    /// - Simpson's 3/8 rule for `n = 3` or the first 4 points when `n ≥ 4, even`:
    ///   \[
    ///   \int_{x_0}^{x_3} f(x) \, dx \approx \frac{3h}{8} \left[ f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3) \right]
    ///   \]
    ///   where \( h = (x_3 - x_0)/3 \).
    /// - Simpson's 1/3 rule for `n ≥ 3, odd` or remaining points when `n > 4, even`:
    ///   \[
    ///   \int_{x_0}^{x_{n-1}} f(x) \, dx \approx \frac{h}{3} \left[ f(x_0) + 4 \sum_{\text{odd } i} f(x_i) + 2 \sum_{\text{even } i} f(x_i) + f(x_{n-1}) \right]
    ///   \]
    ///   where \( h = (x_{n-1} - x_0)/(n-1) \).
    ///
    /// # Arguments
    /// - `x`: Optional slice of x-coordinates for unevenly spaced data. Must have the same length as the input data.
    /// - `dx`: Optional step size for evenly spaced data. Defaults to 1.0 if `x` is not provided.
    /// - `axis`: Optional axis along which to integrate (for `ndarray`). Defaults to the last axis.
    ///
    /// # Errors
    /// Returns an `IntegrationError` if:
    /// - The input has fewer than 2 points.
    /// - The lengths of `x` and the input data do not match.
    /// - The specified axis is out of bounds (for `ndarray`).
    /// - A contiguous slice cannot be obtained (for `ndarray`).
    /// - The output array cannot be created (for `ndarray`).
    fn simpson<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;
}

/// Implementation of the `Integrate` trait for slices.
impl<'a, T, X> Integrate<X> for &'a [T]
where
    T: Num + Copy + From<X> + From<f64>,
    X: Num + Copy + PartialOrd + From<f64>,
{
    type Output = T;

    fn trapezoid<'b>(
        &self,
        x: Option<&'b [X]>,
        dx: Option<X>,
        _axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        let n = self.len();
        if n < 2 {
            return Err(IntegrationError::InsufficientPoints { len: n });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: x_slice.len(),
                    y_len: n,
                });
            }
            let result = (0..n - 1)
                .map(|i| {
                    (self[i] + self[i + 1]) * T::from(0.5) * T::from(x_slice[i + 1] - x_slice[i])
                })
                .fold(T::zero(), |acc, val| acc + val);
            Ok(result)
        } else {
            let dx = dx.unwrap_or(X::from(1.0f64));
            let result = (0..n - 1)
                .map(|i| (self[i] + self[i + 1]) * T::from(0.5) * T::from(dx))
                .fold(T::zero(), |acc, val| acc + val);
            Ok(result)
        }
    }

    fn simpson<'b>(
        &self,
        x: Option<&'b [X]>,
        dx: Option<X>,
        _axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        let n = self.len();
        if n < 2 {
            return Err(IntegrationError::InsufficientPoints { len: n });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: x_slice.len(),
                    y_len: n,
                });
            }
            let h = (x_slice[n - 1] - x_slice[0]) / X::from((n - 1) as f64);
            let result = if n == 2 {
                (self[0] + self[1]) * T::from(0.5) * T::from(h)
            } else if n % 2 == 1 {
                let init = self[0] + self[n - 1];
                let sum = (1..n - 1).fold(init, |acc, i| {
                    acc + if i % 2 == 0 {
                        T::from(2.0) * self[i]
                    } else {
                        T::from(4.0) * self[i]
                    }
                });
                sum * T::from(h) / T::from(3.0)
            } else if n >= 4 {
                let mut result = T::from(3.0) * T::from(h) / T::from(8.0)
                    * (self[0] + T::from(3.0) * self[1] + T::from(3.0) * self[2] + self[3]);
                if n > 4 {
                    let sub_result = (4..n - 1).fold(self[3] + self[n - 1], |acc, i| {
                        acc + if (i - 3) % 2 == 0 {
                            T::from(2.0) * self[i]
                        } else {
                            T::from(4.0) * self[i]
                        }
                    }) * T::from(h)
                        / T::from(3.0);
                    result = result + sub_result;
                }
                result
            } else {
                T::from(3.0) * T::from(h) / T::from(8.0)
                    * (self[0] + T::from(3.0) * self[1] + T::from(3.0) * self[2])
            };
            Ok(result)
        } else {
            let dx = dx.unwrap_or(X::from(1.0f64));
            let h = dx;
            let result = if n == 2 {
                (self[0] + self[1]) * T::from(0.5) * T::from(h)
            } else if n % 2 == 1 {
                let init = self[0] + self[n - 1];
                let sum = (1..n - 1).fold(init, |acc, i| {
                    acc + if i % 2 == 0 {
                        T::from(2.0) * self[i]
                    } else {
                        T::from(4.0) * self[i]
                    }
                });
                sum * T::from(h) / T::from(3.0)
            } else if n >= 4 {
                let mut result = T::from(3.0) * T::from(h) / T::from(8.0)
                    * (self[0] + T::from(3.0) * self[1] + T::from(3.0) * self[2] + self[3]);
                if n > 4 {
                    let sub_result = (4..n - 1).fold(self[3] + self[n - 1], |acc, i| {
                        acc + if (i - 3) % 2 == 0 {
                            T::from(2.0) * self[i]
                        } else {
                            T::from(4.0) * self[i]
                        }
                    }) * T::from(h)
                        / T::from(3.0);
                    result = result + sub_result;
                }
                result
            } else {
                T::from(3.0) * T::from(h) / T::from(8.0)
                    * (self[0] + T::from(3.0) * self[1] + T::from(3.0) * self[2])
            };
            Ok(result)
        }
    }
}

/// Implementation of the `Integrate` trait for `ndarray::ArrayBase`.
impl<S, A, X, D> Integrate<X> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    A: Num + Copy + From<f64> + From<X>,
    X: Num + Copy + PartialOrd + From<f64>,
    D: Dimension + RemoveAxis,
{
    type Output = ArrayBase<OwnedRepr<A>, D::Smaller>;

    fn trapezoid<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        let axis = axis.unwrap_or(self.ndim() as isize - 1) as usize;
        if axis >= self.ndim() {
            return Err(IntegrationError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let n = self.shape()[axis];
        if n < 2 {
            return Err(IntegrationError::InsufficientPoints { len: n });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: x_slice.len(),
                    y_len: n,
                });
            }
        }

        let output_shape = self.raw_dim().remove_axis(Axis(axis));
        let output_len = output_shape.size();
        let mut results = Vec::with_capacity(output_len);

        if self.ndim() == 1 {
            let y = self
                .as_slice()
                .ok_or(IntegrationError::NonContiguousSlice)?;
            if y.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: y.len(),
                    y_len: n,
                });
            }
            let slice_result = if let Some(x_slice) = x {
                (0..n - 1)
                    .map(|i| {
                        (y[i] + y[i + 1]) * A::from(0.5) * A::from(x_slice[i + 1] - x_slice[i])
                    })
                    .fold(A::zero(), |acc, val| acc + val)
            } else {
                let dx = dx.unwrap_or(X::from(1.0f64));
                (0..n - 1)
                    .map(|i| (y[i] + y[i + 1]) * A::from(0.5) * A::from(dx))
                    .fold(A::zero(), |acc, val| acc + val)
            };
            results.push(slice_result);
        } else {
            let iter_axis = if axis == self.ndim() - 1 { 0 } else { axis };
            for slice in self.axis_iter(Axis(iter_axis)) {
                let owned = slice.to_owned();
                let y = owned
                    .as_slice()
                    .ok_or(IntegrationError::NonContiguousSlice)?;
                if y.len() != n {
                    return Err(IntegrationError::MismatchedLengths {
                        x_len: y.len(),
                        y_len: n,
                    });
                }
                let slice_result = if let Some(x_slice) = x {
                    (0..n - 1)
                        .map(|i| {
                            (y[i] + y[i + 1]) * A::from(0.5) * A::from(x_slice[i + 1] - x_slice[i])
                        })
                        .fold(A::zero(), |acc, val| acc + val)
                } else {
                    let dx = dx.unwrap_or(X::from(1.0f64));
                    (0..n - 1)
                        .map(|i| (y[i] + y[i + 1]) * A::from(0.5) * A::from(dx))
                        .fold(A::zero(), |acc, val| acc + val)
                };
                results.push(slice_result);
            }
        }

        let output: ArrayBase<OwnedRepr<A>, D::Smaller> =
            ArrayBase::from_shape_vec(output_shape, results)
                .map_err(|_| IntegrationError::OutputArrayCreationFailed)?;
        Ok(output)
    }

    fn simpson<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        let axis = axis.unwrap_or(self.ndim() as isize - 1) as usize;
        if axis >= self.ndim() {
            return Err(IntegrationError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let n = self.shape()[axis];
        if n < 2 {
            return Err(IntegrationError::InsufficientPoints { len: n });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: x_slice.len(),
                    y_len: n,
                });
            }
        }

        let output_shape = self.raw_dim().remove_axis(Axis(axis));
        let output_len = output_shape.size();
        let mut results = Vec::with_capacity(output_len);

        if self.ndim() == 1 {
            let y = self
                .as_slice()
                .ok_or(IntegrationError::NonContiguousSlice)?;
            if y.len() != n {
                return Err(IntegrationError::MismatchedLengths {
                    x_len: y.len(),
                    y_len: n,
                });
            }
            let slice_result = if let Some(x_slice) = x {
                let h = (x_slice[n - 1] - x_slice[0]) / X::from((n - 1) as f64);
                if n == 2 {
                    (y[0] + y[1]) * A::from(0.5) * A::from(h)
                } else if n % 2 == 1 {
                    let init = y[0] + y[n - 1];
                    let sum = (1..n - 1).fold(init, |acc, i| {
                        acc + if i % 2 == 0 {
                            A::from(2.0) * y[i]
                        } else {
                            A::from(4.0) * y[i]
                        }
                    });
                    sum * A::from(h) / A::from(3.0)
                } else if n >= 4 {
                    let mut result = A::from(3.0) * A::from(h) / A::from(8.0)
                        * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2] + y[3]);
                    if n > 4 {
                        let sub_result = (4..n - 1).fold(y[3] + y[n - 1], |acc, i| {
                            acc + if (i - 3) % 2 == 0 {
                                A::from(2.0) * y[i]
                            } else {
                                A::from(4.0) * y[i]
                            }
                        }) * A::from(h)
                            / A::from(3.0);
                        result = result + sub_result;
                    }
                    result
                } else {
                    A::from(3.0) * A::from(h) / A::from(8.0)
                        * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2])
                }
            } else {
                let dx = dx.unwrap_or(X::from(1.0f64));
                let h = dx;
                if n == 2 {
                    (y[0] + y[1]) * A::from(0.5) * A::from(h)
                } else if n % 2 == 1 {
                    let init = y[0] + y[n - 1];
                    let sum = (1..n - 1).fold(init, |acc, i| {
                        acc + if i % 2 == 0 {
                            A::from(2.0) * y[i]
                        } else {
                            A::from(4.0) * y[i]
                        }
                    });
                    sum * A::from(h) / A::from(3.0)
                } else if n >= 4 {
                    let mut result = A::from(3.0) * A::from(h) / A::from(8.0)
                        * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2] + y[3]);
                    if n > 4 {
                        let sub_result = (4..n - 1).fold(y[3] + y[n - 1], |acc, i| {
                            acc + if (i - 3) % 2 == 0 {
                                A::from(2.0) * y[i]
                            } else {
                                A::from(4.0) * y[i]
                            }
                        }) * A::from(h)
                            / A::from(3.0);
                        result = result + sub_result;
                    }
                    result
                } else {
                    A::from(3.0) * A::from(h) / A::from(8.0)
                        * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2])
                }
            };
            results.push(slice_result);
        } else {
            let iter_axis = if axis == self.ndim() - 1 { 0 } else { axis };
            for slice in self.axis_iter(Axis(iter_axis)) {
                let owned = slice.to_owned();
                let y = owned
                    .as_slice()
                    .ok_or(IntegrationError::NonContiguousSlice)?;
                if y.len() != n {
                    return Err(IntegrationError::MismatchedLengths {
                        x_len: y.len(),
                        y_len: n,
                    });
                }
                let slice_result = if let Some(x_slice) = x {
                    let h = (x_slice[n - 1] - x_slice[0]) / X::from((n - 1) as f64);
                    if n == 2 {
                        (y[0] + y[1]) * A::from(0.5) * A::from(h)
                    } else if n % 2 == 1 {
                        let init = y[0] + y[n - 1];
                        let sum = (1..n - 1).fold(init, |acc, i| {
                            acc + if i % 2 == 0 {
                                A::from(2.0) * y[i]
                            } else {
                                A::from(4.0) * y[i]
                            }
                        });
                        sum * A::from(h) / A::from(3.0)
                    } else if n >= 4 {
                        let mut result = A::from(3.0) * A::from(h) / A::from(8.0)
                            * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2] + y[3]);
                        if n > 4 {
                            let sub_result = (4..n - 1).fold(y[3] + y[n - 1], |acc, i| {
                                acc + if (i - 3) % 2 == 0 {
                                    A::from(2.0) * y[i]
                                } else {
                                    A::from(4.0) * y[i]
                                }
                            }) * A::from(h)
                                / A::from(3.0);
                            result = result + sub_result;
                        }
                        result
                    } else {
                        A::from(3.0) * A::from(h) / A::from(8.0)
                            * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2])
                    }
                } else {
                    let dx = dx.unwrap_or(X::from(1.0f64));
                    let h = dx;
                    if n == 2 {
                        (y[0] + y[1]) * A::from(0.5) * A::from(h)
                    } else if n % 2 == 1 {
                        let init = y[0] + y[n - 1];
                        let sum = (1..n - 1).fold(init, |acc, i| {
                            acc + if i % 2 == 0 {
                                A::from(2.0) * y[i]
                            } else {
                                A::from(4.0) * y[i]
                            }
                        });
                        sum * A::from(h) / A::from(3.0)
                    } else if n >= 4 {
                        let mut result = A::from(3.0) * A::from(h) / A::from(8.0)
                            * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2] + y[3]);
                        if n > 4 {
                            let sub_result = (4..n - 1).fold(y[3] + y[n - 1], |acc, i| {
                                acc + if (i - 3) % 2 == 0 {
                                    A::from(2.0) * y[i]
                                } else {
                                    A::from(4.0) * y[i]
                                }
                            }) * A::from(h)
                                / A::from(3.0);
                            result = result + sub_result;
                        }
                        result
                    } else {
                        A::from(3.0) * A::from(h) / A::from(8.0)
                            * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2])
                    }
                };
                results.push(slice_result);
            }
        }

        let output: ArrayBase<OwnedRepr<A>, D::Smaller> =
            ArrayBase::from_shape_vec(output_shape, results)
                .map_err(|_| IntegrationError::OutputArrayCreationFailed)?;
        Ok(output)
    }
}

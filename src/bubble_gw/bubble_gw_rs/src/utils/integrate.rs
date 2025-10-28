use ndarray::{ArrayBase, Axis, Data, Dimension, OwnedRepr, RemoveAxis};
use num_traits::Num;
use std::fmt;

// Custom error type for integration
#[derive(Debug)]
pub struct IntegrationError {
    message: String,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integration error: {}", self.message)
    }
}

impl std::error::Error for IntegrationError {}

// Define a generic Integrate trait with associated output type
pub trait Integrate<X: PartialOrd> {
    type Output;

    fn trapezoid<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;

    fn simpson<'a>(
        &self,
        x: Option<&'a [X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;
}

// Implement Integrate for slices
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
            return Err(IntegrationError {
                message: "At least 2 points required for trapezoid rule".to_string(),
            });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError {
                    message: "x and y must have the same length".to_string(),
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
            return Err(IntegrationError {
                message: "At least 2 points required for Simpson's rule".to_string(),
            });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError {
                    message: "x and y must have the same length".to_string(),
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

// Implement Integrate for generic ArrayBase
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
            return Err(IntegrationError {
                message: format!(
                    "Axis {} out of bounds for array with {} dimensions",
                    axis,
                    self.ndim()
                ),
            });
        }

        let n = self.shape()[axis];
        if n < 2 {
            return Err(IntegrationError {
                message: "At least 2 points required for trapezoid rule".to_string(),
            });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError {
                    message: "x and y must have the same length".to_string(),
                });
            }
        }

        let output_shape = self.raw_dim().remove_axis(Axis(axis));
        let output_len = output_shape.size();
        let mut results = Vec::with_capacity(output_len);

        if self.ndim() == 1 {
            let y = self.as_slice().ok_or(IntegrationError {
                message: "Cannot get contiguous slice from 1D array".to_string(),
            })?;
            if y.len() != n {
                return Err(IntegrationError {
                    message: format!(
                        "Slice length {} does not match expected dimension {}",
                        y.len(),
                        n
                    ),
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
                let y = owned.as_slice().ok_or(IntegrationError {
                    message: "Cannot get contiguous slice from array after conversion".to_string(),
                })?;
                if y.len() != n {
                    return Err(IntegrationError {
                        message: format!(
                            "Slice length {} does not match expected dimension {} for axis {}",
                            y.len(),
                            n,
                            axis
                        ),
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
            ArrayBase::from_shape_vec(output_shape, results).map_err(|_| IntegrationError {
                message: "Failed to create output array".to_string(),
            })?;
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
            return Err(IntegrationError {
                message: format!(
                    "Axis {} out of bounds for array with {} dimensions",
                    axis,
                    self.ndim()
                ),
            });
        }

        let n = self.shape()[axis];
        if n < 2 {
            return Err(IntegrationError {
                message: "At least 2 points required for Simpson's rule".to_string(),
            });
        }

        if let Some(x_slice) = x {
            if x_slice.len() != n {
                return Err(IntegrationError {
                    message: "x and y must have the same length".to_string(),
                });
            }
        }

        let output_shape = self.raw_dim().remove_axis(Axis(axis));
        let output_len = output_shape.size();
        let mut results = Vec::with_capacity(output_len);

        if self.ndim() == 1 {
            let y = self.as_slice().ok_or(IntegrationError {
                message: "Cannot get contiguous slice from 1D array".to_string(),
            })?;
            if y.len() != n {
                return Err(IntegrationError {
                    message: format!(
                        "Slice length {} does not match expected dimension {}",
                        y.len(),
                        n
                    ),
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
                let y = owned.as_slice().ok_or(IntegrationError {
                    message: "Cannot get contiguous slice from array after conversion".to_string(),
                })?;
                if y.len() != n {
                    return Err(IntegrationError {
                        message: format!(
                            "Slice length {} does not match expected dimension {} for axis {}",
                            y.len(),
                            n,
                            axis
                        ),
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
            ArrayBase::from_shape_vec(output_shape, results).map_err(|_| IntegrationError {
                message: "Failed to create output array".to_string(),
            })?;
        Ok(output)
    }
}

// use ndarray::{
//     Array1, ArrayBase, ArrayView1, Axis, Data, Dimension, OwnedRepr, RemoveAxis, Zip, s,
// };
// use num_traits::{FromPrimitive, Num};
// use std::fmt;
//
// // Custom error type for integration
// #[derive(Debug)]
// pub struct IntegrationError {
//     message: String,
// }
//
// impl fmt::Display for IntegrationError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "Integration error: {}", self.message)
//     }
// }
//
// impl std::error::Error for IntegrationError {}
//
// // Define a generic Integrate trait with associated output type
// pub trait Integrate<X: PartialOrd> {
//     type Output;
//
//     fn trapezoid<'a>(
//         &self,
//         x: Option<&'a [X]>,
//         dx: Option<X>,
//         axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError>;
//
//     fn simpson<'a>(
//         &self,
//         x: Option<&'a [X]>,
//         dx: Option<X>,
//         axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError>;
// }
//
// // Helper function to compute weights for trapezoidal and Simpson's rules
// fn compute_weights<A, X>(
//     n: usize,
//     x: Option<&[X]>,
//     dx: Option<X>,
//     is_simpson: bool,
// ) -> Result<(Array1<A>, A), IntegrationError>
// where
//     A: Num + Copy + From<X> + FromPrimitive + ndarray::ScalarOperand,
//     X: Num + Copy + PartialOrd + FromPrimitive,
// {
//     if n < 2 {
//         return Err(IntegrationError {
//             message: "At least 2 points required for integration".to_string(),
//         });
//     }
//
//     let h = if let Some(x_slice) = x {
//         if x_slice.len() != n {
//             return Err(IntegrationError {
//                 message: "x and y must have the same length".to_string(),
//             });
//         }
//         A::from((x_slice[n - 1] - x_slice[0]) / X::from_usize(n - 1).unwrap_or(X::one()))
//     } else {
//         A::from(dx.unwrap_or(X::from_f64(1.0).unwrap_or(X::one())))
//     };
//
//     let mut weights = Array1::zeros(n);
//
//     if is_simpson {
//         // Simpson's rule
//         if n == 2 {
//             // Trapezoidal rule for 2 points
//             weights[0] = A::from_f64(0.5).unwrap_or(A::one());
//             weights[1] = A::from_f64(0.5).unwrap_or(A::one());
//             Ok((weights, h))
//         } else if n == 3 {
//             // Simpson's 3/8 rule for 3 points
//             weights[0] = A::from_f64(1.0).unwrap_or(A::one());
//             weights[1] = A::from_f64(3.0).unwrap_or(A::one() + A::one() + A::one());
//             weights[2] = A::from_f64(3.0).unwrap_or(A::one() + A::one() + A::one());
//             Ok((weights * A::from_f64(3.0 / 8.0).unwrap_or(A::one()), h))
//         } else if n % 2 == 1 {
//             // Odd n (even intervals): use Simpson's 1/3 rule
//             weights[0] = A::from_f64(1.0).unwrap_or(A::one());
//             weights[n - 1] = A::from_f64(1.0).unwrap_or(A::one());
//             for i in 1..n - 1 {
//                 weights[i] = if i % 2 == 0 {
//                     A::from_f64(2.0).unwrap_or(A::one() + A::one())
//                 } else {
//                     A::from_f64(4.0).unwrap_or(A::one() + A::one() + A::one() + A::one())
//                 };
//             }
//             Ok((weights * A::from_f64(1.0 / 3.0).unwrap_or(A::one()), h))
//         } else {
//             // Even n (odd intervals): use 3/8 rule for first 4 points, 1/3 rule for rest
//             // 3/8 rule for [0, 1, 2, 3]
//             weights[0] = A::from_f64(1.0).unwrap_or(A::one());
//             weights[1] = A::from_f64(3.0).unwrap_or(A::one() + A::one() + A::one());
//             weights[2] = A::from_f64(3.0).unwrap_or(A::one() + A::one() + A::one());
//             weights[3] = A::from_f64(1.0).unwrap_or(A::one());
//             weights
//                 .slice_mut(s![0..4])
//                 .mapv_inplace(|w| w * A::from_f64(3.0 / 8.0).unwrap_or(A::one()));
//
//             if n > 4 {
//                 // 1/3 rule for [3, 4, ..., n-1]
//                 weights[3] = weights[3] + A::from_f64(1.0).unwrap_or(A::one()); // Accumulate at index 3
//                 for i in 4..n - 1 {
//                     weights[i] = if (i - 3) % 2 == 0 {
//                         A::from_f64(2.0).unwrap_or(A::one() + A::one())
//                     } else {
//                         A::from_f64(4.0).unwrap_or(A::one() + A::one() + A::one() + A::one())
//                     };
//                 }
//                 weights[n - 1] = A::from_f64(1.0).unwrap_or(A::one());
//                 weights
//                     .slice_mut(s![3..n])
//                     .mapv_inplace(|w| w * A::from_f64(1.0 / 3.0).unwrap_or(A::one()));
//             }
//             Ok((weights, h))
//         }
//     } else {
//         // Trapezoidal rule
//         weights[0] = A::from_f64(0.5).unwrap_or(A::one());
//         weights[n - 1] = A::from_f64(0.5).unwrap_or(A::one());
//         for i in 1..n - 1 {
//             weights[i] = A::from_f64(1.0).unwrap_or(A::one());
//         }
//         Ok((weights, h))
//     }
// }
//
// // Implement Integrate for slices
// impl<'a, T, X> Integrate<X> for &'a [T]
// where
//     T: Num + Copy + From<X> + FromPrimitive + ndarray::ScalarOperand,
//     X: Num + Copy + PartialOrd + FromPrimitive,
// {
//     type Output = T;
//
//     fn trapezoid<'b>(
//         &self,
//         x: Option<&'b [X]>,
//         dx: Option<X>,
//         _axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError> {
//         let n = self.len();
//         let (weights, h) = compute_weights::<T, X>(n, x, dx, false)?;
//         let mut result = T::zero();
//         for i in 0..n {
//             result = result + weights[i] * self[i];
//         }
//         Ok(result * h)
//     }
//
//     fn simpson<'b>(
//         &self,
//         x: Option<&'b [X]>,
//         dx: Option<X>,
//         _axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError> {
//         let n = self.len();
//         let (weights, h) = compute_weights::<T, X>(n, x, dx, true)?;
//         let mut result = T::zero();
//         for i in 0..n {
//             result = result + weights[i] * self[i];
//         }
//         Ok(result * h)
//     }
// }
//
// // Implement Integrate for generic ArrayBase
// impl<S, A, X, D> Integrate<X> for ArrayBase<S, D>
// where
//     S: Data<Elem = A>,
//     A: Num + Copy + From<X> + FromPrimitive + ndarray::ScalarOperand,
//     X: Num + Copy + PartialOrd + FromPrimitive,
//     D: Dimension + RemoveAxis,
// {
//     type Output = ArrayBase<OwnedRepr<A>, D::Smaller>;
//
//     fn trapezoid<'a>(
//         &self,
//         x: Option<&'a [X]>,
//         dx: Option<X>,
//         axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError> {
//         let axis = axis.unwrap_or(self.ndim() as isize - 1) as usize;
//         if axis >= self.ndim() {
//             return Err(IntegrationError {
//                 message: format!(
//                     "Axis {} out of bounds for array with {} dimensions",
//                     axis,
//                     self.ndim()
//                 ),
//             });
//         }
//
//         let n = self.shape()[axis];
//         let (weights, h) = compute_weights::<A, X>(n, x, dx, false)?;
//         let output_shape = self.raw_dim().remove_axis(Axis(axis));
//         let mut results = Vec::with_capacity(output_shape.size());
//
//         if self.ndim() == 1 {
//             let y = self.as_slice().ok_or(IntegrationError {
//                 message: "Cannot get contiguous slice from 1D array".to_string(),
//             })?;
//             let mut result = A::zero();
//             Zip::from(&weights)
//                 .and(ArrayView1::from(y))
//                 .for_each(|&w, &y_val| result = result + w * y_val);
//             results.push(result * h);
//         } else {
//             let iter_axis = if axis == self.ndim() - 1 { 0 } else { axis };
//             for slice in self.axis_iter(Axis(iter_axis)) {
//                 let y = slice.as_slice().ok_or(IntegrationError {
//                     message: "Cannot get contiguous slice from array".to_string(),
//                 })?;
//                 let mut result = A::zero();
//                 Zip::from(&weights)
//                     .and(ArrayView1::from(y))
//                     .for_each(|&w, &y_val| result = result + w * y_val);
//                 results.push(result * h);
//             }
//         }
//
//         let output: ArrayBase<OwnedRepr<A>, D::Smaller> =
//             ArrayBase::from_shape_vec(output_shape, results).map_err(|_| IntegrationError {
//                 message: "Failed to create output array".to_string(),
//             })?;
//         Ok(output)
//     }
//
//     fn simpson<'a>(
//         &self,
//         x: Option<&'a [X]>,
//         dx: Option<X>,
//         axis: Option<isize>,
//     ) -> Result<Self::Output, IntegrationError> {
//         let axis = axis.unwrap_or(self.ndim() as isize - 1) as usize;
//         if axis >= self.ndim() {
//             return Err(IntegrationError {
//                 message: format!(
//                     "Axis {} out of bounds for array with {} dimensions",
//                     axis,
//                     self.ndim()
//                 ),
//             });
//         }
//
//         let n = self.shape()[axis];
//         let (weights, h) = compute_weights::<A, X>(n, x, dx, true)?;
//         let output_shape = self.raw_dim().remove_axis(Axis(axis));
//         let mut results = Vec::with_capacity(output_shape.size());
//
//         if self.ndim() == 1 {
//             let y = self.as_slice().ok_or(IntegrationError {
//                 message: "Cannot get contiguous slice from 1D array".to_string(),
//             })?;
//             let mut result = A::zero();
//             Zip::from(&weights)
//                 .and(ArrayView1::from(y))
//                 .for_each(|&w, &y_val| result = result + w * y_val);
//             results.push(result * h);
//         } else {
//             let iter_axis = if axis == self.ndim() - 1 { 0 } else { axis };
//             for slice in self.axis_iter(Axis(iter_axis)) {
//                 let y = slice.as_slice().ok_or(IntegrationError {
//                     message: "Cannot get contiguous slice from array".to_string(),
//                 })?;
//                 let mut result = A::zero();
//                 Zip::from(&weights)
//                     .and(ArrayView1::from(y))
//                     .for_each(|&w, &y_val| result = result + w * y_val);
//                 results.push(result * h);
//             }
//         }
//
//         let output: ArrayBase<OwnedRepr<A>, D::Smaller> =
//             ArrayBase::from_shape_vec(output_shape, results).map_err(|_| IntegrationError {
//                 message: "Failed to create output array".to_string(),
//             })?;
//         Ok(output)
//     }
// }

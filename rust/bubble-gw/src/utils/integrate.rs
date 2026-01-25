use ndarray::{ArrayBase, Axis, Data, Dimension, OwnedRepr, RemoveAxis};
use num_traits::Num;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("At least 2 points required for integration, got {len}")]
    InsufficientPoints { len: usize },

    #[error("x and y must have the same length, got x: {x_len}, y: {y_len}")]
    MismatchedLengths { x_len: usize, y_len: usize },

    #[error("Axis {axis} out of bounds for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("Cannot get contiguous slice from array")]
    NonContiguousSlice,

    #[error("Failed to create output array")]
    OutputArrayCreationFailed,
}

pub trait Integrate<X: PartialOrd> {
    type Output;

    fn trapezoid(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;

    fn simpson(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError>;
}

fn integrate_1d<T, X, F>(
    y: &[T],
    x: Option<&[X]>,
    dx: Option<X>,
    weight_fn: F,
) -> Result<T, IntegrationError>
where
    T: Num + Copy + From<X> + From<f64>,
    X: Num + Copy + PartialOrd + From<f64>,
    F: Fn(&[T], X) -> T,
{
    let n = y.len();
    if n < 2 {
        return Err(IntegrationError::InsufficientPoints { len: n });
    }
    if let Some(xs) = x {
        if xs.len() != n {
            return Err(IntegrationError::MismatchedLengths {
                x_len: xs.len(),
                y_len: n,
            });
        }
    }
    let h = x
        .map(|xs| (xs[n - 1] - xs[0]) / X::from((n - 1) as f64))
        .or(dx)
        .unwrap_or(X::from(1.0));
    Ok(weight_fn(y, h))
}

fn integrate_nd<S, A, X, D, F>(
    arr: &ArrayBase<S, D>,
    x: Option<&[X]>,
    dx: Option<X>,
    axis: Option<isize>,
    weight_fn: F,
) -> Result<ArrayBase<OwnedRepr<A>, D::Smaller>, IntegrationError>
where
    S: Data<Elem = A>,
    A: Num + Copy + From<X> + From<f64>,
    X: Num + Copy + PartialOrd + From<f64>,
    D: Dimension + RemoveAxis,
    F: Fn(&[A], X) -> A,
{
    let axis = axis.unwrap_or(arr.ndim() as isize - 1) as usize;
    if axis >= arr.ndim() {
        return Err(IntegrationError::InvalidAxis {
            axis,
            ndim: arr.ndim(),
        });
    }
    let n = arr.shape()[axis];
    if n < 2 {
        return Err(IntegrationError::InsufficientPoints { len: n });
    }
    if let Some(xs) = x {
        if xs.len() != n {
            return Err(IntegrationError::MismatchedLengths {
                x_len: xs.len(),
                y_len: n,
            });
        }
    }
    let out_shape = arr.raw_dim().remove_axis(Axis(axis));
    let mut results = Vec::with_capacity(out_shape.size());
    let iter_axis = if axis == arr.ndim() - 1 { 0 } else { axis };

    if arr.ndim() == 1 {
        let slice = arr.as_slice().ok_or(IntegrationError::NonContiguousSlice)?;
        results.push(integrate_1d(slice, x, dx, &weight_fn)?);
    } else {
        for subview in arr.axis_iter(Axis(iter_axis)) {
            let owned = subview.to_owned();
            let slice = owned
                .as_slice()
                .ok_or(IntegrationError::NonContiguousSlice)?;
            results.push(integrate_1d(slice, x, dx, &weight_fn)?);
        }
    }
    ArrayBase::from_shape_vec(out_shape, results)
        .map_err(|_| IntegrationError::OutputArrayCreationFailed)
}

impl<T, X> Integrate<X> for &[T]
where
    T: Num + Copy + From<X> + From<f64>,
    X: Num + Copy + PartialOrd + From<f64>,
{
    type Output = T;

    fn trapezoid(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        _axis: Option<isize>,
    ) -> Result<T, IntegrationError> {
        integrate_1d(self, x, dx, |y, _| {
            if let Some(xs) = x {
                (0..y.len() - 1)
                    .map(|i| (y[i] + y[i + 1]) * T::from(0.5) * T::from(xs[i + 1] - xs[i]))
                    .fold(T::zero(), |a, v| a + v)
            } else {
                let dx_val = dx.unwrap_or(X::from(1.0));
                (0..y.len() - 1)
                    .map(|i| (y[i] + y[i + 1]) * T::from(0.5) * T::from(dx_val))
                    .fold(T::zero(), |a, v| a + v)
            }
        })
    }

    fn simpson(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        _axis: Option<isize>,
    ) -> Result<T, IntegrationError> {
        integrate_1d(self, x, dx, |y, h| {
            let n = y.len();
            if n == 2 {
                (y[0] + y[1]) * T::from(0.5) * T::from(h)
            } else if n % 2 == 1 {
                let mut sum = y[0] + y[n - 1];
                for i in 1..n - 1 {
                    sum = sum
                        + if i % 2 == 0 {
                            T::from(2.0) * y[i]
                        } else {
                            T::from(4.0) * y[i]
                        };
                }
                sum * T::from(h) / T::from(3.0)
            } else if n >= 4 {
                let mut res = T::from(3.0) * T::from(h) / T::from(8.0)
                    * (y[0] + T::from(3.0) * y[1] + T::from(3.0) * y[2] + y[3]);
                if n > 4 {
                    let mut sub = y[3] + y[n - 1];
                    for i in 4..n - 1 {
                        sub = sub
                            + if (i - 3) % 2 == 0 {
                                T::from(2.0) * y[i]
                            } else {
                                T::from(4.0) * y[i]
                            };
                    }
                    res = res + sub * T::from(h) / T::from(3.0);
                }
                res
            } else {
                T::from(3.0) * T::from(h) / T::from(8.0)
                    * (y[0] + T::from(3.0) * y[1] + T::from(3.0) * y[2])
            }
        })
    }
}

impl<S, A, X, D> Integrate<X> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    A: Num + Copy + From<X> + From<f64>,
    X: Num + Copy + PartialOrd + From<f64>,
    D: Dimension + RemoveAxis,
{
    type Output = ArrayBase<OwnedRepr<A>, D::Smaller>;

    fn trapezoid(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        integrate_nd(self, x, dx, axis, move |y, _| {
            if let Some(xs) = x {
                (0..y.len() - 1)
                    .map(|i| (y[i] + y[i + 1]) * A::from(0.5) * A::from(xs[i + 1] - xs[i]))
                    .fold(A::zero(), |a, v| a + v)
            } else {
                let dx_val = dx.unwrap_or(X::from(1.0));
                (0..y.len() - 1)
                    .map(|i| (y[i] + y[i + 1]) * A::from(0.5) * A::from(dx_val))
                    .fold(A::zero(), |a, v| a + v)
            }
        })
    }

    fn simpson(
        &self,
        x: Option<&[X]>,
        dx: Option<X>,
        axis: Option<isize>,
    ) -> Result<Self::Output, IntegrationError> {
        integrate_nd(self, x, dx, axis, move |y, h| {
            let n = y.len();
            if n == 2 {
                (y[0] + y[1]) * A::from(0.5) * A::from(h)
            } else if n % 2 == 1 {
                let mut sum = y[0] + y[n - 1];
                for i in 1..n - 1 {
                    sum = sum
                        + if i % 2 == 0 {
                            A::from(2.0) * y[i]
                        } else {
                            A::from(4.0) * y[i]
                        };
                }
                sum * A::from(h) / A::from(3.0)
            } else if n >= 4 {
                let mut res = A::from(3.0) * A::from(h) / A::from(8.0)
                    * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2] + y[3]);
                if n > 4 {
                    let mut sub = y[3] + y[n - 1];
                    for i in 4..n - 1 {
                        sub = sub
                            + if (i - 3) % 2 == 0 {
                                A::from(2.0) * y[i]
                            } else {
                                A::from(4.0) * y[i]
                            };
                    }
                    res = res + sub * A::from(h) / A::from(3.0);
                }
                res
            } else {
                A::from(3.0) * A::from(h) / A::from(8.0)
                    * (y[0] + A::from(3.0) * y[1] + A::from(3.0) * y[2])
            }
        })
    }
}

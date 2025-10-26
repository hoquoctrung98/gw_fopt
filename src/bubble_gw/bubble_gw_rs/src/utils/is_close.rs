use ndarray::{Array1, ArrayBase, Data, Dim, Dimension};
use num::complex::Complex;

pub trait IsClose<Rhs = Self> {
    fn is_close(&self, other: &Rhs, abs_tol: f64, rel_tol: f64) -> Result<(), String>;
}

impl<S1, S2, D> IsClose<ArrayBase<S2, D>> for ArrayBase<S1, D>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    fn is_close(&self, other: &ArrayBase<S2, D>, abs_tol: f64, rel_tol: f64) -> Result<(), String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Array shapes differ: actual {:?}, expected {:?}",
                self.shape(),
                other.shape()
            ));
        }
        if !self.relative_eq(other, abs_tol, rel_tol) {
            let mut errors = Vec::new();
            for (idx, (a, b)) in self.iter().zip(other.iter()).enumerate() {
                let abs_diff = (a - b).abs();
                let max_abs = a.abs().max(b.abs());
                let tolerance = rel_tol.mul_add(max_abs, abs_tol).max(abs_tol);
                if abs_diff > tolerance {
                    errors.push(format!(
                        "Element at index {} differs: actual {}, expected {}, abs_diff {}, tolerance {}",
                        idx, a, b, abs_diff, tolerance
                    ));
                }
            }
            return Err(errors.join("\n"));
        }
        Ok(())
    }
}

impl<S> IsClose<&[f64]> for ArrayBase<S, Dim<[usize; 1]>>
where
    S: Data<Elem = f64>,
{
    fn is_close(&self, other: &&[f64], abs_tol: f64, rel_tol: f64) -> Result<(), String> {
        let other_array = Array1::from_vec(other.to_vec());
        self.is_close(&other_array, abs_tol, rel_tol)
    }
}

impl IsClose<f64> for f64 {
    fn is_close(&self, other: &f64, abs_tol: f64, rel_tol: f64) -> Result<(), String> {
        let abs_diff = (self - other).abs();
        let max_abs = self.abs().max(other.abs());
        let tolerance = rel_tol.mul_add(max_abs, abs_tol).max(abs_tol);
        if abs_diff > tolerance {
            return Err(format!(
                "Scalar differs: actual {}, expected {}, abs_diff {}, tolerance {}",
                self, other, abs_diff, tolerance
            ));
        }
        Ok(())
    }
}

// Implement IsClose for Complex<f64>
impl IsClose<Complex<f64>> for Complex<f64> {
    fn is_close(&self, other: &Complex<f64>, abs_tol: f64, rel_tol: f64) -> Result<(), String> {
        let abs_diff_re = (self.re - other.re).abs();
        let abs_diff_im = (self.im - other.im).abs();
        let max_abs_re = self.re.abs().max(other.re.abs());
        let max_abs_im = self.im.abs().max(other.im.abs());
        let tolerance_re = rel_tol.mul_add(max_abs_re, abs_tol).max(abs_tol);
        let tolerance_im = rel_tol.mul_add(max_abs_im, abs_tol).max(abs_tol);
        if abs_diff_re > tolerance_re || abs_diff_im > tolerance_im {
            return Err(format!(
                "Complex scalar differs: actual {} + {}i, expected {} + {}i, abs_diff_re {}, abs_diff_im {}, tolerance_re {}, tolerance_im {}",
                self.re,
                self.im,
                other.re,
                other.im,
                abs_diff_re,
                abs_diff_im,
                tolerance_re,
                tolerance_im
            ));
        }
        Ok(())
    }
}

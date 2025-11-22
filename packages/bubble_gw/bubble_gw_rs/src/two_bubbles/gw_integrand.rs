use num_complex::Complex64;
use puruspe::Jn;

/// Enum to represent the type of integrand to compute.
#[derive(Debug, Clone, Copy)]
pub enum IntegrandType {
    XX,
    YY,
    ZZ,
    XZ,
}

/// Struct to hold integrand parameters and precompute shared terms.
#[derive(Debug, Clone)]
pub struct IntegrandCalculator {
    w: f64,
    k: f64,
    s: f64,
    sign: f64,
    t_cut: f64,
    t_0: f64,
}

impl IntegrandCalculator {
    pub fn new(w: f64, k: f64, s: f64, sign: f64, t_cut: f64, t_0: f64) -> Self {
        Self {
            w,
            k,
            s,
            sign,
            t_cut,
            t_0,
        }
    }

    /// Computes both the real and imaginary parts of the integrand at point u.
    #[inline]
    pub fn compute(&self, u: f64, int_type: IntegrandType) -> Result<Complex64, &'static str> {
        let one_minus_k2_sqrt = (1.0 - self.k * self.k).sqrt();
        // Precompute common terms
        let u_squared_plus_sign = u * u + self.sign;
        let sqrt_term = u_squared_plus_sign.sqrt();
        let bessel_arg = self.w * one_minus_k2_sqrt * self.s * sqrt_term;
        let wsu = self.w * self.s * u;
        let exp_term_real = wsu.cos();
        let exp_term_imag = wsu.sin();
        let c1_val = c1(u * self.s, self.t_cut, self.t_0)?;

        // Precompute Bessel functions
        let (bessel_0, bessel_1, bessel_2) = match int_type {
            IntegrandType::XX | IntegrandType::YY => {
                let b0 = Jn(0, bessel_arg);
                let b2 = Jn(2, bessel_arg);
                (b0, 0.0, b2)
            }
            IntegrandType::ZZ => (Jn(0, bessel_arg), 0.0, 0.0),
            IntegrandType::XZ => (0.0, Jn(1, bessel_arg), 0.0),
        };

        let (real, imag) = match int_type {
            IntegrandType::XX => {
                let factor = u_squared_plus_sign;
                let bessel_diff = bessel_0 - bessel_2;
                (
                    factor * exp_term_real * bessel_diff * c1_val,
                    factor * exp_term_imag * bessel_diff * c1_val,
                )
            }
            IntegrandType::YY => {
                let factor = u_squared_plus_sign;
                let bessel_sum = bessel_0 + bessel_2;
                (
                    factor * exp_term_real * bessel_sum * c1_val,
                    factor * exp_term_imag * bessel_sum * c1_val,
                )
            }
            IntegrandType::ZZ => {
                (exp_term_real * bessel_0 * c1_val, exp_term_imag * bessel_0 * c1_val)
            }
            IntegrandType::XZ => {
                let factor = self.sign * sqrt_term;
                (
                    factor * exp_term_real * bessel_1 * c1_val,
                    factor * exp_term_imag * bessel_1 * c1_val,
                )
            }
        };
        Ok(Complex64::new(real, imag))
    }
}

#[inline]
pub fn c1(t: f64, t_cut: f64, t_0: f64) -> Result<f64, &'static str> {
    if t_0 == 0.0 {
        return Err("t_0 cannot be zero");
    }
    if t < t_cut {
        Ok(1.0)
    } else {
        let exponent = -((t - t_cut).powi(2)) / t_0.powi(2);
        Ok(exponent.exp())
    }
}

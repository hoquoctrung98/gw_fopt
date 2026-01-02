use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::Array2;
use rand::Rng;
use rand::rngs::StdRng;

use super::GeneralLatticeProperties;
use crate::many_bubbles::lattice_bubbles::LatticeBubbles;

/// Nucleation strategy with fixed exponential nucleation rate per unit
/// volume-time: Γ(t) = γ₀ · exp(β (t − t₀))
///
/// At each call, nucleates *zero or one* bubble, advancing time implicitly via
/// dt = d_p0 / (Γ(t) · V_rem(t)).
///
/// # Parameters
/// - `beta`: inverse timescale of rate growth
/// - `gamma0`: base nucleation rate density (bubbles / volume / time)
/// - `t0`: reference time (rate = γ₀ at t = t₀)
/// - `d_p0`: target probability per step (~0.1–0.5 for stability)
/// - `seed`: optional seed for reproducibility
#[derive(Clone, Debug)]
pub struct FixedNucleationRate {
    pub beta: f64,
    pub gamma0: f64,
    pub t0: f64,
    pub d_p0: f64,
    pub seed: Option<u64>,
}

impl FixedNucleationRate {
    /// Compute remaining volume: V_lattice − Σ V_bubble(t)
    /// Approximation: V_bubble(t) = (4π/3) (t − t_n)³ for bubble nucleated at
    /// t_n.
    pub fn volume_remaining<L: GeneralLatticeProperties>(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        t: f64,
    ) -> f64 {
        let lattice_vol = lattice_bubbles.lattice.volume();
        let bubble_vol: f64 = lattice_bubbles
            .interior
            .spacetime
            .iter()
            .map(|v| {
                let t_n = v[0];
                let radius = (t - t_n).max(0.0);
                (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3)
            })
            .sum();
        (lattice_vol - bubble_vol).max(0.0)
    }

    /// Check if point `(x,y,z)` at time `t` is outside all existing bubbles.
    pub fn is_point_valid<L: GeneralLatticeProperties>(
        &self,
        pt: &Point3<f64>,
        t: f64,
        lattice_bubbles: &LatticeBubbles<L>,
    ) -> bool {
        let candidate = Vector4::new(t, pt.x, pt.y, pt.z);

        // Check against all interior bubbles
        for bubble in &lattice_bubbles.interior.spacetime {
            let delta = candidate - bubble;
            // In (−,+,+,+) signature: timelike ⇒ causality violation ⇔ δ² < 0
            if delta.scalar(&delta) < 0.0 {
                return false;
            }
        }

        // Check against all exterior bubbles
        for bubble in &lattice_bubbles.exterior.spacetime {
            let delta = candidate - bubble;
            if delta.scalar(&delta) < 0.0 {
                return false;
            }
        }

        true
    }

    /// Sample a point uniformly in lattice, rejecting if inside existing
    /// bubbles. Returns `Some(point)` on success, `None` if too many
    /// attempts or no volume left.
    pub fn sample_valid_point<L: GeneralLatticeProperties>(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        t: f64,
        rng: &mut StdRng,
    ) -> Option<Point3<f64>> {
        let lattice = &lattice_bubbles.lattice;
        const MAX_ATTEMPTS: i32 = 10_000;

        for _ in 0..MAX_ATTEMPTS {
            let pts = lattice.sample_points(1, rng);
            let pt = pts.into_iter().next()?; // always 1 element

            // `sample_points` ensures pt ∈ lattice — but check for FP edge cases
            if !lattice.contains(&[pt])[0] {
                continue;
            }

            if self.is_point_valid(&pt, t, lattice_bubbles) {
                return Some(pt);
            }
        }

        None
    }

    /// Nucleate zero or one bubble at current time step.
    /// Returns array of shape `(0, 4)` or `(1, 4)`.
    pub fn nucleate_one<L: GeneralLatticeProperties>(
        &self,
        lattice_bubbles: &LatticeBubbles<L>,
        t: f64,
        rng: &mut StdRng,
    ) -> Array2<f64> {
        let v_rem = self.volume_remaining(lattice_bubbles, t);
        if v_rem < 1e-10 {
            return Array2::zeros((0, 4));
        }

        let gamma_t = self.gamma0 * ((self.beta * (t - self.t0)).exp());
        let dt = self.d_p0 / (gamma_t * v_rem).max(f64::EPSILON);
        let new_t = t + dt;

        let x: f64 = rng.random();
        if x > self.d_p0 {
            return Array2::zeros((0, 4));
        }

        if let Some(pt) = self.sample_valid_point(lattice_bubbles, new_t, rng) {
            Array2::from_shape_vec((1, 4), vec![new_t, pt.x, pt.y, pt.z]).unwrap()
        } else {
            Array2::zeros((0, 4))
        }
    }
}

use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, random};

use super::{GeneralLatticeProperties, NucleationError, NucleationStrategy};
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::{
    BoundaryConditions,
    BuiltInLattice,
    CartesianLattice,
    GenerateBubblesExterior,
    LatticeGeometry,
    ParallelepipedLattice,
    SphericalLattice,
};
use crate::many_bubbles::lattice_bubbles::LatticeBubbles;

/// Nucleation strategy with fixed exponential nucleation rate per unit
/// volume-time: Γ(t) = γ₀ · exp(β (t − t₀))
///
/// Assumes `lattice_bubbles` is initially empty. Nucleates bubbles until volume
/// saturation.
#[derive(Clone, Debug)]
pub struct FixedRateNucleation {
    pub beta: f64,
    pub gamma0: f64,
    pub t0: f64,
    pub d_p0: f64,
    pub seed: Option<u64>,
}

impl FixedRateNucleation {
    /// Check if point `(x,y,z)` at time `t` is outside all *newly nucleated*
    /// bubbles.
    fn is_point_valid(&self, pt: &Point3<f64>, t: f64, all_new_bubbles: &[Vector4<f64>]) -> bool {
        let candidate = Vector4::new(t, pt.x, pt.y, pt.z);
        for bubble in all_new_bubbles {
            let delta = candidate - bubble;
            if delta.scalar(&delta) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Sample a point uniformly in lattice, rejecting if inside any *newly
    /// nucleated* bubbles.
    fn sample_valid_point<L: GeneralLatticeProperties>(
        &self,
        lattice: &L,
        t: f64,
        all_new_bubbles: &[Vector4<f64>],
        rng: &mut StdRng,
    ) -> Option<Point3<f64>> {
        const MAX_ATTEMPTS: usize = 10_000;

        for _ in 0..MAX_ATTEMPTS {
            let pts = lattice.sample_points(1, rng);
            let pt = pts.into_iter().next()?;

            if !lattice.contains(&[pt])[0] {
                continue;
            }

            if self.is_point_valid(&pt, t, all_new_bubbles) {
                return Some(pt);
            }
        }

        None
    }
}

macro_rules! impl_fixed_rate_nucleation_for_lattice {
    ($Lattice:ty) => {
        impl NucleationStrategy<$Lattice> for FixedRateNucleation {
            fn nucleate(
                &self,
                lattice_bubbles: &LatticeBubbles<$Lattice>,
                boundary_condition: BoundaryConditions,
            ) -> Result<(Array2<f64>, Array2<f64>), NucleationError> {
                let mut rng = match self.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => StdRng::seed_from_u64(random::<u64>()),
                };

                let lattice = &lattice_bubbles.lattice;
                let lattice_vol = lattice.volume();
                let vol_cutoff = 1e-5 * lattice_vol;

                let t_start = self.t0;
                let mut t = t_start;

                let mut new_interior: Vec<Vector4<f64>> = Vec::new();
                let mut new_exterior: Vec<Vector4<f64>> = Vec::new();

                const MAX_ATTEMPTS: usize = 10_000;
                for _ in 0..MAX_ATTEMPTS {
                    let bubble_vol: f64 = new_interior
                        .iter()
                        .map(|v| {
                            let t_n = v[0];
                            let radius = (t - t_n).max(0.0);
                            (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3)
                        })
                        .sum();
                    let v_rem = (lattice_vol - bubble_vol).max(0.0);

                    if v_rem < vol_cutoff {
                        break;
                    }

                    let exponent = self.beta * (t - self.t0);
                    let gamma_t = self.gamma0 * exponent.exp();
                    if !gamma_t.is_finite() || gamma_t <= 0.0 {
                        break;
                    }

                    let dt = self.d_p0 / (gamma_t * v_rem);
                    if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
                        break;
                    }

                    let new_t = t + dt;

                    let x: f64 = rng.random();
                    if x <= self.d_p0 {
                        let mut all_new =
                            Vec::with_capacity(new_interior.len() + new_exterior.len());
                        all_new.extend_from_slice(&new_interior);
                        all_new.extend_from_slice(&new_exterior);

                        if let Some(pt) =
                            self.sample_valid_point(lattice, new_t, &all_new, &mut rng)
                        {
                            let interior_bubble = Vector4::new(new_t, pt.x, pt.y, pt.z);
                            new_interior.push(interior_bubble);

                            let dummy_interior = Bubbles::new(vec![interior_bubble]);
                            let exterior_bubbles = lattice
                                .generate_bubbles_exterior(&dummy_interior, boundary_condition);
                            new_exterior.extend(exterior_bubbles.spacetime);
                        }
                    }

                    t = new_t;

                    if new_interior.len() >= 1000 {
                        break;
                    }
                }

                // ✅ Fix: get lengths BEFORE consuming vectors
                let n_interior = new_interior.len();
                let n_exterior = new_exterior.len();

                let interior = if n_interior == 0 {
                    Array2::zeros((0, 4))
                } else {
                    let data: Vec<f64> = new_interior
                        .into_iter()
                        .flat_map(|v| [v[0], v[1], v[2], v[3]])
                        .collect();
                    Array2::from_shape_vec((n_interior, 4), data).map_err(|_| {
                        NucleationError::InvalidConfig("interior reshape failed".into())
                    })?
                };

                let exterior = if n_exterior == 0 {
                    Array2::zeros((0, 4))
                } else {
                    let data: Vec<f64> = new_exterior
                        .into_iter()
                        .flat_map(|v| [v[0], v[1], v[2], v[3]])
                        .collect();
                    Array2::from_shape_vec((n_exterior, 4), data).map_err(|_| {
                        NucleationError::InvalidConfig("exterior reshape failed".into())
                    })?
                };

                Ok((interior, exterior))
            }
        }
    };
}

impl_fixed_rate_nucleation_for_lattice!(BuiltInLattice);
impl_fixed_rate_nucleation_for_lattice!(ParallelepipedLattice);
impl_fixed_rate_nucleation_for_lattice!(CartesianLattice);
impl_fixed_rate_nucleation_for_lattice!(SphericalLattice);

use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, random};
use rayon::prelude::*;

use super::{GeneralLatticeProperties, NucleationStrategy};
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;
use crate::many_bubbles::lattice_bubbles::LatticeBubblesError;

const MAX_ATTEMPTS: usize = 10_000;

#[derive(Clone, Debug)]
pub enum VolumeRemainingMethod {
    Approximation,
    MonteCarlo { n_points: usize },
}

#[derive(Clone, Debug)]
pub struct FixedRateNucleation {
    pub beta: f64,
    pub gamma0: f64,
    pub t0: f64,
    pub d_p0: f64,
    pub seed: Option<u64>,
    pub volume_method: VolumeRemainingMethod,
    rng: StdRng,
    points_outside_bubbles: Option<Vec<Point3<f64>>>,
    pub time_history: Vec<f64>,
    pub volume_remaining_history: Vec<f64>,
}

impl FixedRateNucleation {
    pub fn new(
        beta: f64,
        gamma0: f64,
        t0: f64,
        d_p0: f64,
        seed: Option<u64>,
        volume_method: VolumeRemainingMethod,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(random::<u64>()),
        };
        Self {
            beta,
            gamma0,
            t0,
            d_p0,
            seed,
            volume_method,
            rng,
            points_outside_bubbles: None,
            time_history: Vec::new(),
            volume_remaining_history: Vec::new(),
        }
    }

    fn volume_remaining<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        t: f64,
        bubbles_interior: &Bubbles,
        bubbles_exterior: &Bubbles,
    ) -> f64 {
        match &self.volume_method {
            VolumeRemainingMethod::Approximation => {
                let volume_lattice = lattice.volume();
                let volume_bubbles: f64 = bubbles_interior
                    .spacetime
                    .iter()
                    .map(|v| {
                        let t_n = v[0];
                        let radius = (t - t_n).max(0.0);
                        (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3)
                    })
                    .sum();
                (volume_lattice - volume_bubbles).max(0.0)
            },
            VolumeRemainingMethod::MonteCarlo { n_points } => {
                if *n_points == 0 {
                    return lattice.volume();
                }

                if self.points_outside_bubbles.is_none() {
                    let points = lattice.sample_points(*n_points, &mut self.rng);
                    self.points_outside_bubbles = Some(points);
                }

                let points_outside_bubbles = self.points_outside_bubbles.as_mut().unwrap();
                let all_bubbles: Vec<&Vector4<f64>> = bubbles_interior
                    .spacetime
                    .iter()
                    .chain(bubbles_exterior.spacetime.iter())
                    .collect();

                let valid_points: Vec<Point3<f64>> = points_outside_bubbles
                    .par_iter()
                    .filter_map(|pt| {
                        let candidate = Vector4::new(t, pt.x, pt.y, pt.z);
                        for &bubble in &all_bubbles {
                            let delta = candidate - bubble;
                            if delta.scalar(&delta) < 0.0 {
                                return None; // inside bubble
                            }
                        }
                        Some(*pt)
                    })
                    .collect();

                // Replace active points with valid ones
                *points_outside_bubbles = valid_points;

                let active_count = points_outside_bubbles.len();
                let fraction = active_count as f64 / *n_points as f64;
                (fraction * lattice.volume()).max(0.0)
            },
        }
    }

    fn check_points_outside_bubbles(
        &self,
        bubbles_new: &Bubbles,
        bubbles_interior: &Bubbles,
        bubbles_exterior: &Bubbles,
    ) -> bool {
        for bubble_new in &bubbles_new.spacetime {
            for &bubble in bubbles_interior
                .spacetime
                .iter()
                .chain(bubbles_exterior.spacetime.iter())
            {
                let delta = bubble_new - bubble;
                if delta.scalar(&delta) < 0.0 {
                    return false;
                }
            }
        }
        true
    }

    fn sample_points_outside_bubbles<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        t: f64,
        bubbles_interior: &Bubbles,
        bubbles_exterior: &Bubbles,
        n_points: usize,
    ) -> Option<Bubbles> {
        let mut accepted = Vec::with_capacity(n_points);
        let mut attempts = 0;

        while accepted.len() < n_points && attempts < MAX_ATTEMPTS {
            let batch_size = (n_points - accepted.len()).min(100);
            let candidate_pts = lattice.sample_points(batch_size, &mut self.rng);

            for pt in candidate_pts {
                if !lattice.contains(&[pt])[0] {
                    continue;
                }

                let candidate = Vector4::new(t, pt.x, pt.y, pt.z);
                let candidate_bubbles = Bubbles::new(vec![candidate]);
                if self.check_points_outside_bubbles(
                    &candidate_bubbles,
                    bubbles_interior,
                    bubbles_exterior,
                ) {
                    accepted.push(candidate);
                    if accepted.len() >= n_points {
                        break;
                    }
                }
            }

            attempts += batch_size;
        }

        if accepted.len() == n_points {
            Some(Bubbles::new(accepted))
        } else {
            None
        }
    }
}

impl<L: GeneralLatticeProperties> NucleationStrategy<L> for FixedRateNucleation {
    fn nucleate(
        &mut self,
        lattice: &L,
        boundary_condition: BoundaryConditions,
    ) -> Result<(Bubbles, Bubbles), LatticeBubblesError> {
        let volume_lattice = lattice.volume();
        let volume_cutoff = 1e-5 * volume_lattice;

        let t_start = self.t0;
        let mut t = t_start;

        let mut bubbles_interior = Bubbles::new(Vec::new());
        let mut bubbles_exterior = Bubbles::new(Vec::new());

        self.time_history.clear();
        self.volume_remaining_history.clear();

        for _ in 0..MAX_ATTEMPTS {
            let volume_remaining =
                self.volume_remaining(lattice, t, &bubbles_interior, &bubbles_exterior);

            self.time_history.push(t);
            self.volume_remaining_history.push(volume_remaining);

            if volume_remaining < volume_cutoff || volume_remaining < 1e-12 {
                break;
            }

            let exponent = self.beta * (t - self.t0);
            let gamma_t = self.gamma0 * exponent.exp();
            if !gamma_t.is_finite() || gamma_t <= 0.0 {
                break;
            }

            let dt = self.d_p0 / (gamma_t * volume_remaining).max(f64::EPSILON);
            if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
                break;
            }

            let new_t = t + dt;
            if new_t <= t || !new_t.is_finite() {
                break;
            }

            let x: f64 = self.rng.random();
            if x <= self.d_p0 {
                if let Some(new_bubbles) = self.sample_points_outside_bubbles(
                    lattice,
                    new_t,
                    &bubbles_interior,
                    &bubbles_exterior,
                    1,
                ) {
                    if let Some(bubble_new) = new_bubbles.spacetime.into_iter().next() {
                        bubbles_interior.spacetime.push(bubble_new);
                        let dummy_interior = Bubbles::new(vec![bubble_new]);
                        let exterior_bubbles =
                            lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
                        bubbles_exterior
                            .spacetime
                            .extend(exterior_bubbles.spacetime);
                    }
                }
            }

            t = new_t;
        }

        Ok((bubbles_interior, bubbles_exterior))
    }
}

// TODO:
// interesting references to compute volume of union of bubbles
//   + https://inria.hal.science/inria-00409374v1/document
//   + https://www.aei.tuke.sk/papers/2008/4/06_Dzurina.pdf

use core::f64::consts::PI;

use differential_equations::interpolate::Interpolation;
use differential_equations::ode::OrdinaryNumericalMethod;
use differential_equations::prelude::*;
use nalgebra::{Point3, SVector, Vector4, vector};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use thiserror::Error;

use super::GeneralLatticeProperties;
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};

#[derive(Error, Debug)]
pub enum FixedRateNucleationError {
    #[error("Failed to initialize random number generator: {0}")]
    RngInitializationError(#[from] getrandom::Error),

    #[error("Lattice bubbles error: {0}")]
    LatticeBubblesError(#[from] LatticeBubblesError),
}

pub enum FixedRateNucleationMethod {
    FixedProbabilityTimeStepping { d_p0: f64 },
    FixedProbabilitysDistribution,
}

/// Result of nucleation containing bubbles and history.
#[derive(Debug)]
pub struct FixedRateNucleationResult<L: GeneralLatticeProperties> {
    pub lattice_bubbles: LatticeBubbles<L>,
    pub time_history: Array1<f64>,
    pub volume_false_vacuum_history: Array1<f64>,
}

#[derive(Debug)]
pub struct FixedRateNucleation {
    pub beta: f64,
    pub gamma0: f64,
    pub t0: f64,
    pub seed: Option<u64>,
    pub n_points: usize,
    pub max_time_iterations: usize,
    pub cutoff_false_vacuum_fraction: f64,
    rng: StdRng,
    sample_points: Vec<Point3<f64>>,
    first_collision_times: Vec<f64>,
}

impl ODE<f64, SVector<f64, 4>> for FixedRateNucleation {
    fn diff(&self, tau: f64, m: &SVector<f64, 4>, dm_dtau: &mut SVector<f64, 4>) {
        let gamma0_bar = self.gamma0 * self.beta.powi(-4);
        let log_n = gamma0_bar.ln() + tau - (4.0 * PI / 3.0) * m[3];
        let n = log_n.exp();

        dm_dtau[0] = n;
        dm_dtau[1] = m[0];
        dm_dtau[2] = 2.0 * m[1];
        dm_dtau[3] = 3.0 * m[2];
    }
}

impl FixedRateNucleation {
    pub fn new(
        beta: f64,
        gamma0: f64,
        t0: f64,
        seed: Option<u64>,
        n_points: usize,
        max_time_iterations: Option<usize>,
        cutoff_fraction_false_vacuum: Option<f64>,
    ) -> Result<Self, FixedRateNucleationError> {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                // Use getrandom for truly independent seeds across processes
                // Note that calling StdRng::seed_from_u64(rng.random::<u64>())
                // produces wrong distribution from python examples using multiprocessing
                let random_seed = getrandom::u64()?;
                StdRng::seed_from_u64(random_seed)
            },
        };

        let max_time_iterations = max_time_iterations.unwrap_or(1_000_000);
        let cutoff_false_vacuum_fraction = cutoff_fraction_false_vacuum.unwrap_or(0.01);

        Ok(Self {
            beta,
            gamma0,
            t0,
            seed,
            n_points,
            max_time_iterations,
            cutoff_false_vacuum_fraction,
            rng,
            sample_points: Vec::new(),
            first_collision_times: Vec::new(),
        })
    }

    fn update_first_collision_times(&mut self, new_bubble: &Vector4<f64>) {
        let t_n = new_bubble[0];
        let x_n = new_bubble[1];
        let y_n = new_bubble[2];
        let z_n = new_bubble[3];

        for (i, pt) in self.sample_points.iter().enumerate() {
            let dx = pt.x - x_n;
            let dy = pt.y - y_n;
            let dz = pt.z - z_n;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let t_collide = t_n + dist;
            if t_collide < self.first_collision_times[i] {
                self.first_collision_times[i] = t_collide;
            }
        }
    }

    fn volume_remaining<L: GeneralLatticeProperties>(&self, lattice: &L, t: f64) -> f64 {
        if self.n_points == 0 {
            return lattice.volume();
        }
        let count_valid = self
            .first_collision_times
            .iter()
            .filter(|&&t_collide| t_collide > t)
            .count();
        let fraction = count_valid as f64 / self.n_points as f64;
        (fraction * lattice.volume()).max(0.0)
    }

    pub fn solve_bubbles_distribution<S>(
        &self,
        taumax: f64,
        volume_lattice: f64,
        method: &mut S,
    ) -> (Vec<f64>, Vec<f64>)
    where
        S: OrdinaryNumericalMethod<f64, SVector<f64, 4>> + Interpolation<f64, SVector<f64, 4>>,
    {
        // Initial conditions
        let y0 = vector![0.0, 0.0, 0.0, 0.0];
        let tau0 = self.t0 * self.beta;

        let problem = ODEProblem::new(self, tau0, taumax, y0);
        let solution = problem.even(0.001).solve(method).unwrap();
        let tau: Vec<f64> = solution.iter().map(|(t, _)| *t).collect();
        let m0: Vec<f64> = solution.iter().map(|(_, y)| y[0]).collect();

        let n_bubbles: Vec<f64> = m0
            .iter()
            .map(|m| m * volume_lattice * self.beta.powi(3))
            .collect();

        let mut crossing_indices: Vec<usize> = Vec::new();
        let mut prev_floor = n_bubbles[0].floor() as i64;
        for i in 1..n_bubbles.len() {
            let current_floor = n_bubbles[i].floor() as i64;
            if current_floor > prev_floor {
                crossing_indices.push(i);
            }

            prev_floor = current_floor;
        }

        let t: Vec<f64> = crossing_indices
            .iter()
            .map(|&i| tau[i] / self.beta)
            .collect();
        let n_bubbles: Vec<f64> = crossing_indices.iter().map(|&i| n_bubbles[i]).collect();
        return (t, n_bubbles);
    }

    pub fn nucleate_fixed_probability_time_stepping<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        boundary_condition: BoundaryConditions,
        d_p0: f64,
    ) -> Result<FixedRateNucleationResult<L>, LatticeBubblesError> {
        let volume_lattice = lattice.volume();
        let volume_cutoff = self.cutoff_false_vacuum_fraction * volume_lattice;

        let mut t = self.t0;
        let mut bubbles_interior = Bubbles::new(Vec::new());
        let mut bubbles_exterior = Bubbles::new(Vec::new());
        let mut time_history = Vec::new();
        let mut volume_false_vacuum_history = Vec::new();

        // Initialize Monte Carlo state
        if self.n_points > 0 {
            let points = lattice.sample_points(self.n_points, &mut self.rng);
            self.sample_points = points;
            self.first_collision_times = vec![f64::INFINITY; self.n_points];
        }

        for _ in 0..self.max_time_iterations {
            let volume_remaining = self.volume_remaining(lattice, t);

            if volume_remaining < volume_cutoff {
                break;
            }

            let exponent = self.beta * (t - self.t0);
            let gamma_t = self.gamma0 * exponent.exp();

            let x: f64 = self.rng.random();
            if x <= d_p0 {
                let new_bubble = {
                    let mut candidate_pt = None;
                    for i in 0..self.sample_points.len() {
                        if self.first_collision_times[i] > t {
                            candidate_pt = Some(self.sample_points[i]);
                            break;
                        }
                    }

                    if let Some(pt) = candidate_pt {
                        let candidate_vec = Vector4::new(t, pt.x, pt.y, pt.z);
                        Some(candidate_vec)
                    } else {
                        None
                    }
                };

                if let Some(bubble_new) = new_bubble {
                    time_history.push(t);
                    volume_false_vacuum_history.push(volume_remaining);

                    bubbles_interior.spacetime.push(bubble_new);
                    let dummy_interior = Bubbles::new(vec![bubble_new]);
                    let exterior_bubbles =
                        lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
                    let exterior_spacetime_ref = &exterior_bubbles.spacetime;

                    self.update_first_collision_times(&bubble_new);
                    for exterior_bubble in exterior_spacetime_ref {
                        self.update_first_collision_times(exterior_bubble);
                    }

                    bubbles_exterior.spacetime.extend(exterior_spacetime_ref);
                }
            }

            let dt = (d_p0 / gamma_t) * (volume_lattice / volume_remaining);
            t += dt;
        }

        let lattice_bubbles = LatticeBubbles::new(
            bubbles_interior.to_array2(),
            Some(bubbles_exterior.to_array2()),
            lattice.clone(),
        )?;
        let time_history = Array1::from_vec(time_history);
        let volume_false_vacuum_history = Array1::from_vec(volume_false_vacuum_history);

        Ok(FixedRateNucleationResult {
            lattice_bubbles,
            time_history,
            volume_false_vacuum_history,
        })
    }

    pub fn nucleate_fixed_probability_distribution<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        boundary_condition: BoundaryConditions,
    ) -> Result<FixedRateNucleationResult<L>, LatticeBubblesError> {
        let volume_lattice = lattice.volume();
        let volume_cutoff = self.cutoff_false_vacuum_fraction * volume_lattice;

        let mut bubbles_interior = Bubbles::new(Vec::new());
        let mut bubbles_exterior = Bubbles::new(Vec::new());

        let mut method = ImplicitRungeKutta::radau5().rtol(1e-9).atol(1e-12);
        let (time_history, _) =
            self.solve_bubbles_distribution(40.0, lattice.volume(), &mut method);

        let mut volume_false_vacuum_history = Vec::new();

        // Initialize Monte Carlo state
        if self.n_points > 0 {
            let points = lattice.sample_points(self.n_points, &mut self.rng);
            self.sample_points = points;
            self.first_collision_times = vec![f64::INFINITY; self.n_points];
        }

        for &t in &time_history {
            let volume_remaining = self.volume_remaining(lattice, t);

            if volume_remaining < volume_cutoff {
                break;
            }

            let new_bubble = {
                let mut candidate_pt = None;
                for i in 0..self.sample_points.len() {
                    if self.first_collision_times[i] > t {
                        candidate_pt = Some(self.sample_points[i]);
                        break;
                    }
                }

                if let Some(pt) = candidate_pt {
                    let candidate_vec = Vector4::new(t, pt.x, pt.y, pt.z);
                    Some(candidate_vec)
                } else {
                    None
                }
            };

            if let Some(bubble_new) = new_bubble {
                volume_false_vacuum_history.push(volume_remaining);

                bubbles_interior.spacetime.push(bubble_new);
                let dummy_interior = Bubbles::new(vec![bubble_new]);
                let exterior_bubbles =
                    lattice.generate_bubbles_exterior(&dummy_interior, boundary_condition);
                let exterior_spacetime_ref = &exterior_bubbles.spacetime;

                self.update_first_collision_times(&bubble_new);
                for exterior_bubble in exterior_spacetime_ref {
                    self.update_first_collision_times(exterior_bubble);
                }

                bubbles_exterior.spacetime.extend(exterior_spacetime_ref);
            }
        }

        let lattice_bubbles = LatticeBubbles::new(
            bubbles_interior.to_array2(),
            Some(bubbles_exterior.to_array2()),
            lattice.clone(),
        )?;
        let time_history =
            Array1::from_vec(time_history[..volume_false_vacuum_history.len()].to_vec());
        let volume_false_vacuum_history = Array1::from_vec(volume_false_vacuum_history);

        Ok(FixedRateNucleationResult {
            lattice_bubbles,
            time_history,
            volume_false_vacuum_history,
        })
    }

    pub fn nucleate<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        boundary_condition: BoundaryConditions,
        method: FixedRateNucleationMethod,
    ) -> Result<FixedRateNucleationResult<L>, LatticeBubblesError> {
        match method {
            FixedRateNucleationMethod::FixedProbabilityTimeStepping { d_p0 } => {
                return self.nucleate_fixed_probability_time_stepping(
                    lattice,
                    boundary_condition,
                    d_p0,
                );
            },
            FixedRateNucleationMethod::FixedProbabilitysDistribution => {
                return self.nucleate_fixed_probability_distribution(lattice, boundary_condition);
            },
        }
    }
}

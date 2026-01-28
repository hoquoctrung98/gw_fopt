// TODO:
// interesting references to compute volume of union of bubbles
//   + https://inria.hal.science/inria-00409374v1/document
//   + https://www.aei.tuke.sk/papers/2008/4/06_Dzurina.pdf

use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

use super::{GeneralLatticeProperties, NucleationStrategy};
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

#[derive(Clone, Debug)]
pub enum VolumeRemainingMethod {
    Approximation,
    MonteCarlo { n_points: usize },
}

#[derive(Debug)]
pub struct FixedRateNucleation {
    pub beta: f64,
    pub gamma0: f64,
    pub t0: f64,
    pub d_p0: f64,
    pub seed: Option<u64>,
    pub volume_method: VolumeRemainingMethod,
    pub max_time_steps: usize,
    pub volume_remaining_fraction_cutoff: f64,
    rng: StdRng,
    mc_points: Option<Vec<Point3<f64>>>,
    first_collision_times: Option<Vec<f64>>,
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
        max_time_steps: Option<usize>,
        volume_remaining_cutoff: Option<f64>,
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

        let max_attempts = max_time_steps.unwrap_or(1_000_000);
        let volume_remaining_cutoff = volume_remaining_cutoff.unwrap_or(0.66);

        Ok(Self {
            beta,
            gamma0,
            t0,
            d_p0,
            seed,
            volume_method,
            max_time_steps: max_attempts,
            volume_remaining_fraction_cutoff: volume_remaining_cutoff,
            rng,
            mc_points: None,
            first_collision_times: None,
            time_history: Vec::new(),
            volume_remaining_history: Vec::new(),
        })
    }

    fn update_first_collision_times(&mut self, new_bubble: &Vector4<f64>) {
        if self.mc_points.is_none() {
            return;
        }

        let points = self.mc_points.as_ref().unwrap();
        let fct = self.first_collision_times.as_mut().unwrap();
        let t_n = new_bubble[0];
        let x_n = new_bubble[1];
        let y_n = new_bubble[2];
        let z_n = new_bubble[3];

        for (i, pt) in points.iter().enumerate() {
            let dx = pt.x - x_n;
            let dy = pt.y - y_n;
            let dz = pt.z - z_n;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let t_collide = t_n + dist;
            if t_collide < fct[i] {
                fct[i] = t_collide;
            }
        }
    }

    fn volume_remaining<L: GeneralLatticeProperties>(
        &self,
        lattice: &L,
        t: f64,
        bubbles_interior: &Bubbles,
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
                volume_lattice * (-volume_bubbles / volume_lattice).exp()
            },
            VolumeRemainingMethod::MonteCarlo { n_points } => {
                if *n_points == 0 {
                    return lattice.volume();
                }

                let fct = self.first_collision_times.as_ref().unwrap();
                let count_valid = fct.iter().filter(|&&t_collide| t_collide > t).count();
                let fraction = count_valid as f64 / *n_points as f64;
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

        while accepted.len() < n_points && attempts < self.max_time_steps {
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
    ) -> Result<LatticeBubbles<L>, LatticeBubblesError> {
        let volume_lattice = lattice.volume();
        let volume_cutoff = self.volume_remaining_fraction_cutoff * volume_lattice;

        let mut t = self.t0;

        let mut bubbles_interior = Bubbles::new(Vec::new());
        let mut bubbles_exterior = Bubbles::new(Vec::new());

        self.time_history.clear();
        self.volume_remaining_history.clear();

        // Initialize MC state
        if matches!(self.volume_method, VolumeRemainingMethod::MonteCarlo { .. }) {
            if let VolumeRemainingMethod::MonteCarlo { n_points } = self.volume_method {
                if n_points > 0 {
                    let points = lattice.sample_points(n_points, &mut self.rng);
                    self.mc_points = Some(points);
                    self.first_collision_times = Some(vec![f64::INFINITY; n_points]);
                }
            }
        }

        for _ in 0..self.max_time_steps {
            let volume_remaining = self.volume_remaining(lattice, t, &bubbles_interior);

            if volume_remaining < volume_cutoff {
                break;
            }

            let exponent = self.beta * (t - self.t0);
            let gamma_t = self.gamma0 * exponent.exp();

            let x: f64 = self.rng.random();
            if x <= self.d_p0 {
                let new_bubble =
                    if matches!(self.volume_method, VolumeRemainingMethod::MonteCarlo { .. })
                        && self.mc_points.is_some()
                    {
                        let first_collision_times = self.first_collision_times.as_ref().unwrap();
                        let points = self.mc_points.as_ref().unwrap();

                        let mut candidate_pt = None;
                        for i in 0..points.len() {
                            if first_collision_times[i] > t {
                                candidate_pt = Some(points[i]);
                                break;
                            }
                        }

                        if let Some(pt) = candidate_pt {
                            let candidate_vec = Vector4::new(t, pt.x, pt.y, pt.z);
                            Some(candidate_vec)
                        } else {
                            None
                        }
                    } else {
                        if let Some(new_bubbles) = self.sample_points_outside_bubbles(
                            lattice,
                            t,
                            &bubbles_interior,
                            &bubbles_exterior,
                            1,
                        ) {
                            new_bubbles.spacetime.into_iter().next()
                        } else {
                            None
                        }
                    };

                if let Some(bubble_new) = new_bubble {
                    self.time_history.push(t);
                    self.volume_remaining_history.push(volume_remaining);

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

            let dt = (self.d_p0 / gamma_t) * (volume_lattice / volume_remaining);
            t += dt;
        }

        let lattice_bubbles = LatticeBubbles::new(
            bubbles_interior.to_array2(),
            Some(bubbles_exterior.to_array2()),
            lattice.clone(),
        )?;
        Ok(lattice_bubbles)
    }
}

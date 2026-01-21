// TODO:
// interesting references to compute volume of union of bubbles
//   + https://inria.hal.science/inria-00409374v1/document
//   + https://www.aei.tuke.sk/papers/2008/4/06_Dzurina.pdf

use nalgebra::{Point3, Vector4};
use nalgebra_spacetime::Lorentzian;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, random};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::{GeneralLatticeProperties, NucleationStrategy};
use crate::many_bubbles::bubbles::Bubbles;
use crate::many_bubbles::lattice::BoundaryConditions;
use crate::many_bubbles::lattice_bubbles::{LatticeBubbles, LatticeBubblesError};

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
    points_outside_bubbles: Option<Vec<Point3<f64>>>,
    pub time_history: Vec<f64>,
    pub volume_remaining_history: Vec<f64>,
    thread_pool: ThreadPool,
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
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let default_num_threads = rayon::current_num_threads();
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(default_num_threads)
            .build()
            .unwrap();
        let max_attempts = max_time_steps.unwrap_or(1_000_000);
        let volume_remaining_cutoff = volume_remaining_cutoff.unwrap_or(0.66);

        Self {
            beta,
            gamma0,
            t0,
            d_p0,
            seed,
            volume_method,
            max_time_steps: max_attempts,
            volume_remaining_fraction_cutoff: volume_remaining_cutoff,
            rng,
            points_outside_bubbles: None,
            time_history: Vec::new(),
            volume_remaining_history: Vec::new(),
            thread_pool,
        }
    }

    pub fn set_num_threads(&mut self, num_threads: usize) -> Result<(), LatticeBubblesError> {
        self.thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| LatticeBubblesError::Other(format!("ThreadPool build error: {}", e)))?;
        Ok(())
    }

    fn volume_remaining<L: GeneralLatticeProperties>(
        &mut self,
        lattice: &L,
        t: f64,
        bubbles_interior: &Bubbles,
        bubbles_exterior: &Bubbles,
    ) -> f64 {
        match &self.volume_method {
            // volume_remaining = volume_lattice * exp(volume_bubbles / volume_lattice)
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
                // The exponential factor partially taken into account the overlapse of bubbles
                return volume_lattice * (-volume_bubbles / volume_lattice).exp();
            },
            // volume_remaining = volume_lattice * (n_points_outside_bubbles / n_points)
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

                let valid_points = self.thread_pool.install(|| {
                    points_outside_bubbles
                        .par_iter()
                        .filter_map(|pt| {
                            let candidate = Vector4::new(t, pt.x, pt.y, pt.z);
                            for &bubble in &all_bubbles {
                                let delta = candidate - bubble;
                                if delta.scalar(&delta) < 0.0 {
                                    return None;
                                }
                            }
                            Some(*pt)
                        })
                        .collect::<Vec<Point3<f64>>>()
                });

                *points_outside_bubbles = valid_points;

                let fraction = points_outside_bubbles.len() as f64 / *n_points as f64;
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

        for _ in 0..self.max_time_steps {
            let volume_remaining =
                self.volume_remaining(lattice, t, &bubbles_interior, &bubbles_exterior);

            if volume_remaining < volume_cutoff {
                break;
            }

            let exponent = self.beta * (t - self.t0);
            let gamma_t = self.gamma0 * exponent.exp();

            let x: f64 = self.rng.random();
            if x <= self.d_p0 {
                let new_bubble =
                    if matches!(self.volume_method, VolumeRemainingMethod::MonteCarlo { .. })
                        && self.points_outside_bubbles.is_some()
                        && !self.points_outside_bubbles.as_ref().unwrap().is_empty()
                    {
                        let pt = self.points_outside_bubbles.as_mut().unwrap().pop().unwrap();
                        Some(Vector4::new(t, pt.x, pt.y, pt.z))
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
                    bubbles_exterior
                        .spacetime
                        .extend(exterior_bubbles.spacetime);
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

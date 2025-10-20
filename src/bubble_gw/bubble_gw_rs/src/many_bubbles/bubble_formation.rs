// use ndarray::Array2;
// use rand::random;
// use rand::rngs::StdRng;
// use rand::{Rng, SeedableRng};
// use rayon::prelude::*;
// use std::collections::HashSet;
// use std::hash::{Hash, Hasher};
//
// /// Enum representing the reason why the simulation stopped.
// #[derive(Debug, Clone, PartialEq)]
// pub enum SimulationEndStatus {
//     /// The simulation stopped because the time exceeded the specified `t_max`.
//     TimeLimitReached {
//         t_end: f64,
//         volume_remaining_fraction: f64,
//         time_iteration: usize,
//     },
//     /// The simulation stopped because the remaining volume fraction fell below `min_volume_remaining_fraction`.
//     VolumeFractionReached {
//         t_end: f64,
//         volume_remaining_fraction: f64,
//         time_iteration: usize,
//     },
//     /// The simulation stopped because the maximum number of iterations was reached.
//     MaxTimeIterationsReached {
//         t_end: f64,
//         volume_remaining_fraction: f64,
//         time_iteration: usize,
//     },
//     /// The simulation stopped because the remaining volume is below the threshold (1e-6) or no valid points remain.
//     VolumeDepleted {
//         t_end: f64,
//         volume_remaining_fraction: f64,
//         time_iteration: usize,
//     },
// }
//
// /// A helper struct for hashing quantized 3D coordinates to ensure uniqueness.
// ///
// /// This struct converts floating-point coordinates to integers by scaling and rounding,
// /// enabling efficient storage and comparison in a `HashSet`. It is used primarily for
// /// handling periodic boundary conditions in Cartesian lattices.
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// struct QuantizedPoint(i64, i64, i64);
//
// impl QuantizedPoint {
//     /// Creates a new `QuantizedPoint` from floating-point coordinates.
//     ///
//     /// # Arguments
//     ///
//     /// * `coords` - A tuple of three `f64` values representing the (x, y, z) coordinates.
//     ///
//     /// # Returns
//     ///
//     /// A `QuantizedPoint` with coordinates scaled by \(10^{10}\) and rounded to integers.
//     fn new(coords: (f64, f64, f64)) -> Self {
//         let quantize = |x: f64| (x * 1e10).round() as i64;
//         QuantizedPoint(quantize(coords.0), quantize(coords.1), quantize(coords.2))
//     }
// }
//
// impl Hash for QuantizedPoint {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.0.hash(state);
//         self.1.hash(state);
//         self.2.hash(state);
//     }
// }
//
// /// Enum representing the type of lattice.
// #[derive(Debug, Clone, PartialEq)]
// pub enum LatticeType {
//     Cartesian,
//     Sphere,
// }
//
// /// Represents the simulation domain, either a Cartesian box or a sphere.
// ///
// /// The lattice defines the spatial boundaries and grid resolution for the bubble
// /// formation simulation. It supports volume calculations and grid point generation.
// #[derive(Debug, Clone)]
// pub struct Lattice {
//     /// The type of lattice (Cartesian or Sphere).
//     pub lattice_type: LatticeType,
//     /// The dimensions of the lattice: `[lx, ly, lz]` for Cartesian, `[r, 0.0, 0.0]` for Sphere.
//     pub sizes: [f64; 3],
//     /// The number of grid points along each dimension.
//     pub n_grid: usize,
// }
//
// impl Lattice {
//     /// Creates a new lattice with the specified type, sizes, and grid resolution.
//     ///
//     /// # Arguments
//     ///
//     /// * `lattice_type` - A string specifying the lattice type ("cartesian" or "sphere").
//     /// * `sizes` - A vector of dimensions: `[lx, ly, lz]` for Cartesian, `[r]` for Sphere.
//     /// * `n` - The number of grid points along each dimension.
//     ///
//     /// # Returns
//     ///
//     /// * `Ok(Lattice)` - A new `Lattice` instance.
//     /// * `Err(String)` - An error message if the lattice type is invalid or sizes are incorrect.
//     pub fn new(lattice_type: &str, sizes: Vec<f64>, n: usize) -> Result<Self, String> {
//         let lattice_type = match lattice_type.to_lowercase().as_str() {
//             "cartesian" => {
//                 if sizes.len() != 3 {
//                     return Err("For cartesian lattice, sizes must have length 3".to_string());
//                 }
//                 LatticeType::Cartesian
//             }
//             "sphere" => {
//                 if sizes.len() != 1 {
//                     return Err("For sphere lattice, sizes must have length 1".to_string());
//                 }
//                 LatticeType::Sphere
//             }
//             _ => return Err("Invalid lattice_type".to_string()),
//         };
//         let sizes_array = match sizes.len() {
//             1 => [sizes[0], 0.0, 0.0],
//             3 => [sizes[0], sizes[1], sizes[2]],
//             _ => unreachable!(),
//         };
//         Ok(Lattice {
//             lattice_type,
//             sizes: sizes_array,
//             n_grid: n,
//         })
//     }
//
//     /// Computes the total volume of the lattice.
//     ///
//     /// # Returns
//     ///
//     /// * For Cartesian: \( lx \times ly \times lz \).
//     /// * For Sphere: \( \frac{4}{3} \pi r^3 \).
//     pub fn get_volume(&self) -> f64 {
//         match self.lattice_type {
//             LatticeType::Cartesian => self.sizes[0] * self.sizes[1] * self.sizes[2],
//             LatticeType::Sphere => (4.0 / 3.0) * std::f64::consts::PI * self.sizes[0].powi(3),
//         }
//     }
//
//     /// Generates a grid of points within the lattice.
//     ///
//     /// # Returns
//     ///
//     /// An `Array2<f64>` with shape `(N, 3)`, where each row is a point `[x, y, z]`.
//     /// * For Cartesian: A uniform \( n \times n \times n \) grid spanning `[0, lx] × [0, ly] × [0, lz]`.
//     /// * For Sphere: Points within a cube \([-r, r]^3\), filtered to lie within the sphere (\( x^2 + y^2 + z^2 \leq r^2 \)).
//     pub fn generate_grid(&self) -> Array2<f64> {
//         match self.lattice_type {
//             LatticeType::Cartesian => {
//                 let lx = self.sizes[0];
//                 let ly = self.sizes[1];
//                 let lz = self.sizes[2];
//                 let n = self.n_grid;
//                 let x: Vec<f64> = (0..n).map(|i| i as f64 * lx / (n - 1) as f64).collect();
//                 let y: Vec<f64> = (0..n).map(|i| i as f64 * ly / (n - 1) as f64).collect();
//                 let z: Vec<f64> = (0..n).map(|i| i as f64 * lz / (n - 1) as f64).collect();
//                 let mut grid_points = Vec::with_capacity(n * n * n * 3);
//                 for i in 0..n {
//                     for j in 0..n {
//                         for k in 0..n {
//                             grid_points.extend_from_slice(&[x[i], y[j], z[k]]);
//                         }
//                     }
//                 }
//                 Array2::from_shape_vec((n * n * n, 3), grid_points).unwrap()
//             }
//             LatticeType::Sphere => {
//                 let r = self.sizes[0];
//                 let n = self.n_grid;
//                 let x: Vec<f64> = (0..n)
//                     .map(|i| -r + 2.0 * i as f64 * r / (n - 1) as f64)
//                     .collect();
//                 let mut grid_points = Vec::new();
//                 for i in 0..n {
//                     for j in 0..n {
//                         for k in 0..n {
//                             let point = [x[i], x[j], x[k]];
//                             if point.iter().map(|&v| v * v).sum::<f64>() <= r * r {
//                                 grid_points.extend_from_slice(&[x[i], x[j], x[k]]);
//                             }
//                         }
//                     }
//                 }
//                 Array2::from_shape_vec((grid_points.len() / 3, 3), grid_points).unwrap()
//             }
//         }
//     }
//
//     /// Returns the bounds of the lattice for each dimension.
//     ///
//     /// # Returns
//     ///
//     /// A vector of tuples `(min, max)` for each dimension:
//     /// * For Cartesian: `[(0, lx), (0, ly), (0, lz)]`.
//     /// * For Sphere: `[(-r, r), (-r, r), (-r, r)]`.
//     pub fn get_lattice_bounds(&self) -> Vec<(f64, f64)> {
//         match self.lattice_type {
//             LatticeType::Cartesian => vec![
//                 (0.0, self.sizes[0]),
//                 (0.0, self.sizes[1]),
//                 (0.0, self.sizes[2]),
//             ],
//             LatticeType::Sphere => vec![(-self.sizes[0], self.sizes[0]); 3],
//         }
//     }
// }
//
// /// Trait defining the nucleation strategy.
// pub trait NucleationStrategy: Clone {
//     fn nucleate(
//         &mut self,
//         t: &mut f64,
//         valid_points: &Array2<f64>,
//         rng: &mut StdRng,
//         vw: f64,
//         volume_remaining: f64,
//         existing_bubbles: Option<&[Bubble]>,
//     ) -> Option<Array2<f64>>;
//     fn initial_time(&self) -> f64;
//     fn validate_schedule(&self, lattice: &Lattice, vw: f64) -> Result<(), String>;
// }
//
// /// Represents a single bubble.
// #[derive(Clone, Copy)]
// pub struct Bubble {
//     center: [f64; 3],
//     time: f64,
// }
//
// /// Holds simulation parameters.
// #[derive(Clone)]
// pub struct SimParams {
//     vw: f64,
//     volume_total: f64,
//     volume_remaining: f64,
//     last_update_time: f64,
// }
//
// /// Implements a Poisson process for random bubble nucleation.
// #[derive(Debug, Clone, PartialEq)]
// pub struct PoissonNucleation {
//     /// Base nucleation rate.
//     gamma0: f64,
//     /// Exponential growth factor for the nucleation rate.
//     beta: f64,
//     /// Reference time for the nucleation rate.
//     t0: f64,
//     /// Probability parameter controlling nucleation likelihood.
//     d_p0: f64,
// }
//
// impl PoissonNucleation {
//     /// Creates a new `PoissonNucleation` instance.
//     ///
//     /// # Arguments
//     ///
//     /// * `gamma0` - Base nucleation rate (must be positive).
//     /// * `beta` - Exponential growth factor.
//     /// * `t0` - Reference time.
//     /// * `d_p0` - Probability parameter.
//     ///
//     /// # Returns
//     ///
//     /// * `Ok(PoissonNucleation)` - A new instance.
//     /// * `Err(String)` - If `gamma0 <= 0`.
//     pub fn new(gamma0: f64, beta: f64, t0: f64, d_p0: f64) -> Result<Self, String> {
//         if gamma0 <= 0.0 {
//             return Err("Gamma0 must be positive".to_string());
//         }
//         Ok(PoissonNucleation {
//             gamma0,
//             beta,
//             t0,
//             d_p0,
//         })
//     }
// }
//
// impl NucleationStrategy for PoissonNucleation {
//     fn nucleate(
//         &mut self,
//         t: &mut f64,
//         valid_points: &Array2<f64>,
//         rng: &mut StdRng,
//         _vw: f64,
//         volume_remaining: f64,
//         _existing_bubbles: Option<&[Bubble]>,
//     ) -> Option<Array2<f64>> {
//         let gamma_t = self.gamma0 * (self.beta * (*t - self.t0)).exp() * volume_remaining;
//         let dt = self.d_p0 / gamma_t;
//         *t += dt;
//         if valid_points.is_empty() || volume_remaining < 0. {
//             return None;
//         }
//         let x: f64 = rng.random();
//         if x <= self.d_p0 {
//             let n = valid_points.nrows();
//             let idx = rand::seq::index::sample(rng, n, 1)
//                 .into_vec()
//                 .get(0)
//                 .copied()
//                 .unwrap();
//             let point = valid_points.row(idx).to_vec();
//             Some(Array2::from_shape_vec((1, 3), point).unwrap())
//         } else {
//             None
//         }
//     }
//
//     fn initial_time(&self) -> f64 {
//         self.t0
//     }
//
//     fn validate_schedule(&self, _lattice: &Lattice, _vw: f64) -> Result<(), String> {
//         Ok(())
//     }
// }
//
// /// Implements manual nucleation.
// #[derive(Debug, Clone, PartialEq)]
// pub struct ManualNucleation {
//     /// A schedule of nucleation times and corresponding bubble centers.
//     schedule: Vec<(f64, Vec<[f64; 3]>)>,
//     /// Time step for advancing the simulation.
//     dt: f64,
// }
//
// impl ManualNucleation {
//     /// Creates a new `ManualNucleation` instance.
//     ///
//     /// # Arguments
//     ///
//     /// * `schedule` - A vector of tuples `(time, centers)`, where `centers` is a list of `[x, y, z]` coordinates.
//     /// * `dt` - The time step for advancing the simulation (must be positive).
//     ///
//     /// # Returns
//     ///
//     /// * `Ok(ManualNucleation)` - A new instance.
//     /// * `Err(String)` - If the schedule is empty, `dt <= 0`, or any nucleation time is negative.
//     pub fn new(schedule: Vec<(f64, Vec<[f64; 3]>)>, dt: f64) -> Result<Self, String> {
//         if schedule.is_empty() {
//             return Err("Manual nucleation requires a non-empty schedule".to_string());
//         }
//         if dt <= 0.0 {
//             return Err("Time step (dt) must be positive".to_string());
//         }
//         for (t, centers) in &schedule {
//             if *t < 0.0 {
//                 return Err("Nucleation times must be non-negative".to_string());
//             }
//             if centers.is_empty() {
//                 return Err("Each nucleation time must have at least one center".to_string());
//             }
//         }
//         Ok(ManualNucleation { schedule, dt })
//     }
//
//     fn is_point_valid(
//         &self,
//         point: [f64; 3],
//         t: f64,
//         existing_bubbles: &[([f64; 3], f64)],
//         vw: f64,
//     ) -> bool {
//         for &(center, tn) in existing_bubbles {
//             let radius = vw * (t - tn);
//             if radius > 0.0 {
//                 let dist: f64 = ((point[0] - center[0]).powi(2)
//                     + (point[1] - center[1]).powi(2)
//                     + (point[2] - center[2]).powi(2))
//                 .sqrt();
//                 if dist <= radius {
//                     return false;
//                 }
//             }
//         }
//         true
//     }
// }
//
// impl NucleationStrategy for ManualNucleation {
//     fn nucleate(
//         &mut self,
//         t: &mut f64,
//         _valid_points: &Array2<f64>,
//         _rng: &mut StdRng,
//         vw: f64,
//         _volume_remaining: f64,
//         existing_bubbles: Option<&[Bubble]>,
//     ) -> Option<Array2<f64>> {
//         let mut new_centers: Vec<[f64; 3]> = Vec::new();
//         let existing_bubbles = existing_bubbles.unwrap_or(&[]);
//         let existing_bubble_vec: Vec<([f64; 3], f64)> = existing_bubbles
//             .iter()
//             .map(|b| (b.center, b.time))
//             .collect();
//         for (nucleation_time, centers) in &self.schedule {
//             if (*nucleation_time - *t).abs() < 1e-10 {
//                 for center in centers {
//                     if self.is_point_valid(*center, *t, &existing_bubble_vec, vw) {
//                         new_centers.push(*center);
//                     }
//                 }
//             }
//         }
//         *t += self.dt;
//         if new_centers.is_empty() {
//             None
//         } else {
//             Some(
//                 Array2::from_shape_vec(
//                     (new_centers.len(), 3),
//                     new_centers.into_iter().flatten().collect(),
//                 )
//                 .unwrap(),
//             )
//         }
//     }
//
//     fn initial_time(&self) -> f64 {
//         self.schedule
//             .iter()
//             .map(|(t, _)| *t)
//             .min_by(|a, b| a.partial_cmp(b).unwrap())
//             .unwrap_or(0.0)
//     }
//
//     fn validate_schedule(&self, lattice: &Lattice, vw: f64) -> Result<(), String> {
//         let bounds = lattice.get_lattice_bounds();
//         for (t, centers) in &self.schedule {
//             for center in centers {
//                 for (i, &(min_bound, max_bound)) in bounds.iter().enumerate() {
//                     if center[i] < min_bound || center[i] > max_bound {
//                         return Err(format!(
//                             "Bubble center {:?} at time {} is outside lattice bounds [{}, {}] for dimension {}",
//                             center, t, min_bound, max_bound, i
//                         ));
//                     }
//                 }
//             }
//         }
//         let mut sorted_times: Vec<f64> = self.schedule.iter().map(|(t, _)| *t).collect();
//         sorted_times.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
//         let mut existing_bubbles: Vec<([f64; 3], f64)> = Vec::new();
//         for t in sorted_times {
//             let centers = self
//                 .schedule
//                 .iter()
//                 .find(|(time, _)| (*time - t).abs() < 1e-10)
//                 .map(|(_, centers)| centers)
//                 .unwrap();
//             let mut valid_centers = Vec::new();
//             for center in centers {
//                 if self.is_point_valid(*center, t, &existing_bubbles, vw) {
//                     valid_centers.push(*center);
//                 } else {
//                     return Err(format!(
//                         "Bubble at {:?} at time {} overlaps with earlier bubbles",
//                         center, t
//                     ));
//                 }
//             }
//             existing_bubbles.extend(valid_centers.into_iter().map(|c| (c, t)));
//         }
//         Ok(())
//     }
// }
//
// /// The main simulator for bubble nucleation and growth.
// #[derive(Clone)]
// pub struct BubbleFormationSimulator<S: NucleationStrategy> {
//     /// Simulation parameters.
//     params: SimParams,
//     /// The lattice defining the simulation domain.
//     lattice: Lattice,
//     /// The nucleation strategy (Poisson or Manual).
//     strategy: S,
//     /// A list of all nucleated bubbles.
//     bubbles: Vec<Bubble>,
//     /// The grid of points in the lattice.
//     grid: Array2<f64>,
//     /// Indices of grid points not yet consumed by bubbles.
//     outside_points: Vec<usize>,
//     /// Random number generator for reproducibility.
//     rng: StdRng,
//     /// History of volume remaining over time as `(dt, volume_remaining)` tuples.
//     volume_history: Vec<(f64, f64)>,
//     /// The status indicating why the simulation stopped.
//     pub end_status: Option<SimulationEndStatus>,
// }
//
// impl<S: NucleationStrategy> BubbleFormationSimulator<S> {
//     /// Creates a new bubble formation simulator.
//     pub fn new(lattice: Lattice, vw: f64, strategy: S, seed: Option<u64>) -> Result<Self, String> {
//         if vw <= 0.0 || vw > 1.0 {
//             return Err("Wall velocity must be in (0, 1]".to_string());
//         }
//         let grid = lattice.generate_grid();
//         let volume_total = lattice.get_volume();
//         let outside_points = (0..grid.nrows()).collect::<Vec<usize>>();
//         let rng = match seed {
//             Some(seed_value) => StdRng::seed_from_u64(seed_value),
//             None => StdRng::seed_from_u64(random::<u64>()),
//         };
//
//         let mut simulator = BubbleFormationSimulator {
//             params: SimParams {
//                 vw,
//                 volume_total,
//                 volume_remaining: volume_total,
//                 last_update_time: 0.0,
//             },
//             lattice,
//             strategy,
//             bubbles: Vec::new(),
//             grid,
//             outside_points,
//             rng,
//             volume_history: Vec::new(),
//             end_status: None,
//         };
//
//         if let LatticeType::Sphere = simulator.lattice.lattice_type {
//             let r = simulator.lattice.sizes[0];
//             simulator.outside_points.retain(|&i| {
//                 let row = simulator.grid.row(i);
//                 let d = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
//                 d <= r * r
//             });
//         }
//
//         simulator.validate()?;
//
//         Ok(simulator)
//     }
//
//     /// Sets the seed for the random number generator.
//     ///
//     /// # Arguments
//     ///
//     /// * `seed` - The seed value for reproducibility.
//     pub fn set_seed(&mut self, seed: u64) {
//         self.rng = StdRng::seed_from_u64(seed);
//     }
//
//     /// Validates the simulation setup, ensuring manual nucleation centers do not overlap.
//     ///
//     /// # Returns
//     ///
//     /// * `Ok(())` - If the setup is valid.
//     /// * `Err(String)` - If any manual nucleation centers overlap with existing bubbles.
//     fn validate(&self) -> Result<(), String> {
//         self.strategy
//             .validate_schedule(&self.lattice, self.params.vw)
//     }
//
//     fn update_outside_mask(&mut self, new_bubbles: &Vec<([f64; 3], f64)>, t: f64) {
//         if new_bubbles.is_empty() {
//             return;
//         }
//
//         let points_to_remove: Vec<usize> = self
//             .outside_points
//             .par_iter()
//             .filter_map(|&i| {
//                 let px = self.grid[[i, 0]];
//                 let py = self.grid[[i, 1]];
//                 let pz = self.grid[[i, 2]];
//                 for &(center, tn) in new_bubbles {
//                     let radius = (self.params.vw * (t - tn)).max(0.0);
//                     let dist: f64 = ((px - center[0]).powi(2)
//                         + (py - center[1]).powi(2)
//                         + (pz - center[2]).powi(2))
//                     .sqrt();
//                     if dist <= radius {
//                         return Some(i);
//                     }
//                 }
//                 None
//             })
//             .collect();
//
//         self.outside_points
//             .retain(|i| !points_to_remove.contains(i));
//
//         self.params.last_update_time = t;
//     }
//
//     /// Runs the bubble formation simulation until the specified termination conditions.
//     ///
//     /// # Arguments
//     ///
//     /// * `t_max` - An optional maximum simulation time.
//     /// * `min_volume_remaining_fraction` - An optional minimum fraction of the total volume.
//     ///   Defaults to 0.0 if not provided. The simulation stops if the remaining volume falls below this fraction.
//     /// * `max_time_iterations` - An optional maximum number of time iterations for Poisson nucleation.
//     ///
//     /// The simulation's end status is stored in the `end_status` field.
//     pub fn run_simulation(
//         &mut self,
//         t_max: Option<f64>,
//         min_volume_remaining_fraction: Option<f64>,
//         max_time_iterations: Option<usize>,
//     ) {
//         self.volume_history.clear();
//         self.end_status = None;
//         let min_frac = min_volume_remaining_fraction.unwrap_or(0.0);
//         let mut t = self.strategy.initial_time();
//         let mut iteration_count = 0;
//
//         loop {
//             // Termination conditions
//             if t_max.map_or(false, |t_max_val| t >= t_max_val) {
//                 self.end_status = Some(SimulationEndStatus::TimeLimitReached {
//                     t_end: t,
//                     volume_remaining_fraction: self.params.volume_remaining
//                         / self.params.volume_total,
//                     time_iteration: iteration_count,
//                 });
//                 return;
//             }
//             if self.params.volume_remaining < min_frac * self.params.volume_total {
//                 self.end_status = Some(SimulationEndStatus::VolumeFractionReached {
//                     t_end: t,
//                     volume_remaining_fraction: self.params.volume_remaining
//                         / self.params.volume_total,
//                     time_iteration: iteration_count,
//                 });
//                 return;
//             }
//             if max_time_iterations.map_or(false, |max_iter| iteration_count >= max_iter) {
//                 self.end_status = Some(SimulationEndStatus::MaxTimeIterationsReached {
//                     t_end: t,
//                     volume_remaining_fraction: self.params.volume_remaining
//                         / self.params.volume_total,
//                     time_iteration: iteration_count,
//                 });
//                 return;
//             }
//             if self.params.volume_remaining < 1e-6 || self.get_valid_points(t).is_empty() {
//                 self.end_status = Some(SimulationEndStatus::VolumeDepleted {
//                     t_end: t,
//                     volume_remaining_fraction: self.params.volume_remaining
//                         / self.params.volume_total,
//                     time_iteration: iteration_count,
//                 });
//                 return;
//             }
//
//             iteration_count += 1;
//             let initial_t = t;
//             let valid_points = self.get_valid_points(t);
//             let existing_bubbles = Some(self.bubbles.as_slice());
//             let new_bubbles: Vec<([f64; 3], f64)> = if let Some(new_centers) =
//                 self.strategy.nucleate(
//                     &mut t,
//                     &valid_points,
//                     &mut self.rng,
//                     self.params.vw,
//                     self.params.volume_remaining,
//                     existing_bubbles,
//                 ) {
//                 let new_t = t; // Time after nucleation
//                 self.update_outside_mask(
//                     &new_centers
//                         .outer_iter()
//                         .map(|row| ([row[0], row[1], row[2]], new_t))
//                         .collect(),
//                     new_t,
//                 );
//                 new_centers
//                     .outer_iter()
//                     .map(|row| ([row[0], row[1], row[2]], new_t))
//                     .collect()
//             } else {
//                 Vec::new()
//             };
//             for (center, tn) in new_bubbles {
//                 self.bubbles.push(Bubble { center, time: tn });
//             }
//             let valid_points_updated = self.get_valid_points(t);
//             self.params.volume_remaining = self.update_volume_remaining(&valid_points_updated);
//             self.volume_history
//                 .push((t - initial_t, self.params.volume_remaining));
//         }
//     }
//
//     pub fn volume_remaining(&self) -> f64 {
//         self.params.volume_remaining
//     }
//
//     /// Returns the valid grid points not yet consumed by bubbles at the specified time.
//     ///
//     /// # Arguments
//     ///
//     /// * `t` - An optional time to evaluate valid points. If `None`, uses the last update time.
//     ///
//     /// # Returns
//     ///
//     /// An `Array2<f64>` with shape `(N, 3)`, where each row is a valid point `[x, y, z]`.
//     pub fn get_valid_points(&mut self, t: f64) -> Array2<f64> {
//         let last_t = self.params.last_update_time;
//         if t > last_t {
//             let points_to_remove: Vec<usize> = self
//                 .outside_points
//                 .par_iter()
//                 .filter_map(|&i| {
//                     let px = self.grid[[i, 0]];
//                     let py = self.grid[[i, 1]];
//                     let pz = self.grid[[i, 2]];
//                     for bubble in &self.bubbles {
//                         let tn = bubble.time;
//                         if tn > t {
//                             continue;
//                         }
//                         let radius = self.params.vw * (t - tn).max(0.0);
//                         let dist: f64 = ((px - bubble.center[0]).powi(2)
//                             + (py - bubble.center[1]).powi(2)
//                             + (pz - bubble.center[2]).powi(2))
//                         .sqrt();
//                         if dist <= radius {
//                             return Some(i);
//                         }
//                     }
//                     None
//                 })
//                 .collect();
//
//             self.outside_points
//                 .retain(|i| !points_to_remove.contains(i));
//
//             self.params.last_update_time = t;
//         }
//
//         Array2::from_shape_vec(
//             (self.outside_points.len(), 3),
//             self.outside_points
//                 .iter()
//                 .flat_map(|&i| vec![self.grid[[i, 0]], self.grid[[i, 1]], self.grid[[i, 2]]])
//                 .collect(),
//         )
//         .unwrap()
//     }
//
//     /// Updates and returns the remaining volume based on valid points.
//     ///
//     /// The remaining volume is calculated as:
//     /// \( V_{\text{remaining}} = V_{\text{total}} \times \frac{\text{number of valid points}}{\text{total grid points}} \).
//     ///
//     /// # Arguments
//     ///
//     /// * `valid_points` - The current valid points as an `Array2<f64>`.
//     ///
//     /// # Returns
//     ///
//     /// The updated remaining volume.
//     pub fn update_volume_remaining(&mut self, valid_points: &Array2<f64>) -> f64 {
//         let fraction_remaining = valid_points.nrows() as f64 / self.grid.nrows() as f64;
//         self.params.volume_remaining = self.params.volume_total * fraction_remaining;
//         self.params.volume_remaining
//     }
//
//     /// Identifies bubbles that intersect the lattice boundaries at the specified time.
//     ///
//     /// # Arguments
//     ///
//     /// * `t` - The simulation time to evaluate.
//     ///
//     /// # Returns
//     ///
//     /// A vector of tuples `(index, nucleation_time)` for bubbles intersecting the boundaries.
//     pub fn get_boundary_intersecting_bubbles(&self, t: f64) -> Vec<(usize, f64)> {
//         let mut boundary_bubbles = Vec::new();
//         for (i, bubble) in self.bubbles.iter().enumerate() {
//             let tn = bubble.time;
//             let radius = (self.params.vw * (t - tn)).max(0.0);
//             if radius <= 0.0 {
//                 continue;
//             }
//             let x = bubble.center[0];
//             let y = bubble.center[1];
//             let z = bubble.center[2];
//             match self.lattice.lattice_type {
//                 LatticeType::Cartesian => {
//                     let lx = self.lattice.sizes[0];
//                     let ly = self.lattice.sizes[1];
//                     let lz = self.lattice.sizes[2];
//                     if x - radius < 0.0
//                         || x + radius > lx
//                         || y - radius < 0.0
//                         || y + radius > ly
//                         || z - radius < 0.0
//                         || z + radius > lz
//                     {
//                         boundary_bubbles.push((i, tn));
//                     }
//                 }
//                 LatticeType::Sphere => {
//                     let r = self.lattice.sizes[0];
//                     let center_dist = (x * x + y * y + z * z).sqrt();
//                     let max_radius = r - center_dist;
//                     if max_radius < 0.0 || radius > max_radius {
//                         boundary_bubbles.push((i, tn));
//                     }
//                 }
//             }
//         }
//         boundary_bubbles
//     }
//
//     /// Generates exterior bubbles for periodic boundary conditions in Cartesian lattices.
//     ///
//     /// # Returns
//     ///
//     /// An `Array2<f64>` with shape `(N, 4)`, where each row is `[time, x, y, z]` for exterior bubbles.
//     /// Returns an empty array for non-Cartesian lattices.
//     pub fn generate_exterior_bubbles(&self) -> Array2<f64> {
//         if self.lattice.lattice_type != LatticeType::Cartesian {
//             return Array2::zeros((0, 4));
//         }
//
//         let lx = self.lattice.sizes[0];
//         let ly = self.lattice.sizes[1];
//         let lz = self.lattice.sizes[2];
//
//         let shifts: [[f64; 3]; 6] = [
//             [lx, 0.0, 0.0],
//             [-lx, 0.0, 0.0],
//             [0.0, ly, 0.0],
//             [0.0, -ly, 0.0],
//             [0.0, 0.0, lz],
//             [0.0, 0.0, -lz],
//         ];
//
//         let mut exterior_bubbles = Vec::new();
//         let mut seen_centers = HashSet::new();
//
//         for bubble in self.bubbles.iter() {
//             let tn = bubble.time;
//             let center = bubble.center;
//
//             for shift in shifts.iter() {
//                 let shifted_center = [
//                     center[0] + shift[0],
//                     center[1] + shift[1],
//                     center[2] + shift[2],
//                 ];
//                 let q_point =
//                     QuantizedPoint::new((shifted_center[0], shifted_center[1], shifted_center[2]));
//
//                 if seen_centers.insert(q_point) {
//                     exterior_bubbles.extend_from_slice(&[
//                         tn,
//                         shifted_center[0],
//                         shifted_center[1],
//                         shifted_center[2],
//                     ]);
//                 }
//             }
//         }
//
//         if exterior_bubbles.is_empty() {
//             Array2::zeros((0, 4))
//         } else {
//             Array2::from_shape_vec((exterior_bubbles.len() / 4, 4), exterior_bubbles).unwrap()
//         }
//     }
//
//     /// Returns the lattice used in the simulation.
//     pub fn lattice(&self) -> &Lattice {
//         &self.lattice
//     }
//
//     /// Returns the bubble wall velocity.
//     pub fn vw(&self) -> f64 {
//         self.params.vw
//     }
//
//     /// Returns the grid of points in the lattice.
//     pub fn grid(&self) -> &Array2<f64> {
//         &self.grid
//     }
//
//     /// Returns the total volume of the lattice.
//     pub fn volume_total(&self) -> f64 {
//         self.params.volume_total
//     }
//
//     /// Returns the center of the bubble at the specified index.
//     ///
//     /// # Arguments
//     ///
//     /// * `idx` - The index of the bubble.
//     ///
//     /// # Returns
//     ///
//     /// The `[x, y, z]` coordinates of the bubble's center.
//     pub fn get_center(&self, idx: usize) -> [f64; 3] {
//         self.bubbles[idx].center
//     }
//
//     /// Returns all bubbles in the simulation.
//     ///
//     /// # Returns
//     ///
//     /// An `Array2<f64>` with shape `(N, 4)`, where each row is `[time, x, y, z]`.
//     pub fn get_bubbles(&self) -> Array2<f64> {
//         let num_bubbles = self.bubbles.len();
//         if num_bubbles == 0 {
//             return Array2::zeros((0, 4));
//         }
//
//         let mut bubbles_array = Array2::zeros((num_bubbles, 4));
//         for (i, bubble) in self.bubbles.iter().enumerate() {
//             bubbles_array[[i, 0]] = bubble.time;
//             bubbles_array[[i, 1]] = bubble.center[0];
//             bubbles_array[[i, 2]] = bubble.center[1];
//             bubbles_array[[i, 3]] = bubble.center[2];
//         }
//         bubbles_array
//     }
//
//     /// Returns the history of volume remaining over time.
//     ///
//     /// # Returns
//     ///
//     /// A vector of `(dt, volume_remaining)` tuples.
//     pub fn get_volume_history(&self) -> Vec<(f64, f64)> {
//         self.volume_history.clone()
//     }
// }

use ndarray::Array2;
use rand::random;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub enum SimulationEndStatus {
    TimeLimitReached {
        t_end: f64,
        volume_remaining_fraction: f64,
        time_iteration: usize,
    },
    VolumeFractionReached {
        t_end: f64,
        volume_remaining_fraction: f64,
        time_iteration: usize,
    },
    MaxTimeIterationsReached {
        t_end: f64,
        volume_remaining_fraction: f64,
        time_iteration: usize,
    },
    VolumeDepleted {
        t_end: f64,
        volume_remaining_fraction: f64,
        time_iteration: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedPoint(i64, i64, i64);

impl QuantizedPoint {
    fn new(coords: (f64, f64, f64)) -> Self {
        let quantize = |x: f64| (x * 1e10).round() as i64;
        QuantizedPoint(quantize(coords.0), quantize(coords.1), quantize(coords.2))
    }
}

impl std::hash::Hash for QuantizedPoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
        self.2.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeType {
    Cartesian,
    Sphere,
}

#[derive(Debug, Clone)]
pub struct Lattice {
    pub lattice_type: LatticeType,
    pub sizes: [f64; 3],
    pub n_grid: usize,
}

impl Lattice {
    pub fn new(lattice_type: &str, sizes: Vec<f64>, n: usize) -> Result<Self, String> {
        let lattice_type = match lattice_type.to_lowercase().as_str() {
            "cartesian" => {
                if sizes.len() != 3 {
                    return Err("For cartesian lattice, sizes must have length 3".to_string());
                }
                LatticeType::Cartesian
            }
            "sphere" => {
                if sizes.len() != 1 {
                    return Err("For sphere lattice, sizes must have length 1".to_string());
                }
                LatticeType::Sphere
            }
            _ => return Err("Invalid lattice_type".to_string()),
        };
        let sizes_array = match sizes.len() {
            1 => [sizes[0], 0.0, 0.0],
            3 => [sizes[0], sizes[1], sizes[2]],
            _ => unreachable!(),
        };
        Ok(Lattice {
            lattice_type,
            sizes: sizes_array,
            n_grid: n,
        })
    }

    pub fn get_volume(&self) -> f64 {
        match self.lattice_type {
            LatticeType::Cartesian => self.sizes[0] * self.sizes[1] * self.sizes[2],
            LatticeType::Sphere => (4.0 / 3.0) * std::f64::consts::PI * self.sizes[0].powi(3),
        }
    }

    pub fn generate_grid(&self) -> Array2<f64> {
        match self.lattice_type {
            LatticeType::Cartesian => {
                let lx = self.sizes[0];
                let ly = self.sizes[1];
                let lz = self.sizes[2];
                let n = self.n_grid;
                let x: Vec<f64> = (0..n).map(|i| i as f64 * lx / (n - 1) as f64).collect();
                let y: Vec<f64> = (0..n).map(|i| i as f64 * ly / (n - 1) as f64).collect();
                let z: Vec<f64> = (0..n).map(|i| i as f64 * lz / (n - 1) as f64).collect();
                let mut grid_points = Vec::with_capacity(n * n * n * 3);
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            grid_points.extend_from_slice(&[x[i], y[j], z[k]]);
                        }
                    }
                }
                Array2::from_shape_vec((n * n * n, 3), grid_points).unwrap()
            }
            LatticeType::Sphere => {
                let r = self.sizes[0];
                let n = self.n_grid;
                let x: Vec<f64> = (0..n)
                    .map(|i| -r + 2.0 * i as f64 * r / (n - 1) as f64)
                    .collect();
                let mut grid_points = Vec::new();
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            let point = [x[i], x[j], x[k]];
                            if point.iter().map(|&v| v * v).sum::<f64>() <= r * r {
                                grid_points.extend_from_slice(&[x[i], x[j], x[k]]);
                            }
                        }
                    }
                }
                Array2::from_shape_vec((grid_points.len() / 3, 3), grid_points).unwrap()
            }
        }
    }

    pub fn get_lattice_bounds(&self) -> Vec<(f64, f64)> {
        match self.lattice_type {
            LatticeType::Cartesian => vec![
                (0.0, self.sizes[0]),
                (0.0, self.sizes[1]),
                (0.0, self.sizes[2]),
            ],
            LatticeType::Sphere => vec![(-self.sizes[0], self.sizes[0]); 3],
        }
    }
}

pub trait NucleationStrategy: Clone {
    fn nucleate(
        &mut self,
        t: &mut f64,
        valid_points: &Array2<f64>,
        rng: &mut StdRng,
        vw: f64,
        volume_remaining: f64,
        existing_bubbles: Option<&[Bubble]>,
    ) -> Option<Array2<f64>>;
    fn initial_time(&self) -> f64;
    fn validate_schedule(&self, lattice: &Lattice, vw: f64) -> Result<(), String>;
}

#[derive(Clone, Copy)]
pub struct Bubble {
    center: [f64; 3],
    time: f64,
}

#[derive(Clone)]
pub struct SimParams {
    vw: f64,
    volume_total: f64,
    volume_remaining: f64,
    last_update_time: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoissonNucleation {
    gamma0: f64,
    beta: f64,
    t0: f64,
    d_p0: f64,
}

impl PoissonNucleation {
    pub fn new(gamma0: f64, beta: f64, t0: f64, d_p0: f64) -> Result<Self, String> {
        if gamma0 <= 0.0 {
            return Err("Gamma0 must be positive".to_string());
        }
        Ok(PoissonNucleation {
            gamma0,
            beta,
            t0,
            d_p0,
        })
    }

    fn is_point_valid(&self, point: &[f64], t: f64, existing_bubbles: &[Bubble], vw: f64) -> bool {
        for bubble in existing_bubbles {
            let tn = bubble.time;
            if tn > t {
                continue;
            }
            let radius = vw * (t - tn).max(0.0);
            let dist: f64 = ((point[0] - bubble.center[0]).powi(2)
                + (point[1] - bubble.center[1]).powi(2)
                + (point[2] - bubble.center[2]).powi(2))
            .sqrt();
            if dist <= radius {
                return false;
            }
        }
        true
    }
}

impl NucleationStrategy for PoissonNucleation {
    fn nucleate(
        &mut self,
        t: &mut f64,
        valid_points: &Array2<f64>,
        rng: &mut StdRng,
        vw: f64,
        volume_remaining: f64,
        existing_bubbles: Option<&[Bubble]>,
    ) -> Option<Array2<f64>> {
        let gamma_t = self.gamma0 * (self.beta * (*t - self.t0)).exp() * volume_remaining;
        let dt = self.d_p0 / gamma_t;
        let new_t = *t + dt;
        if valid_points.is_empty() || volume_remaining < 1e-10 {
            *t = new_t;
            return None;
        }
        let x: f64 = rng.random();
        if x <= self.d_p0 {
            let n = valid_points.nrows();
            let idx = rng.random_range(0..n);
            let point = valid_points.row(idx).to_vec();
            let existing_bubbles = existing_bubbles.unwrap_or(&[]);
            if self.is_point_valid(&point, new_t, existing_bubbles, vw) {
                *t = new_t;
                Some(Array2::from_shape_vec((1, 3), point).unwrap())
            } else {
                *t = new_t;
                None
            }
        } else {
            *t = new_t;
            None
        }
    }

    fn initial_time(&self) -> f64 {
        self.t0
    }

    fn validate_schedule(&self, _lattice: &Lattice, _vw: f64) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ManualNucleation {
    schedule: Vec<(f64, Vec<[f64; 3]>)>,
    dt: f64,
}

impl ManualNucleation {
    pub fn new(schedule: Vec<(f64, Vec<[f64; 3]>)>, dt: f64) -> Result<Self, String> {
        if schedule.is_empty() {
            return Err("Manual nucleation requires a non-empty schedule".to_string());
        }
        if dt <= 0.0 {
            return Err("Time step (dt) must be positive".to_string());
        }
        for (t, centers) in &schedule {
            if *t < 0.0 {
                return Err("Nucleation times must be non-negative".to_string());
            }
            if centers.is_empty() {
                return Err("Each nucleation time must have at least one center".to_string());
            }
        }
        Ok(ManualNucleation { schedule, dt })
    }

    fn is_point_valid(
        &self,
        point: [f64; 3],
        t: f64,
        existing_bubbles: &[([f64; 3], f64)],
        vw: f64,
    ) -> bool {
        for &(center, tn) in existing_bubbles {
            let radius = vw * (t - tn);
            if radius > 0.0 {
                let dist: f64 = ((point[0] - center[0]).powi(2)
                    + (point[1] - center[1]).powi(2)
                    + (point[2] - center[2]).powi(2))
                .sqrt();
                if dist <= radius {
                    return false;
                }
            }
        }
        true
    }
}

impl NucleationStrategy for ManualNucleation {
    fn nucleate(
        &mut self,
        t: &mut f64,
        _valid_points: &Array2<f64>,
        _rng: &mut StdRng,
        vw: f64,
        _volume_remaining: f64,
        existing_bubbles: Option<&[Bubble]>,
    ) -> Option<Array2<f64>> {
        let mut new_centers: Vec<[f64; 3]> = Vec::new();
        let existing_bubbles = existing_bubbles.unwrap_or(&[]);
        let existing_bubble_vec: Vec<([f64; 3], f64)> = existing_bubbles
            .iter()
            .map(|b| (b.center, b.time))
            .collect();
        for (nucleation_time, centers) in &self.schedule {
            if (*nucleation_time - *t).abs() < 1e-10 {
                for center in centers {
                    if self.is_point_valid(*center, *t, &existing_bubble_vec, vw) {
                        new_centers.push(*center);
                    }
                }
            }
        }
        *t += self.dt;
        if new_centers.is_empty() {
            None
        } else {
            Some(
                Array2::from_shape_vec(
                    (new_centers.len(), 3),
                    new_centers.into_iter().flatten().collect(),
                )
                .unwrap(),
            )
        }
    }

    fn initial_time(&self) -> f64 {
        self.schedule
            .iter()
            .map(|(t, _)| *t)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    fn validate_schedule(&self, lattice: &Lattice, vw: f64) -> Result<(), String> {
        let bounds = lattice.get_lattice_bounds();
        for (t, centers) in &self.schedule {
            for center in centers {
                for (i, &(min_bound, max_bound)) in bounds.iter().enumerate() {
                    if center[i] < min_bound || center[i] > max_bound {
                        return Err(format!(
                            "Bubble center {:?} at time {} is outside lattice bounds [{}, {}] for dimension {}",
                            center, t, min_bound, max_bound, i
                        ));
                    }
                }
            }
        }
        let mut sorted_times: Vec<f64> = self.schedule.iter().map(|(t, _)| *t).collect();
        sorted_times.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
        let mut existing_bubbles: Vec<([f64; 3], f64)> = Vec::new();
        for t in sorted_times {
            let centers = self
                .schedule
                .iter()
                .find(|(time, _)| (*time - t).abs() < 1e-10)
                .map(|(_, centers)| centers)
                .unwrap();
            let mut valid_centers = Vec::new();
            for center in centers {
                if self.is_point_valid(*center, t, &existing_bubbles, vw) {
                    valid_centers.push(*center);
                } else {
                    return Err(format!(
                        "Bubble at {:?} at time {} overlaps with earlier bubbles",
                        center, t
                    ));
                }
            }
            existing_bubbles.extend(valid_centers.into_iter().map(|c| (c, t)));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct BubbleFormationSimulator<S: NucleationStrategy> {
    params: SimParams,
    lattice: Lattice,
    strategy: S,
    bubbles: Vec<Bubble>,
    grid: Array2<f64>,
    outside_points: Vec<usize>,
    rng: StdRng,
    volume_history: Vec<(f64, f64)>,
    pub end_status: Option<SimulationEndStatus>,
}

impl<S: NucleationStrategy> BubbleFormationSimulator<S> {
    pub fn new(lattice: Lattice, vw: f64, strategy: S, seed: Option<u64>) -> Result<Self, String> {
        if vw <= 0.0 || vw > 1.0 {
            return Err("Wall velocity must be in (0, 1]".to_string());
        }
        let grid = lattice.generate_grid();
        let volume_total = lattice.get_volume();
        let outside_points = (0..grid.nrows()).collect::<Vec<usize>>();
        let rng = match seed {
            Some(seed_value) => StdRng::seed_from_u64(seed_value),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let mut simulator = BubbleFormationSimulator {
            params: SimParams {
                vw,
                volume_total,
                volume_remaining: volume_total,
                last_update_time: 0.0,
            },
            lattice,
            strategy,
            bubbles: Vec::new(),
            grid,
            outside_points,
            rng,
            volume_history: Vec::new(),
            end_status: None,
        };

        if let LatticeType::Sphere = simulator.lattice.lattice_type {
            let r = simulator.lattice.sizes[0];
            simulator.outside_points.retain(|&i| {
                let row = simulator.grid.row(i);
                let d = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
                d <= r * r
            });
        }

        simulator.validate()?;

        Ok(simulator)
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn validate(&self) -> Result<(), String> {
        self.strategy
            .validate_schedule(&self.lattice, self.params.vw)
    }

    fn update_outside_mask(&mut self, new_bubbles: &Vec<([f64; 3], f64)>, t: f64) {
        if new_bubbles.is_empty() {
            return;
        }

        let points_to_remove: Vec<usize> = self
            .outside_points
            .par_iter()
            .with_min_len(100) // Tune chunk size for better load balance
            .filter_map(|&i| {
                let px = self.grid[[i, 0]];
                let py = self.grid[[i, 1]];
                let pz = self.grid[[i, 2]];
                for &(center, tn) in new_bubbles {
                    let radius = (self.params.vw * (t - tn)).max(0.0);
                    let dist: f64 = ((px - center[0]).powi(2)
                        + (py - center[1]).powi(2)
                        + (pz - center[2]).powi(2))
                    .sqrt();
                    if dist <= radius {
                        return Some(i);
                    }
                }
                None
            })
            .collect();

        // Parallel filtering for outside_points
        self.outside_points = self
            .outside_points
            .par_iter()
            .with_min_len(100)
            .filter(|&&i| !points_to_remove.iter().any(|&idx| idx == i))
            .copied()
            .collect();

        self.params.last_update_time = t;
    }

    pub fn run_simulation(
        &mut self,
        t_max: Option<f64>,
        min_volume_remaining_fraction: Option<f64>,
        max_time_iterations: Option<usize>,
    ) {
        self.volume_history.clear();
        self.end_status = None;
        let min_frac = min_volume_remaining_fraction.unwrap_or(0.0);
        let mut t = self.strategy.initial_time();
        let mut iteration_count = 0;

        loop {
            if t_max.map_or(false, |t_max_val| t >= t_max_val) {
                self.end_status = Some(SimulationEndStatus::TimeLimitReached {
                    t_end: t,
                    volume_remaining_fraction: self.params.volume_remaining
                        / self.params.volume_total,
                    time_iteration: iteration_count,
                });
                return;
            }
            if self.params.volume_remaining < min_frac * self.params.volume_total {
                self.end_status = Some(SimulationEndStatus::VolumeFractionReached {
                    t_end: t,
                    volume_remaining_fraction: self.params.volume_remaining
                        / self.params.volume_total,
                    time_iteration: iteration_count,
                });
                return;
            }
            if max_time_iterations.map_or(false, |max_iter| iteration_count >= max_iter) {
                self.end_status = Some(SimulationEndStatus::MaxTimeIterationsReached {
                    t_end: t,
                    volume_remaining_fraction: self.params.volume_remaining
                        / self.params.volume_total,
                    time_iteration: iteration_count,
                });
                return;
            }
            if self.params.volume_remaining < 1e-6 || self.get_valid_points(t).is_empty() {
                self.end_status = Some(SimulationEndStatus::VolumeDepleted {
                    t_end: t,
                    volume_remaining_fraction: self.params.volume_remaining
                        / self.params.volume_total,
                    time_iteration: iteration_count,
                });
                return;
            }

            iteration_count += 1;
            let initial_t = t;
            let valid_points = self.get_valid_points(t);
            let existing_bubbles = Some(self.bubbles.as_slice());
            let new_bubbles: Vec<([f64; 3], f64)> = if let Some(new_centers) =
                self.strategy.nucleate(
                    &mut t,
                    &valid_points,
                    &mut self.rng,
                    self.params.vw,
                    self.params.volume_remaining,
                    existing_bubbles,
                ) {
                let new_t = t;
                self.update_outside_mask(
                    &new_centers
                        .outer_iter()
                        .map(|row| ([row[0], row[1], row[2]], new_t))
                        .collect(),
                    new_t,
                );
                new_centers
                    .outer_iter()
                    .map(|row| ([row[0], row[1], row[2]], new_t))
                    .collect()
            } else {
                Vec::new()
            };
            for (center, tn) in new_bubbles {
                self.bubbles.push(Bubble { center, time: tn });
            }
            let valid_points_updated = self.get_valid_points(t);
            self.params.volume_remaining = self.update_volume_remaining(&valid_points_updated);
            self.volume_history
                .push((t - initial_t, self.params.volume_remaining));
        }
    }

    pub fn volume_remaining(&self) -> f64 {
        self.params.volume_remaining
    }

    pub fn get_valid_points(&mut self, t: f64) -> Array2<f64> {
        let last_t = self.params.last_update_time;
        let mut valid_coords = Vec::new();
        if t > last_t {
            // Single-pass collection of remove indices and valid coordinates
            let (points_to_remove, valid_coords_temp): (Vec<usize>, Vec<[f64; 3]>) = self
                .outside_points
                .par_iter()
                .with_min_len(100) // Tune chunk size for better load balance
                .map(|&i| {
                    let px = self.grid[[i, 0]];
                    let py = self.grid[[i, 1]];
                    let pz = self.grid[[i, 2]];
                    for bubble in &self.bubbles {
                        let tn = bubble.time;
                        if tn > t {
                            continue;
                        }
                        let radius = self.params.vw * (t - tn).max(0.0);
                        let dist: f64 = ((px - bubble.center[0]).powi(2)
                            + (py - bubble.center[1]).powi(2)
                            + (pz - bubble.center[2]).powi(2))
                        .sqrt();
                        if dist <= radius {
                            return (Some(i), None);
                        }
                    }
                    (None, Some([px, py, pz]))
                })
                .fold(
                    || (Vec::new(), Vec::new()),
                    |(mut remove, mut valid), (r, v)| {
                        if let Some(idx) = r {
                            remove.push(idx);
                        }
                        if let Some(coords) = v {
                            valid.push(coords);
                        }
                        (remove, valid)
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |(mut r1, mut v1), (r2, v2)| {
                        r1.extend(r2);
                        v1.extend(v2);
                        (r1, v1)
                    },
                );

            valid_coords = valid_coords_temp;

            // Parallel filtering for outside_points
            self.outside_points = self
                .outside_points
                .par_iter()
                .with_min_len(100)
                .filter(|&&i| !points_to_remove.iter().any(|&idx| idx == i))
                .copied()
                .collect();

            self.params.last_update_time = t;
        } else {
            valid_coords = self
                .outside_points
                .par_iter()
                .with_min_len(100)
                .map(|&i| [self.grid[[i, 0]], self.grid[[i, 1]], self.grid[[i, 2]]])
                .collect();
        }

        Array2::from_shape_vec(
            (valid_coords.len(), 3),
            valid_coords.into_iter().flatten().collect(),
        )
        .unwrap()
    }

    pub fn update_volume_remaining(&mut self, valid_points: &Array2<f64>) -> f64 {
        let fraction_remaining = valid_points.nrows() as f64 / self.grid.nrows() as f64;
        self.params.volume_remaining = self.params.volume_total * fraction_remaining;
        self.params.volume_remaining
    }

    pub fn get_boundary_intersecting_bubbles(&self, t: f64) -> Vec<(usize, f64)> {
        let mut boundary_bubbles = Vec::new();
        for (i, bubble) in self.bubbles.iter().enumerate() {
            let tn = bubble.time;
            let radius = (self.params.vw * (t - tn)).max(0.0);
            if radius <= 0.0 {
                continue;
            }
            let x = bubble.center[0];
            let y = bubble.center[1];
            let z = bubble.center[2];
            match self.lattice.lattice_type {
                LatticeType::Cartesian => {
                    let lx = self.lattice.sizes[0];
                    let ly = self.lattice.sizes[1];
                    let lz = self.lattice.sizes[2];
                    if x - radius < 0.0
                        || x + radius > lx
                        || y - radius < 0.0
                        || y + radius > ly
                        || z - radius < 0.0
                        || z + radius > lz
                    {
                        boundary_bubbles.push((i, tn));
                    }
                }
                LatticeType::Sphere => {
                    let r = self.lattice.sizes[0];
                    let center_dist = (x * x + y * y + z * z).sqrt();
                    let max_radius = r - center_dist;
                    if max_radius < 0.0 || radius > max_radius {
                        boundary_bubbles.push((i, tn));
                    }
                }
            }
        }
        boundary_bubbles
    }

    pub fn generate_exterior_bubbles(&self) -> Array2<f64> {
        if self.lattice.lattice_type != LatticeType::Cartesian {
            return Array2::zeros((0, 4));
        }

        let lx = self.lattice.sizes[0];
        let ly = self.lattice.sizes[1];
        let lz = self.lattice.sizes[2];

        let shifts: [[f64; 3]; 6] = [
            [lx, 0.0, 0.0],
            [-lx, 0.0, 0.0],
            [0.0, ly, 0.0],
            [0.0, -ly, 0.0],
            [0.0, 0.0, lz],
            [0.0, 0.0, -lz],
        ];

        let mut exterior_bubbles = Vec::new();
        let mut seen_centers = HashSet::new();

        for bubble in self.bubbles.iter() {
            let tn = bubble.time;
            let center = bubble.center;

            for shift in shifts.iter() {
                let shifted_center = [
                    center[0] + shift[0],
                    center[1] + shift[1],
                    center[2] + shift[2],
                ];
                let q_point =
                    QuantizedPoint::new((shifted_center[0], shifted_center[1], shifted_center[2]));

                if seen_centers.insert(q_point) {
                    exterior_bubbles.extend_from_slice(&[
                        tn,
                        shifted_center[0],
                        shifted_center[1],
                        shifted_center[2],
                    ]);
                }
            }
        }

        if exterior_bubbles.is_empty() {
            Array2::zeros((0, 4))
        } else {
            Array2::from_shape_vec((exterior_bubbles.len() / 4, 4), exterior_bubbles).unwrap()
        }
    }

    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    pub fn vw(&self) -> f64 {
        self.params.vw
    }

    pub fn grid(&self) -> &Array2<f64> {
        &self.grid
    }

    pub fn volume_total(&self) -> f64 {
        self.params.volume_total
    }

    pub fn get_center(&self, idx: usize) -> [f64; 3] {
        self.bubbles[idx].center
    }

    pub fn get_bubbles(&self) -> Array2<f64> {
        let num_bubbles = self.bubbles.len();
        if num_bubbles == 0 {
            return Array2::zeros((0, 4));
        }

        let mut bubbles_array = Array2::zeros((num_bubbles, 4));
        for (i, bubble) in self.bubbles.iter().enumerate() {
            bubbles_array[[i, 0]] = bubble.time;
            bubbles_array[[i, 1]] = bubble.center[0];
            bubbles_array[[i, 2]] = bubble.center[1];
            bubbles_array[[i, 3]] = bubble.center[2];
        }
        bubbles_array
    }

    pub fn get_volume_history(&self) -> Vec<(f64, f64)> {
        self.volume_history.clone()
    }
}

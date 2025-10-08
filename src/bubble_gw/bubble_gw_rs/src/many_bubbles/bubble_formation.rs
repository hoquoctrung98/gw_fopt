use bitvec::prelude::*;
use ndarray::{Array2, Array3};
use rand::random;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};

// Helper struct for hashing quantized coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedPoint(i64, i64, i64);

impl QuantizedPoint {
    fn new(coords: (f64, f64, f64)) -> Self {
        let quantize = |x: f64| (x * 1e10).round() as i64;
        QuantizedPoint(quantize(coords.0), quantize(coords.1), quantize(coords.2))
    }
}

impl Hash for QuantizedPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
    pub n: usize,
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
            n,
        })
    }

    pub fn get_volume(&self) -> f64 {
        match self.lattice_type {
            LatticeType::Cartesian => self.sizes[0] * self.sizes[1] * self.sizes[2],
            LatticeType::Sphere => (4.0 / 3.0) * std::f64::consts::PI * self.sizes[0].powi(3),
        }
    }

    pub fn generate_grid(&self) -> (Array2<f64>, Array3<Vec<usize>>) {
        match self.lattice_type {
            LatticeType::Cartesian => {
                let lx = self.sizes[0];
                let ly = self.sizes[1];
                let lz = self.sizes[2];
                let n = self.n;
                let x: Vec<f64> = (0..n).map(|i| i as f64 * lx / (n - 1) as f64).collect();
                let y: Vec<f64> = (0..n).map(|i| i as f64 * ly / (n - 1) as f64).collect();
                let z: Vec<f64> = (0..n).map(|i| i as f64 * lz / (n - 1) as f64).collect();
                let mut grid_points = Vec::with_capacity(n * n * n * 3);
                const N_CELLS: usize = 10;
                let cell_size_x = lx / N_CELLS as f64;
                let cell_size_y = ly / N_CELLS as f64;
                let cell_size_z = lz / N_CELLS as f64;
                let mut cell_map = Array3::from_elem([N_CELLS, N_CELLS, N_CELLS], Vec::new());
                let mut idx = 0;

                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            grid_points.extend_from_slice(&[x[i], y[j], z[k]]);
                            let cell_x = (x[i] / cell_size_x).floor() as usize;
                            let cell_y = (y[j] / cell_size_y).floor() as usize;
                            let cell_z = (z[k] / cell_size_z).floor() as usize;
                            let cell_x = cell_x.min(N_CELLS - 1);
                            let cell_y = cell_y.min(N_CELLS - 1);
                            let cell_z = cell_z.min(N_CELLS - 1);
                            cell_map[[cell_x, cell_y, cell_z]].push(idx);
                            idx += 1;
                        }
                    }
                }
                let grid = Array2::from_shape_vec((n * n * n, 3), grid_points).unwrap();
                (grid, cell_map)
            }
            LatticeType::Sphere => {
                let r = self.sizes[0];
                let n = self.n;
                let x: Vec<f64> = (0..n)
                    .map(|i| -r + 2.0 * i as f64 * r / (n - 1) as f64)
                    .collect();
                let mut grid_points = Vec::new();
                const N_CELLS: usize = 10;
                let cell_size = (2.0 * r) / N_CELLS as f64;
                let mut cell_map = Array3::from_elem([N_CELLS, N_CELLS, N_CELLS], Vec::new());
                let mut idx = 0;

                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            let point = [x[i], x[j], x[k]];
                            if point.iter().map(|&v| v * v).sum::<f64>() <= r * r {
                                grid_points.extend_from_slice(&[x[i], x[j], x[k]]);
                                let cell_x = ((x[i] + r) / cell_size).floor() as usize;
                                let cell_y = ((x[j] + r) / cell_size).floor() as usize;
                                let cell_z = ((x[k] + r) / cell_size).floor() as usize;
                                let cell_x = cell_x.min(N_CELLS - 1);
                                let cell_y = cell_y.min(N_CELLS - 1);
                                let cell_z = cell_z.min(N_CELLS - 1);
                                cell_map[[cell_x, cell_y, cell_z]].push(idx);
                                idx += 1;
                            }
                        }
                    }
                }
                let grid = Array2::from_shape_vec((grid_points.len() / 3, 3), grid_points).unwrap();
                (grid, cell_map)
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

pub trait SimulationState {
    fn dt(&self) -> f64;
    fn v_remaining(&self) -> f64;
    fn get_valid_points(&mut self, t: Option<f64>) -> Array2<f64>;
    fn update_remaining_volume_bulk(&mut self, _t: f64, valid_points: &Array2<f64>) -> f64;
}

#[derive(Debug, Clone)]
pub enum NucleationStrategy {
    Poisson(PoissonNucleation),
    Manual(ManualNucleation),
}

impl NucleationStrategy {
    pub fn nucleate(&self, t: f64, state: &mut BubbleFormationSimulator) -> Array2<f64> {
        match self {
            NucleationStrategy::Poisson(inner) => inner.nucleate(t, state),
            NucleationStrategy::Manual(inner) => inner.nucleate(t, state),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoissonNucleation {
    gamma0: f64,
    beta: f64,
    t0: f64,
}

impl PoissonNucleation {
    pub fn new(params: BTreeMap<String, f64>) -> Result<Self, String> {
        let gamma0 = *params
            .get("Gamma0")
            .ok_or("Missing Gamma0 in poisson_params")?;
        let beta = *params.get("beta").ok_or("Missing beta in poisson_params")?;
        let t0 = *params.get("t0").ok_or("Missing t0 in poisson_params")?;
        if gamma0 <= 0.0 {
            return Err("Gamma0 must be positive".to_string());
        }
        Ok(PoissonNucleation { gamma0, beta, t0 })
    }

    pub fn nucleate(&self, t: f64, state: &mut BubbleFormationSimulator) -> Array2<f64> {
        let gamma_t = self.gamma0 * (self.beta * (t - self.t0)).exp();
        let num_bubbles = (gamma_t * state.dt() * state.v_remaining()) as usize;
        let valid_points = state.get_valid_points(Some(t));
        if valid_points.is_empty() || num_bubbles == 0 || state.v_remaining() < 1e-10 {
            return Array2::zeros((0, 3));
        }
        let n = valid_points.nrows().min(num_bubbles);
        let indices: Vec<usize> =
            rand::seq::index::sample(&mut state.rng, valid_points.nrows(), n).into_vec();
        Array2::from_shape_vec(
            (n, 3),
            indices
                .into_iter()
                .flat_map(|row| valid_points.row(row).to_vec())
                .collect(),
        )
        .unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct ManualNucleation {
    schedule: Vec<(f64, Vec<[f64; 3]>)>,
    max_time: f64,
}

impl ManualNucleation {
    pub fn new(schedule: Vec<(f64, Vec<[f64; 3]>)>) -> Result<Self, String> {
        if schedule.is_empty() {
            return Err("Manual nucleation requires a non-empty schedule".to_string());
        }
        let max_time = schedule
            .iter()
            .map(|(t, _)| *t)
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(ManualNucleation { schedule, max_time })
    }

    pub fn nucleate(&self, t: f64, state: &mut BubbleFormationSimulator) -> Array2<f64> {
        let dt = state.dt();
        let time_range = (t - dt, t);
        let mut new_centers: Vec<[f64; 3]> = Vec::new();
        for (nucleation_time, centers) in &self.schedule {
            if *nucleation_time > time_range.0 && *nucleation_time <= time_range.1 {
                new_centers.extend(centers.iter().cloned());
            }
        }
        if new_centers.is_empty() {
            Array2::zeros((0, 3))
        } else {
            Array2::from_shape_vec(
                (new_centers.len(), 3),
                new_centers.into_iter().flatten().collect(),
            )
            .unwrap()
        }
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

    pub fn max_nucleation_time(&self) -> f64 {
        self.max_time
    }
}

#[derive(Clone, Copy)]
struct Bubble {
    center: [f64; 3],
    time: f64,
}

pub struct SimParams {
    vw: f64,
    dt: f64,
    v_total: f64,
    v_remaining: f64,
    last_update_time: f64,
    cell_size: [f64; 3],
}

pub struct BubbleFormationSimulator {
    params: SimParams,
    lattice: Lattice,
    strategy: NucleationStrategy,
    bubbles: Vec<Bubble>,
    grid: Array2<f64>,
    is_outside: BitVec<usize, Lsb0>,
    cell_map: Array3<Vec<usize>>,
    rng: StdRng,
}

impl BubbleFormationSimulator {
    pub fn new(
        lattice: Lattice,
        vw: f64,
        dt: f64,
        strategy: Option<NucleationStrategy>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let (grid, cell_map) = lattice.generate_grid();
        let v_total = lattice.get_volume();
        let strategy = strategy.unwrap_or_else(|| {
            let mut params = BTreeMap::new();
            params.insert("Gamma0".to_string(), 0.1);
            params.insert("beta".to_string(), 1.0);
            params.insert("t0".to_string(), 0.0);
            NucleationStrategy::Poisson(PoissonNucleation::new(params).unwrap())
        });
        #[allow(unused_mut)]
        let mut is_outside =
            BitVec::<usize, Lsb0>::from_iter(std::iter::repeat(true).take(grid.nrows()));

        let cell_size = match lattice.lattice_type {
            LatticeType::Cartesian => {
                const N_CELLS: f64 = 10.0;
                [
                    lattice.sizes[0] / N_CELLS,
                    lattice.sizes[1] / N_CELLS,
                    lattice.sizes[2] / N_CELLS,
                ]
            }
            LatticeType::Sphere => {
                const N_CELLS: f64 = 10.0;
                let r = lattice.sizes[0];
                let cell_size = (2.0 * r) / N_CELLS;
                [cell_size, cell_size, cell_size]
            }
        };

        let rng = match seed {
            Some(seed_value) => StdRng::seed_from_u64(seed_value),
            None => StdRng::seed_from_u64(random::<u64>()),
        };

        let mut simulator = BubbleFormationSimulator {
            params: SimParams {
                vw,
                dt,
                v_total,
                v_remaining: v_total,
                last_update_time: 0.0,
                cell_size,
            },
            lattice: lattice.clone(),
            strategy,
            bubbles: Vec::new(),
            grid,
            is_outside,
            cell_map,
            rng,
        };

        if let LatticeType::Sphere = lattice.lattice_type {
            let r = lattice.sizes[0];
            for (i, row) in simulator.grid.outer_iter().enumerate() {
                let d = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
                if d > r * r {
                    simulator.is_outside.set(i, false);
                }
            }
        }

        simulator.validate()?;

        Ok(simulator)
    }

    fn validate(&self) -> Result<(), String> {
        if let NucleationStrategy::Manual(manual) = &self.strategy {
            let mut existing_bubbles: Vec<([f64; 3], f64)> = Vec::new();
            let mut sorted_times: Vec<f64> = manual.schedule.iter().map(|(t, _)| *t).collect();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for t in sorted_times {
                let centers = manual
                    .schedule
                    .iter()
                    .find(|(time, _)| (*time - t).abs() < 1e-10)
                    .map(|(_, centers)| centers)
                    .unwrap();
                let mut valid_centers = Vec::new();
                for center in centers {
                    if manual.is_point_valid(*center, t, &existing_bubbles, self.vw()) {
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
        }
        Ok(())
    }

    pub fn run_simulation(&mut self, t_final: f64, verbose: bool) -> Result<(), String> {
        let max_nucleation_time = if let NucleationStrategy::Manual(manual) = &self.strategy {
            manual.max_nucleation_time()
        } else {
            t_final
        };

        let effective_t_final = if max_nucleation_time < t_final {
            (max_nucleation_time + self.params.dt).min(t_final)
        } else {
            t_final
        };

        let max_steps = (effective_t_final / self.params.dt).ceil() as usize;
        let t_arr: Vec<f64> = (0..max_steps).map(|i| i as f64 * self.params.dt).collect();
        let strategy = self.strategy.clone();
        const MAX_ITERATIONS: usize = 10_000;
        let mut iteration_count = 0;

        for &t in &t_arr {
            if iteration_count >= MAX_ITERATIONS {
                if verbose {
                    println!(
                        "Terminating at t = {:.2} due to reaching maximum iterations ({})",
                        t, MAX_ITERATIONS
                    );
                }
                break;
            }
            iteration_count += 1;

            let new_centers = strategy.nucleate(t, self);
            if !new_centers.is_empty() {
                let new_bubbles: Vec<([f64; 3], f64)> = new_centers
                    .outer_iter()
                    .map(|row| ([row[0], row[1], row[2]], t))
                    .collect();
                self.update_outside_mask(&new_bubbles, t);
                for (center, tn) in new_bubbles {
                    self.bubbles.push(Bubble { center, time: tn });
                }
            }

            let valid_points = self.get_valid_points(Some(t));
            self.params.v_remaining = self.update_remaining_volume_bulk(t, &valid_points);
            if verbose {
                println!(
                    "Simulating time step: t = {:.2}, v_remaining = {:.6e}, valid points = {}",
                    t,
                    self.params.v_remaining,
                    valid_points.nrows()
                );
            }
            if self.params.v_remaining < 1e-6 || valid_points.is_empty() {
                if verbose {
                    println!(
                        "Terminating at t = {:.2} due to v_remaining = {:.6e}, or no valid points",
                        t, self.params.v_remaining
                    );
                }
                break;
            }
        }
        Ok(())
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

    fn update_outside_mask(&mut self, new_bubbles: &Vec<([f64; 3], f64)>, t: f64) {
        if new_bubbles.is_empty() {
            return;
        }

        const N_CELLS: usize = 10;
        let buffer = 1;

        for &(center, tn) in new_bubbles {
            let radius = (self.params.vw * (t - tn)).max(0.0);
            if radius <= 0.0 {
                continue;
            }

            let (cell_x, cell_y, cell_z) = match self.lattice.lattice_type {
                LatticeType::Cartesian => (
                    (center[0] / self.params.cell_size[0]).floor() as usize,
                    (center[1] / self.params.cell_size[1]).floor() as usize,
                    (center[2] / self.params.cell_size[2]).floor() as usize,
                ),
                LatticeType::Sphere => {
                    let r = self.lattice.sizes[0];
                    (
                        ((center[0] + r) / self.params.cell_size[0]).floor() as usize,
                        ((center[1] + r) / self.params.cell_size[1]).floor() as usize,
                        ((center[2] + r) / self.params.cell_size[2]).floor() as usize,
                    )
                }
            };
            let cell_x = cell_x.min(N_CELLS - 1);
            let cell_y = cell_y.min(N_CELLS - 1);
            let cell_z = cell_z.min(N_CELLS - 1);

            let cell_radius_x = (radius / self.params.cell_size[0]).ceil() as i32 + buffer as i32;
            let cell_radius_y = (radius / self.params.cell_size[1]).ceil() as i32 + buffer as i32;
            let cell_radius_z = (radius / self.params.cell_size[2]).ceil() as i32 + buffer as i32;

            let mut nearby_indices = Vec::new();
            for dx in -cell_radius_x..=cell_radius_x {
                for dy in -cell_radius_y..=cell_radius_y {
                    for dz in -cell_radius_z..=cell_radius_z {
                        let cx = (cell_x as i32 + dx) as usize;
                        let cy = (cell_y as i32 + dy) as usize;
                        let cz = (cell_z as i32 + dz) as usize;
                        if cx >= N_CELLS || cy >= N_CELLS || cz >= N_CELLS {
                            continue;
                        }
                        nearby_indices.extend(&self.cell_map[[cx, cy, cz]]);
                    }
                }
            }

            let indices_to_update: Vec<usize> = nearby_indices
                .par_iter()
                .filter_map(|&i| {
                    let is_outside: bool = self.is_outside[i];
                    if i >= self.grid.nrows() || !is_outside {
                        None
                    } else {
                        let px = self.grid[[i, 0]];
                        let py = self.grid[[i, 1]];
                        let pz = self.grid[[i, 2]];
                        let dist: f64 = ((px - center[0]).powi(2)
                            + (py - center[1]).powi(2)
                            + (pz - center[2]).powi(2))
                        .sqrt();
                        if dist <= radius {
                            Some(i)
                        } else {
                            None
                        }
                    }
                })
                .collect();

            indices_to_update.chunks(64).for_each(|chunk| {
                for &i in chunk {
                    self.is_outside.set(i, false);
                }
            });
        }

        self.params.last_update_time = t;
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

    pub fn v_total(&self) -> f64 {
        self.params.v_total
    }

    pub fn get_center(&self, idx: usize) -> [f64; 3] {
        self.bubbles[idx].center
    }
}

impl SimulationState for BubbleFormationSimulator {
    fn dt(&self) -> f64 {
        self.params.dt
    }

    fn v_remaining(&self) -> f64 {
        self.params.v_remaining
    }

    fn get_valid_points(&mut self, t: Option<f64>) -> Array2<f64> {
        let t = t.unwrap_or(self.params.last_update_time);
        let last_t = self.params.last_update_time;

        if t > last_t {
            for bubble in &self.bubbles {
                let tn = bubble.time;
                if tn > t {
                    continue;
                }
                let new_radius = self.params.vw * (t - tn).max(0.0);
                let last_radius = self.params.vw * (last_t - tn).max(0.0);
                if new_radius <= last_radius {
                    continue;
                }

                const N_CELLS: usize = 10;
                let buffer = 1;

                let center = bubble.center;
                let (cell_x, cell_y, cell_z) = match self.lattice.lattice_type {
                    LatticeType::Cartesian => (
                        (center[0] / self.params.cell_size[0]).floor() as usize,
                        (center[1] / self.params.cell_size[1]).floor() as usize,
                        (center[2] / self.params.cell_size[2]).floor() as usize,
                    ),
                    LatticeType::Sphere => {
                        let r = self.lattice.sizes[0];
                        (
                            ((center[0] + r) / self.params.cell_size[0]).floor() as usize,
                            ((center[1] + r) / self.params.cell_size[1]).floor() as usize,
                            ((center[2] + r) / self.params.cell_size[2]).floor() as usize,
                        )
                    }
                };
                let cell_x = cell_x.min(N_CELLS - 1);
                let cell_y = cell_y.min(N_CELLS - 1);
                let cell_z = cell_z.min(N_CELLS - 1);

                let cell_radius_x =
                    (new_radius / self.params.cell_size[0]).ceil() as i32 + buffer as i32;
                let cell_radius_y =
                    (new_radius / self.params.cell_size[1]).ceil() as i32 + buffer as i32;
                let cell_radius_z =
                    (new_radius / self.params.cell_size[2]).ceil() as i32 + buffer as i32;

                let mut nearby_indices = Vec::new();
                for dx in -cell_radius_x..=cell_radius_x {
                    for dy in -cell_radius_y..=cell_radius_y {
                        for dz in -cell_radius_z..=cell_radius_z {
                            let cx = (cell_x as i32 + dx) as usize;
                            let cy = (cell_y as i32 + dy) as usize;
                            let cz = (cell_z as i32 + dz) as usize;
                            if cx >= N_CELLS || cy >= N_CELLS || cz >= N_CELLS {
                                continue;
                            }
                            nearby_indices.extend(&self.cell_map[[cx, cy, cz]]);
                        }
                    }
                }

                let indices_to_update: Vec<usize> = nearby_indices
                    .par_iter()
                    .filter_map(|&i| {
                        let is_outside: bool = self.is_outside[i];
                        if i >= self.grid.nrows() || !is_outside {
                            None
                        } else {
                            let px = self.grid[[i, 0]];
                            let py = self.grid[[i, 1]];
                            let pz = self.grid[[i, 2]];
                            let dist: f64 = ((px - center[0]).powi(2)
                                + (py - center[1]).powi(2)
                                + (pz - center[2]).powi(2))
                            .sqrt();
                            if last_radius < dist && dist <= new_radius {
                                Some(i)
                            } else {
                                None
                            }
                        }
                    })
                    .collect();

                indices_to_update.chunks(64).for_each(|chunk| {
                    for &i in chunk {
                        self.is_outside.set(i, false);
                    }
                });
            }

            self.params.last_update_time = t;
        }

        let valid_indices: Vec<usize> = self
            .is_outside
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect();
        Array2::from_shape_vec(
            (valid_indices.len(), 3),
            valid_indices
                .into_iter()
                .flat_map(|i| vec![self.grid[[i, 0]], self.grid[[i, 1]], self.grid[[i, 2]]])
                .collect(),
        )
        .unwrap()
    }

    fn update_remaining_volume_bulk(&mut self, _t: f64, valid_points: &Array2<f64>) -> f64 {
        let fraction_remaining = valid_points.nrows() as f64 / self.grid.nrows() as f64;
        self.params.v_remaining = self.params.v_total * fraction_remaining;
        self.params.v_remaining
    }
}

impl BubbleFormationSimulator {
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
}

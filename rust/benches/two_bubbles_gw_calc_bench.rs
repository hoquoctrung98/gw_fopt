use std::hint::black_box;
use std::time::{Duration, Instant};

use bubble_gw::two_bubbles::gw_calc::{
    GravitationalWaveCalculator as OldGravitationalWaveCalculator,
    InitialFieldStatus as OldInitialFieldStatus,
};
use bubble_gw::two_bubbles::new_gw_calc::{
    ExponentialTimeCutoff,
    GravitationalWaveCalculator as NewGravitationalWaveCalculator,
    InitialFieldStatus as NewInitialFieldStatus,
};
use ndarray::{Array1, Array3};

const N_FIELDS: usize = 2;
const N_S: usize = 64;
const N_Z: usize = 73;
const N_W: usize = 12;
const N_K: usize = 21;
const RATIO_T_CUT: Option<f64> = Some(0.8);
const RATIO_T_0: Option<f64> = Some(0.12);
const TOL: f64 = 1e-7;
const MAX_ITER: u32 = 30;
const REPEATS: usize = 3;

fn synthetic_fields() -> (Array3<f64>, Array3<f64>, Array1<f64>, f64) {
    let ds = 0.28;
    let z_grid = Array1::linspace(-1.35, 1.15, N_Z);

    let phi1 = Array3::from_shape_fn((N_FIELDS, N_S, N_Z), |(field, i_s, i_z)| {
        let s: f64 = i_s as f64 * ds;
        let z: f64 = z_grid[i_z];
        let field_shift: f64 = field as f64 + 1.0;
        let envelope = (-0.025 * (s - 4.3).powi(2) - 0.18 * z.powi(2)).exp();
        let oscillatory = (0.9 * field_shift * s + 1.7 * z).sin();
        let mixed = (0.35 * s * z + 0.2 * field_shift).cos();
        envelope * oscillatory + 0.11 * mixed + 0.03 * field_shift * s * z
    });

    let phi2 = Array3::from_shape_fn((N_FIELDS, N_S, N_Z), |(field, i_s, i_z)| {
        let s: f64 = i_s as f64 * ds;
        let z: f64 = z_grid[i_z];
        let field_shift: f64 = field as f64 + 1.0;
        let envelope = (-0.018 * (s - 3.1).powi(2) - 0.22 * (z + 0.15).powi(2)).exp();
        let oscillatory = (0.6 * field_shift * z - 0.45 * s).cos();
        let mixed = (0.25 * s.powi(2) - 0.4 * z + field_shift).sin();
        0.8 * envelope * oscillatory + 0.09 * mixed - 0.025 * field_shift * s * z
    });

    (phi1, phi2, z_grid, ds)
}

fn setup_old_calculator(num_threads: usize) -> OldGravitationalWaveCalculator {
    let (phi1, phi2, z_grid, ds) = synthetic_fields();
    let mut calc = OldGravitationalWaveCalculator::new(
        OldInitialFieldStatus::TwoBubbles,
        phi1,
        phi2,
        z_grid,
        ds,
        RATIO_T_CUT,
        RATIO_T_0,
    )
    .expect("failed to construct old calculator");
    calc.set_num_threads(num_threads)
        .expect("failed to set old calculator thread count");
    calc.set_integral_params(TOL, MAX_ITER)
        .expect("failed to set old calculator integral params");
    calc
}

fn setup_new_calculator(
    num_threads: usize,
) -> NewGravitationalWaveCalculator<ExponentialTimeCutoff> {
    let (phi1, phi2, z_grid, ds) = synthetic_fields();
    let smax = (N_S - 1) as f64 * ds;
    let time_cutoff = ExponentialTimeCutoff::new(smax, RATIO_T_CUT, RATIO_T_0);
    let mut calc = NewGravitationalWaveCalculator::new(
        NewInitialFieldStatus::TwoBubbles,
        phi1,
        phi2,
        z_grid,
        ds,
        time_cutoff,
    )
    .expect("failed to construct new calculator");
    calc.set_num_threads(num_threads)
        .expect("failed to set new calculator thread count");
    calc.set_integral_params(TOL, MAX_ITER)
        .expect("failed to set new calculator integral params");
    calc
}

fn time_repeated<F>(mut f: F) -> Vec<Duration>
where
    F: FnMut(),
{
    let mut durations = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }
    durations
}

fn mean_duration(durations: &[Duration]) -> Duration {
    let total: Duration = durations.iter().copied().sum();
    total / durations.len() as u32
}

fn print_result(label: &str, durations: &[Duration]) {
    let mean = mean_duration(durations);
    let best = durations.iter().min().copied().unwrap();
    let worst = durations.iter().max().copied().unwrap();
    println!(
        "{label:<34} mean={:>9.3?} best={:>9.3?} worst={:>9.3?} repeats={}",
        mean,
        best,
        worst,
        durations.len()
    );
}

fn bench_pair(num_threads: usize, w_arr: &[f64], cos_thetak_arr: &[f64]) {
    let old_calc = setup_old_calculator(num_threads);
    let new_calc = setup_new_calculator(num_threads);

    let old_durations = time_repeated(|| {
        black_box(
            old_calc
                .compute_angular_gw_spectrum(black_box(w_arr), black_box(cos_thetak_arr))
                .expect("old calculator spectrum failed"),
        );
    });

    let new_durations = time_repeated(|| {
        black_box(
            new_calc
                .compute_angular_gw_spectrum(black_box(w_arr), black_box(cos_thetak_arr))
                .expect("new calculator spectrum failed"),
        );
    });

    print_result(&format!("old_gw_calc threads={num_threads}"), &old_durations);
    print_result(&format!("new_gw_calc threads={num_threads}"), &new_durations);

    let old_mean = mean_duration(&old_durations).as_secs_f64();
    let new_mean = mean_duration(&new_durations).as_secs_f64();
    println!(
        "{:<34} {:.3}x",
        format!("new/old ratio threads={num_threads}"),
        new_mean / old_mean
    );
    println!();
}

fn main() {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let w_arr = Array1::geomspace(0.25, 4.0, N_W).unwrap().to_vec();
    let cos_thetak_arr = Array1::linspace(-0.9, 0.9, N_K).to_vec();

    println!("two_bubbles compute_angular_gw_spectrum benchmark");
    println!(
        "input: n_fields={N_FIELDS}, n_s={N_S}, n_z={N_Z}, n_w={N_W}, n_k={N_K}, tol={TOL}, max_iter={MAX_ITER}"
    );
    println!("available_threads={available_threads}, repeats={REPEATS}");
    println!();

    bench_pair(1, &w_arr, &cos_thetak_arr);
    if available_threads > 1 {
        bench_pair(available_threads, &w_arr, &cos_thetak_arr);
    }
}

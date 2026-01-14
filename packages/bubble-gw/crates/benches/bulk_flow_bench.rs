use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow;
use bubble_gw::many_bubbles::lattice::EmptyLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2, arr2};

fn setup_bulk_flow() -> GeneralizedBulkFlow<EmptyLattice> {
    let bubbles_interior = arr2(&[
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 4.0],
        [0.0, 4.0, 2.0, 4.0],
        [0.0, 0.0, 2.0, 1.0],
    ]);
    let bubbles_exterior = Array2::zeros((0, 4));
    let mut bf = GeneralizedBulkFlow::new(
        LatticeBubbles::new(bubbles_interior, Some(bubbles_exterior), EmptyLattice {}).unwrap(),
    )
    .expect("Failed to create BulkFlowSegment");
    bf.set_resolution(50, 100, true)
        .expect("Failed to set resolution");
    let coefficients_sets = vec![vec![0.0], vec![1.0]];
    let powers_sets = vec![vec![3.0], vec![3.0]];
    bf.set_gradient_scaling_params(coefficients_sets, powers_sets, None)
        .expect("Failed to set params");
    bf
}

fn bench_compute_c_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_c_integral");

    let w_arr: Vec<f64> = Array1::geomspace(1e-2, 1e2, 120).unwrap().to_vec();

    // Benchmark nalgebra (optimized)
    group.bench_function(BenchmarkId::new("Nalgebra", "nθ=50 nφ=100 n_t=800"), |b| {
        b.iter_batched(
            || setup_bulk_flow(),
            |mut bf| {
                bf.compute_c_integral(&w_arr, Some(0.0), 15.0, 800, None)
                    .expect("Nalgebra failed");
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_compute_c_integral);
criterion_main!(benches);

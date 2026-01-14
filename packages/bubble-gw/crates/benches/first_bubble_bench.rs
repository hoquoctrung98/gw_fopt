use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow as BulkFlowNalgebra;
use bubble_gw::many_bubbles::lattice::EmptyLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

fn setup_bulk_flow_nalgebra() -> BulkFlowNalgebra<EmptyLattice> {
    let lattice = Lattice::new(
        LatticeType::Cartesian {
            sizes: [10., 10., 10.],
        },
        100,
    )
    .expect("Failed to create lattice");
    let bubbles_config =
        generate_random_bubbles(lattice, BoundaryConditions::Periodic, -0.1, 0.0, 20, Some(0))
            .expect("Failed to create bubbles_config");
    let bubbles_config =
        LatticeBubbles::new(bubbles_config.interior, bubbles_config.exterior, EmptyLattice {})
            .unwrap();
    let mut bf = BulkFlowNalgebra::new(bubbles_config).expect("Failed to create BulkFlow");
    let coefficients_sets = vec![vec![0.0], vec![1.0]];
    let powers_sets = vec![vec![3.0], vec![3.0]];
    bf.set_gradient_scaling_params(coefficients_sets, powers_sets, None)
        .expect("Failed to set params");
    bf
}

fn bench_compute_c_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_resolution");

    // Benchmark original
    group.bench_function(BenchmarkId::new("Original", "nθ=50 nφ=100 n_t=800"), |b| {
        b.iter_batched(
            || setup_bulk_flow(),
            |mut bf| {
                bf.set_resolution(100, 100, true)
                    .expect("Set resolution failed");
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark segmented (optimized)
    group.bench_function(BenchmarkId::new("Segmented", "nθ=50 nφ=100 n_t=800"), |b| {
        b.iter_batched(
            || setup_bulk_flow_segment(),
            |mut bf| {
                bf.set_resolution(100, 100, true)
                    .expect("Set resolution failed for segmented version")
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark segmented (optimized)
    group.bench_function(BenchmarkId::new("Nalgebra", "nθ=50 nφ=100 n_t=800"), |b| {
        b.iter_batched(
            || setup_bulk_flow_nalgebra(),
            |mut bf| {
                bf.set_resolution(100, 100, true)
                    .expect("Set resolution failed for Nalgebra version")
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_compute_c_integral);
criterion_main!(benches);

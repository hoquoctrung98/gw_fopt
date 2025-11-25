use bubble_gw_rs::utils::{integrate};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use integrate::Integrate as NewIntegrate;

fn bench_1d_slice<T, F>(c: &mut Criterion, name: &str, data_size: usize, mut setup: F)
where
    T: Copy + num_traits::Float + From<f64> + Send + Sync + 'static,
    F: Fn(usize) -> (Vec<T>, Option<Vec<T>>),
{
    let mut group = c.benchmark_group(format!("1D Slice - {} - {} points", name, data_size));

    for method in ["trapezoid", "simpson"] {
        group.bench_with_input(BenchmarkId::new("new", method), &data_size, |b, &size| {
            b.iter_batched(
                || setup(size),
                |(y, x)| {
                    let y_slice = y.as_slice();
                    let y_ref = &y_slice; // <-- &y_slice: &&[T]

                    match method {
                        "trapezoid" => {
                            let _ = <&[T] as NewIntegrate<T>>::trapezoid(
                                y_ref, // <-- pass &y_slice
                                x.as_deref(),
                                None,
                                None,
                            )
                            .unwrap();
                        }
                        "simpson" => {
                            let _ = <&[T] as NewIntegrate<T>>::simpson(
                                y_ref, // <-- pass &y_slice
                                x.as_deref(),
                                None,
                                None,
                            )
                            .unwrap();
                        }
                        _ => unreachable!(),
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn setup_even_spacing(size: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let y: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
    (y, None)
}

fn setup_uneven_spacing(size: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut y = Vec::with_capacity(size);
    let mut x = Vec::with_capacity(size);
    let mut xi = 0.0_f64;
    for i in 0..size {
        y.push((i as f64).sin());
        x.push(xi);
        xi += 1.0 + 0.1 * ((i as f64) % 3.0);
    }
    (y, Some(x))
}

fn bench_trapezoid_simpson_1d(c: &mut Criterion) {
    let sizes = [10, 100, 1_000, 10_000, 100_000];

    for &size in &sizes {
        bench_1d_slice(c, "even", size, setup_even_spacing);
        bench_1d_slice(c, "uneven", size, setup_uneven_spacing);
    }
}

criterion_group!(benches, bench_trapezoid_simpson_1d);
criterion_main!(benches);

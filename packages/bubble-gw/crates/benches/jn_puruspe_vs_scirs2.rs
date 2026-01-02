use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn bench_jn_puruspe_vs_scirs2_j0(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut input1 = [0f64; 100];
    rng.fill(&mut input1);

    let mut group = c.benchmark_group("bench_jn_puruspe_vs_scirs2");
    group.bench_function("puruspe, J0, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), x)))
    });
    group.bench_function("scirs2_special , J0, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), x)))
    });

    group.bench_function("puruspe, J0, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), 100. * x)))
    });
    group.bench_function("scirs2_special, J0, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), 100. * x)))
    });
}

pub fn bench_jn_puruspe_vs_scirs2_j1(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut input1 = [0f64; 100];
    rng.fill(&mut input1);

    let mut group = c.benchmark_group("bench_jn_puruspe_vs_scirs2");
    group.bench_function("puruspe, J1, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), x)))
    });
    group.bench_function("scirs2_special , J1, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), x)))
    });

    group.bench_function("puruspe, J1, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), 100. * x)))
    });
    group.bench_function("scirs2_special, J1, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), 100. * x)))
    });
}

pub fn bench_jn_puruspe_vs_scirs2_j2(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut input1 = [0f64; 100];
    rng.fill(&mut input1);

    let mut group = c.benchmark_group("bench_jn_puruspe_vs_scirs2");
    group.bench_function("puruspe, J2, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), x)))
    });
    group.bench_function("scirs2_special , J2, range(0, 1)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), x)))
    });

    group.bench_function("puruspe, J2, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| puruspe::Jn(black_box(0), 100. * x)))
    });
    group.bench_function("scirs2_special, J2, range(0, 100)", |b| {
        b.iter(|| input1.map(|x| scirs2_special::jn(black_box(0), 100. * x)))
    });
}

criterion_group!(
    benches,
    bench_jn_puruspe_vs_scirs2_j0,
    bench_jn_puruspe_vs_scirs2_j1,
    bench_jn_puruspe_vs_scirs2_j2
);
criterion_main!(benches);

use std::time::Instant;

use peroxide::fuga::*;

fn main() {
    let f = |x: f64| 3.0 * x.powi(2) + 2.0 * x + 5.0;

    let tol = 1.0e-4;
    let max_iter = 24u32;

    let methods = vec![
        ("G7K15 ", Integral::G7K15(tol, max_iter)),
        ("G15K31 ", Integral::G15K31(tol, max_iter)),
        ("G25K51 ", Integral::G25K51(tol, max_iter)),
        ("G10K21 ", Integral::G10K21(tol, max_iter)),
        ("G20K41 ", Integral::G20K41(tol, max_iter)),
        ("G30K61 ", Integral::G30K61(tol, max_iter)),
        // ("G10K21R", Integral::G10K21R(tol, max_iter)),
        // ("G20K41R", Integral::G20K41R(tol, max_iter)),
        // ("G30K61R", Integral::G30K61R(tol, max_iter)),
    ];

    for (name, method) in methods {
        let start = Instant::now();
        let res = integrate(f, (5.0, 10.0), method);
        // let res = gauss_kronrod_quadrature(f, (5.0, 10.0), method);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("{} | {:.3} ms | {:.6}", name, ms, res);
    }
}

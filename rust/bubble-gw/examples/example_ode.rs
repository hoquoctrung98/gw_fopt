use core::f64;
use std::error::Error;

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

struct FixedRateNucleation {
    gamma0: f64,
    beta: f64,
}

impl ODE<f64, SVector<f64, 4>> for FixedRateNucleation {
    fn diff(&self, tau: f64, m: &SVector<f64, 4>, dm_dtau: &mut SVector<f64, 4>) {
        let gamma0_bar = self.gamma0 * self.beta.powi(-4);
        let log_n = gamma0_bar.ln() + tau - (4.0 * f64::consts::PI / 3.0) * m[3];
        let n = log_n.exp();

        dm_dtau[0] = n;
        dm_dtau[1] = m[0];
        dm_dtau[2] = 2.0 * m[1];
        dm_dtau[3] = 3.0 * m[2];
    }
}

impl FixedRateNucleation {
    fn solve_bubbles_distribution(&self, volume_lattice: f64) -> (Vec<f64>, Vec<f64>) {
        let y0 = vector![0.0, 0.0, 0.0, 0.0];
        let tau0 = 0.0;
        let taumax = 40.0;
        // let mut method = ImplicitRungeKutta::radau5().rtol(1e-9).atol(1e-12);
        let mut method = ExplicitRungeKutta::rkf45().rtol(1e-9).atol(1e-12);
        let problem = ODEProblem::new(self, tau0, taumax, y0);

        let solution = problem.even(0.001).solve(&mut method).unwrap();
        let tau: Vec<f64> = solution.iter().map(|(t, _)| *t).collect();
        let m0: Vec<f64> = solution.iter().map(|(_, y)| y[0]).collect();

        let n_bubbles: Vec<f64> = m0
            .iter()
            .map(|m| m * volume_lattice * self.beta.powi(3))
            .collect();

        let mut crossing_indices: Vec<usize> = Vec::new();
        let mut prev_floor = n_bubbles[0].floor() as i64;
        for i in 1..n_bubbles.len() {
            let current_floor = n_bubbles[i].floor() as i64;
            if current_floor > prev_floor {
                crossing_indices.push(i);
            }

            prev_floor = current_floor;
        }

        let tau: Vec<f64> = crossing_indices.iter().map(|&i| tau[i]).collect();
        let n_bubbles: Vec<f64> = crossing_indices.iter().map(|&i| n_bubbles[i]).collect();
        return (tau, n_bubbles);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let beta: f64 = 0.05;
    let gamma0: f64 = 1e-12;

    let ode = FixedRateNucleation { gamma0, beta };
    let volume_lattice = (20.0 / ode.beta).powi(3);
    let (tau, n_bubbles) = ode.solve_bubbles_distribution(volume_lattice);

    println!("\nInteger-crossing points (from below → above):");
    println!("tau, t, N_bubbles");

    for (t, n) in tau.iter().zip(&n_bubbles) {
        println!("{:10.6}, {:10.6}, {:10.6}", t, t / beta, n);
    }

    println!("{}", n_bubbles.len());

    Ok(())
}

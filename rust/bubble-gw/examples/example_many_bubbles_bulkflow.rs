use std::error::Error;

use bubble_gw::many_bubbles::generalized_bulk_flow_segment::GeneralizedBulkFlow;
use bubble_gw::many_bubbles::lattice::{BoundaryConditions, CartesianLattice};
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use nalgebra::{Point3, Vector3};
use ndarray::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let bubbles_interior = arr2(&[
        [0.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 4.0],
        [0.0, 4.0, 2.0, 4.0],
        [0.0, 0.0, 2.0, 1.0],
    ]);
    let lbox = 10.; // lattice size
    let lattice = CartesianLattice::new(
        Point3::new(0., 0., 0.),
        [
            Vector3::new(lbox, 0., 0.),
            Vector3::new(0., lbox, 0.),
            Vector3::new(0., 0., lbox),
        ],
    );

    let mut lattice_bubbles = LatticeBubbles::new(
        bubbles_interior,
        None, // start with no bubbles_exterior
        lattice,
    )?;
    // generate bubbles_exterior corresponding to Periodic boundary condition
    lattice_bubbles.with_boundary_condition(BoundaryConditions::Periodic);

    let mut generalized_bulk_flow = GeneralizedBulkFlow::new(lattice_bubbles)?;
    generalized_bulk_flow.set_resolution(100, 200, true)?;

    let w_arr = Array1::geomspace(1e-2, 1e2, 100).unwrap().to_vec();
    let coefficients_sets = vec![vec![0.0], vec![1.0]];
    let powers_sets = vec![vec![3.0], vec![3.0]];
    generalized_bulk_flow.set_gradient_scaling_params(coefficients_sets, powers_sets, None)?;

    let a_idx = 0;
    let t = 8.0;
    let first_bubble = generalized_bulk_flow
        .first_colliding_bubbles()
        .unwrap()
        .slice(s![a_idx, .., ..]);
    let delta_tab_grid = generalized_bulk_flow.compute_delta_tab(a_idx, &first_bubble)?;
    let collision_status =
        generalized_bulk_flow.compute_collision_status(a_idx, t, &first_bubble, &delta_tab_grid)?;
    let delta_ta = t - generalized_bulk_flow.bubbles_interior().spacetime[a_idx][0];

    let (b_plus, b_minus) = generalized_bulk_flow.compute_b_integral(
        25,
        &collision_status,
        &delta_tab_grid,
        delta_ta,
    )?;
    println!("b_plus[0] at cosθ_idx=25: {}", b_plus[0]);
    println!("b_minus[0] at cosθ_idx=25: {}", b_minus[0]);
    println!("b_plus[1] at cosθ_idx=25: {}", b_plus[1]);
    println!("b_minus[1] at cosθ_idx=25: {}", b_minus[1]);

    let (a_plus, a_minus) = generalized_bulk_flow.compute_a_integral(
        a_idx,
        &w_arr,
        t,
        &first_bubble,
        &delta_tab_grid,
    )?;

    println!("a_plus[0, 50]: {}", a_plus[[0, 50]]);
    println!("a_minus[0, 50]: {}", a_minus[[0, 50]]);
    println!("a_plus[1, 50]: {}", a_plus[[1, 50]]);
    println!("a_minus[1, 50]: {}", a_minus[[1, 50]]);

    let c_matrix = generalized_bulk_flow.compute_c_integral(&w_arr, Some(0.0), 20.0, 1000, None)?;
    println!("c_matrix[0, 0, 0]: {}", c_matrix[[0, 0, 0]]);
    println!("c_matrix[1, 0, 0]: {}", c_matrix[[1, 0, 0]]);
    println!("c_matrix[0, 1, 0]: {}", c_matrix[[0, 1, 0]]);
    println!("c_matrix[1, 1, 0]: {}", c_matrix[[1, 1, 0]]);

    Ok(())
}

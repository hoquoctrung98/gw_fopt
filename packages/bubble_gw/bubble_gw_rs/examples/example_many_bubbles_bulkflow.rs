use bubble_gw_rs::many_bubbles::bubble_formation::{
    BoundaryConditions, Lattice, LatticeType, generate_bubbles_exterior,
};
use bubble_gw_rs::many_bubbles::bubbles::Bubbles;
use bubble_gw_rs::many_bubbles::bulk_flow::{BulkFlow, BulkFlowError};
use ndarray::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let bubbles_interior = arr2(&[
        [0.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 4.0],
        [0.0, 4.0, 2.0, 4.0],
        [0.0, 0.0, 2.0, 1.0],
    ]);
    let lattice = Lattice::new(
        LatticeType::Cartesian {
            sizes: [10., 10., 10.],
        },
        100,
    )
    .unwrap();
    let bubbles_exterior =
        generate_bubbles_exterior(&lattice, &bubbles_interior, BoundaryConditions::Periodic);

    let mut bulk_flow = BulkFlow::new(Bubbles::new(bubbles_interior, bubbles_exterior, true)?)?;
    bulk_flow.set_resolution(100, 200, true)?;

    let w_arr = Array1::geomspace(1e-2, 1e2, 100).unwrap().to_vec();
    let coefficients_sets = vec![vec![0.0], vec![1.0]];
    let powers_sets = vec![vec![3.0], vec![3.0]];
    bulk_flow.set_gradient_scaling_params(coefficients_sets, powers_sets, None)?;

    let a_idx = 0;
    let t = 8.0;
    let first_bubble = bulk_flow
        .first_colliding_bubbles()
        .ok_or_else(|| BulkFlowError::UninitializedField("first_colliding_bubbles".to_string()))?
        .slice(s![a_idx, .., ..]);
    let delta_tab_grid = bulk_flow.compute_delta_tab(a_idx, &first_bubble)?;
    let collision_status =
        bulk_flow.compute_collision_status(a_idx, t, &first_bubble, &delta_tab_grid)?;
    let delta_ta = t - bulk_flow.bubbles_interior()[[a_idx, 0]];

    let (b_plus, b_minus) =
        bulk_flow.compute_b_integral(25, &collision_status, &delta_tab_grid, delta_ta)?;
    println!("b_plus[0] at cosθ_idx=25: {}", b_plus[0]);
    println!("b_minus[0] at cosθ_idx=25: {}", b_minus[0]);
    println!("b_plus[1] at cosθ_idx=25: {}", b_plus[1]);
    println!("b_minus[1] at cosθ_idx=25: {}", b_minus[1]);

    let (a_plus, a_minus) =
        bulk_flow.compute_a_integral(a_idx, &w_arr, t, &first_bubble, &delta_tab_grid)?;

    println!("a_plus[0, 50]: {}", a_plus[[0, 50]]);
    println!("a_minus[0, 50]: {}", a_minus[[0, 50]]);
    println!("a_plus[1, 50]: {}", a_plus[[1, 50]]);
    println!("a_minus[1, 50]: {}", a_minus[[1, 50]]);

    let c_matrix = bulk_flow.compute_c_integral(&w_arr, Some(0.0), 20.0, 1000, None)?;
    println!("c_matrix[0, 0, 0]: {}", c_matrix[[0, 0, 0]]);
    println!("c_matrix[1, 0, 0]: {}", c_matrix[[1, 0, 0]]);
    println!("c_matrix[0, 1, 0]: {}", c_matrix[[0, 1, 0]]);
    println!("c_matrix[1, 1, 0]: {}", c_matrix[[1, 1, 0]]);

    Ok(())
}

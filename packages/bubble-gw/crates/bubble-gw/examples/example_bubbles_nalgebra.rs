use std::error::Error;

use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow as BulkFlowNalgebra;
use bubble_gw::many_bubbles::lattice::CartesianLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles as LatticeBubblesNalgebra;
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, arr2};

fn main() -> Result<(), Box<dyn Error>> {
    let l_box = 20.;
    let lattice_nalgebra = CartesianLattice::new(
        // Point3::new(L / 2., L / 2., L / 2.),
        Point3::new(0., 0., 0.),
        [
            Vector3::new(l_box, 0., 0.),
            Vector3::new(0., l_box, 0.),
            Vector3::new(0., 0., l_box),
        ],
    );
    let bubbles_interior = arr2(&[[0.0, 5.5, 5.0, 5.0], [0.0, 4.5, 5.0, 10.0]]);
    let mut lattice_bubbles = LatticeBubblesNalgebra::with_bubbles(
        bubbles_interior.clone(),
        Array2::zeros((0, 4)),
        lattice_nalgebra,
    )
    .unwrap();
    lattice_bubbles
        .with_boundary_condition(bubble_gw::many_bubbles::lattice::BoundaryConditions::Periodic);
    let mut generalized_bulk_flow = BulkFlowNalgebra::new(lattice_bubbles).unwrap();
    generalized_bulk_flow.set_num_threads(1)?;
    generalized_bulk_flow.set_resolution(10, 10, true)?;
    generalized_bulk_flow.compute_first_colliding_bubble(0)?;

    Ok(())
}

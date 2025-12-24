use bubble_gw::many_bubbles::bubbles::LatticeBubbles;
use bubble_gw::many_bubbles::bulk_flow::BulkFlow;
use bubble_gw::many_bubbles::lattice::{
    BoundaryConditions, Lattice, LatticeType, generate_bubbles_exterior,
};
use bubble_gw::many_bubbles_nalgebra::{
    bulk_flow::BulkFlow as BulkFlowNalgebra, lattice::CartesianLattice,
    lattice_bubbles::LatticeBubbles as LatticeBubblesNalgebra,
};
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, arr2, s};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let L = 20.;
    let lattice_nalgebra = CartesianLattice::new(
        // Point3::new(L / 2., L / 2., L / 2.),
        Point3::new(0., 0., 0.),
        [
            Vector3::new(L, 0., 0.),
            Vector3::new(0., L, 0.),
            Vector3::new(0., 0., L),
        ],
    );
    let bubbles_interior = arr2(&[[0.0, 5.5, 5.0, 5.0], [0.0, 4.5, 5.0, 10.0]]);
    let mut lattice_bubbles = LatticeBubblesNalgebra::new(
        bubbles_interior.clone(),
        Array2::zeros((0, 4)),
        lattice_nalgebra,
        false,
    )
    .unwrap();
    lattice_bubbles.with_boundary_condition(
        bubble_gw::many_bubbles_nalgebra::lattice::BoundaryConditions::Periodic,
    );
    let mut bulk_flow_nalgebra = BulkFlowNalgebra::new(lattice_bubbles).unwrap();
    bulk_flow_nalgebra.set_num_threads(1);
    bulk_flow_nalgebra.set_resolution(10, 10, true).unwrap();
    bulk_flow_nalgebra.compute_first_colliding_bubble(0);

    let mut bulk_flow_original = BulkFlow::new(
        LatticeBubbles::new(
            bubbles_interior.clone(),
            generate_bubbles_exterior(
                &Lattice::new(LatticeType::Cartesian { sizes: [L, L, L] }, 10).unwrap(),
                &bubbles_interior,
                BoundaryConditions::Periodic,
            ),
            false,
        )
        .unwrap(),
    )
    .unwrap();
    bulk_flow_original.set_num_threads(1);
    bulk_flow_original.set_resolution(10, 10, true).unwrap();
    let foo = bulk_flow_original.compute_first_colliding_bubble(0);

    Ok(())
}

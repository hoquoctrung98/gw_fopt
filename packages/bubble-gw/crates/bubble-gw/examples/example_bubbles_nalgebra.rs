use bubble_gw::many_bubbles::{
    generalized_bulk_flow::GeneralizedBulkFlow as BulkFlowNalgebra, lattice::CartesianLattice,
    lattice_bubbles::LatticeBubbles as LatticeBubblesNalgebra,
};
use bubble_gw::many_bubbles_legacy::bubbles::LatticeBubbles;
use bubble_gw::many_bubbles_legacy::bulk_flow::BulkFlow;
use bubble_gw::many_bubbles_legacy::lattice::{
    BoundaryConditions, Lattice, LatticeType, generate_bubbles_exterior,
};
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, arr2};
use std::error::Error;

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
        false,
    )
    .unwrap();
    lattice_bubbles
        .with_boundary_condition(bubble_gw::many_bubbles::lattice::BoundaryConditions::Periodic);
    let mut bulk_flow_nalgebra = BulkFlowNalgebra::new(lattice_bubbles).unwrap();
    bulk_flow_nalgebra.set_num_threads(1)?;
    bulk_flow_nalgebra.set_resolution(10, 10, true)?;
    bulk_flow_nalgebra.compute_first_colliding_bubble(0)?;

    let mut bulk_flow_original = BulkFlow::new(
        LatticeBubbles::new(
            bubbles_interior.clone(),
            generate_bubbles_exterior(
                &Lattice::new(
                    LatticeType::Cartesian {
                        sizes: [l_box, l_box, l_box],
                    },
                    10,
                )
                .unwrap(),
                &bubbles_interior,
                BoundaryConditions::Periodic,
            ),
            false,
        )
        .unwrap(),
    )
    .unwrap();
    bulk_flow_original.set_num_threads(1)?;
    bulk_flow_original.set_resolution(10, 10, true).unwrap();

    Ok(())
}

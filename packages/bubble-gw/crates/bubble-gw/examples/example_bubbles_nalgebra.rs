use bubble_gw::{
    many_bubbles_nalgebra::lattice::CartesianLattice,
    many_bubbles_nalgebra::lattice_bubbles::LatticeBubbles,
};
use nalgebra::{Point3, Vector3};
use ndarray::arr2;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let L = 10.;
    let lattice = CartesianLattice::new(
        // Point3::new(L / 2., L / 2., L / 2.),
        Point3::new(0., 0., 0.),
        [
            Vector3::new(L, 0., 0.),
            Vector3::new(0., L, 0.),
            Vector3::new(0., 0., L),
        ],
    );
    let bubbles_interior = arr2(&[[0.0, 5.5, 5.0, 5.0], [0.0, 4.5, 5.0, 10.0]]);
    let lattice_bubbles =
        LatticeBubbles::new(bubbles_interior.clone(), bubbles_interior.clone(), lattice, true)
            .unwrap();

    Ok(())
}

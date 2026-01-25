use std::error::Error;

use bubble_gw::many_bubbles::lattice::CartesianLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use bubble_gw::many_bubbles::{self};
use nalgebra::{Point3, Vector3};

fn main() -> Result<(), Box<dyn Error>> {
    let lbox = 200.;
    let lattice = many_bubbles::lattice::CartesianLattice::new(
        Point3::new(0., 0., 0.),
        [
            Vector3::new(lbox, 0., 0.),
            Vector3::new(0., lbox, 0.),
            Vector3::new(0., 0., lbox),
        ],
    );
    let bubbles_interior = LatticeBubbles::<CartesianLattice>::load_bubbles_from_csv(
        "/home/hoquoctrung/workspace/projects/SISSA_projects/code/gw_fopt/notebooks/toby_data/toby_bubbles_N64.csv",
        false,
    ).unwrap();

    let lattice_bubbles = LatticeBubbles::new(bubbles_interior, None, lattice);
    Ok(())
}

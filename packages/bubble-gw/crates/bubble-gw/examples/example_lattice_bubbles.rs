use std::error::Error;

use bubble_gw::many_bubbles;
use nalgebra::{Point3, Vector3};

fn main() -> Result<(), Box<dyn Error>> {
    let lbox = 5.;
    let lattice = many_bubbles::lattice::CartesianLattice::new(
        Point3::new(0., 0., 0.),
        [
            Vector3::new(lbox, 0., 0.),
            Vector3::new(0., lbox, 0.),
            Vector3::new(0., 0., lbox),
        ],
    );

    let lattice = many_bubbles::lattice::BuiltInLattice::Cartesian(lattice);
    let mut lattice_bubbles = many_bubbles::lattice_bubbles::LatticeBubbles::new(lattice);
    // let nucleation_strategy =
    // many_bubbles::bubbles_nucleation::UniformAtFixedTime {     n_bubbles:
    // 200,     t0: 0.,
    //     seed: None,
    // };
    let nucleation_strategy = many_bubbles::bubbles_nucleation::FixedRateNucleation::new(
        0.1,
        1.,
        0.0,
        0.01,
        None,
        many_bubbles::bubbles_nucleation::VolumeRemainingMethod::Approximation,
    );
    lattice_bubbles.nucleate_and_update(
        nucleation_strategy,
        many_bubbles::lattice::BoundaryConditions::Periodic,
    )?;
    println!("{:?}", lattice_bubbles.interior.n_bubbles());
    Ok(())
}

use std::error::Error;

use bubble_gw::many_bubbles::bubbles_nucleation::{NucleationStrategy, VolumeRemainingMethod};
use bubble_gw::many_bubbles::lattice::BoundaryConditions;
use bubble_gw::many_bubbles::{self};
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
    let mut nucleation_strategy = many_bubbles::bubbles_nucleation::FixedRateNucleation::new(
        0.1,
        1.,
        0.0,
        0.01,
        None,
        VolumeRemainingMethod::Approximation,
    );
    let lattice_bubbles = nucleation_strategy.nucleate(&lattice, BoundaryConditions::Periodic)?;
    println!("{:?}", lattice_bubbles.interior.n_bubbles());
    Ok(())
}

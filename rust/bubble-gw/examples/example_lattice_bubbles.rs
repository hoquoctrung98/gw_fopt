use std::error::Error;

use bubble_gw::many_bubbles::bubbles_nucleation::FixedRateNucleationMethod;
use bubble_gw::many_bubbles::lattice::BoundaryConditions;
use bubble_gw::many_bubbles::{self};
use nalgebra::{Point3, Vector3};

fn main() -> Result<(), Box<dyn Error>> {
    let lbox = 11.;
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
        0.05,
        1.,
        0.0,
        0.01,
        Some(0),
        FixedRateNucleationMethod::Approximation,
        None,
        None,
    )?;
    let lattice_bubbles = nucleation_strategy.nucleate(&lattice, BoundaryConditions::Periodic)?;

    // dbg!(nucleation_strategy.volume_remaining_history);

    let nucleation_time_arr: Vec<f64> = lattice_bubbles
        .interior
        .spacetime
        .iter()
        .map(|bubble| bubble[0])
        .collect();

    dbg!(nucleation_time_arr);
    dbg!(lattice_bubbles.interior.spacetime.len());
    Ok(())
}

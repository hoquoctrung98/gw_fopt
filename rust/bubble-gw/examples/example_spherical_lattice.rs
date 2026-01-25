use std::error::Error;

use bubble_gw::many_bubbles::bubbles::Bubbles;
use bubble_gw::many_bubbles::lattice::{
    BoundaryConditions,
    GenerateBubblesExterior,
    SphericalLattice,
};
use nalgebra::{Point3, Vector4};

fn main() -> Result<(), Box<dyn Error>> {
    let bubbles_interior = Bubbles::new(vec![Vector4::new(0., 0., 3., 0.)]);
    let lattice = SphericalLattice::new(Point3::new(0., 1., 0.), 10.);
    let bubbles_exterior =
        lattice.generate_bubbles_exterior(bubbles_interior, BoundaryConditions::Reflection);
    println!("{bubbles_exterior:?}");

    Ok(())
}

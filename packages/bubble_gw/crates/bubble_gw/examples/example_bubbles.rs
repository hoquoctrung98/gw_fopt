use bubble_gw::many_bubbles::bubble_formation::generate_random_bubbles;
use bubble_gw::many_bubbles::bubbles::LatticeBubbles;
use bubble_gw::many_bubbles::lattice::{BoundaryConditions, Lattice, LatticeType};
use ndarray::arr2;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let lattice = Lattice::new(
        LatticeType::Cartesian {
            sizes: [10., 10., 10.],
        },
        100,
    )
    .unwrap();
    let bubbles_config =
        generate_random_bubbles(lattice, BoundaryConditions::Periodic, -0.1, 0.0, 20, Some(0))?;
    let interior_path = "./bubbles_interior.csv";
    let exterior_path = "./bubbles_exterior.csv";
    let has_header = true;
    bubbles_config.write_bubbles_interior_to_csv(&interior_path, has_header)?;
    bubbles_config.save_bubbles_exterior_to_csv(&exterior_path, has_header)?;

    // read bubbles from csv
    let mut new_bubbles_config =
        LatticeBubbles::from_csv_files(&interior_path, &exterior_path, true, has_header)?;
    println!("{:?}", new_bubbles_config);

    let extra_bubbles = arr2(&[
        [-7.83765486e-2, 3.18613046e0, 3.53075225e0, 5.56789382e0],
        [-5.39420343e-2, 9.24262657e0, 2.27040645e0, 5.59332295e0],
        [-2.68886584e-2, 7.73460184e0, 2.58446342e-1, 5.84159262e0],
        [-3.81845405e-2, 6.13769208e0, 7.85237693e0, 6.65888552e0],
        [-9.15593573e-2, 8.39645306e-1, 5.18711871e0, 7.52937952e0],
        [-1.50360038e-3, 6.66013000e0, 6.72066944e0, 7.91788143e0],
        [-7.37829367e-2, 7.71986906e0, 2.18696911e0, 7.93713457e0],
        [-3.89967168e-2, 4.76126404e-1, 8.00901707e0, 8.13782902e0],
        [-3.62250577e-2, 4.78849413e0, 8.76077110e0, 8.70660394e0],
        [-9.36768522e-2, 9.86890568e0, 2.16797563e0, 9.77053397e0],
    ]);

    new_bubbles_config.add_interior_bubbles(extra_bubbles, true)?;

    Ok(())
}

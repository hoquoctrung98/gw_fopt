use bubble_gw_rs::many_bubbles::bubble_formation::{
    BoundaryConditions, Lattice, LatticeType, generate_bubbles_exterior,
};
use bubble_gw_rs::many_bubbles::bubbles::{Bubbles, BubblesError};
use ndarray::{Array2, arr2};
use rand::{self, Rng, SeedableRng};

use rand::random;
use rand::rngs::StdRng;
use std::borrow::Borrow;
use std::error::Error;

pub fn generate_random_bubbles(
    lattice: impl Borrow<Lattice>,
    t_begin: f64,
    t_end: f64,
    n_bubbles: usize,
    seed: Option<u64>,
) -> Result<Bubbles, BubblesError> {
    let lattice = lattice.borrow();
    let mut rng = match seed {
        Some(seed_value) => StdRng::seed_from_u64(seed_value),
        None => StdRng::seed_from_u64(random::<u64>()),
    };
    let bounds = lattice.lattice_bounds();
    let [x_r, y_r, z_r] = [bounds[0], bounds[1], bounds[2]].map(|(lo, hi)| lo..=hi);

    // Start with a valid empty configuration
    let mut bubbles = Bubbles::new(
        Array2::zeros((0, 4)),
        Array2::zeros((0, 4)),
        false, // no sorting yet
    )?;

    // Generate interior bubbles one by one with rejection sampling
    for _ in 0..n_bubbles {
        loop {
            let t = rng.random_range(t_begin..=t_end);
            let x = rng.random_range(x_r.clone());
            let y = rng.random_range(y_r.clone());
            let z = rng.random_range(z_r.clone());

            let candidate = Array2::from_shape_vec((1, 4), vec![t, x, y, z])
                .expect("Failed to create candidate row");

            if bubbles.add_interior_bubbles(candidate, true).is_ok() {
                break;
            }
        }
    }

    let bubbles_exterior =
        generate_bubbles_exterior(lattice, &bubbles.interior, BoundaryConditions::Reflection);

    bubbles.add_exterior_bubbles(bubbles_exterior, true)?;

    bubbles.sort_by_time()?;
    Ok(bubbles)
}

fn main() -> Result<(), Box<dyn Error>> {
    let lattice = Lattice::new(
        LatticeType::Cartesian {
            sizes: [10., 10., 10.],
        },
        100,
    )
    .unwrap();
    let bubbles_config = generate_random_bubbles(lattice, -0.1, 0.0, 10, Some(0))?;
    let interior_path = "./bubbles_interior.csv";
    let exterior_path = "./bubbles_exterior.csv";
    let has_header = true;
    bubbles_config.save_interior_to_csv(&interior_path, has_header)?;
    bubbles_config.save_exterior_to_csv(&exterior_path, has_header)?;

    // read bubbles from csv
    let mut new_bubbles_config =
        Bubbles::from_csv_files(&interior_path, &exterior_path, true, has_header)?;
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

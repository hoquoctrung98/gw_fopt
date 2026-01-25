use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use bubble_gw::many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow;
use bubble_gw::many_bubbles::lattice::EmptyLattice;
use bubble_gw::many_bubbles::lattice_bubbles::LatticeBubbles;
use bubble_gw::utils::is_close::IsClose;
use ndarray::{Array1, Array2, arr2};
use num::complex::Complex64;

const ABS_TOL: f64 = 1e-2;
const REL_TOL: f64 = 1e-2;

#[test]
fn test_bulk_flow_two_bubbles() -> Result<(), Box<dyn Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR")).unwrap();
    let file = File::open("./tests/envelope_2bubbles_scan=n_phix.csv")?;
    let mut rdr = csv::Reader::from_reader(BufReader::new(file));
    let headers = rdr.headers()?.clone();

    let idx = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("Column '{name}' not found"))
    };

    let i_n_cos = idx("n_cos_thetax");
    let i_n_phi = idx("n_phix");
    let i_w = idx("w");
    let i_c_plus_re = idx("c_plus_re");
    let i_c_plus_im = idx("c_plus_im");
    let i_c_cross_re = idx("c_cross_re");
    let i_c_cross_im = idx("c_cross_im");

    let mut records = Vec::new();
    for r in rdr.records() {
        records.push(r?);
    }

    let mut groups: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, rec) in records.iter().enumerate() {
        let n_cos: i64 = rec[i_n_cos].parse()?;
        let n_phi: i64 = rec[i_n_phi].parse()?;
        groups.entry((n_cos, n_phi)).or_default().push(i);
    }

    let bubbles_interior = arr2(&[[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
    let bubbles_exterior = Array2::<f64>::zeros((0, 4));
    let coefficients_sets = vec![vec![0.0], vec![1.0]];
    let powers_sets = vec![vec![3.0], vec![3.0]];

    for (&(n_cos_val, n_phi_val), rows) in &groups {
        println!("Testing n_cos={n_cos_val}, n_phi={n_phi_val} → {} points", rows.len());

        // Sort by w
        let mut w_with_idx: Vec<(f64, usize)> = rows
            .iter()
            .map(|&i| (records[i][i_w].parse::<f64>().unwrap(), i))
            .collect();
        w_with_idx.sort_by(|a, b| a.0.total_cmp(&b.0));

        let w_arr = Array1::from_iter(w_with_idx.iter().map(|&(w, _)| w)).to_vec();
        let sorted_rows: Vec<usize> = w_with_idx.iter().map(|&(_, i)| i).collect();

        let expected_plus: Vec<Complex64> = sorted_rows
            .iter()
            .map(|&i| {
                let re = records[i][i_c_plus_re].parse::<f64>().unwrap();
                let im = records[i][i_c_plus_im].parse::<f64>().unwrap();
                Complex64::new(re, im)
            })
            .collect();

        let expected_cross: Vec<Complex64> = sorted_rows
            .iter()
            .map(|&i| {
                let re = records[i][i_c_cross_re].parse::<f64>().unwrap();
                let im = records[i][i_c_cross_im].parse::<f64>().unwrap();
                Complex64::new(re, im)
            })
            .collect();

        let mut bulk = GeneralizedBulkFlow::new(LatticeBubbles::new(
            bubbles_interior.clone(),
            Some(bubbles_exterior.clone()),
            EmptyLattice {},
        )?)?;
        bulk.set_resolution(n_cos_val as usize, n_phi_val as usize, true)?;
        bulk.set_gradient_scaling_params(coefficients_sets.clone(), powers_sets.clone(), None)?;

        let c_matrix = bulk.compute_c_integral(&w_arr, Some(0.0), 8.0, 1000, None)?;

        // Extract computed results for coeff=1.0
        let computed_plus: Vec<Complex64> = c_matrix.slice(ndarray::s![0, 0, ..]).to_vec();
        let computed_cross: Vec<Complex64> = c_matrix.slice(ndarray::s![1, 0, ..]).to_vec();

        // Use IsClose!
        computed_plus
            .is_close(&expected_plus, ABS_TOL, REL_TOL)
            .map_err(|e| format!("c_plus failed:\n{e}"))?;
        computed_cross
            .is_close(&expected_cross, ABS_TOL, REL_TOL)
            .map_err(|e| format!("c_cross failed:\n{e}"))?;

        println!("PASSED: {n_cos_val}×{n_phi_val} with {} points", w_arr.len());
    }

    Ok(())
}

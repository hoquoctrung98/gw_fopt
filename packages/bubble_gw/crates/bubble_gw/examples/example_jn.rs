use bubble_gw::utils::is_close::IsClose;
use puruspe::Jn;
use scirs2_special::jn;

fn main() {
    let puruspe_arr: Vec<f64> = (0..=2).map(|n| 1.0000000001 * Jn(n, 1.)).collect();
    let scirs2_arr: Vec<f64> = (0..=2).map(|n| jn(n, 1.)).collect();
    puruspe_arr.is_close(&scirs2_arr, 1e-10, 1e-10).unwrap();
}

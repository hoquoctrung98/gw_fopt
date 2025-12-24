use bubble_gw::many_bubbles::bubbles::Bubbles;
use nalgebra::{Rotation3, Vector4};

fn main() {
    let bubbles = Bubbles::new(vec![
        Vector4::new(1.0, 1.0, 0.0, 0.0),
        Vector4::new(2.0, 0.0, 1.0, 0.0),
    ]);

    // Rotate 90° around Z axis
    let rot_z_90 =
        Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), std::f64::consts::PI / 2.0);

    // let rotated = bubbles.rotate_spatial(rot_z_90);
    // println!("{:?}", rotated);
    // println!("foo");
    // → [ (1.0, 0.0, 1.0, 0.0), (2.0, -1.0, 0.0, 0.0) ]

    // Or in-place:
    // let mut bubbles = bubbles;
    // bubbles.rotate_spatial_mut(rot_z_90);
}

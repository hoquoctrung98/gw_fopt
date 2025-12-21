use nalgebra::{Vector2, Vector3, Vector4};
use nalgebra_spacetime::Lorentzian;

fn main() {
    let v1 = Vector4::new(1.0, 2.0, 1.0, 3.0);

    // A vector with three components.
    let v = Vector3::new(1, 2, 3);

    // ✅ Runtime-safe fallback (no panic):
    match v.get(5) {
        Some(val) => println!("{:?}", val),
        None => println!("Index 5 out of bounds"),
    }

    // Rest of your code is fine:
    println!("{:?}", v1.spacelike_norm());
    println!("{:?}", v1.timelike_norm());
    println!("{:?}", v1.scalar(&v1));
    println!("{:?}", v1.dual().dot(&v1));
    println!("{:?}", v1.dot(&v1));
    let num = Vector2::new(2, 2);
    let v1 = Vector3::new(num, 2 * num, 3 * num);
    let num = Vector2::new(3, 5);
    let v2 = Vector3::new(num, num, num);
    let v3 = v1 + v2;
    println!("{:?}", v3);
}

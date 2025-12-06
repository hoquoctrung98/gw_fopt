use bubble_gw::utils::segment::*;
use std::fmt::Debug;

fn print_segments<T: Debug + PartialEq>(
    title: &str,
    input: &[T],
    seg: &ConstrainedSegment<'_, T, impl SegmentConstraint<T>>,
) {
    println!("{title}");
    println!("  Input: {:?}", input);
    print!("  Runs :");
    for run in seg.as_const_subsegments() {
        print!(" {:?}", run);
    }
    println!("\n");
}

fn example_different_constraints() {
    println!("=== Example: Different constraints ===\n");

    let data = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];

    if let Some(seg) = data.try_into_constrained_segment(&Unconstrained) {
        print_segments("Unconstrained", &data, &seg);
    }

    if let Some(seg) = data.try_into_constrained_segment(&UniqueValues) {
        print_segments("UniqueValues", &data, &seg);
    } else {
        println!("UniqueValues → REJECTED\n");
    }

    if let Some(seg) = data.try_into_constrained_segment(&Sorted) {
        print_segments("Sorted", &data, &seg);
    }

    let max3 = MaxRunLength(3);
    if let Some(seg) = data.try_into_constrained_segment(&max3) {
        print_segments("MaxRunLength(3)", &data, &seg);
    }
}

fn example_different_inputs() {
    println!("=== Example: Many different input types (Unconstrained) ===\n");

    // &[T]
    let arr = [10; 7];
    if let Some(seg) = arr.try_into_constrained_segment(&Unconstrained) {
        print_segments("array [10; 7×]", &arr, &seg);
    }

    // Vec<T>
    let vec_data = vec!['a', 'a', 'b', 'b', 'b', 'c'];
    if let Some(seg) = vec_data.try_into_constrained_segment(&Unconstrained) {
        print_segments("Vec<char>", &vec_data, &seg);
    }

    // &str → bytes
    let text = "aaabbcccc";
    if let Some(seg) = text.as_bytes().try_into_constrained_segment(&Unconstrained) {
        print_segments("&str as bytes", text.as_bytes(), &seg);
    }

    // Vec<String>
    let words = vec!["hello".to_string(); 4];
    if let Some(seg) = words.try_into_constrained_segment(&Unconstrained) {
        print_segments("Vec<String>", &words, &seg);
    }

    // Empty
    let empty: &[i32] = &[];
    if let Some(seg) = empty.try_into_constrained_segment(&Unconstrained) {
        print_segments("empty slice", empty, &seg);
    }
}

fn main() {
    example_different_constraints();
    example_different_inputs();
}

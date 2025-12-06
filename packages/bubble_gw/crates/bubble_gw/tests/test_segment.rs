use bubble_gw::utils::segment::*;

fn collect_runs<'a, T: PartialEq + Clone, C: SegmentConstraint<T>>(
    seg: &ConstrainedSegment<'a, T, C>,
) -> Vec<Vec<T>> {
    seg.as_const_subsegments().map(|s| s.to_vec()).collect()
}

#[test]
fn test_unconstrained() {
    let data: Vec<i32> = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];
    let constraint = Unconstrained;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1, 1], vec![2, 2], vec![4, 4, 4], vec![5, 5]]);
}

#[test]
fn test_unique_values_accept() {
    let data: Vec<i32> = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];
    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1, 1], vec![2, 2], vec![4, 4, 4], vec![5, 5]]);
}

#[test]
fn test_unique_values_reject() {
    let data: Vec<i32> = vec![1, 1, 2, 2, 1, 1];
    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_none());
}

#[test]
fn test_sorted_accept() {
    let data: Vec<i32> = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];
    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1, 1], vec![2, 2], vec![4, 4, 4], vec![5, 5]]);
}

#[test]
fn test_sorted_reject() {
    let data: Vec<i32> = vec![1, 1, 3, 2, 2];
    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_none());
}

#[test]
fn test_max_run_length_accept() {
    let data: Vec<i32> = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];
    let constraint = MaxRunLength(3);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1, 1], vec![2, 2], vec![4, 4, 4], vec![5, 5]]);
}

#[test]
fn test_max_run_length_reject() {
    let data: Vec<i32> = vec![1, 1, 1, 1, 2, 2];
    let constraint = MaxRunLength(3);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_none());
}

#[test]
fn test_allowed_values_accept() {
    let data: Vec<i32> = vec![1, 1, 1, 2, 2, 4, 4, 4, 5, 5];
    let allowed = AllowedValues(&[1i32, 2, 4, 5][..]);
    let seg = data.try_into_constrained_segment(&allowed);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1, 1], vec![2, 2], vec![4, 4, 4], vec![5, 5]]);
}

#[test]
fn test_allowed_values_reject() {
    let data: Vec<i32> = vec![1, 1, 3, 3];
    let allowed = AllowedValues(&[1i32, 2, 4][..]);
    let seg = data.try_into_constrained_segment(&allowed);
    assert!(seg.is_none());
}

#[test]
fn test_empty_slice() {
    let data: Vec<i32> = vec![];
    let constraint = Unconstrained;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![] as Vec<Vec<i32>>);

    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let constraint = MaxRunLength(3);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let allowed = AllowedValues(&[1i32, 2][..]);
    let seg = data.try_into_constrained_segment(&allowed);
    assert!(seg.is_some());
}

#[test]
fn test_single_element() {
    let data: Vec<i32> = vec![42];
    let constraint = Unconstrained;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![42]]);

    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let constraint = MaxRunLength(1);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let allowed = AllowedValues(&[42i32][..]);
    let seg = data.try_into_constrained_segment(&allowed);
    assert!(seg.is_some());
}

#[test]
fn test_all_unique_no_repeats() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1], vec![2], vec![3], vec![4], vec![5]]);

    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());

    let constraint = MaxRunLength(1);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
}

#[test]
fn test_all_same_valid() {
    let data: Vec<i32> = vec![7, 7, 7];
    let constraint = MaxRunLength(3);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![7, 7, 7]]);

    let constraint = UniqueValues;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some()); // Only one unique value

    let constraint = Sorted;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
}

#[test]
fn test_all_same_invalid() {
    let data: Vec<i32> = vec![7, 7, 7, 7];
    let constraint = MaxRunLength(3);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_none());
}

#[test]
fn test_max_run_length_custom_next_run_end() {
    let data: Vec<i32> = vec![1, 1, 2, 2, 2, 3];
    let constraint = MaxRunLength(2);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_none()); // Run of 3 twos would fail verify

    let data: Vec<i32> = vec![1, 1, 2, 2, 3, 3];
    let constraint = MaxRunLength(2);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec![1, 1], vec![2, 2], vec![3, 3]]);
}

#[test]
fn test_iterator_exhaustion() {
    let data: Vec<i32> = vec![1, 1, 2];
    let constraint = Unconstrained;
    let seg = data.try_into_constrained_segment(&constraint).unwrap();
    let mut iter = seg.as_const_subsegments();
    assert_eq!(iter.next(), Some(&data[0..2]));
    assert_eq!(iter.next(), Some(&data[2..3]));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None); // Fused
}

#[test]
fn test_different_types() {
    let data: Vec<char> = vec!['a', 'a', 'b', 'b', 'b', 'c'];
    let constraint = Unconstrained;
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec!['a', 'a'], vec!['b', 'b', 'b'], vec!['c']]);

    let data: Vec<String> = vec!["hello".to_string(); 4];
    let constraint = MaxRunLength(4);
    let seg = data.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
    let runs = collect_runs(&seg.unwrap());
    assert_eq!(runs, vec![vec!["hello".to_string(); 4]]);

    let bytes = "aaabbcccc".as_bytes();
    let constraint = Sorted;
    let seg = bytes.try_into_constrained_segment(&constraint);
    assert!(seg.is_some());
}

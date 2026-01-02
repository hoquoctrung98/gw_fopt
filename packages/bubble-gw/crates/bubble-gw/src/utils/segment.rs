use std::fmt::Debug;
use std::hash::Hash;

/// Core trait — a constraint that the whole slice must satisfy
pub trait SegmentConstraint<T> {
    fn verify(&self, slice: &[T]) -> bool;

    /// Fast path: how far does the current run go?
    fn next_run_end(&self, slice: &[T], start: usize) -> usize
    where
        T: PartialEq,
    {
        let value = &slice[start];
        slice[start..]
            .iter()
            .position(|x| x != value)
            .map_or(slice.len(), |off| start + off)
    }
}

/// A validated view: the slice **is guaranteed** to satisfy the constraint C
#[derive(Clone, Debug)]
pub struct ConstrainedSegment<'a, T, C>
where
    C: SegmentConstraint<T>,
{
    slice: &'a [T],
    constraint: &'a C,
}

impl<'a, T: PartialEq, C: SegmentConstraint<T>> ConstrainedSegment<'a, T, C> {
    /// Get an iterator over the iterator over contiguous runs of equal elements
    pub fn as_const_subsegments(&self) -> ConstSubSegments<'a, T, C> {
        ConstSubSegments {
            slice: self.slice,
            idx: 0,
            constraint: self.constraint,
        }
    }
}

/// The iterator over runs of equal elements
#[derive(Clone, Debug)]
pub struct ConstSubSegments<'a, T, C>
where
    C: SegmentConstraint<T>,
{
    slice: &'a [T],
    idx: usize,
    constraint: &'a C,
}

impl<'a, T: PartialEq, C: SegmentConstraint<T>> Iterator for ConstSubSegments<'a, T, C> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.slice.len() {
            return None;
        }
        let start = self.idx;
        self.idx = self.constraint.next_run_end(self.slice, start);
        Some(&self.slice[start..self.idx])
    }
}

impl<'a, T: PartialEq, C: SegmentConstraint<T>> std::iter::FusedIterator
    for ConstSubSegments<'a, T, C>
{
}

/// The conversion trait — exactly the API you wanted
pub trait TryIntoConstrainedSegment<T> {
    fn try_into_constrained_segment<'a, C>(
        &'a self,
        constraint: &'a C,
    ) -> Option<ConstrainedSegment<'a, T, C>>
    where
        C: SegmentConstraint<T>,
        T: PartialEq;
}

/// Blanket impl for everything that can be borrowed as a slice
impl<T: PartialEq, S: ?Sized> TryIntoConstrainedSegment<T> for S
where
    S: AsRef<[T]>,
{
    fn try_into_constrained_segment<'a, C>(
        &'a self,
        constraint: &'a C,
    ) -> Option<ConstrainedSegment<'a, T, C>>
    where
        C: SegmentConstraint<T>,
    {
        let slice = self.as_ref();
        if constraint.verify(slice) {
            Some(ConstrainedSegment { slice, constraint })
        } else {
            None
        }
    }
}

/* ============================= Built-in constraints
 * ============================= */

#[derive(Clone, Debug)]
pub struct Unconstrained;
impl<T: PartialEq> SegmentConstraint<T> for Unconstrained {
    fn verify(&self, _: &[T]) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct UniqueValues;
impl<T: PartialEq + Eq + Hash> SegmentConstraint<T> for UniqueValues {
    fn verify(&self, slice: &[T]) -> bool {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut i = 0;
        while i < slice.len() {
            let val = &slice[i];
            let mut j = i + 1;
            while j < slice.len() && slice[j] == *val {
                j += 1;
            }
            if !seen.insert(val) {
                return false;
            }
            i = j;
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct Sorted;
impl<T: PartialOrd> SegmentConstraint<T> for Sorted {
    fn verify(&self, slice: &[T]) -> bool {
        slice.windows(2).all(|w| w[0] <= w[1])
    }
}

#[derive(Clone, Debug)]
pub struct MaxRunLength(pub usize);

impl<T: PartialEq> SegmentConstraint<T> for MaxRunLength {
    fn verify(&self, slice: &[T]) -> bool {
        let mut i = 0;
        while i < slice.len() {
            let start = i;
            while i < slice.len() && slice[i] == slice[start] {
                i += 1;
            }
            if i - start > self.0 {
                return false;
            }
        }
        true
    }

    fn next_run_end(&self, slice: &[T], start: usize) -> usize
    where
        T: PartialEq,
    {
        let cap = start + self.0 + 1;
        let mut end = start + 1;
        while end < slice.len() && end < cap && slice[end] == slice[start] {
            end += 1;
        }
        end
    }
}

#[derive(Clone, Debug)]
pub struct AllowedValues<T>(pub &'static [T])
where
    T: PartialEq + Eq + Hash + 'static;

impl<T: PartialEq + Eq + Hash> SegmentConstraint<T> for AllowedValues<T> {
    fn verify(&self, slice: &[T]) -> bool {
        let allowed: std::collections::HashSet<_> = self.0.iter().collect();
        let mut i = 0;
        while i < slice.len() {
            let val = &slice[i];
            if !allowed.contains(val) {
                return false;
            }
            while i < slice.len() && slice[i] == *val {
                i += 1;
            }
        }
        true
    }
}

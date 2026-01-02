use std::borrow::Borrow;

use nalgebra::{Isometry3, Point3};
use rand::rngs::StdRng;

use super::{
    BoundaryConditions,
    GenerateBubblesExterior,
    LatticeGeometry,
    SamplePointsInsideLattice,
    TransformationIsometry3,
};
use crate::many_bubbles::bubbles::Bubbles;

#[derive(Clone, Debug, PartialEq)]
pub struct EmptyLattice {}

impl LatticeGeometry for EmptyLattice {
    fn volume(&self) -> f64 {
        0.0
    }

    fn reference_point(&self) -> Point3<f64> {
        Point3::origin()
    }

    fn parameters(&self) -> Vec<f64> {
        Vec::new()
    }

    fn contains(&self, _points: &[Point3<f64>]) -> Vec<bool> {
        Vec::new()
    }
}

impl TransformationIsometry3 for EmptyLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, _iso: I) -> Self {
        EmptyLattice {}
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, _iso: I) {}
}

// Generate no exterior bubbles as we have no information about the lattice
impl GenerateBubblesExterior for EmptyLattice {
    fn generate_bubbles_exterior(
        &self,
        _bubbles_interior: impl Borrow<Bubbles>,
        _boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        Bubbles::new(Vec::new())
    }
}

impl SamplePointsInsideLattice for EmptyLattice {
    fn sample_points(&self, n_points: usize, _rng: &mut StdRng) -> Vec<Point3<f64>> {
        if n_points == 0 {
            Vec::new()
        } else {
            panic!("Cannot sample points from EmptyLattice (volume = 0)");
        }
    }
}

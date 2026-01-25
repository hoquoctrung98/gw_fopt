use std::borrow::Borrow;

use nalgebra::{Isometry3, Point3};
use rand::rngs::StdRng;

use super::{
    BoundaryConditions,
    CartesianLattice,
    EmptyLattice,
    GenerateBubblesExterior,
    LatticeGeometry,
    ParallelepipedLattice,
    SamplePointsInsideLattice,
    SphericalLattice,
    TransformationIsometry3,
};
use crate::many_bubbles::bubbles::Bubbles;

#[derive(Clone, Debug)]
pub enum BuiltInLattice {
    Parallelepiped(ParallelepipedLattice),
    Cartesian(CartesianLattice),
    Spherical(SphericalLattice),
    Empty(EmptyLattice),
}

impl LatticeGeometry for BuiltInLattice {
    fn volume(&self) -> f64 {
        match self {
            Self::Parallelepiped(l) => l.volume(),
            Self::Cartesian(l) => l.volume(),
            Self::Spherical(l) => l.volume(),
            Self::Empty(l) => l.volume(),
        }
    }

    fn reference_point(&self) -> Point3<f64> {
        match self {
            Self::Parallelepiped(l) => l.reference_point(),
            Self::Cartesian(l) => l.reference_point(),
            Self::Spherical(l) => l.reference_point(),
            Self::Empty(l) => l.reference_point(),
        }
    }

    fn parameters(&self) -> Vec<f64> {
        match self {
            Self::Parallelepiped(l) => l.parameters(),
            Self::Cartesian(l) => l.parameters(),
            Self::Spherical(l) => l.parameters(),
            Self::Empty(l) => l.parameters(),
        }
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        match self {
            Self::Parallelepiped(l) => l.contains(points),
            Self::Cartesian(l) => l.contains(points),
            Self::Spherical(l) => l.contains(points),
            Self::Empty(l) => l.contains(points),
        }
    }
}

impl TransformationIsometry3 for BuiltInLattice {
    fn transform<I: Into<Isometry3<f64>>>(&self, iso: I) -> Self {
        match self {
            Self::Parallelepiped(l) => Self::Parallelepiped(l.transform(iso)),
            Self::Cartesian(l) => Self::Cartesian(l.transform(iso)),
            Self::Spherical(l) => Self::Spherical(l.transform(iso)),
            Self::Empty(l) => Self::Empty(l.transform(iso)),
        }
    }

    fn transform_mut<I: Into<Isometry3<f64>>>(&mut self, iso: I) {
        match self {
            Self::Parallelepiped(l) => l.transform_mut(iso),
            Self::Cartesian(l) => l.transform_mut(iso),
            Self::Spherical(l) => l.transform_mut(iso),
            Self::Empty(l) => l.transform_mut(iso),
        }
    }
}

impl GenerateBubblesExterior for BuiltInLattice {
    fn generate_bubbles_exterior(
        &self,
        bubbles_interior: impl Borrow<Bubbles>,
        boundary_condition: BoundaryConditions,
    ) -> Bubbles {
        let bubbles = match self {
            BuiltInLattice::Parallelepiped(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            },
            BuiltInLattice::Cartesian(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            },
            BuiltInLattice::Spherical(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            },
            BuiltInLattice::Empty(lattice) => {
                lattice.generate_bubbles_exterior(bubbles_interior, boundary_condition)
            },
        };
        bubbles
    }
}

impl SamplePointsInsideLattice for BuiltInLattice {
    fn sample_points(&self, n_points: usize, rng: &mut StdRng) -> Vec<Point3<f64>> {
        match self {
            Self::Parallelepiped(l) => l.sample_points(n_points, rng),
            Self::Cartesian(l) => l.sample_points(n_points, rng),
            Self::Spherical(l) => l.sample_points(n_points, rng),
            Self::Empty(l) => l.sample_points(n_points, rng),
        }
    }
}

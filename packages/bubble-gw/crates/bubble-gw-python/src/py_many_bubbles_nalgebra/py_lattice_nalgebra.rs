use bubble_gw::many_bubbles_nalgebra::lattice_nalgebra::{
    CartesianLattice, EmptyLattice, LatticeGeometry, LatticeTransform, ParallelepipedLattice,
    SphericalLattice,
};
use nalgebra::{Isometry3, Point3};

#[derive(Clone, Debug)]
pub enum ConcreteLattice {
    Parallelepiped(ParallelepipedLattice),
    Cartesian(CartesianLattice),
    Sphere(SphericalLattice),
    Empty(EmptyLattice),
}

impl LatticeGeometry for ConcreteLattice {
    fn volume(&self) -> f64 {
        match self {
            Self::Parallelepiped(l) => l.volume(),
            Self::Cartesian(l) => l.volume(),
            Self::Sphere(l) => l.volume(),
            Self::Empty(l) => l.volume(),
        }
    }

    fn reference_point(&self) -> Point3<f64> {
        match self {
            Self::Parallelepiped(l) => l.reference_point(),
            Self::Cartesian(l) => l.reference_point(),
            Self::Sphere(l) => l.reference_point(),
            Self::Empty(l) => l.reference_point(),
        }
    }

    fn parameters(&self) -> Vec<f64> {
        match self {
            Self::Parallelepiped(l) => l.parameters(),
            Self::Cartesian(l) => l.parameters(),
            Self::Sphere(l) => l.parameters(),
            Self::Empty(l) => l.parameters(),
        }
    }

    fn contains(&self, points: &[Point3<f64>]) -> Vec<bool> {
        match self {
            Self::Parallelepiped(l) => l.contains(points),
            Self::Cartesian(l) => l.contains(points),
            Self::Sphere(l) => l.contains(points),
            Self::Empty(l) => l.contains(points),
        }
    }
}

impl LatticeTransform for ConcreteLattice {
    fn transform(&self, iso: &Isometry3<f64>) -> Self {
        match self {
            Self::Parallelepiped(l) => Self::Parallelepiped(l.transform(iso)),
            Self::Cartesian(l) => Self::Cartesian(l.transform(iso)),
            Self::Sphere(l) => Self::Sphere(l.transform(iso)),
            Self::Empty(l) => Self::Empty(l.transform(iso)),
        }
    }

    fn transform_mut(&mut self, iso: &Isometry3<f64>) {
        match self {
            Self::Parallelepiped(l) => l.transform_mut(iso),
            Self::Cartesian(l) => l.transform_mut(iso),
            Self::Sphere(l) => l.transform_mut(iso),
            Self::Empty(l) => l.transform_mut(iso),
        }
    }
}

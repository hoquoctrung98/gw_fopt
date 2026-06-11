//! `bubble_gw` is the Rust computational core of `gw_fopt` for gravitational
//! waves sourced by first-order phase-transition bubble collisions.
//!
//! The crate has two main physics workflows:
//!
//! 1. `two_bubbles`: exact gravitational-wave calculation from a
//!    precomputed $(s, z)$ field evolution of two colliding bubbles.
//! 2. `many_bubbles`: many-bubble geometry, nucleation, lattice bookkeeping,
//!    and generalized bulk-flow / envelope-style approximations.
//!
//! ## Two-bubble exact calculation
//!
//! The two-bubble pipeline assumes that the field evolution has already been
//! computed on a $(1+1)D$ lattice for the two regions $\phi_+(s, z)$ and
//! $\phi_-(s, z)$. The
//! Fourier-space stress tensor is then assembled component by component and
//! projected onto the gravitational-wave spectrum.
//!
//! <div>\[
//! \frac{d E_{\mathrm{GW}}}{d \omega\, d \Omega}
//! = G \omega^2 \left|
//! \cos^2 \theta_k\, \widetilde{T}_{xx}
//! - \widetilde{T}_{yy}
//! + \sin^2 \theta_k\, \widetilde{T}_{zz}
//! - \sin(2\theta_k)\, \widetilde{T}_{xz}
//! \right|^2.
//! \]</div>
//!
//! In this crate, the entry point is
//! [`two_bubbles::gw_calc::GravitationalWaveCalculator`]. It accepts sampled
//! field arrays, lattice spacing, and a time cutoff, then computes either
//! tensor components or the angular / averaged spectrum.
//!
//! Relevant APIs:
//!
//! - [`two_bubbles::gw_calc::GravitationalWaveCalculator`]
//! - [`two_bubbles::gw_calc::InitialFieldStatus`]
//! - [`two_bubbles::Integral`]
//! - [`time_cutoff::TimeCutoff`]
//! - [`time_cutoff::ExponentialTimeCutoff`]
//! - [`time_cutoff::UnitTimeCutoff`]
//!
//! For the full derivation and the companion Python workflow, see
//! `docs/two_bubbles.md` and `docs/examples/two_bubbles.py` in the repository.
//!
//! ## Many-bubble generalized bulk flow
//!
//! The many-bubble workflow models a population of nucleated bubbles inside a
//! lattice and computes collision-aware bulk-flow integrals whose final purpose
//! is to obtain the gravitational-wave spectrum.
//!
//! The radiated energy per logarithmic frequency and solid angle is
//! <div>\[
//! \frac{d E_\mathrm{GW}}{d \ln \omega \, d \Omega_{\mathbf{k}}}
//! = 2 G \omega^3 \Lambda_{ij,lm}(\hat{\mathbf{k}})
//! \widetilde{T}^{*}_{ij}(\hat{\mathbf{k}},\omega)
//! \widetilde{T}_{lm}(\hat{\mathbf{k}},\omega)
//! = 2 G \omega^3 \left(
//! \mathrm{Tr}\left[ P_{\hat{\mathbf{k}}} \widetilde{T}^{\ast}
//! P_{\hat{\mathbf{k}}} \widetilde{T} \right]
//! - \frac{1}{2}
//! \left| \mathrm{Tr}\left[ P_{\hat{\mathbf{k}}} \widetilde{T} \right] \right|^2
//! \right).
//! \]</div>
//!
//! Here $P_{ij}(\hat{\mathbf{k}}) = \delta_{ij} - \hat{\mathbf{k}}_i \hat{\mathbf{k}}_j$ is the projector transverse to the
//! propagation direction and $\Lambda$ is the corresponding
//! transverse-traceless projector:
//! <div>\[
//! \Lambda_{ij,lm}(\hat{\mathbf{k}})
//! = P_{il}(\hat{\mathbf{k}}) P_{jm}(\hat{\mathbf{k}})
//! - \frac{1}{2} P_{ij}(\hat{\mathbf{k}}) P_{lm}(\hat{\mathbf{k}}).
//! \]</div>
//!
//! The central intermediate object is the tensor
//! $C_{ij}(\hat{\mathbf{k}}, \omega)$, built from wall elements on all bubbles:
//!
//! <div>\[
//! \widetilde{T}_{ij}(\hat{\mathbf{k}},\omega)
//! = \frac{1}{2\pi}\int dt \, d^3x \,
//! e^{i\omega(t-\hat{\mathbf{k}} \cdot \mathbf{x})}T_{ij}(\mathbf{x},t)
//! = \Delta V_{\mathrm{vac}}\, C_{ij}(\hat{\mathbf{k}},\omega).
//! \]</div>
//!
//! <div>\[
//! C_{ij}(\hat{\mathbf{k}},\omega)
//! = \frac{1}{6 \pi} \sum_n \int dt\,
//! e^{i \omega (t - \hat{\mathbf{k}} \cdot \mathbf{x}_n)}
//! A_{n, ij}(\hat{\mathbf{k}}, \omega, t).
//! \]</div>
//! <div>\[
//! A_{n, ij}(\hat{\mathbf{k}},\omega, t)
//! = \int_{S_n} d \Omega_{\mathbf{x}} \,
//! e^{-i \omega (t - t_n) \hat{\mathbf{k}} \cdot \mathbf{\hat{x}}}
//! \hat{\mathbf{x}}_i \hat{\mathbf{x}}_j
//! \left[ (t - t_n)^3  f(t, t_n, t_{n,c})\right].
//! \]</div>
//!
//! The collision history enters through the first-collision time
//! $t_{n,c}(\cos(\theta_x), \phi_x)$ and the scaling function
//! $f(t, t_n, t_{n,c})$, which interpolates between envelope and
//! post-collision bulk-flow behavior.
//! This approximation assumes vanishing initial bubble radius $R_0 = 0$ and
//! wall speed $v_w = 1$.
//!
//! The many-bubble code is split into three layers:
//!
//! - lattice and geometry:
//!   [`many_bubbles::lattice`], [`many_bubbles::lattice_bubbles`],
//!   [`many_bubbles::spacetime`]
//! - nucleation strategies:
//!   [`many_bubbles::bubbles_nucleation`]
//! - gravitational-wave integrals:
//!   [`many_bubbles::generalized_bulk_flow`]
//!
//! The main user-facing entry points are:
//!
//! - [`many_bubbles::lattice_bubbles::LatticeBubbles`]
//! - [`many_bubbles::generalized_bulk_flow::GeneralizedBulkFlow`]
//! - [`many_bubbles::bubbles_nucleation::FixedRateNucleation`]
//! - [`many_bubbles::bubbles_nucleation::SpontaneousNucleation`]
//!
//! For the physical background and example plots, see
//! `docs/generalized_bulkflow.md`, `docs/lattice_bubbles.md`, and
//! `docs/examples/generalized_bulkflow.py` in the repository.
//!
//! ## Module guide
//!
//! - [`two_bubbles`]: exact two-bubble GW calculation from field profiles
//! - [`many_bubbles`]: lattices, nucleation, and bulk-flow approximations
//! - [`time_cutoff`]: time cutoff functions reused across workflows
//! - [`utils`]: numerical utilities shared by the core algorithms

pub mod many_bubbles;
pub mod time_cutoff;
pub mod two_bubbles;
pub mod utils;

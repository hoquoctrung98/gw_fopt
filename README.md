# gw_fopt

**gw_fopt** is a high-performance Python package with Rust bindings, designed to compute the gravitational wave (GW) spectrum arising from bubbles collisions in first order phase transition.
Currently, this is a workspace containing two different packages: **bubble_dynamics** and **bubble_gw**, each taking care of different computational aspects

+ **bubble_dynamics** is a Python library that mainly solves the equation of motion of two bubbles system in $(1+1)D$ spacetime (instead of the full $(3+1)D$ equation of motion, thanks to the $O(2,1)$ symmetry of this system). This package also includes tools to extract the surface tension of the wall before and after collision, which is later used in fitting to extract coefficients for the generalized bulk-flow method in computing the GW spectrum.

+ **bubble_gw** leveraging Rust for improved computational efficiency and PyO3 for generating a Python interface. This package computes the GW spectrum via different modules:

  + **bubble_gw.two_bubbles** compute the exact GW spectrum of two bubbles system by taking the input of fields evolution on the $(1+1)D$ lattice computed from **bubbles_dynamics** and numerically compute the GW spectrum via  the fourier transformed stress-energy tensor.
The core computation of exact two bubbles spectrum is a reimplementation of the [two_bubbles_code-v1.0.1](https://zenodo.org/records/5127538.).

  + **bubble_gw.many_bubbles** compute the approximate GW spectrum via the generalized bulk-flow, with the input are lattice sizes and the bubble configuration of an arbitrary number of bubbles.

## Installation

**gw_fopt** requires the Rust toolchain to be installed on your system.
The simplest way to install **gw_fopt** is via the **uv**, a Python package and project manager

1. Ensure you have the Rust toolchain installed

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

1. Install **uv**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

1. Build and install **gw_fopt**: from the workspace directory run

  ```bash
  uv sync
  ```

  This command creates an virtual environment in .venv, install the necessary packages, compile and install the Rust code in release mode.
In order to use the installed packages, you need to source the virtual environment corresponding to your shell, for example if you use `bash`, from the workspace directory run

  ```bash
  source .venv/bin/activate
  ```

## Examples of using the package **gw_fopt**

+ [two bubbles](./docs/two_bubbles.md): Here we use `bubbles_dynamics` to solve the equation of motion on $(1+1)D$ lattice of a quartic potential and plugging this to `bubbles_gw.two_bubbles` to compute the exact GW spectrum.
Also from the field evolution, we can compute the surface tension of the wall as a function of time, which will be useful in fitting and extracting the coefficients for the generalized bulk-flow approximation.

| Field evolution                                                           | Corresponding GW spectrum                                         |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| ![evolution_field](docs/examples/figures/two_bubbles/evolution_field.png) | ![gw_spectrum](docs/examples/figures/two_bubbles/gw_spectrum.png) |

+ [lattice bubbles](./docs/lattice_bubbles.md): Here we show how to define a lattice (Cartesian or Spherical), attach the bubbles into a lattice and generate the exterior bubbles via boundary conditions (currently support "periodic" and "reflection" boundary conditions).
One can also define an Isometry3 transformation (i.e spatial translation + rotation) and apply to the whole lattice+bubbles system.

| Bubbles in a cartesian lattice with periodic boundary condition                            | Bubbles in a spherical lattice with reflection boundary condition                                |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| ![bubble centers](./docs/examples/figures/many_bubbles/many_bubbles_at_time_cartesian.png) | ![bubble at fixed time](./docs/examples/figures/many_bubbles/many_bubbles_at_time_spherical.png) |

+ [generalized bulk-flow](./docs/generalized_bulkflow.md): Here we illustrate how to approximate the GW spectrum using generalized bulk-flow scheme, with the input being the bubbles configuration (i.e a list of four-vectors $(t_c, x_c, y_c, z_c)$ of the center of the nucleated bubbles).

| First collided bubbles                                                            | Collision time with first collided bubbles                               |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| ![first collided bubbles](docs/examples/figures/many_bubbles/first_collision.png) | ![collision time](docs/examples/figures/many_bubbles/collision_time.png) |

Generalized bulk-flow with various $\xi$-powers for two bubbles system with no boundary conditions
![gw spectrum approximation](./docs/examples/figures/two_bubbles/gw_spectrum_apprx.png)

## Some docs for useful utilities

+ [sample](./docs/sample.md): This util can be used to generate the samples for the parameter scan.
It is especially useful when working with an expensive (i.e time consuming) function, where we can divide the input parameters into several disjoint batches with gradually larger density.
One then get the output of the expensive function with higher and higher resolution over the whole range of parameter scan per iteration.
This helps to avoid the case where we start with too many parameters and have to wait for a long time to get all the data in one batch, and risk loosing the output in the whole range of parameters if the computation suddenly stopped (e.g we have an input where the scan function throw an error and stop running in the middle of the long parameter scan.)

| Uniform sampling                                     | Log sampling                                 |
| ---------------------------------------------------- | -------------------------------------------- |
| ![uniform](docs/examples/figures/sample_uniform.png) | ![log](docs/examples/figures/sample_log.png) |

## Roadmap and status

State key:

+ 🟢 - fully implemented
+ 🟡 - partially implemented
+ 🟠 - implemented, but partially broken
+ 🔴 - not yet implemented

| Feature                                                                                                                                       | Status |
| :-------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **#bubble_dynamics**                                                                                                                          |        |
| Evolution of two-bubbles on lattice using leapfrog                                                                                            |   🟢    |
| Evolution of two-bubbles on lattice using adaptive mesh refinement?                                                                           |   🔴    |
| Extract the surface tension along the bubble wall                                                                                             |   🟢    |
| # **bubble_gw**                                                                                                                               |        |
| Compute GW spectrum of two-bubble collision with input from $(1+1)D$ lattice simulation                                                       |   🟢    |
| Support for common Lattices (parallelepiped, cartesian, spherical) where we attach/generate Bubbles                                           |   🟢    |
| Support LatticeBubbles: a bundle of bubbles and lattice to perform checks on bubbles inside/outside lattice and bubbles formed inside bubbles |   🟢    |
| Implement core computations of generalized bulk-flow along $z$ direction                                                                      |   🟢    |
| Generate bubbles with given nucleation rate $\beta$                                                                                           |   🟠    |
| Generalized bulk-flow computations along arbitrary $\hat{\bm{k}}$ directions without rotating the input Bubbles?                              |   🔴    |

> [!NOTE]
> From personal experience in using bubble_gw from Python code, it could happen that the the Python kernel died while executing the Rust code.
> This is because Rust library faced an unrecoverable error (e.g out of bounds memory access) and panic instead of throwing error to be handled by the user.
> We tried to handle extensively these edge cases whenever we encounter them.
> If you face such a situation, please open an issue.

<!-- Additional details for each step in the roadmap below: -->

## License

Licensed under either of

+ MIT license ([LICENSE-MIT](./LICENSE-MIT))

+ Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))

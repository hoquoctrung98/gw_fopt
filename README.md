# gw_fopt
gw_fopt is a high-performance Python package with Rust bindings, designed to compute the gravitational wave (GW) spectrum arising from bubbles collisions in first order phase transition.
Currently, this is a workspace containing two different packages: `bubble_dynamics` and `bubble_gw`, each taking care of different computational aspects
+ `bubble_dynamics` mainly solves the equation of motion of two bubbles system in $1+1$ spacetime (instead of the full $3+1$ equation of motion, thanks to the $O(2,1)$ symmetry of this system). This package also includes tools to extract the surface tension of the wall before and after collision, which is later used in fitting to extract coefficients for the generalized bulk-flow method in computing the GW spectrum.
+ `bubble_gw` compute the GW spectrum via different modules:
  +`bubble_gw.two_bubbles` compute the exact GW spectrum of two bubbles system by taking the input of fields evolution on the $1+1$ lattice and numerically compute the GW spectrum via  the fourier transformed stress-energy tensor.
  + +`bubble_gw.many_bubbles` compute the approximate GW spectrum via the generalized bulk-flow, with the input are lattice sizes and the bubble configuration of an arbitrary number of bubbles.

## Installation

gw_fopt requires the Rust toolchain to be installed on your system.
The simplest way to install gw_fopt is via the uv package manager.

1. Ensure you have Rust installed

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

2. Install maturin

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

3. Build and install gw_fopt:

  ```bash
  uv sync
  ```

  This command create an virtual environment in .venv, install the necessary packages, compile and install the Rust code in release mode.


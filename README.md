# pygw_fopt
pygw_fopt is a high-performance Python package with Rust bindings, designed to compute the gravitational wave (GW) spectrum arising from bubbles collisions in first order phase transition.

## Installation

pygw_fopt requires the Rust toolchain to be installed on your system.
The simplest way to install pygw_fopt is via the uv package manager.

1. Ensure you have Rust installed

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

2. Install maturin

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

3. Build and install pygw_fopt:

  ```bash
  uv sync
  ```

  This command create an virtual environment in .venv, install the necessary packages, compile and install the Rust code in release mode.

## Usage example

```python
import numpy as np
from pygw_fopt.bubble_simulator import LatticeSetup, PDEBubbleSolver
from pygw_fopt.bubble_simulator.potentials import U1Potential, QuarticPotential
from bubble_gw import two_bubbles
from bubble_gw.utils import sample

# potential and relevant parameters
lambdabar = 0.2
potential = QuarticPotential(lambdabar=lambdabar)
phi_absMin0 = np.array([(1 + np.sqrt(1 - 8/9*lambdabar))/2])
phi_metaMin0 = np.array([0.0])

# lattice and initial profile parameters
scale_dz = 0.25
scale_z = 3
d = 20.0 # bubbles separation
setup = LatticeSetup(potential)
setup.set_tunnelling_phi(phi_absMin=phi_absMin0, phi_metaMin=phi_metaMin0)
setup.set_d(d=d)
initial = setup.two_bubbles(type="full", scale_dz=scale_dz, scale_z=scale_z)
z_grid, phi_initial, d = initial
phi_initial = phi_initial.T
dz = abs(z_grid[1] - z_grid[0]) # z-coordinate step
smax = d * 0.8 # simulation time
ds = dz * 0.5 # time step

solver = PDEBubbleSolver(phi1_initial=phi_initial, z_grid=z_grid, ds=ds, dz=dz, potential=potential, d=d)
solver.evolve(smax, history_interval=6)
solver.compute_phi_region2()

# GW computation
w_arr = sample(0.01, 10.0, 10, 2, 0, 'log') # array of frequencies
cos_thetak_arr = sample(0.0, 1.0, 10, 2, 0, 'uniform') # array of different direction of cos(theta_k)
gw_calc = two_bubbles.GravitationalWaveCalculator(initial_field_status="two_bubbles", phi1=solver.phi1, phi2=solver.phi2, z_grid=solver.z_grid, ds=solver.ds*solver.history_interval)
gw_calc.set_integral_params(tol=1e-5, max_iter=20)
w_mesh, cos_thetak_mesh = np.meshgrid(w_arr, cos_thetak_arr, indexing='ij')
dE_dlogw_dcosthetak= gw_calc.compute_angular_gw_spectrum(w_arr=w_mesh.flatten(), cos_thetak_arr=cos_thetak_mesh.flatten())
```

## Explanation of the Example

+ Inputs:
  + phi1 and phi2: 3D NumPy arrays of shape (n_s, n_fields, n_z) representing $\phi_+$ and $\phi_-$ , respectively.
  In case having phi1 of shape (n_s, n_z), we can add one more axis by e.g

    ```python
        phi1=np.expand_dims(phi1, axis=1)
    ```

  + z: 1D array of spatial coordinates.
  + ds: Step size in the (s)-direction (time-like coordinate).
  + d: Distance between bubble centers.

+ Frequency Grid: w_arr specifies the frequencies ($\omega$) at which the GW spectrum is evaluated, using a logarithmic scale.

+ Computation: compute_angular_gw_spectrum calculates the differential energy spectrum $dE/d\log\omega d \cos \theta_k$ for each frequency in w_arr, integrating over n_k momentum directions.

+ Output: A list of dictionaries, each containing:
  + dE_dlogw_dcosthetak: The GW energy per logarithmic frequency interval per change of measured angle.
  + klist: The momentum values used in the integration.

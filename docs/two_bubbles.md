# Example of computing GW spectrum in 2 bubbles set-up

Here we consider the quartic potential
$$
V(\phi) = \dfrac{m^2}{2} \phi^2 - \dfrac{\delta}{3} \phi^3 + \dfrac{\lambda}{4} \phi^4.
$$
Define $\overline{\lambda}=\dfrac{9 \lambda m^2}{2 \delta^2}$, the range for a first order phase transition to happen is $\overline{\lambda} \in (0, 1)$.
The chosen parameters in the following results are
$$
\overline{\lambda} = 0.84, \quad
m = 1, \quad
\delta = \dfrac{\sqrt{9 - 8 \overline{\lambda}} + 3}{4 \overline{\lambda}}, \quad
\lambda = \dfrac{(\sqrt{9 - 8 \overline{\lambda}} - 3)^2}{8 \overline{\lambda}},
$$
A sketch of the potential in this example is shown below.
![potential](examples/figures/potential.png)

One can solve for the bubble profile using e.g CosmoTransitions.
This profile can be use to build the initial condition for the lattice simulation with 2 bubbles nucleated at the same time and has a separation $d$.

## Usage example

```python
import numpy as np
from bubble_dynamics.bubble_simulator import LatticeSetup, PDEBubbleSolver
from bubble_dynamics.bubble_simulator.potentials import U1Potential, GouldQuarticPotential
from bubble_gw import two_bubbles
from bubble_gw.utils import sample

# potential and relevant parameters
lambdabar = 0.2
potential = GouldQuarticPotential(lambdabar=lambdabar)
phi_absMin0 = np.array([(1 + np.sqrt(1 - 8/9*lambdabar))/2])
phi_metaMin0 = np.array([0.0])

# lattice and initial profile parameters
scale_dz = 0.25
scale_z = 3
d = 20.0 # bubbles separation
setup = LatticeSetup(potential)
setup.set_tunnelling_phi(phi_absMin=phi_absMin0, phi_metaMin=phi_metaMin0)
setup.set_d(d=d)
initial = setup.two_bubbles(layout="full", scale_dz=scale_dz, scale_z=scale_z)
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
dE_dlogw_dcosthetak= gw_calc.compute_angular_gw_spectrum(w_arr=w_arr, cos_thetak_arr=cos_thetak)
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

## Evolution of field profile

![evolution_field](examples/figures/evolution_field.png)

## Evolution of gradient energy density

![evolution_gradient_energy_density](examples/figures/evolution_gradient_energy_density.png)

## Surface tension of the wall as a function of time

![surface_tension](examples/figures/surface_tension.png)

## GW spectrum of two bubbles collision

Below is the GW spectrum computed with the input being field evolution
![gw_spectrum](examples/figures/gw_spectrum.png)

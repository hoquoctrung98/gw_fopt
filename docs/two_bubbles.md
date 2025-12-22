# Example working with two bubbles system

## Physics Background

The scalar fields $\phi_{\pm}(s, z)$ are solutions to the following PDEs, which describe the dynamics of two colliding bubbles:
$$
\pm \dfrac{\partial^2 \phi_{\pm}}{\partial s^2} \pm \dfrac{2}{s} \dfrac{\partial \phi_{\pm}}{\partial s} - \dfrac{\partial^2 \phi_{\pm}}{\partial z^2} + \dfrac{\partial V}{\partial \phi_{\pm}} = 0
$$

+ $\phi_+(s, z)$ : Represents the field in the region where $t^2 > x^2 + y^2$. This is typically obtained from lattice simulations.
+ $\phi_-(s, z)$ : Represents the field in the region where $t^2 < x^2 + y^2$. It is analytically defined as:
  $$
  \phi_-(t, z) = \phi_0 \left(\sqrt{s^2 + (z - d/2)^2}\right) + \phi_0\left(\sqrt{s^2 + (z + d/2)^2}\right)
  $$
  where $\phi_0$ is the bubble profile, and $d$ is the separation distance between the bubble centers.

In this package, $\phi_+$ is referred to as phi1, and $\phi_-$ as phi2, following the naming convention of two_bubbles_code. These inputs can be generated using the original two_bubbles_code package or similar tools.
The GW spectrum is computed by integrating over the field derivatives and weights, accounting for contributions from different spatial components (e.g., (zz), (xx), (yy), and (xz)).
More details on the computations can be found at e.g [On bubble collisions in strongly supercooled phase transitions](https://arxiv.org/abs/1912.00997).

## Example of computing GW spectrum in 2 bubbles set-up

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

## Example code
An example of working with two bubbles system can be found at [](./examples/two_bubbles_evolution.py)

## Evolution of field profile

![evolution_field](examples/figures/evolution_field.png)

## Evolution of gradient energy density

![evolution_gradient_energy_density](examples/figures/evolution_gradient_energy_density.png)

## Surface tension of the wall as a function of time

![surface_tension](examples/figures/surface_tension.png)

## GW spectrum of two bubbles collision

Below is the GW spectrum computed with the input being field evolution
![gw_spectrum](examples/figures/gw_spectrum.png)

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

## Evolution of field profile

![evolution_field](examples/figures/evolution_field.png)

## Evolution of gradient energy density

![evolution_gradient_energy_density](examples/figures/evolution_gradient_energy_density.png)

## Surface tension of the wall as a function of time

![surface_tension](examples/figures/surface_tension.png)

## GW spectrum of two bubbles collision

Below is the GW spectrum computed with the input being field evolution
![gw_spectrum](examples/figures/gw_spectrum.png)

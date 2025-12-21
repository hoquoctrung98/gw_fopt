# bubble_gw

bubble_gw is a high-performance Python package with Rust bindings, designed to compute the gravitational wave (GW) spectrum arising from bubbles collisions in first order phase transition.
It has two main functionalities: 
+ compute the exact GW spectrum from a two bubbles collision process:
  The computation of exact two bubbles spectrum is a reimplementation of the [two_bubbles_code-v1.0.1](https://zenodo.org/records/5127538.), originally written in Python, leveraging Rust for improved computational efficiency and PyO3 for seamless Python integration.
  This package calculates the GW spectrum from numerical solutions to partial differential equations (PDEs) describing scalar field dynamics in bubble collisions, a phenomenon relevant to cosmological phase transitions in the early universe. 
  Specifically, it takes as input the scalar field profiles $\phi_{\pm}(s, z)$ , which represent the field evolution in two distinct spacetime regions, and computes the resulting GW energy spectrum.
+ Computation of approximated GW spectrum using generalized bulk-flow approach for many-bubbles configuration.

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

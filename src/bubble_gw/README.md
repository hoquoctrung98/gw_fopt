# bubble_gw

bubble_gw is a high-performance Python package with Rust bindings, designed to compute the gravitational wave (GW) spectrum arising from the collision of two cosmological bubbles.
It is a reimplementation of the [two_bubbles_code-v1.0.1](https://zenodo.org/records/5127538.), originally written in Python, leveraging Rust for improved computational efficiency and PyO3 for seamless Python integration.
This package calculates the GW spectrum from numerical solutions to partial differential equations (PDEs) describing scalar field dynamics in bubble collisions, a phenomenon relevant to cosmological phase transitions in the early universe. 
Specifically, it takes as input the scalar field profiles $\phi_{\pm}(s, z)$ , which represent the field evolution in two distinct spacetime regions, and computes the resulting GW energy spectrum.

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

## Installation

bubble_gw requires the Rust toolchain and maturin, a tool for building Python extensions from Rust. To install it in your current Python environment, follow these steps:

1. Ensure you have Rust installed

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

2. Install maturin

  ```bash
  pip install maturin
  ```

3. Build and install bubble_gw:

  ```bash
  maturin develop --release
  ```

  This compiles the Rust code in release mode and installs the Python package locally.

## Usage example

Dependencies include numpy for array handling and pandas for data manipulation in the examples below.

```python
# Create an array of desired frequency
import numpy as np
import pandas as pd
import bubble_gw

# Example input data (replace with actual data from lattice simulations or two_bubbles_code)
n_fields, n_s, n_z = 2, 100, 50  # Dimensions: number of fields, s (time), z (spatial)
phi1 = np.zeros((n_fields, n_s, n_z))  # Replace with  phi_+ from simulation
phi2 = np.zeros((n_fields, n_s, n_z))  # Replace with phi_-
z = np.linspace(-10, 10, n_z)  # Spatial grid
ds = 0.1  # Time step
d = 5.0   # Bubble separation distance
w_arr = np.geomspace(1e-2, 1e0, 10)  # Frequency range for GW spectrum
n_k = 50  # Number of k-values for integration

# Initialize the calculator
gw_calc = bubble_gw.GravitationalWaveCalculator(
  phi1, phi2, z, ds, d, cutoff_ratio=0.9
)

# Compute the GW spectrum
results = gw_calc.compute_gw_integral(w_arr=w_arr, n_k=n_k)

# Convert results to a DataFrame for analysis
df_gw = pd.DataFrame(results)
print(df_gw[["dE_dlogw", "klist", "integrand_klist"]])```
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
  + cutoff_ratio: Optional parameter (defaults to 0.9) defining the time cutoff for GW contributions.

+ Frequency Grid: w_arr specifies the frequencies ($\omega$) at which the GW spectrum is evaluated, using a logarithmic scale.

+ Computation: compute_gw_integral calculates the differential energy spectrum $dE/d\log\omega$ for each frequency in w_arr, integrating over n_k momentum directions.

+ Output: A list of dictionaries, each containing:
  + dE_dlogw: The GW energy per logarithmic frequency interval.
  + klist: The momentum values used in the integration.
  + integrand_klist: The integrand values for each (k).

The resulting df_gw DataFrame makes it easy to analyze or plot the spectrum using Python tools like matplotlib.

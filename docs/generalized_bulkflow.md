# Generalized bulk-flow approximations

## Physics Background

The gravitational energy radiated along the $\hat{\textbf{z}}$ direction is
$$
\frac{d E_{GW}}{d \omega d \Omega} =  G \omega^2 (|T_{xx}-T_{yy}|^2+2|T_{xy}|^2+2|T_{yx}|^2)\,,\\
=4 G \Delta V^2 \omega^2 (|C_+|^2 + |C_{\times}|^2),
$$
with the functions $C_{+,\times}$ are explicitly given by
$$
	C_{+,\times}(\omega) = \dfrac{1}{6 \pi} \sum_{n=1}^N \int d t \ e^{i \omega (t - z_n)} A_{n, \pm}(\omega, t), \\
	A_{n, +,\times}(\omega, t) = \int_{-1}^1 d \zeta \ e^{-i \omega (t - t_n) \zeta} B_{n, \pm} (\zeta, t), \\
	B_{n, +,\times}(\zeta, t) = \dfrac{1 - \zeta^2}{2} \int_0^{2 \pi} d \phi \ g_{+,\times}(\phi) \left[ (t - t_n)^3  f(t,t_{n,c})\right]\,.
$$
In the final line, we define $g_+ \equiv \cos 2\phi$, $g_\times \equiv \sin 2\phi$ and $\zeta \equiv \cos\theta$.
Here, $z_n$ and $t_n$ denote the $z$-coordinate and the nucleation time of the $n$-th bubble, respectively.
As before, $t_{n,c}(\zeta, \phi)$ denotes the time at which the wall element of the $n$-th bubble undergoes a the first collision with other bubbles.
Finally, the scaling function can be decomposed into the envelope contribution (i.e before first collision) and generalized bulk-flow contribution (i.e after first collision) as follows
$$
    f_\sigma(t, t_n, t_{n, c}) \equiv 
    \Theta(t_{n, c}-t)
    + \Theta(t - t_{n, c}) \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^3 \sum_\xi a_\xi \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^\xi,
$$
where the coefficients $a_\xi$ can be extracted from fitting the surface tension of the bubble wall from e.g $(1+1)D$ simulation using package **bubble_dynamics**
$$
    \sigma(t,t_n, t_{n, c}) = \Theta(t_{n, c}-t)\sigma_0 (t-t_n)
    + \Theta(t - t_{n, c}) \sigma_0 (t_{n, c} - t_n) \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^2 \sum_\xi a_\xi \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^\xi,
$$

## Example of computing GW spectrum in many bubbles set-up

The bubble configuration file is given at [input configuration](./examples/inputs/confY.txt), which is a table containing four columns corresponding to $(t_c, x_c, y_c, z_c)$ of the nucleated bubbles, and each row corresponds to one bubble.
The example code to reproduce the results below is [generalized_bulkflow.ipynb](./examples/generalized_bulkflow.ipynb).

### First collided bubbles
Below is the colormap which indicates the first collision between the points on the reference bubble (here denoted as bubble 0) versus other bubbles.
Here we use "Interior" to indicate the bubbles inside the lattice, and "Exterior" to call those outside the lattice, which typically generated using e.g periodic or reflection boundary conditions.
For all plots generated below, we use no boundary conditions for the case "Two bubbles" and periodic boundary condition for the case "Many bubbles".

| Two bubbles | Many bubbles |
|-|-|
|![first collided bubbles - two bubbles](examples/figures/two_bubbles/first_collision.png) | ![first collided bubbles - many bubbles](examples/figures/many_bubbles/first_collision.png)|

### Collision time with first collided bubbles
Below we plot the quantity $t_{n,c}(\phi, \cos \theta)$, which is the moment of the first collision between the point on the reference bubble with coordinates $(\phi, \cos \theta)$ and other bubbles.
In the case "Two bubbles", since there are no exterior bubbles, the interior bubbles always have a part that never collide with the other bubble, hence we have to manually cut the maximum time at $t=10$ to plot on the colorbar.
This is not the case in "Many bubbles" where we put the periodic boundary conditions and hence introduced a natural time cut-off in the plot.

| Two bubbles | Many bubbles |
|-|-|
| ![collision time - two bubbles](examples/figures/two_bubbles/collision_time.png)| ![collision time - many bubbles](examples/figures/many_bubbles/collision_time.png)|

### Collision status at a fixed time

| Two bubbles | Many bubbles |
|-|-|
| ![collision status - two bubbles](examples/figures/two_bubbles/collision_status.png)| ![collision status - many bubbles](examples/figures/many_bubbles/collision_status.png)|

### GW spectrum via generalized bulk-flow with various powers for two bubbles
In the following plot, $t_\text{end} = 0.8 d$.
![gw spectrum approximation](examples/figures/two_bubbles/gw_spectrum_apprx.png)
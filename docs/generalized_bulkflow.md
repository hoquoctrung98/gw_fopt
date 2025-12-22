# Generalized bulk-flow approximations

## Physics Background
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
where the coefficients $a_\xi$ can be extracted from fitting the surface tension of the bubble wall from e.g $(1+1)$D simulation using package **bubble_dynamics**
$$
    \sigma(t,t_n, t_{n, c}) = \Theta(t_{n, c}-t)\sigma_0 (t-t_n)
    + \Theta(t - t_{n, c}) \sigma_0 (t_{n, c} - t_n) \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^2 \sum_\xi a_\xi \left(\frac{t_{n, c}-t_n}{t-t_n}\right)^\xi,
$$

## Example of computing GW spectrum in many bubbles set-up

The bubble configuration file is given at [input configuration](./examples/inputs/confY.txt), which is a table containing four columns corresponding to $(t_c, x_c, y_c, z_c)$ of the nucleated bubbles, and each row corresponds to one bubble.

### First collided bubbles

![first collided bubbles](examples/figures/many_bubbles/first_collision.png)

### Collision time with first collided bubbles

![collision time](examples/figures/many_bubbles/collision_time.png)

### Collision status at a fixed time

![collision status](examples/figures/many_bubbles/collision_status.png)
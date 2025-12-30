# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <hoquoctrung98@gmail.com>
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bubble_dynamics.bubble_simulator import LatticeSetup, PDEBubbleSolver
from bubble_dynamics.bubble_simulator.potentials import (
    QuarticPotential,
)
from bubble_dynamics.visualizer import TwoBubblesEvolutionVisualizer
from bubble_gw import two_bubbles
from bubble_gw.utils import sample

# %%
sns.set(style="ticks", font="Dejavu Sans")
sns.set_palette("bright")
# set default font for both text and mathtext
mpl.rcParams["mathtext.default"] = "regular"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
# mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update(
    {
        "axes.linewidth": 1,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm} \usepackage{xcolor}",
        # Enforce default LaTeX font.
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.weight": "bold",
        "figure.facecolor": "white",
        "animation.html": "jshtml",
    }
)

# do not show figures on screen
plt.ioff()

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

# %% [markdown]
# A general form of the quartic potential
# $$
# V(\phi) = \dfrac{c_2^2}{2} \phi^2 - \dfrac{c_3}{3} \phi^3 + \dfrac{c_4}{4} \phi^4
# $$

# %%
lambdabar = 0.5  # lambdabar = 9*c4*c2**2/(2*c3**2)
couplings = (
    1.0,
    3 * (np.sqrt(9 - 8 * lambdabar) + 3) / (4 * lambdabar),
    (np.sqrt(9 - 8 * lambdabar) + 3) ** 2 / (8 * lambdabar),
)
c2, c3, c4 = couplings
# Construct the potential using built-in `QuarticPotential` class.
# User may also define their own potential as well
potential = QuarticPotential(c2=c2, c3=c3, c4=c4)

# True Vacuum and False Vacuum
phi_tv = np.array([c2 / np.sqrt(8 * c4 * lambdabar) * (3 + np.sqrt(9 - 8 * lambdabar))])
phi_fv = np.array([0.0])
phi_local_max = np.array(
    [c2 / np.sqrt(8 * c4 * lambdabar) * (3 - np.sqrt(9 - 8 * lambdabar))]
)  # local maximum
m_meta = c2  # mass in the unbroken phase
m_abs = c2 * np.sqrt(
    (-8 * lambdabar + 3 * np.sqrt(9 - 8 * lambdabar) + 9) / (4 * lambdabar)
)  # mass in the broken phase
rho_vac = (m_abs**4 - m_meta**4) / (12 * c4)  # vacuum energy

phi_range = (-1, 2)  # plotting range
fig, ax = plt.subplots(figsize=(10, 6))
potential.plot_potential(fig=fig, ax=ax, phi_range=phi_range, num_points=100000)
ax.set_xlim(left=phi_range[0], right=phi_range[1])
ax.set_ylim(-0.5, 2)
fig.savefig(
    rf"./figures/two_bubbles/potential.png", bbox_inches="tight", facecolor="white"
)

setup = LatticeSetup(potential)
setup.set_tunnelling_phi(phi_tv=phi_tv, phi_fv=phi_fv)
fig, ax = setup.plot_profiles(npoints=1000)
ax.legend()
fig.savefig(
    rf"./figures/two_bubbles/bubble_profiles.png",
    bbox_inches="tight",
    facecolor="white",
)

# Simulation parameters
scale_dz = 0.5  # smaller means better z resolution
scale_z = 3  # larger means larger simulation size on z-axis
d = 20  # bubbles separation
smax = d * 2  # simulation time

setup.set_tunnelling_phi(phi_tv=phi_tv, phi_fv=phi_fv)
# set the distance between two bubbles at s=0
# you can also call method set_gamma instead of set_d if you know the boost factor of the wall beforehand
setup.set_d(d=d)
z_grid, phi_initial, d = setup.two_bubbles(
    layout="full", scale_dz=scale_dz, scale_z=scale_z
)
phi_initial = phi_initial.T  # initial 2-bubbles profile
dz = abs(z_grid[1] - z_grid[0])  # space step
# time step. The factor ds/dz should be smaller than 1 so that the leapfrog integrator does not diverge
ds = dz * 0.9

# number of time steps before saving, smaller means better time resolution
history_interval = 4
solver = PDEBubbleSolver(
    phi1_initial=phi_initial, z_grid=z_grid, ds=ds, dz=dz, potential=potential, d=d
)
# Evolve field over time from s=0 to s~smax.
# Note that the end of the simulation is not exactly smax, but within smax +- ds
phi_evolution = solver.evolve(smax, history_interval=history_interval)

# %%
analyzer = TwoBubblesEvolutionVisualizer(
    phi1=solver.phi1,
    d=solver.d,
    s_grid=solver.s_grid,
    z_grid=solver.z_grid,
)
# width of a window whose center tracing the maximum of gradient energy density
# surface tension is obtained by integrating gradient energy density along this window
integration_width = 2 * setup.compute_radii().max()
# width of a window centering around the wall location,
# over which we find the maximum of gradient energy density
window_width = 0.5 * integration_width
analyzer.set_width(integration_width=integration_width, window_width=window_width)
analyzer.compute_surface_tension()

fig, ax = analyzer.plot_field_evolution()
fig.savefig(
    f"./figures/two_bubbles/evolution_field.png", bbox_inches="tight", facecolor="white"
)

fig, ax = analyzer.plot_gradient_energy_density(plot_boundaries=True, cutoff=1e-4)
fig.savefig(
    f"./figures/two_bubbles/evolution_gradient_energy_density.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.set_size_inches(6, 3)

fig, ax = analyzer.plot_surface_tension(
    small_s_range=(4e-1, 6e-1), large_s_range=(2.12, 4), normalized=True
)
fig.savefig(
    f"./figures/two_bubbles/surface_tension.png", bbox_inches="tight", facecolor="white"
)

# %%
solver.compute_phi_region2()  # Compute the field $\phi_-$ on the patch $t^2 < x^2 + y^2$

# GW computation
w_arr = (1.0 / d) * sample(1e-1, 1e2, 100, 2, 0, "log")  # array of frequencies
cos_thetak_arr = sample(0, 1, 5, 2, 0, "uniform")  # array of directions
gw_calc = two_bubbles.GravitationalWaveCalculator(
    initial_field_status="two_bubbles",
    phi1=solver.phi1,
    phi2=solver.phi2,
    z_grid=solver.z_grid,
    ds=solver.ds * solver.history_interval,
)
gw_calc.set_integral_params(tol=1e-7, max_iter=20)
dE_dlogw_dcosthetak = gw_calc.compute_angular_gw_spectrum(
    w_arr=w_arr, cos_thetak_arr=cos_thetak_arr
)
# compute the spectrum dE/dlogw using trapezoidal integrator
# factor of 2 because we only compute dE_dlogw_dcosthetak for positive cos_thetak_arr
dE_dlogw = 2 * np.trapezoid(dE_dlogw_dcosthetak, axis=0, x=cos_thetak_arr)

fig, ax = plt.subplots()
ax.plot(w_arr, dE_dlogw, marker="o", ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(
    r"$\widehat{\Omega}_\text{GW} = \dfrac{1}{\rho_\text{vac}^2} \dfrac{dE_\text{GW}}{d\log \omega} = \dfrac{2}{\rho_\text{vac}^2} \displaystyle\int_0^1 d \cos \theta_k \dfrac{dE_\text{GW}}{d \log \omega d \cos \theta_k}$"
)
ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
ax.grid(True, which="both", alpha=0.5)
fig.savefig(
    f"./figures/two_bubbles/gw_spectrum.png", bbox_inches="tight", facecolor="white"
)

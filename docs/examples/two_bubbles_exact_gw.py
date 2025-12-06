#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <qho@sissa.it>
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from bubble_dynamics.bubble_simulator import LatticeSetup, PDEBubbleSolver
from bubble_dynamics.bubble_simulator.potentials import QuarticPotential
from bubble_dynamics.visualizer import TwoBubblesEvolutionVisualizer
from bubble_gw import two_bubbles
from bubble_gw.utils import sample

# Potential = V(\phi) = \dfrac{c2^2}{2} \phi^2 - \dfrac{c3}{3} \phi^3 + \dfrac{c4}{4} \phi^4
# Parameters of the potential
# lambdabar = 9*c4*c2**2/(2*c3**2)
lambdabar = 0.84
couplings = (1., 3*(np.sqrt(9 - 8*lambdabar) + 3)/(4*lambdabar), (np.sqrt(9 - 8*lambdabar) + 3)**2/(8*lambdabar))
c2, c3, c4 = couplings
potential = QuarticPotential(c2=c2, c3=c3, c4=c4)

# True Vacuum and False Vacuum
phi_absMin = np.array([c2/np.sqrt(8*c4*lambdabar) * (3 + np.sqrt(9 - 8*lambdabar))])
phi_metaMin = np.array([0.0])
phi_metaMax = np.array([c2/np.sqrt(8*c4*lambdabar) * (3 - np.sqrt(9 - 8*lambdabar))]) # local maximum
m_meta = c2 # mass in the unbroken phase
m_abs = c2*np.sqrt((-8*lambdabar + 3*np.sqrt(9 - 8*lambdabar) + 9)/(4*lambdabar)) # mass in the broken phase
rho_vac = (m_abs**4 - m_meta**4)/(12*c4) # vacuum energy

phi_range = (-1, 2) # plotting range
fig, ax = potential.plot_potential(phi_range=phi_range, num_points=100000)
ax.set_xlim(left=phi_range[0], right=phi_range[1])
ax.set_ylim(-0.5, 2)
fig.savefig(rf"./figures/potential.png", bbox_inches="tight", facecolor="white")

setup = LatticeSetup(potential)
setup.set_tunnelling_phi(phi_absMin=phi_absMin, phi_metaMin=phi_metaMin)
fig, ax = setup.plot_profiles(npoints=1000)
ax.legend()
fig.savefig(rf"./figures/bubble_profiles.png", bbox_inches="tight", facecolor="white")

# Simulation parameters
scale_dz = 0.5 # smaller means better z resolution
scale_z = 3 # larger means larger simulation size on z-axis
d = 50 # bubbles separation
smax = d * 2 # simulation time

setup.set_tunnelling_phi(phi_absMin=phi_absMin, phi_metaMin=phi_metaMin)
setup.set_d(d=d)
z_grid, phi_initial, d = setup.two_bubbles(layout="full", scale_dz=scale_dz, scale_z=scale_z)
phi_initial = phi_initial.T # initial 2-bubbles profile
dz = abs(z_grid[1] - z_grid[0]) # space step
ds = dz*0.9 # time step

history_interval = 10 # number of time steps before saving, smaller means better time resolution
solver = PDEBubbleSolver(phi1_initial=phi_initial, z_grid=z_grid, ds=ds, dz=dz, potential=potential, d=d)
phi_evolution = solver.evolve(smax, history_interval=history_interval) # Evolve field over time
solver.compute_phi_region2()

# GW computation
w_arr = sample(1e-3, 1e1, 100, 2, 0, 'log') # array of frequencies
gw_calc = two_bubbles.GravitationalWaveCalculator(initial_field_status="two_bubbles", phi1=solver.phi1, phi2=solver.phi2, z_grid=solver.z_grid, ds=solver.ds*solver.history_interval)
gw_calc.set_integral_params(tol=1e-7, max_iter=20)
dE_dlogw_dcosthetak= gw_calc.compute_angular_gw_spectrum(w_arr=w_arr, cos_thetak_arr=[0.])

fig, ax = plt.subplots()
ax.plot(w_arr, dE_dlogw_dcosthetak.mean(axis=0), marker='o', ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True)
fig.savefig(f"./figures/gw_spectrum.png", bbox_inches="tight", facecolor="white")

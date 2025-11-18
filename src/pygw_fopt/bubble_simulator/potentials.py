import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema  # For finding local extrema

from .generic_potential import GenericPotential

# Here |phi|^2 = (phi_re^2+phi_im^2)/2 for correct normalization of kinetic term
class U1Potential(GenericPotential):
    def __init__(self, delta_V, kappa, v):
        self.Ndim = 2
        self.params = delta_V, kappa, v
        self.params_str = rf"$\Delta V={delta_V}, \kappa={kappa}, v={v}$"
    
    def V0(self, X):
        """Compute the potential V0."""
        phi_re, phi_im = X[...,0], X[...,1]
        phi2 = (phi_re**2 + phi_im**2)/2.
        # Avoid log(0) by setting a small threshold
        phi2 = np.where(phi2 < 1e-100, 1e-100, phi2)
        delta_V, kappa, v = self.params
        rval = 1 + kappa * phi2 / v**2 + (phi2**2 / v**4) * ((kappa + 2) * np.log(phi2 / v**2) - (kappa + 1))
        return 2*delta_V*rval
    
    def dV0(self, X):
        """Compute the derivative of the potential with respect to each field component."""
        phi_re, phi_im = X[...,0], X[...,1]
        phi2 = (phi_re**2 + phi_im**2)/2.
        delta_V, kappa, v = self.params
        
        # Avoid log(0) by setting a small threshold
        phi2 = np.where(phi2 < 1e-100, 1e-100, phi2)
        
        tmp = -kappa*(phi2 - v**2) + 2*(2 + kappa)*phi2*np.log(phi2/v**2)
        dV_dphi_re = tmp * phi_re / v**4
        dV_dphi_im = tmp * phi_im / v**4
        
        # Combine into a single array for the gradient
        dV = np.zeros_like(X)
        dV[..., 0] = dV_dphi_re
        dV[..., 1] = dV_dphi_im
        
        return 2*delta_V*dV
    
    def plot_potential(self, phi_range, fixed_phi_im=0.0, num_points=1000):
        """
        Plot the potential V0(phi_re, phi_im=fixed_phi_im) over a given range and identify local maxima/minima.
        
        Parameters:
        - phi_range: Tuple (phi_re_min, phi_re_max) specifying the range of phi_re
        - fixed_phi_im: The fixed value of phi_im (default 0.0)
        - num_points: Number of points to evaluate the potential (default 1000)
        
        Returns:
        - fig, ax: Matplotlib figure and axis objects
        """
        # Generate phi_re values
        phi_re = np.linspace(phi_range[0], phi_range[1], num_points)
        
        # Create phi array with fixed phi_im
        phi = np.zeros((num_points, self.Ndim))
        phi[:, 0] = phi_re  # phi_re varies
        phi[:, 1] = fixed_phi_im  # phi_im is fixed
        
        # Compute potential
        V = self.V0(phi)
        
        # Find local minima and maxima
        minima_idx = argrelextrema(V, np.less)[0]
        maxima_idx = argrelextrema(V, np.greater)[0]
        
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(phi_re, V, 'b-', label='$V_0(\\phi_{\\text{re}}, \\phi_{\\text{im}}=' + f'{fixed_phi_im}' + ')$')
        
        # Scatter plot for minima and maxima if they exist
        if len(minima_idx) > 0:
            ax.scatter(phi_re[minima_idx], V[minima_idx], color='green', label='Minima', zorder=5)
        if len(maxima_idx) > 0:
            ax.scatter(phi_re[maxima_idx], V[maxima_idx], color='red', label='Maxima', zorder=5)
        
        # Formatting
        ax.set_xlabel('$\\phi_{\\text{re}}$')
        ax.set_ylabel('$V_0(\\phi)$', rotation=0, labelpad=15)
        delta_V, kappa, v = self.params
        ax.set_title(rf'Potential with $\Delta V = {delta_V}$, $\\kappa = {kappa}$, $v = {v}$, $\\phi_{{\\text{{im}}}} = {fixed_phi_im}$')
        ax.set_xlim(phi_range[0], phi_range[1])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Print extrema for reference
        if len(minima_idx) > 0:
            print(f"Minima at phi_re = {phi_re[minima_idx]}, V = {V[minima_idx]}")
        if len(maxima_idx) > 0:
            print(f"Maxima at phi_re = {phi_re[maxima_idx]}, V = {V[maxima_idx]}")
        
        return fig, ax

# class QuarticPotential(GenericPotential):
#     def __init__(self, lambdabar):
#         self.Ndim = 1
#         self.params = lambdabar
    
#     def V0(self, phi):
#         """Compute the potential V0."""
#         lambdabar = self.params
#         return phi**2 * (-phi/3 + phi**2/4 + lambdabar/9)
    
#     def dV0(self, phi):
#         """Compute the derivative of the potential with respect to each field component."""
#         lambdabar = self.params
#         return phi**3 - phi**2 + 2/9*lambdabar*phi
    
#     def plot_potential(self, phi_range, num_points = 1000):
#         # Generate phi_re values
#         phi = np.linspace(phi_range[0], phi_range[1], num_points)

#         # Compute potential
#         V = self.V0(phi)

#         # Find local minima and maxima
#         minima_idx = argrelextrema(V, np.less)[0]
#         maxima_idx = argrelextrema(V, np.greater)[0]

#         # Create plot
#         fig, ax = plt.subplots()

#         ax.plot(phi, V)

#         # Scatter plot for minima and maxima if they exist
#         if len(minima_idx) > 0:
#             ax.scatter(phi[minima_idx], V[minima_idx], color='green', label='Minima', zorder=5)
#         if len(maxima_idx) > 0:
#             ax.scatter(phi[maxima_idx], V[maxima_idx], color='red', label='Maxima', zorder=5)

#         # Formatting
#         ax.set_xlabel('$\\phi_{\\text{re}}$')
#         ax.set_ylabel('$V_0(\\phi)$', rotation=0, labelpad=15)
#         # ax.set_xlim(phi_range[0], phi_range[1])
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend()

#         # Print extrema for reference
#         if len(minima_idx) > 0:
#             print(f"Minima at phi_re = {phi[minima_idx]}, V = {V[minima_idx]}")
#         if len(maxima_idx) > 0:
#             print(f"Maxima at phi_re = {phi[maxima_idx]}, V = {V[maxima_idx]}")

#         return fig, ax

class QuarticPotential(GenericPotential):
    def __init__(self, c2, c3, c4):
        self.Ndim = 1
        self.params = c2, c3, c4
    
    def V0(self, phi):
        """Compute the potential V0."""
        c2, c3, c4 = self.params
        return c2**2*phi**2/2. - c3*phi**3/3. + c4*phi**4/4.
    
    def dV0(self, phi):
        """Compute the derivative of the potential with respect to each field component."""
        c2, c3, c4 = self.params
        return c2**2*phi - c3*phi**2 + c4*phi**3
   
    def d2V0(self, phi):
        """Compute the derivative of the potential with respect to each field component."""
        c2, c3, c4 = self.params
        return c2**2 - 2*c3*phi + 3*c4*phi**2

    
    def plot_potential(self, phi_range, num_points = 1000):
        # Generate phi_re values
        phi = np.linspace(phi_range[0], phi_range[1], num_points)

        # Compute potential
        V = self.V0(phi)

        # Find local minima and maxima
        minima_idx = argrelextrema(V, np.less)[0]
        maxima_idx = argrelextrema(V, np.greater)[0]

        # Create plot
        fig, ax = plt.subplots()

        ax.plot(phi, V)

        # Scatter plot for minima and maxima if they exist
        if len(minima_idx) > 0:
            ax.scatter(phi[minima_idx], V[minima_idx], color='green', label='Minima', zorder=5)
        if len(maxima_idx) > 0:
            ax.scatter(phi[maxima_idx], V[maxima_idx], color='red', label='Maxima', zorder=5)

        # Formatting
        ax.set_xlabel('$\\phi_{\\text{re}}$')
        ax.set_ylabel('$V_0(\\phi)$', rotation=0, labelpad=15)
        # ax.set_xlim(phi_range[0], phi_range[1])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Print extrema for reference
        if len(minima_idx) > 0:
            print(f"Minima at phi_re = {phi[minima_idx]}, V = {V[minima_idx]}")
        if len(maxima_idx) > 0:
            print(f"Maxima at phi_re = {phi[maxima_idx]}, V = {V[maxima_idx]}")

        return fig, ax
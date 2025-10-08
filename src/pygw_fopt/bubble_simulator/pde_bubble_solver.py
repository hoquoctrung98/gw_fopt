import numpy as np
from typing import Union, Optional, List
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from .generic_potential import GenericPotential

class PDEBubbleSolver:
    """Class to solve a coupled set of nonlinear wave equations for bubble dynamics."""
    
    def __init__(
        self,
        phi1_initial: Union[NDArray[np.float64], List[float]],
        z_grid: Union[NDArray[np.float64], List[float]],
        ds: float,
        dz: float,
        potential: GenericPotential,
        d: float
    ) -> None:
        self.Ndim: int = potential.Ndim
        self.phi1_initial = phi1_initial
        self.z_grid: NDArray[np.float64] = np.array(z_grid, dtype=np.float64)
        self.ds: float = min(ds, 0.9*dz)
        self.dz: float = dz
        self.potential: GenericPotential = potential
        self.d: float = d  # Distance between bubble centers
        self.n_z: int = len(z_grid)
        self.history_interval: Optional[int] = None
        self.phi1: Optional[NDArray[np.float64]] = None
        self.phi2: Optional[NDArray[np.float64]] = None
        self.n_s: Optional[int] = None  # Number of history points
        self.s_max: Optional[float] = None  # Maximum simulation time
        self.s_grid: Optional[NDArray[np.float64]] = None  # Time points of phi1
        self.energy_density: Optional[NDArray[np.float64]] = None
        self.energy_how_often: Optional[int] = None
    
    def _spatial_derivative(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        phi_padded = np.pad(phi, ((0, 0), (1, 1)), mode='reflect')
        return (phi_padded[:, :-2] - 2 * phi_padded[:, 1:-1] + phi_padded[:, 2:]) / (self.dz**2)
    
    def evolvepi(
        self,
        phi: NDArray[np.float64],
        pi: NDArray[np.float64],
        s: float,
        ds: float
    ) -> NDArray[np.float64]:
        damping = (1 - 2 * ds / (s + ds))
        phi_transposed = phi.T  # Shape: (n_z, Ndim)
        forcing = s * ds / (s + ds) * (-self.potential.dV0(phi_transposed).T + self._spatial_derivative(phi))
        return pi * damping + forcing
    
    def evolvepi_first_half_step(self, phi_initial: NDArray[np.float64], baby_steps: int = 20) -> NDArray[np.float64]:
        pi = np.zeros_like(phi_initial, dtype=np.float64)  # Use phi_initial instead of self.phi
        baby_ds = 0.5 * self.ds / (baby_steps - 1)
        phi = phi_initial.copy()  # Work with a local copy
        for i in range(1, baby_steps):
            pi = self.evolvepi(phi, pi, (i-1)*baby_ds, baby_ds)
            phi += baby_ds * pi
        return pi
    
    def evolve(self, smax: float, history_interval: int = 1) -> NDArray[np.float64]:
        n_steps = int(np.ceil(smax / self.ds))
        # n_history = (n_steps + history_interval - 1) // history_interval + 1
        n_history = n_steps//history_interval
        n_steps = n_history * history_interval
        self.n_s = n_history + 1
        self.s_max = n_history * self.ds * history_interval
        self.history_interval = history_interval
        
        # Initialize phi1 with the initial condition directly
        phi_init = np.array(self.phi1_initial, dtype=np.float64)  # Temporary variable from __init__
        if phi_init.ndim == 1:
            phi_init = phi_init[np.newaxis, :]
        self.phi1 = np.zeros((self.Ndim, self.n_s, self.n_z), dtype=np.float64)
        self.s_grid = np.linspace(0, self.s_max, self.n_s)
        self.phi1[:, 0, :] = phi_init
        
        pi = self.evolvepi_first_half_step(phi_init)  # Pass initial condition
        phi = phi_init.copy()  # Local variable for evolution
        for i in range(1, n_steps + 1):
            if i > 1:
                pi = self.evolvepi(phi, pi, (i-1)*self.ds, self.ds)
            phi += self.ds * pi
            if i % self.history_interval == 0:
                self.phi1[:, i // self.history_interval, :] = phi
        return self.phi1
    
    def calculate_energy_density(self, how_often: int = 10) -> NDArray[np.float64]:
        phiall = self.phi1  # Shape: (Ndim, n_s, n_z)
        n_s = phiall.shape[1]
        idx = np.arange(0, n_s-2, how_often // self.history_interval)
        idx = idx[idx < n_s-2]
        n_idx = len(idx)
        
        energy_density = np.zeros((n_idx, self.n_z-1), dtype=np.float64)
        
        for i in range(n_idx):
            phi_diff_s = (phiall[:, idx[i]+1, :-1] - phiall[:, idx[i], :-1]) / (self.ds * self.history_interval)
            kin_s = 0.5 * np.sum(phi_diff_s**2, axis=0)
            
            phi_diff_z = (phiall[:, idx[i], 1:] - phiall[:, idx[i], :-1]) / self.dz
            kin_z = 0.5 * np.sum(phi_diff_z**2, axis=0)
            
            phi_flat = phiall[:, idx[i], :-1].T  # Shape: (n_z-1, Ndim)
            pot = self.potential.V0(phi_flat.T)  # Expects (Ndim, n_z-1)
            
            energy_density[i] = kin_s + kin_z + pot[0]
        
        self.energy_density = energy_density
        self.energy_how_often = how_often
        return energy_density
      
    def compute_phi_region2(self, bubble_type="half"):
        n_s, Ndim, n_z = self.n_s, self.Ndim, self.n_z
        d = self.d
        self.phi2 = np.zeros((Ndim, n_s, n_z))  # Match phi1 shape: (Ndim, n_s, n_z)

        if bubble_type == "one":
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]  # Initial time slice for field n
                phi0_interp = interp1d(self.z_grid, phi0, kind='linear', fill_value='extrapolate')
                S, Z = np.meshgrid(self.s_grid, self.z_grid, indexing='ij')
                R = np.sqrt(S**2 + Z**2)
                self.phi2[n, :, :] = phi0_interp(R)  # Assign to field n across all time and space
        else:  # "half" case
            idx_zcenter = np.argmin(np.abs(self.z_grid))
            for n in range(Ndim):
                phi0 = self.phi1[n, 0, :]  # Initial time slice for field n

                # Right side
                phi_right = np.abs(phi0[idx_zcenter:])
                idx_phimid_right = np.argmax(phi_right) + idx_zcenter
                z_shifted_right = self.z_grid[idx_phimid_right:] - self.z_grid[idx_phimid_right]
                z_shifted_right = np.concatenate((-z_shifted_right[::-1], z_shifted_right))
                phi0_right = phi0[idx_phimid_right:]
                phi0_right = np.concatenate((phi0_right[::-1], phi0_right))
                phi0_interp_right = interp1d(z_shifted_right, phi0_right, kind='linear', fill_value=(0., 0.), bounds_error=False)

                # Left side
                phi_left = np.abs(phi0[:idx_zcenter])
                idx_phimid_left = np.argmax(phi_left)
                z_shifted_left = self.z_grid[:idx_phimid_left + 1] - self.z_grid[idx_phimid_left]
                z_shifted_left = np.concatenate((-z_shifted_left[::-1], z_shifted_left[1:]))
                phi0_left = phi0[:idx_phimid_left + 1]
                phi0_left = np.concatenate((phi0_left[::-1], phi0_left[1:]))
                phi0_interp_left = interp1d(z_shifted_left, phi0_left, kind='linear', fill_value=(0., 0.), bounds_error=False)

                # Vectorized computation
                S, Z = np.meshgrid(self.s_grid, self.z_grid, indexing='ij')
                r_right = np.sqrt(S**2 + (Z - d/2)**2)
                r_left = np.sqrt(S**2 + (Z + d/2)**2)
                self.phi2[n, :, :] = phi0_interp_right(r_right) + phi0_interp_left(r_left)
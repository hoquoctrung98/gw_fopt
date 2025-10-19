import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

class SimulationState(ABC):
    @property
    @abstractmethod
    def dt(self) -> float:
        """Time step for the simulation."""
        pass

    @property
    @abstractmethod
    def V_remaining(self) -> float:
        """Remaining volume not contained within bubbles."""
        pass

    @abstractmethod
    def _get_valid_points(self, t: float = None) -> np.ndarray:
        """Get grid points outside all bubbles at time t."""
        pass

    @abstractmethod
    def _update_remaining_volume_bulk(self, t: float) -> float:
        """Update and return the remaining volume at time t."""
        pass


class NucleationStrategy(ABC):
    @abstractmethod
    def nucleate(self, t: float, state: SimulationState) -> np.ndarray:
        """
        Generate new bubble centers at time t.
        
        Args:
            t (float): Current simulation time.
            state: Reference to the state for accessing grid, remaining volume, etc.
        
        Returns:
            np.ndarray: Array of new bubble centers.
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the strategy's parameters during initialization."""
        pass


class BubbleFormationSimulator(SimulationState):
    def __init__(
        self,
        L: float,
        N: int,
        vw: float,
        dt: float,
        nucleation_strategy: NucleationStrategy
    ):
        # Store dt and V_remaining as instance variables
        self._dt = dt  # Use _dt for internal storage
        self.L = L
        self.N = N
        self.vw = vw
        self.strategy = nucleation_strategy

        self.x = np.linspace(0, L, N, endpoint=False)
        self.grid = np.array(np.meshgrid(self.x, self.x, self.x)).T.reshape(-1, 3)
        self.bubbles = []
        self.V_total = L**3
        self._V_remaining = self.V_total  # Use _V_remaining for internal storage
        self.bubble_df = None
        self.t_arr = None

        self.is_outside = np.ones(len(self.grid), dtype=bool)
        self.bubble_centers = []
        self.bubble_times = []
        self.dist_to_centers = []

        self.strategy.validate()

    @property
    def dt(self) -> float:
        """Time step for the simulation (read-only property)."""
        return self._dt

    @property
    def V_remaining(self) -> float:
        """Remaining volume not contained within bubbles (read-only property)."""
        return self._V_remaining

    def _get_valid_points(self, t: float = None) -> np.ndarray:
        if not self.bubbles:
            return self.grid
        if t is None:
            t = self.t_arr[-1] if self.t_arr is not None else 0
        for i, (center, tn) in enumerate(zip(self.bubble_centers, self.bubble_times)):
            radius = self.vw * (t - tn)
            radius = max(radius, 0)
            if radius > 0:
                self.is_outside &= (self.dist_to_centers[i] > radius)
        return self.grid[self.is_outside]

    def _update_remaining_volume_bulk(self, t: float) -> float:
        valid_points = self._get_valid_points(t)
        fraction_remaining = len(valid_points) / len(self.grid)
        self._V_remaining = self.V_total * fraction_remaining  # Update internal variable
        return self._V_remaining

    def _update_outside_mask(self, new_bubbles: list, t: float):
        if not new_bubbles:
            return
        new_centers = np.array([b[0] for b in new_bubbles])
        new_times = np.array([b[1] for b in new_bubbles])
        for center, tn in zip(new_centers, new_times):
            self.bubble_centers.append(center)
            self.bubble_times.append(tn)
            dist = np.linalg.norm(self.grid - center, axis=1)
            self.dist_to_centers.append(dist)
        for i, (center, tn) in enumerate(zip(new_centers, new_times)):
            radius = self.vw * (t - tn)
            if radius > 0:
                self.is_outside &= (self.dist_to_centers[-len(new_centers) + i] > radius)

    def run_simulation(self, T: float, verbose: bool = False):
        t = 0
        self.t_arr = np.arange(0, T + self.dt, self.dt)
        bubble_data = []
        for t in self.t_arr:
            if verbose:
                print(f"Simulating time step: t = {t:.2f}")
            new_centers = self.strategy.nucleate(t, self)
            if len(new_centers) > 0:
                new_bubbles = [(center, t) for center in new_centers]
                self.bubbles.extend(new_bubbles)
                self._update_outside_mask(new_bubbles, t)
                bubble_data.extend([(c[0], c[1], c[2], t) for c in new_centers])
        if bubble_data:
            self.bubble_df = pd.DataFrame(
                bubble_data,
                columns=["center_x", "center_y", "center_z", "nucleation_time"]
            )
            self.bubble_df.insert(0, "bubble_idx", range(len(self.bubble_df)))
        else:
            self.bubble_df = pd.DataFrame(columns=["center_x", "center_y", "center_z", "nucleation_time"])
        self._update_remaining_volume_bulk(T)

    def draw_lattice(self, t: float, max_points_per_side: int = None, smoothness=10):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        if isinstance(smoothness, (int, float)):
            smoothness_u = smoothness_v = int(smoothness)
        elif isinstance(smoothness, (list, tuple)) and len(smoothness) == 2:
            smoothness_u, smoothness_v = map(int, smoothness)
        else:
            raise ValueError("smoothness must be an integer or a list/tuple of two integers.")

        inside_bubble = np.zeros(len(self.grid), dtype=bool)
        for idx, (center, tn) in enumerate(self.bubbles):
            radius = self.vw * (t - tn)
            if radius <= 0:
                continue
            dist = np.linalg.norm(self.grid - center, axis=1)
            inside_bubble |= dist <= radius

        if max_points_per_side is not None and max_points_per_side < self.N:
            x_plot = np.linspace(0, self.L, max_points_per_side, endpoint=False)
            plot_grid = np.array(np.meshgrid(x_plot, x_plot, x_plot)).T.reshape(-1, 3)
            inside_bubble_plot = np.zeros(len(plot_grid), dtype=bool)
            for idx, (center, tn) in enumerate(self.bubbles):
                radius = self.vw * (t - tn)
                if radius <= 0:
                    continue
                dist = np.linalg.norm(plot_grid - center, axis=1)
                inside_bubble_plot |= dist <= radius
            outside_points = plot_grid[~inside_bubble_plot]
            inside_points = plot_grid[inside_bubble_plot]
        else:
            outside_points = self.grid[~inside_bubble]
            inside_points = self.grid[inside_bubble]

        if len(outside_points) > 0:
            ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], c="blue", s=1, label="Outside Bubbles")
        if len(inside_points) > 0:
            ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], c="red", s=1, label="Inside Bubbles")

        for center, tn in self.bubbles:
            radius = self.vw * (t - tn)
            if radius > 0:
                u, v = np.mgrid[0:2*np.pi:(smoothness_u*1j), 0:np.pi:(smoothness_v*1j)]
                x = center[0] + radius * np.cos(u) * np.sin(v)
                y = center[1] + radius * np.sin(u) * np.sin(v)
                z = center[2] + radius * np.cos(v)
                ax.plot_surface(x, y, z, color="orange", alpha=0.3, edgecolor="none")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_zlim(0, self.L)
        ax.set_title(f"Time = {t:.2f}")
        ax.legend()
        return fig, ax

    def plot_bubble_formation_histogram(self) -> tuple:
        """
        Plot a histogram of bubble formation over time, shifted half a step to the right so the left edge 
        of the first bar aligns with t = 0.0.

        The x-axis represents time (with bin width equal to dt), and the y-axis represents the number of bubbles formed
        within each time bin (t, t+dt). Y-axis ticks are ensured to be integers.

        Returns:
            fig, ax: Matplotlib figure and axis objects for the histogram plot.
        """
        if self.t_arr is None or len(self.bubbles) == 0:
            raise ValueError("Simulation has not been run yet. Please call `run_simulation` first.")

        # Extract nucleation times of all bubbles
        nucleation_times = np.array([b[1] for b in self.bubbles])

        # Use time array for bins, starting half a step earlier to shift right
        bin_width = self.dt
        shift = bin_width / 2  # Shift by 0.05 for dt = 0.1
        bins = np.arange(-shift, self.t_arr[-1] + bin_width, bin_width)  # Start at -0.05, step by 0.1, up to 3.1

        # Compute the histogram with explicit bins, ensuring no data loss at edges
        counts, bin_edges = np.histogram(nucleation_times, bins=bins, density=False)

        # Create the histogram plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            bin_edges[:-1] + shift,  # Shift bar positions right by 0.05
            counts,
            width=bin_width,
            align='edge',  # Align bars to the left edge for exact positioning
            alpha=0.7,
            color='blue',
            edgecolor='black',
            label=f"Bin width = {bin_width:.3f}"
        )

        # Adjust x-axis limits to show the full histogram, aligning the first bar's left edge at 0.0
        ax.set_xlim(0, self.t_arr[-1] + bin_width)  # From 0 to 3.1 to show all bars fully

        # Set plot labels and title
        ax.set_title("Bubble Formation Histogram", fontsize=16)
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Number of Bubbles Formed", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Ensure y-axis ticks are integers
        if counts.size > 0 and np.any(counts > 0):  # Check for any non-zero counts
            max_count = int(np.ceil(max(counts)))  # Round up for clear visualization
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylim(0, max_count + 1)
        else:
            ax.set_ylim(0, 1)  # Default range if no bubbles

        return fig, ax


class PoissonNucleation(NucleationStrategy):
    def __init__(self, poisson_params: dict):
        self.poisson_params = poisson_params
        self.validate()

    def validate(self):
        if not all(key in self.poisson_params for key in ["Gamma0", "beta", "t0"]):
            raise ValueError("Poisson nucleation requires 'Gamma0', 'beta', and 't0' in poisson_params.")
        self.Gamma0 = self.poisson_params["Gamma0"]
        self.beta = self.poisson_params["beta"]
        self.t0 = self.poisson_params["t0"]

    def nucleate(self, t: float, simulator: BubbleFormationSimulator) -> np.ndarray:
        simulator._update_remaining_volume_bulk(t)
        Gamma_t = self.Gamma0 * np.exp(self.beta * (t - self.t0))
        num_bubbles = int(Gamma_t * simulator.dt * simulator.V_remaining)
        valid_points = simulator._get_valid_points(t)
        if len(valid_points) == 0 or num_bubbles == 0:
            return np.array([])
        selected_indices = np.random.choice(len(valid_points), size=min(num_bubbles, len(valid_points)), replace=False)
        return valid_points[selected_indices]


class ManualNucleation(NucleationStrategy):
    def __init__(self, manual_nucleation_schedule: dict):
        self.schedule = manual_nucleation_schedule
        self.validate()

    def validate(self):
        if self.schedule is None:
            raise ValueError("Manual nucleation requires a nucleation schedule.")
        # Could reuse _validate_manual_schedule logic here, but as a method of this class
        sorted_times = sorted(self.schedule.keys())
        existing_bubbles = []
        for t in sorted_times:
            centers = self.schedule[t]
            valid_centers = []
            for center in centers:
                center_array = np.array(center)
                if self._is_point_valid(center_array, t, existing_bubbles):
                    valid_centers.append(center)
                else:
                    raise ValueError(f"Bubble at {center} at time {t} overlaps with earlier bubbles.")
            existing_bubbles.extend([(np.array(c), t) for c in valid_centers])
            self.schedule[t] = valid_centers

    def _is_point_valid(self, point: np.ndarray, t: float, existing_bubbles: list) -> bool:
        for center, tn in existing_bubbles:
            radius = self.vw * (t - tn) if hasattr(self, 'vw') else 0  # vw would need to come from simulator
            if radius > 0 and np.linalg.norm(point - center) <= radius:
                return False
        return True

    def nucleate(self, t: float, simulator: 'BubbleFormationSimulator') -> np.ndarray:
        time_range = (t - simulator.dt, t)
        new_centers = []
        for nucleation_time, centers in self.schedule.items():
            if time_range[0] < nucleation_time <= time_range[1]:
                new_centers.extend(centers)
        return np.array(new_centers) if new_centers else np.array([])

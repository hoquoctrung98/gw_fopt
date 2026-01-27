import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from .bubble_formation_simulator import BubbleFormationSimulator
from joblib import Parallel, delayed

class BubbleEnvelopeMC:
    def __init__(self, simulator: BubbleFormationSimulator):
        """
        Initialize the Monte Carlo envelope for bubble computations.
        """
        self.simulator = simulator

    def compute_total_surface_area_over_time(
        self,
        base_samples: int = 10000,
        include_isolated_bubbles: bool = True,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Compute the total non-overlapping surface area at each time step.

        Parameters:
            base_samples (int): Total number of Monte Carlo samples across all bubbles.
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the computation.
            n_jobs (int): Number of parallel jobs for computation (-1 uses all available cores).

        Returns:
            np.ndarray: Array of total surface areas at each time step.
        """
        if self.simulator.t_arr is None:
            raise ValueError(
                "Simulation has not been run yet. Please call `run_simulation` first."
            )

        # Precompute bubble radii for all time steps
        bubble_centers = self.simulator.bubble_df[["center_x", "center_y", "center_z"]].to_numpy()
        nucleation_times = self.simulator.bubble_df["nucleation_time"].to_numpy()
        bubble_radii_over_time = np.maximum(
            self.simulator.vw * (self.simulator.t_arr[:, None] - nucleation_times), 0
        )
        total_surface_area_over_time = 4 * np.pi * bubble_radii_over_time**2

        # Precompute valid indices (bubbles with positive radii at each time step)
        valid_indices_over_time = bubble_radii_over_time > 0

        # Define a helper function for computing non-overlapping surface area at a single time step
        def compute_for_time_step(t_idx):
            valid_indices = valid_indices_over_time[t_idx]
            valid_centers = bubble_centers[valid_indices]
            valid_radii = bubble_radii_over_time[t_idx][valid_indices]
            total_surface_area = total_surface_area_over_time[t_idx][valid_indices]

            if len(valid_centers) == 0:
                return 0

            # Allocate samples proportionally
            sample_weights = total_surface_area / np.sum(total_surface_area)
            bubble_samples = np.floor(sample_weights * base_samples).astype(int)
            remaining_samples = int(base_samples - np.sum(bubble_samples))
            if remaining_samples > 0:
                fractional_parts = (sample_weights * base_samples) - bubble_samples
                additional_samples = np.argsort(fractional_parts)[-remaining_samples:]
                bubble_samples[additional_samples] += 1

            non_overlapping_count = 0
            for idx, (center, radius, samples) in enumerate(
                zip(valid_centers, valid_radii, bubble_samples)
            ):
                if samples == 0:
                    continue

                # Generate points uniformly on the surface of the sphere
                surface_points = self._sample_sphere_surface(center, radius, samples)

                # Clip points to the lattice boundaries
                surface_points = np.clip(surface_points, 0, self.simulator.L)

                if len(surface_points) == 0:
                    continue

                # Check overlap with other bubbles
                other_indices = np.arange(len(valid_centers)) != idx
                other_centers = valid_centers[other_indices]
                other_radii = valid_radii[other_indices]

                # Determine if the bubble is isolated
                is_isolated = all(
                    np.linalg.norm(center - c) >= r + radius + 1e-8
                    for c, r in zip(other_centers, other_radii)
                )

                if include_isolated_bubbles or not is_isolated:
                    if len(other_centers) == 0:
                        non_overlapping_count += len(surface_points)
                    else:
                        dist_matrix = distance_matrix(surface_points, other_centers)
                        non_overlapping_mask = np.all(
                            dist_matrix >= other_radii + 1e-8, axis=1
                        )
                        non_overlapping_count += np.sum(non_overlapping_mask)

            # Compute total non-overlapping surface area for this time step
            total_surface_area = np.sum(
                total_surface_area_over_time[t_idx][valid_indices]
            )
            return (non_overlapping_count / base_samples) * total_surface_area

        # Parallelize computation across time steps
        total_surface_areas = Parallel(n_jobs=n_jobs)(
            delayed(compute_for_time_step)(t_idx) for t_idx in range(len(self.simulator.t_arr))
        )

        return np.array(total_surface_areas)

    def plot_total_surface_area_vs_time(
        self, base_samples: int = 10000, include_isolated_bubbles: bool = False
    ):
        """
        Plot the total non-overlapping surface area as a function of time.

        Parameters:
            base_samples (int): Total number of Monte Carlo samples across all bubbles.
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the computation.
        """
        if self.simulator.t_arr is None:
            raise ValueError(
                "Simulation has not been run yet. Please call `run_simulation` first."
            )

        total_surface_areas = self.compute_total_surface_area_over_time(
            base_samples, include_isolated_bubbles
        )

        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.simulator.t_arr, total_surface_areas, color="blue")
        ax.set_title("Total Non-Overlapping Surface Area vs Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Surface Area")
        ax.grid(True)
        fig.legend()
        return fig, ax

    def compute_non_overlapping_surface_area(
        self, t: float, base_samples: int = 10000, include_isolated_bubbles: bool = True
    ) -> pd.DataFrame:
        """
        Compute the non-overlapping surface area of bubbles using Monte Carlo sampling.

        Parameters:
            t (float): Current simulation time.
            base_samples (int): Number of Monte Carlo samples for a reference bubble (e.g., a bubble with radius L/2).
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the computation.

        Returns:
            pd.DataFrame: DataFrame with non-overlapping surface areas.
        """
        results = []
        bubble_centers = self.simulator.bubble_df[["center_x", "center_y", "center_z"]].to_numpy()
        bubble_radii = self.simulator.vw * (t - self.simulator.bubble_df["nucleation_time"].to_numpy())

        # Define a reference bubble (e.g., a bubble with radius L/2)
        reference_radius = self.simulator.L / 2
        reference_surface_area = 4 * np.pi * reference_radius**2

        # Ensure radii are positive
        valid_indices = bubble_radii > 0
        valid_centers = bubble_centers[valid_indices]
        valid_radii = bubble_radii[valid_indices]

        # Iterate over valid bubbles
        for idx, (center, radius) in enumerate(zip(valid_centers, valid_radii)):
            # Compute the surface area of the current bubble
            total_surface_area = 4 * np.pi * radius**2

            # Determine the number of samples for this bubble
            if reference_surface_area == 0:
                samples = base_samples  # Avoid division by zero
            else:
                samples = max(
                    1, int((total_surface_area / reference_surface_area) * base_samples)
                )

            # Generate points uniformly on the surface of the sphere
            surface_points = self._sample_sphere_surface(center, radius, samples)

            # Clip points to the lattice boundaries instead of filtering them out entirely
            surface_points = np.clip(surface_points, 0, self.simulator.L)

            if len(surface_points) == 0:
                results.append({"bubble_idx": idx, "non_overlapping_surface_area": 0})
                continue

            # Vectorized overlap check with other bubbles
            other_indices = (
                np.arange(len(valid_centers)) != idx
            )  # Mask to exclude the current bubble
            other_centers = valid_centers[other_indices]
            other_radii = valid_radii[other_indices]

            if len(other_centers) == 0:
                non_overlapping_count = len(surface_points)  # No other bubbles to check
            else:
                # Compute distance matrix between sampled points and other bubble centers
                dist_matrix = distance_matrix(surface_points, other_centers)

                # Ensure other_radii is broadcasted correctly
                if dist_matrix.shape[1] != len(other_radii):
                    raise ValueError(
                        "Mismatch in dimensions between distance matrix and radii."
                    )

                # Add tolerance to avoid floating-point precision issues
                non_overlapping_mask = np.all(dist_matrix >= other_radii + 1e-8, axis=1)
                non_overlapping_count = np.sum(non_overlapping_mask)

            # Compute non-overlapping surface area
            non_overlapping_surface_area = (
                non_overlapping_count / len(surface_points)
            ) * total_surface_area

            # Check if the bubble is isolated
            is_isolated = all(
                np.linalg.norm(center - c) >= r + radius + 1e-8
                for c, r in zip(other_centers, other_radii)
            )

            # Include only non-isolated bubbles if the flag is False
            if include_isolated_bubbles or not is_isolated:
                results.append(
                    {
                        "bubble_idx": idx,
                        "non_overlapping_surface_area": non_overlapping_surface_area,
                    }
                )
            else:
                results.append({"bubble_idx": idx, "non_overlapping_surface_area": 0})

        # Map valid indices back to original indices
        original_indices = np.arange(len(bubble_radii))  # Original bubble indices
        valid_index_map = {
            j: i for i, j in enumerate(original_indices[valid_indices])
        }  # Map valid indices to original indices

        # Initialize full_results with zeros
        full_results = [
            {"bubble_idx": i, "non_overlapping_surface_area": 0}
            for i in range(len(bubble_radii))
        ]

        # Update full_results using the valid_index_map
        for result in results:
            if result["bubble_idx"] in valid_index_map:
                full_results[valid_index_map[result["bubble_idx"]]] = result

        return pd.DataFrame(full_results)

    def _sample_sphere_surface(
        self, center: np.ndarray, radius: float, num_samples: int
    ) -> np.ndarray:
        """
        Sample points uniformly on the surface of a sphere and ensure they lie within the lattice boundaries.

        Parameters:
            center (np.ndarray): Center of the sphere.
            radius (float): Radius of the sphere.
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of sampled points on the sphere surface, clipped to the lattice boundaries.
        """
        theta = np.random.uniform(0, 2 * np.pi, num_samples)  # Azimuthal angle
        phi = np.arccos(np.random.uniform(-1, 1, num_samples))  # Polar angle

        # Generate points on the sphere surface
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)

        # Clip points to the lattice boundaries [0, L]
        points = np.column_stack([x, y, z])
        clipped_points = np.clip(points, 0, self.simulator.L)

        return clipped_points

    def draw_non_overlapping_surfaces(
        self, t: float, num_samples: int = 10000, include_isolated_bubbles: bool = True
    ):
        """
        Visualize the non-overlapping surfaces of bubbles.

        Parameters:
            t (float): Current simulation time.
            num_samples (int): Total number of Monte Carlo samples across all bubbles.
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the visualization.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        bubble_centers = self.simulator.bubble_df[["center_x", "center_y", "center_z"]].to_numpy()
        bubble_radii = self.simulator.vw * (t - self.simulator.bubble_df["nucleation_time"].to_numpy())

        total_surface_area = 4 * np.pi * bubble_radii**2
        total_surface_area[bubble_radii <= 0] = 0

        if np.sum(total_surface_area) == 0:
            return

        sample_weights = total_surface_area / np.sum(total_surface_area)
        bubble_samples = np.floor(sample_weights * num_samples).astype(int)

        for idx, (center, radius, samples) in enumerate(
            zip(bubble_centers, bubble_radii, bubble_samples)
        ):
            if radius <= 0 or samples == 0:
                continue

            # Generate points uniformly on the surface of the sphere
            surface_points = self._sample_sphere_surface(center, radius, samples)

            # Check overlap with other bubbles
            other_bubbles = [
                (c, r)
                for i, (c, r) in enumerate(zip(bubble_centers, bubble_radii))
                if i != idx
            ]
            non_overlapping_points = [
                p
                for p in surface_points
                if all(np.linalg.norm(p - c) >= r for c, r in other_bubbles)
            ]

            # Check if the bubble is isolated
            is_isolated = all(
                np.linalg.norm(center - c) >= r + radius for c, r in other_bubbles
            )

            # Plot non-overlapping surface points
            if include_isolated_bubbles or not is_isolated:
                if non_overlapping_points:
                    non_overlapping_points = np.array(non_overlapping_points)
                    ax.scatter(
                        non_overlapping_points[:, 0],
                        non_overlapping_points[:, 1],
                        non_overlapping_points[:, 2],
                        c="green",
                        s=1,
                    )

        ax.set_xlim(0, self.simulator.L)
        ax.set_ylim(0, self.simulator.L)
        ax.set_zlim(0, self.simulator.L)
        ax.legend()
        return fig, ax

    def compute_A_matrices_over_time(
        self,
        w: float,
        k_hat: np.ndarray,
        base_samples: int = 10000,
        include_isolated_bubbles: bool = True,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Compute the A-matrix for all bubbles at each time step.
        
        Parameters:
            w (float): Frequency parameter in the exponential term.
            k_hat (np.ndarray): Input vector (3 components) that should be normalized before proceeding.
            base_samples (int): Total number of Monte Carlo samples across all bubbles.
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the computation.
            n_jobs (int): Number of parallel jobs for computation (-1 uses all available cores).
        
        Returns:
            pd.DataFrame: DataFrame containing A-matrices for all bubbles at each time step.
        """
        if self.simulator.t_arr is None:
            raise ValueError("Simulation has not been run yet. Please call `run_simulation` first.")
        if len(k_hat) != 3:
            raise ValueError("k_hat must be a 3-dimensional vector.")
        
        # Normalize k_hat
        k_hat = k_hat / np.linalg.norm(k_hat)
        
        # Precompute bubble radii for all time steps
        bubble_centers = self.simulator.bubble_df[["center_x", "center_y", "center_z"]].to_numpy()
        nucleation_times = self.simulator.bubble_df["nucleation_time"].to_numpy()
        bubble_radii_over_time = np.maximum(self.simulator.vw * (self.simulator.t_arr[:, None] - nucleation_times), 0)
        
        # Precompute valid indices (bubbles with positive radii at each time step)
        valid_indices_over_time = bubble_radii_over_time > 0
        
        def compute_A_for_time_step(t_idx):
            valid_indices = valid_indices_over_time[t_idx]
            valid_centers = bubble_centers[valid_indices]
            valid_radii = bubble_radii_over_time[t_idx][valid_indices]
            
            if len(valid_centers) == 0:
                return []
            
            # Allocate samples proportionally
            total_surface_area = 4 * np.pi * valid_radii**2
            sample_weights = total_surface_area / np.sum(total_surface_area)
            bubble_samples = np.floor(sample_weights * base_samples).astype(int)
            remaining_samples = int(base_samples - np.sum(bubble_samples))
            
            if remaining_samples > 0:
                fractional_parts = (sample_weights * base_samples) - bubble_samples
                additional_samples = np.argsort(fractional_parts)[-remaining_samples:]
                bubble_samples[additional_samples] += 1
            
            results = []
            t = self.simulator.t_arr[t_idx]
            
            for idx, (center, radius, samples) in enumerate(zip(valid_centers, valid_radii, bubble_samples)):
                if samples == 0 or radius <= 0:
                    results.append({"bubble_idx": idx, "A_matrix": np.zeros((3, 3), dtype=complex)})
                    continue
                
                # Generate points uniformly on the surface of the sphere
                surface_points = self._sample_sphere_surface(center, radius, samples)
                
                # Clip points to the lattice boundaries
                surface_points = np.clip(surface_points, 0, self.simulator.L)
                
                if len(surface_points) == 0:
                    results.append({"bubble_idx": idx, "A_matrix": np.zeros((3, 3), dtype=complex)})
                    continue
                
                # Check overlap with other bubbles
                other_indices = np.arange(len(valid_centers)) != idx
                other_centers = valid_centers[other_indices]
                other_radii = valid_radii[other_indices]
                
                if len(other_centers) == 0:
                    non_overlapping_mask = np.ones(len(surface_points), dtype=bool)
                else:
                    dist_matrix = cdist(surface_points, other_centers)
                    non_overlapping_mask = np.all(dist_matrix >= other_radii + 1e-8, axis=1)  # Add tolerance
                
                non_overlapping_points = surface_points[non_overlapping_mask]
                
                if len(non_overlapping_points) == 0:
                    results.append({"bubble_idx": idx, "A_matrix": np.zeros((3, 3), dtype=complex)})
                    continue
                
                # Compute unit vectors x_hat for each point on the bubble surface
                x_hat = (non_overlapping_points - center) / radius
                
                # Compute the exponential term for each point
                exponential_term = np.exp(
                    -1j * w * self.simulator.vw * (t - nucleation_times[valid_indices][idx]) * np.dot(x_hat, k_hat)
                )
                
                # Initialize the symmetric A matrix
                A = np.zeros((3, 3), dtype=complex)
                
                # Compute the integral for the upper triangular part of A
                for i in range(3):
                    for j in range(i, 3):
                        integrand = x_hat[:, i] * x_hat[:, j] * exponential_term
                        A[i, j] = np.mean(integrand) * (4 * np.pi)  # Monte Carlo estimate of the integral
                
                # Fill the lower triangular part using symmetry
                A += A.T  # Add the transpose to fill the lower triangle
                A /= 2  # Divide the entire matrix by 2 to ensure symmetry
                
                # Check if the bubble is isolated
                is_isolated = all(
                    np.linalg.norm(center - c) >= r + radius + 1e-8
                    for c, r in zip(other_centers, other_radii)
                )
                
                # Include only non-isolated bubbles if the flag is False
                if include_isolated_bubbles or not is_isolated:
                    results.append({"bubble_idx": idx, "A_matrix": A})
                else:
                    results.append({"bubble_idx": idx, "A_matrix": np.zeros((3, 3), dtype=complex)})
            
            return results

        # Parallelize computation across time steps
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(compute_A_for_time_step)(t_idx) for t_idx in range(len(self.simulator.t_arr))
        )
        
        # Flatten the results and create a DataFrame
        flattened_results = []
        for t_idx, results in enumerate(all_results):
            for result in results:
                flattened_results.append(
                    {
                        "time": self.simulator.t_arr[t_idx],
                        "bubble_idx": result["bubble_idx"],
                        "A_matrix": result["A_matrix"],
                    }
                )
        
        return pd.DataFrame(flattened_results)

    def compute_C_matrix(
        self,
        w: float,
        k_hat: np.ndarray,
        base_samples: int = 10000,
        include_isolated_bubbles: bool = True,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Compute the matrix C(i, j, w, k_hat) using the precomputed A matrices over time.

        Parameters:
            w (float): Frequency parameter in the exponential term.
            k_hat (np.ndarray): Input vector (3 components) that should be normalized before proceeding.
            base_samples (int): Total number of Monte Carlo samples across all bubbles.
            include_isolated_bubbles (bool): Whether to include isolated bubbles in the computation.
            n_jobs (int): Number of parallel jobs for computation (-1 uses all available cores).

        Returns:
            np.ndarray: The resulting 3x3 complex-valued matrix C.
        """
        if self.simulator.t_arr is None:
            raise ValueError(
                "Simulation has not been run yet. Please call `run_simulation` first."
            )

        if len(k_hat) != 3:
            raise ValueError("k_hat must be a 3-dimensional vector.")

        # Normalize k_hat
        k_hat = k_hat / np.linalg.norm(k_hat)

        # Precompute bubble centers and nucleation times
        bubble_centers = self.simulator.bubble_df[["center_x", "center_y", "center_z"]].to_numpy()
        nucleation_times = self.simulator.bubble_df["nucleation_time"].to_numpy()

        # Compute A matrices over time
        A_matrices_df = self.compute_A_matrices_over_time(
            w,
            k_hat,
            base_samples=base_samples,
            include_isolated_bubbles=include_isolated_bubbles,
            n_jobs=n_jobs,
        )

        # Initialize C matrix
        C = np.zeros((3, 3), dtype=complex)

        # Group A matrices by bubble index
        grouped_A = A_matrices_df.groupby("bubble_idx")

        # Iterate over all bubbles
        for bubble_idx, group in grouped_A:
            # Extract bubble-specific data
            xn = bubble_centers[bubble_idx]
            tn = nucleation_times[bubble_idx]
            times = group["time"].to_numpy()
            A_matrices = np.stack(
                group["A_matrix"].to_list(), axis=0
            )  # Shape: (num_time_steps, 3, 3)

            # Compute the dot product k_hat . x_n
            k_dot_xn = np.dot(k_hat, xn)

            # Compute the exponential term exp(i * w * (t - k_hat . x_n))
            exponential_term = np.exp(1j * w * (times - k_dot_xn))

            # Compute the cubic term (t - t_n)^3
            cubic_term = (times - tn) ** 3

            # Compute the integrand for this bubble
            integrand = (
                exponential_term[:, None, None] * cubic_term[:, None, None] * A_matrices
            )

            # Integrate over time using the trapezoidal rule
            bubble_contribution = np.trapz(integrand, x=times, axis=0)

            # Accumulate contributions to C
            C += bubble_contribution

        # Normalize the result
        C /= 6 * np.pi

        return C
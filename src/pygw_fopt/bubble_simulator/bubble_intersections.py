import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import scipy.integrate as integrate
import joblib
from joblib import Parallel, delayed


from .bubble_formation_simulator import BubbleFormationSimulator
from ..utils.segment_complement import SegmentComplement

class BubbleIntersections:
    def __init__(self, simulator: BubbleFormationSimulator):
        """
        Initialize the class with bubble data and simulation parameters (vw, t_arr, dt).
        """
        # Extract bubble data as NumPy arrays instead of DataFrame
        # Assuming bubble_df has columns: ['bubble_idx', 'center_x', 'center_y', 'center_z', 'nucleation_time']
        bubble_data = simulator.bubble_df.to_numpy()
        self.bubble_centers = bubble_data[:, 1:4].astype(float)  # Shape: (n_bubbles, 3) for [center_x, center_y, center_z]
        self.bubble_nucleation_times = bubble_data[:, 4].astype(float)  # Shape: (n_bubbles,)
        self.bubble_indices = bubble_data[:, 0].astype(int)  # Shape: (n_bubbles,)
        self.vw = simulator.vw
        self.t_arr = simulator.t_arr  # Store the simulation time array
        self.dt = simulator.dt
        self.intersections = None  # Store intersections as a list of lists/dicts instead of DataFrame

    def compute_radius(self, nucleation_time, t):
        """
        Computes the radius of a bubble given vw, t, and nucleation_time.
        Ensures the radius is non-negative.
        """
        if t < nucleation_time:
            return 0
        return self.vw * (t - nucleation_time)

    def compute_circle_of_intersection(self, center1, radius1, center2, radius2):
        """
        Computes the circle of intersection between two spheres.
        Returns the center and radius of the circle in 3D space.
        """
        d = np.linalg.norm(center2 - center1)  # Distance between centers
        if d > radius1 + radius2 or d < abs(radius1 - radius2):
            return None  # No intersection (spheres too far apart or one inside without touching)
        
        # Compute the point along the line joining the centers
        x = (radius1**2 - radius2**2 + d**2) / (2 * d)
        h = np.sqrt(np.maximum(0, radius1**2 - x**2))  # Radius of the intersection circle, ensure non-negative
        
        if h == 0:
            return None  # No intersection circle (tangent or degenerate case)
        
        # Compute the center of the circle
        v = (center2 - center1) / d
        circle_center = center1 + x * v
        
        return circle_center, h

    def generate_circle_points(self, circle_center, radius, normal_vector, num_points=10000):
        """
        Generates points on the circle in 3D using the parametric equation.
        """
        # Find two orthogonal vectors in the plane of the circle
        u = np.array([1, 0, 0]) if normal_vector[1] != 0 or normal_vector[2] != 0 else np.array([0, 1, 0])
        v1 = np.cross(normal_vector, u)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal_vector, v1)
        v2 /= np.linalg.norm(v2)
        
        # Generate points on the circle
        t = np.linspace(0, 2 * np.pi, num_points)
        points = circle_center[:, None] + radius * (
            np.cos(t) * v1[:, None] + np.sin(t) * v2[:, None]
        )
        return points.T  # Shape: (num_points, 3)

    def convert_to_spherical(self, points, reference_center):
        """
        Converts Cartesian points to spherical coordinates (theta, phi) relative to the reference center.
        """
        r = points - reference_center
        theta = np.arccos(r[:, 2] / np.linalg.norm(r, axis=1))  # Angle from z-axis
        phi = np.arctan2(r[:, 1], r[:, 0])  # Angle from x-axis in xy-plane
        phi = np.mod(phi, 2 * np.pi)  # Ensure phi is in [0, 2*pi)
        return theta, phi

    def compute_intersections(self, t):
        """
        Computes intersections of all bubbles with each other at time `t`.
        Returns a list of dictionaries with intersection data for all bubbles.
        """
        all_intersections = []
        
        for i in range(len(self.bubble_indices)):
            if self.bubble_nucleation_times[i] > t:
                continue  # Skip bubbles that haven't nucleated by time `t`
            
            bubble_idx = self.bubble_indices[i]
            bubble_intersections = self.compute_intersections_bubble_idx(bubble_idx, t)
            if bubble_intersections:  # Check if list is not empty
                all_intersections.extend(bubble_intersections)
        
        self.intersections = all_intersections if all_intersections else []
        return self.intersections

    def find_all_roots_in_range(self, f, u_min=0, u_max=2 * np.pi, num_intervals=8, tol=1e-6):
        """
        Find all roots of the function f(u) within the range [u_min, u_max].

        Parameters:
            f (callable): The function for which roots are to be found.
            u_min (float): Start of the range.
            u_max (float): End of the range.
            num_intervals (int): Number of sub-intervals to divide the range into.
            tol (float): Tolerance for considering two solutions as identical.

        Returns:
            list: List of unique roots within the specified range.
        """
        # Divide the range [u_min, u_max] into sub-intervals
        interval_bounds = np.linspace(u_min, u_max, num_intervals + 1)
        roots = []

        # Search for roots in each sub-interval
        for i in range(num_intervals):
            a, b = interval_bounds[i], interval_bounds[i + 1]

            # Check if f(a) and f(b) have opposite signs (necessary for brentq)
            if f(a) * f(b) <= 0:
                try:
                    # Use root_scalar with brentq method to find a root in the interval [a, b]
                    sol = root_scalar(f, bracket=[a, b], method="brentq", xtol=tol)
                    if sol.converged:
                        root = sol.root
                        # Avoid duplicate roots by checking against already found roots
                        if not any(abs(root - r) < tol for r in roots):
                            roots.append(root)
                except ValueError:
                    # If no root is found in this interval, continue to the next one
                    continue

            # Stop early if we already found the maximum number of solutions (2 for f(u))
            if len(roots) == 2:
                break

        return roots

    def compute_intersections_bubble_idx(self, bubble_idx, t):
        """
        Computes intersections of the bubble with index `bubble_idx` with all other bubbles at time `t`.
        Returns a list of dictionaries with intersection data.
        """
        # Find the reference bubble's data by index
        idx = np.where(self.bubble_indices == bubble_idx)[0]
        if len(idx) == 0:
            return []  # Return empty list if bubble_idx not found
        idx = idx[0]  # Take the first match (assuming unique indices)
        ref_center = self.bubble_centers[idx]
        ref_nucleation_time = self.bubble_nucleation_times[idx]
        
        # Skip computation if the reference bubble hasn't nucleated by time `t`
        if t < ref_nucleation_time:
            return []
        
        ref_radius = self.compute_radius(ref_nucleation_time, t)
        
        intersections = []
        
        for j in range(len(self.bubble_indices)):
            if self.bubble_indices[j] == bubble_idx or self.bubble_nucleation_times[j] > t:
                continue  # Skip self-intersection and bubbles not yet nucleated
            
            other_center = self.bubble_centers[j]
            other_radius = self.compute_radius(self.bubble_nucleation_times[j], t)
            
            if ref_radius == 0 or other_radius == 0:
                continue  # Skip if either bubble has zero radius
            
            result = self.compute_circle_of_intersection(ref_center, ref_radius, other_center, other_radius)
            if result is not None:
                circle_center, circle_radius = result
                
                # Precompute spherical coordinates of the circle's center
                theta_center, phi_center = self.convert_to_spherical(np.array([circle_center]), ref_center)
                cos_theta_center = np.cos(theta_center)
                
                intersections.append({
                    'other_bubble_idx': self.bubble_indices[j],
                    'circle_center': circle_center,
                    'circle_radius': circle_radius,
                    'circle_angles': (phi_center[0], cos_theta_center[0]),  # Store (phi, cos(theta))
                    'nucleation_time': self.bubble_nucleation_times[j],  # Add nucleation_time of the other bubble
                })
        
        return intersections

    def find_circle_line_intersections(self, bubble_idx, t, cos_theta_line, eps=1e-6):
        """
        Computes all intersection points between the horizontal line (cos(theta) = cos_theta_line)
        and the intersection circles for a given bubble at time t.
        Returns a dictionary with intersection data instead of a DataFrame.
        
        Parameters:
            bubble_idx (int): Index of the reference bubble.
            t (float): Current simulation time.
            cos_theta_line (float): Value of cos(theta) for the horizontal line.
            eps (float): Tolerance for considering two solutions as identical.
        
        Returns:
            dict or None: Dictionary containing the original intersection data plus the intersection points and middle point status, or None if no intersections.
        """
        # Step 1: Compute intersections for the specific bubble
        intersections = self.compute_intersections_bubble_idx(bubble_idx, t)
        if not intersections:
            return None  # Return None instead of raising ValueError for consistency
        
        # Extract data for the reference bubble
        idx = np.where(self.bubble_indices == bubble_idx)[0][0]
        ref_center = self.bubble_centers[idx]
        ref_nucleation_time = self.bubble_nucleation_times[idx]
        ref_radius = self.compute_radius(ref_nucleation_time, t)
        z_idx = ref_center[2]  # z-coordinate of the reference bubble's center
        
        # Step 2: Find intersection points for each circle
        updated_intersections = []
        
        for intersection in intersections:
            circle_center = intersection["circle_center"]
            circle_radius = intersection["circle_radius"]
            z_C = circle_center[2]  # z-coordinate of the intersection circle's center
            R_C = circle_radius     # Radius of the intersection circle
            
            # Define vectors v1 and v2 from generate_circle_points
            normal_vector = ref_center - circle_center
            u = np.array([1, 0, 0]) if normal_vector[1] != 0 or normal_vector[2] != 0 else np.array([0, 1, 0])
            v1 = np.cross(normal_vector, u)
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(normal_vector, v1)
            v2 /= np.linalg.norm(v2)
            v1z = v1[2]  # z-component of v1
            v2z = v2[2]  # z-component of v2
            
            # Define the function f(u) to solve
            def f(u):
                return z_C + R_C * (np.cos(u) * v1z + np.sin(u) * v2z) - z_idx - ref_radius * cos_theta_line
            
            # Find all roots of f(u) in the range [0, 2*pi]
            u_values = self.find_all_roots_in_range(f, u_min=0, u_max=2 * np.pi, num_intervals=8, tol=eps)
            
            # Compute intersection points in Cartesian coordinates
            intersection_points_phi = []
            intersection_points_cos_theta = []
            for u in u_values:
                point = circle_center + circle_radius * (np.cos(u) * v1 + np.sin(u) * v2)
                # Convert to spherical coordinates
                theta_point, phi_point = self.convert_to_spherical(np.array([point]), ref_center)
                cos_theta_point = np.cos(theta_point)
                # Store intersection points
                intersection_points_phi.append(phi_point[0])
                intersection_points_cos_theta.append(cos_theta_point[0])
            
            # Sort intersection points by increasing phi
            sorted_indices = np.argsort(intersection_points_phi)
            sorted_phi = [intersection_points_phi[i] for i in sorted_indices]
            sorted_cos_theta = [intersection_points_cos_theta[i] for i in sorted_indices]
            
            # Determine the middle point status (if two intersection points exist)
            middle_point_inside = None
            if len(sorted_phi) == 2:  # Two intersection points
                phi_middle = (sorted_phi[0] + sorted_phi[1]) / 2
                cos_theta_middle = cos_theta_line
                
                # Convert the middle point to Cartesian coordinates on the reference circle
                sin_theta_middle = np.sqrt(1 - cos_theta_middle**2)
                x_middle = ref_radius * sin_theta_middle * np.cos(phi_middle)
                y_middle = ref_radius * sin_theta_middle * np.sin(phi_middle)
                z_middle = ref_radius * cos_theta_middle
                middle_point_cartesian = ref_center + np.array([x_middle, y_middle, z_middle])
                
                # Compute the distance to the other bubble's center
                other_center = np.array([intersection["circle_center"]])
                other_radius = self.compute_radius(intersection["nucleation_time"], t)
                distance_to_other = np.linalg.norm(middle_point_cartesian - other_center)
                middle_point_inside = distance_to_other <= other_radius
            
            # Append intersection points and middle point status to the original data
            updated_intersection = {
                "other_bubble_idx": intersection["other_bubble_idx"],
                "circle_center": intersection["circle_center"],
                "circle_radius": intersection["circle_radius"],
                "circle_angles": intersection["circle_angles"],
                "intersection_points_phi": sorted_phi,
                "intersection_points_cos_theta": sorted_cos_theta,
                "middle_point_inside": middle_point_inside,
            }
            updated_intersections.append(updated_intersection)
        
        # Step 3: Create a dictionary with the updated intersection data
        updated_ref_data = {
            "bubble_idx": self.bubble_indices[idx],
            "center": ref_center.tolist(),  # Convert to list for JSON serialization or external use
            "nucleation_time": ref_nucleation_time,
            "intersections": updated_intersections,
        }
        
        return updated_ref_data

    def plot_intersections(
        self,
        bubble_idx,
        t,
        plot_circle_centers=False,
        num_points=10000,
        cos_theta_line=None,
        eps=1e-6,
    ):
        """
        Plots all intersections of the bubble with index `bubble_idx` at time `t`.
        If `cos_theta_line` is provided, finds the intersection points between the horizontal line and each intersection circle.
        Computes complementary segments of phi using SegmentComplement and plots them on the horizontal line.

        Parameters:
            bubble_idx (int): Index of the reference bubble.
            t (float): Current simulation time.
            plot_circle_centers (bool): Whether to plot the centers of the intersection circles.
            num_points (int): Number of points to generate on each circle.
            cos_theta_line (float): Value of cos(theta) for the horizontal line.
            eps (float): Tolerance for root uniqueness.
        """
        # Step 1: Call find_circle_line_intersections to get the updated dictionary
        ref_data = self.find_circle_line_intersections(bubble_idx, t, cos_theta_line, eps=eps)
        if ref_data is None:
            raise ValueError(f"No data available for bubble_idx={bubble_idx}.")

        # Extract data for the reference bubble
        ref_center = np.array(ref_data["center"])
        fig, ax = plt.subplots(figsize=(8, 6))

        # Ensure intersections is a list of dictionaries
        intersections = ref_data["intersections"] if ref_data["intersections"] else []

        # Generate a colormap for unique colors
        cmap = plt.get_cmap("tab10")  # Use a qualitative colormap

        # Plot intersections and scatter-plot intersection points
        for i, intersection in enumerate(intersections):
            color = cmap(i % cmap.N)  # Assign a unique color for this intersection

            # Plot the intersection circle
            circle_center = intersection["circle_center"]
            circle_radius = intersection["circle_radius"]
            normal_vector = ref_center - circle_center
            circle_points = self.generate_circle_points(circle_center, circle_radius, normal_vector, num_points=num_points)
            theta, phi = self.convert_to_spherical(circle_points, ref_center)
            cos_theta = np.cos(theta)
            ax.scatter(phi, cos_theta, marker=".", s=1, color=color)

            # Optionally plot the circle center
            if plot_circle_centers:
                phi_center, cos_theta_center = intersection["circle_angles"]
                ax.scatter(phi_center, cos_theta_center, marker="x", s=20, color=color)

            # Scatter-plot intersection points if they exist
            if cos_theta_line is not None:
                intersection_points_phi = intersection.get("intersection_points_phi", [])
                middle_point_inside = intersection.get("middle_point_inside", None)

                if len(intersection_points_phi) == 2:
                    ax.scatter(
                        intersection_points_phi,
                        [cos_theta_line, cos_theta_line],
                        marker="o",
                        s=20,
                        color=color,
                    )

        # Step 2: Create arrays of "intersection_points_phi" and "middle_point_inside"
        if cos_theta_line is not None:
            # Collect only cases with exactly 2 intersection points
            intersection_points_phi_list = []
            middle_point_inside_list = []

            for intersection in intersections:
                intersection_points_phi = intersection.get("intersection_points_phi", [])
                middle_point_inside = intersection.get("middle_point_inside", None)

                if len(intersection_points_phi) == 2:
                    intersection_points_phi_list.append(intersection_points_phi)
                    middle_point_inside_list.append(middle_point_inside)

            # Flatten the intersection_points_phi_list into a single list of tuples
            sub_segments = [(phi1, phi2) for phi_pair in intersection_points_phi_list for phi1, phi2 in [phi_pair]]

            # Create an instance of SegmentComplement
            segment_complement = SegmentComplement(bound_segments=(0, 2 * np.pi))

            # Add sub-segments with complement flags
            segment_complement.add_sub_segments(
                sub_segments=sub_segments,
                complement_flags=middle_point_inside_list,
                allow_out_of_bounds=False,
            )

            # Get complementary segments
            complementary_segments = segment_complement.get_complementary_segments()

            # Plot complementary segments on the horizontal line
            for segment in complementary_segments:
                phi_start, phi_end = segment
                ax.plot(
                    [phi_start, phi_end],
                    [cos_theta_line, cos_theta_line],
                    color="black",
                    linestyle="-",
                    lw=2,
                    zorder=10
                )

            # Draw the horizontal dashed line
            ax.axhline(cos_theta_line, color="gray", linestyle="--", lw=1, label="Horizontal Line")

        # Set axis limits and labels
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\cos(\theta)$")
        ax.set_title(f"Intersections for Bubble {bubble_idx} at t={t}")

        # Customize ticks
        phi_ticks = np.arange(0, 2 * np.pi + 0.01, np.pi / 4)
        phi_labels = ["$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$", r"$2\pi$"]
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels(phi_labels)

        cos_theta_ticks = np.arange(-1, 1.1, 0.25)
        ax.set_yticks(cos_theta_ticks)

        # Add grid and legend
        ax.grid(True)
        if cos_theta_line is not None:
            ax.legend()

        return fig, ax

    def compute_B_matrix(self, bubble_idx, t, cos_theta, eps=1e-6):
        """
        Computes the B matrix by integrating over complementary segments of phi for a given bubble.

        Parameters:
            bubble_idx (int): Index of the reference bubble.
            t (float): Current simulation time.
            cos_theta (float): Value of cos(theta) for the horizontal line.
            eps (float): Tolerance for root uniqueness.

        Returns:
            np.ndarray: B matrix with shape (2,) containing B_+ and B_- values.
        """
        B_matrix = np.array([0., 0.])

        # Step 1: Call find_circle_line_intersections to get the updated dictionary
        ref_data = self.find_circle_line_intersections(bubble_idx, t, cos_theta, eps=eps)
        if ref_data is None:
            return B_matrix

        # Ensure intersections is a list of dictionaries
        intersections = ref_data["intersections"] if ref_data["intersections"] else []

        # Step 2: Collect intersection points and flags for segments with exactly 2 points
        intersection_points_phi_list = []
        middle_point_inside_list = []

        for intersection in intersections:
            intersection_points_phi = intersection.get("intersection_points_phi", [])
            middle_point_inside = intersection.get("middle_point_inside", None)

            if len(intersection_points_phi) == 2:
                intersection_points_phi_list.append(intersection_points_phi)
                middle_point_inside_list.append(middle_point_inside)

        # Flatten the intersection points into sub-segments
        sub_segments = [(phi1, phi2) for phi_pair in intersection_points_phi_list for phi1, phi2 in [phi_pair]]

        # Create an instance of SegmentComplement
        segment_complement = SegmentComplement(bound_segments=(0, 2 * np.pi))

        # Add sub-segments with complement flags
        segment_complement.add_sub_segments(
            sub_segments=sub_segments,
            complement_flags=middle_point_inside_list,
            allow_out_of_bounds=False,
        )

        # Get complementary segments
        complementary_segments = segment_complement.get_complementary_segments()
        
        # Vectorize the integration over segments
        for segment in complementary_segments:
            B_matrix[0] += np.sin(2.0 * segment[1]) - np.sin(2.0 * segment[0])
            B_matrix[1] += -np.cos(2.0 * segment[1]) + np.cos(2.0 * segment[0])
        
        # Apply the scaling factor
        B_matrix *= 0.25 * (1.0 - cos_theta**2)
        return B_matrix

    def compute_A_matrix(self, bubble_idx, t, w_arr, n_cos_theta=1000, eps=1e-6):
        """
        Compute the A matrix by integrating B matrix over cos_theta, vectorized over w_arr.

        Parameters:
            bubble_idx (int): Index of the reference bubble.
            t (float): Current simulation time.
            w_arr (float or array-like): Array of w values to compute A matrix for.
            n_cos_theta (int): Number of points in cos_theta grid for integration.
            eps (float): Tolerance for root uniqueness in compute_B_matrix.

        Returns:
            np.ndarray: A matrix with shape (len(w_arr), 2) containing A values for each w.
        """
        # Convert w_arr to a NumPy array
        w_arr = np.asarray(w_arr)
        if w_arr.ndim == 0:
            w_arr = w_arr[np.newaxis]  # Ensure w_arr is at least 1D

        # Get nucleation time for the bubble
        idx = np.where(self.bubble_indices == bubble_idx)[0]
        if len(idx) == 0:
            raise ValueError(f"Bubble index {bubble_idx} not found.")
        idx = idx[0]  # Take the first match (assuming unique indices)
        tn = self.bubble_nucleation_times[idx]

        # Create cos_theta array for integration
        cos_theta_arr = np.linspace(-1, 1, n_cos_theta)

        # Compute B matrix for all cos_theta values (shape: (n_cos_theta, 2))
        B_matrix_arr = np.array([self.compute_B_matrix(bubble_idx, t, cos_theta, eps=eps) 
                                for cos_theta in cos_theta_arr])

        # Vectorize the exponential term over w_arr
        # Shape of exp_term: (n_cos_theta, len(w_arr))
        exp_term = np.exp(-1j * self.vw * w_arr * (t - tn) * cos_theta_arr[:, np.newaxis])
        
        # Compute integrand: multiply exp_term (n_cos_theta, len(w_arr)) by B_matrix_arr (n_cos_theta, 2)
        # Broadcasting gives integrand_arr shape: (n_cos_theta, len(w_arr), 2)
        integrand_arr = exp_term[:, :, np.newaxis] * B_matrix_arr[:, np.newaxis, :]

        # Perform Simpson's rule integration over cos_theta axis (axis=0)
        # Result shape: (len(w_arr), 2)
        A_matrix = integrate.simpson(integrand_arr, x=cos_theta_arr, axis=0)

        return A_matrix

    def compute_C_matrix_integrand(self, t, w_arr, n_cos_theta=1000, eps=1e-6):
        """
        Compute the integrand for the C matrix at a specific time t for all bubbles that have nucleated.

        Parameters:
            t (float): Current simulation time.
            w_arr (np.ndarray): Array of angular frequencies.
            n_cos_theta (int): Number of points in cos_theta grid for A matrix computation.
            eps (float): Tolerance for root uniqueness.

        Returns:
            np.ndarray: Integrand array with shape (len(w_arr), 2) for this time step.
        """
        integrand = np.zeros((len(w_arr), 2), dtype=complex)
        
        # Iterate over all bubbles
        for idx, bubble_idx in enumerate(self.bubble_indices):
            tn = self.bubble_nucleation_times[idx]
            if t < tn:
                continue  # Skip bubbles that haven't nucleated yet
            
            # Compute A matrix for this bubble at time t
            A_matrix = self.compute_A_matrix(bubble_idx, t, w_arr, n_cos_theta=n_cos_theta, eps=eps)  # Shape: (len(w_arr), 2)
            
            # Compute time-dependent factors
            time_diff = t - tn
            time_factor = time_diff**3
            exp_factor = np.exp(1j * w_arr * time_diff)  # Shape: (len(w_arr),)
            
            # Add contribution to integrand
            integrand += time_factor * exp_factor[:, np.newaxis] * A_matrix
        
        return integrand

    def compute_C_matrix(self, w_arr, n_cores=None, n_cos_theta=1000, eps=1e-6, chunk_size=1000):
        """
        Compute the C matrix (C+-) for given angular frequencies w_arr, summing over all bubbles,
        with parallelization over time steps using joblib and processing in chunks to optimize memory.

        Parameters:
            w_arr (float or array-like): Array of angular frequencies to compute C for.
            n_cores (int, optional): Number of CPU cores to use. Defaults to CPU count.
            n_cos_theta (int): Number of points in cos_theta grid for A matrix computation.
            eps (float): Tolerance for root uniqueness in compute_B_matrix.
            chunk_size (int): Number of time steps to process in each chunk.

        Returns:
            np.ndarray: C matrix with shape (len(w_arr), 2) containing C_+ and C_- values for each w.
        """
        # Ensure w_arr is a NumPy array
        w_arr = np.asarray(w_arr)
        if w_arr.ndim == 0:
            w_arr = w_arr[np.newaxis]
        
        # Initialize C matrix
        C_matrix = np.zeros((len(w_arr), 2), dtype=complex)
        
        # Use the simulation time array and parameters
        t_arr = self.t_arr
        
        # Determine number of cores
        if n_cores is None:
            n_cores = joblib.cpu_count()
        
        # Calculate number of chunks
        n_steps = len(t_arr)
        chunk_size = min(chunk_size, n_steps)  # Ensure chunk_size doesn’t exceed t_arr length
        n_chunks = (n_steps + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_steps)
            t_chunk = t_arr[start_idx:end_idx]  # Subset of time steps
            
            # Parallel computation of integrand for this chunk
            integrand_list = Parallel(n_jobs=n_cores)(
                delayed(self.compute_C_matrix_integrand)(
                    t,
                    w_arr,
                    n_cos_theta=n_cos_theta,
                    eps=eps
                )
                for t in t_chunk
            )
            
            # Convert list to array for this chunk
            integrand_chunk = np.array(integrand_list)  # Shape: (len(t_chunk), len(w_arr), 2)
            
            # Integrate over this chunk using Simpson’s rule
            chunk_integral = integrate.simpson(integrand_chunk, x=t_chunk, axis=0)  # Shape: (len(w_arr), 2)
            
            # Accumulate the result
            C_matrix += chunk_integral
        
        # Normalize by 1/(6π)
        C_matrix *= 1 / (6 * np.pi)
        
        return C_matrix
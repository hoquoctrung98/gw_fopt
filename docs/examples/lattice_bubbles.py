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
#     display_name: gw-fopt
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <hoquoctrung98@gmail.com>
"""

import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gw_fopt.bubble_gw import many_bubbles

# %%
# %matplotlib ipympl
sns.set_theme(style="ticks", font="Dejavu Sans")
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

# %%
L = 10.0
# An empty lattice is merely a container with no bubbles validation
lattice_empty = many_bubbles.EmptyLattice()
# A cartesian lattice
lattice_cartesian = many_bubbles.CartesianLattice(
    origin=[0.0, 0.0, 0.0], basis=[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
)
# A spherical lattice
lattice_spherical = many_bubbles.SphericalLattice(center=[0.0, 0.0, 0.0], radius=10)

# %%
print(repr(lattice_cartesian))
print(str(lattice_spherical))

# %%
# The input bubbles_interior and bubbles_exterior is a numpy array of shape (n_bubbles, 4)
bubbles_interior = np.array([[0.0, 0.0, 0.0, L], [1.0, L, 0.0, 0.0]])
bubbles_exterior = np.array([[0.0, 0.0, 0.0, 2 * L], [1.0, 3 * L, 0.0, 0.0]])

# Below we create a combination of bubbles on a lattice
lattice_bubbles_empty = many_bubbles.LatticeBubbles(
    bubbles_interior=bubbles_interior,
    bubbles_exterior=bubbles_exterior,
    lattice=lattice_empty,
)

# Note that is is optional to pass bubbles_exterior as an input argument
# we can always generate the bubbles_exterior with a built-in boundary conditions later
lattice_bubbles_cartesian = many_bubbles.LatticeBubbles(
    bubbles_interior=bubbles_interior,
    bubbles_exterior=None,
    lattice=lattice_cartesian,
)
lattice_bubbles_spherical = many_bubbles.LatticeBubbles(
    bubbles_interior=bubbles_interior,
    bubbles_exterior=None,
    lattice=lattice_spherical,
)

# Here are examples of setting bubbles_exterior with a boundary condition
lattice_bubbles_cartesian.with_boundary_condition(boundary_condition="periodic")
lattice_bubbles_spherical.with_boundary_condition(boundary_condition="reflection")

# %%
# Here we perform bubble validation on the lattice, if not satisfied an error is thrown
# + bubbles_interior must be inside the lattice
# + bubbles_exterior must be outside the lattice
# + causality is imposed by all lattice to ensure no bubble is formed inside other bubbles.
# Note that the causality checks only applied for (Interior-Interior) and (Interior-Exterior), and not (Exterior-Exterior), as some boundary conditions might violate this

try:
    bubbles_interior = np.array([[0.0, 0.0, 0.0, L], [1.0, L * 1.0001, 0.0, 0.0]])
    lattice_bubbles = many_bubbles.LatticeBubbles(
        bubbles_interior=bubbles_interior,
        bubbles_exterior=None,
        lattice=lattice_cartesian,
    )
except Exception:
    print(traceback.format_exc())

try:
    bubbles_interior = np.array([[0.0, 0.0, 0.0, L], [1.0, L, 0.0, 0.0]])
    bubbles_exterior = np.array([[0.0, 0.0, 0.0, 0.5 * L], [1.0, 0.5 * L, 0.0, 0.0]])
    lattice_bubbles = many_bubbles.LatticeBubbles(
        bubbles_interior=bubbles_interior,
        bubbles_exterior=bubbles_exterior,
        lattice=lattice_spherical,
    )
except Exception:
    print(traceback.format_exc())

try:
    bubbles_interior = np.array([[0.0, 0.0, 0.0, L], [2 * L, 0.0, 0.0, 0.0]])
    lattice_bubbles = many_bubbles.LatticeBubbles(
        bubbles_interior=bubbles_interior,
        bubbles_exterior=None,
        lattice=lattice_empty,
    )
except Exception:
    print(traceback.format_exc())

# %% [markdown]
# ## Transforming the LatticeBubbles system with an Isometry3

# %% [markdown]
# Isometry3 is composed of a spatial translation + a spatial rotation (which is specified by either euler_angles or rotation_matrix, but not both).
#
# Under Isometry3, both the lattice and the (bubbles_interior + bubbles_exterior) are transformed, in such a way that the relative positions of the system remains the same.

# %%
bubbles_interior = np.array(
    [[0.0, 10.0, 0.0, 1.0], [1.0, 0.0, 10.0, 0.0], [2.0, 0.0, 0.0, 10.0]]
)
lattice_bubbles_cartesian = many_bubbles.LatticeBubbles(
    bubbles_interior=bubbles_interior,
    bubbles_exterior=None,
    lattice=lattice_cartesian,
)
lattice_bubbles_cartesian.with_boundary_condition(boundary_condition="periodic")
print(lattice_bubbles_cartesian.bubbles_interior)
print(lattice_bubbles_cartesian.bubbles_exterior)

# %%
# Create an Isometry3 transformation and apply it to LatticeBubbles, which transform both the lattice and the bubbles
iso = many_bubbles.Isometry3(
    translation=[1.0, 2.0, 3.0], euler_angles=None, rotation_matrix=None
)
# One can either call method `transform` to create a copy of the LatticeBubbles after transformation,
# or call method `transform_mut` to perform the in-place transformation on the object
transformed_lattice_bubbles_cartesian = lattice_bubbles_cartesian.transform(iso)
print(transformed_lattice_bubbles_cartesian.bubbles_interior)
print(transformed_lattice_bubbles_cartesian.bubbles_exterior)

# %%
euler_angles_six_faces = [
    [0.0, 0.0, 0.0],  # +z direction
    [np.pi, 0.0, 0.0],  # -z direction
    [0.0, np.pi / 2, 0.0],  # +x direction
    [0.0, -np.pi / 2, 0.0],  # -x direction
    [-np.pi / 2, 0.0, 0.0],  # +y direction
    [np.pi / 2, 0.0, 0.0],  # -y direction
]
for euler_angles in euler_angles_six_faces:
    iso = many_bubbles.Isometry3(euler_angles=euler_angles, rotation_matrix=None)
    transformed_lattice_bubbles_cartesian = lattice_bubbles_cartesian.transform(iso)
    print(transformed_lattice_bubbles_cartesian.bubbles_interior)

# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class LatticeBubblesVisualizer:
    """
    A visualizer for LatticeBubbles objects, providing methods to plot
    bubble centers and actual bubble spheres with time-dependent radii.
    """

    def __init__(self, lattice_bubbles):
        if not hasattr(lattice_bubbles, "bubbles_interior") or not hasattr(
            lattice_bubbles, "bubbles_exterior"
        ):
            raise ValueError(
                "lattice_bubbles must have .bubbles_interior and .bubbles_exterior attributes"
            )

        self.interior = np.asarray(lattice_bubbles.bubbles_interior)
        self.exterior = np.asarray(lattice_bubbles.bubbles_exterior)

        if self.interior.shape[1] != 4 or (
            self.exterior.size > 0 and self.exterior.shape[1] != 4
        ):
            raise ValueError(
                "Both bubble arrays must have shape (n, 4) with columns [t_bubble, x, y, z]"
            )

        self.lattice_bubbles = lattice_bubbles
        self.lattice = lattice_bubbles.lattice
        self.lattice_type = self.lattice.name()

        if self.lattice_type in ["ParallelepipedLattice", "CartesianLattice"]:
            self.origin = np.array(self.lattice.origin)
            self.basis = np.array(self.lattice.basis)
            a, b, c = self.basis
            self.vertices = np.array(
                [
                    self.origin,
                    self.origin + a,
                    self.origin + b,
                    self.origin + c,
                    self.origin + a + b,
                    self.origin + a + c,
                    self.origin + b + c,
                    self.origin + a + b + c,
                ]
            )
            self.lattice_min = np.min(self.vertices, axis=0)
            self.lattice_max = np.max(self.vertices, axis=0)
        elif self.lattice_type == "SphericalLattice":
            self.center = np.array(self.lattice.center)
            self.radius = self.lattice.radius
            self.lattice_min = self.center - self.radius
            self.lattice_max = self.center + self.radius
        else:
            self.lattice_type = None  # Not supported for plotting

    def plot_lattice(
        self,
        fig,
        ax,
        fill_lattice=True,
        skeleton_alpha=0.8,
        fill_alpha=0.3,
        skeleton_color="k",
        fill_color="lightgray",
        show_grid=True,
        grid_color="gray",
        grid_alpha=0.2,
        grid_lw=0.5,
        equal_aspect=True,
    ):
        """
        Plot the lattice shape with filled surfaces, skeleton edges, and optional grid.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes (3D)
        fill_lattice : bool, default True
            Whether to fill the lattice surfaces.
        skeleton_alpha : float, default 0.8
            Alpha for the thick skeleton edges.
        fill_alpha : float, default 0.3
            Alpha for filled surfaces.
        skeleton_color / fill_color : str
            Colors for skeleton and fill.
        show_grid : bool, default True
            Whether to draw a faint grid (wireframe) on the faces.
        grid_color : str, default "gray"
            Color of the grid lines.
        grid_alpha : float, default 0.2
            Transparency of the grid.
        grid_lw : float, default 0.5
            Line width of the grid.
        equal_aspect : bool, default True
            Set equal aspect ratio to fully show the lattice.
        """
        if self.lattice_type is None:
            print("Lattice type not supported for plotting.")
            return

        if self.lattice_type in ["ParallelepipedLattice", "CartesianLattice"]:
            vertices = self.vertices
            o = self.origin
            a, b, c = self.basis

            # Thick skeleton edges
            if skeleton_alpha > 0:
                edges = [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [1, 5],
                    [2, 4],
                    [2, 6],
                    [3, 5],
                    [3, 6],
                    [4, 7],
                    [5, 7],
                    [6, 7],
                ]
                for i, j in edges:
                    pts = vertices[[i, j]]
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color=skeleton_color,
                        alpha=skeleton_alpha,
                        lw=1.5,
                    )

            # Filled faces
            if fill_lattice:
                face_indices = [
                    [0, 1, 4, 2],  # bottom
                    [3, 5, 7, 6],  # top
                    [0, 1, 5, 3],  # yz front
                    [2, 4, 7, 6],  # yz back
                    [0, 2, 6, 3],  # xz left
                    [1, 4, 7, 5],  # xz right
                ]
                faces = [vertices[idx] for idx in face_indices]
                collection = Poly3DCollection(
                    faces,
                    facecolors=fill_color,
                    linewidths=0,
                    alpha=fill_alpha,
                    zsort="average",
                )
                ax.add_collection3d(collection)

            # Grid on each face (faint wireframe)
            if show_grid:
                # Helper to plot grid on a parallelogram face defined by two vectors
                def plot_face_grid(origin, vec1, vec2, color, alpha, lw):
                    n_lines = 8  # number of grid lines in each direction
                    for i in range(1, n_lines):
                        t = i / n_lines
                        # line parallel to vec1
                        ax.plot(
                            [
                                origin[0] + t * vec2[0],
                                origin[0] + vec1[0] + t * vec2[0],
                            ],
                            [
                                origin[1] + t * vec2[1],
                                origin[1] + vec1[1] + t * vec2[1],
                            ],
                            [
                                origin[2] + t * vec2[2],
                                origin[2] + vec1[2] + t * vec2[2],
                            ],
                            color=color,
                            alpha=alpha,
                            lw=lw,
                        )
                        # line parallel to vec2
                        ax.plot(
                            [
                                origin[0] + t * vec1[0],
                                origin[0] + vec2[0] + t * vec1[0],
                            ],
                            [
                                origin[1] + t * vec1[1],
                                origin[1] + vec2[1] + t * vec1[1],
                            ],
                            [
                                origin[2] + t * vec1[2],
                                origin[2] + vec2[2] + t * vec1[2],
                            ],
                            color=color,
                            alpha=alpha,
                            lw=lw,
                        )

                # Draw grid on all six faces
                plot_face_grid(o, a, b, grid_color, grid_alpha, grid_lw)  # bottom
                plot_face_grid(o + c, a, b, grid_color, grid_alpha, grid_lw)  # top
                plot_face_grid(o, a, c, grid_color, grid_alpha, grid_lw)  # front
                plot_face_grid(o + b, a, c, grid_color, grid_alpha, grid_lw)  # back
                plot_face_grid(o, b, c, grid_color, grid_alpha, grid_lw)  # left
                plot_face_grid(o + a, b, c, grid_color, grid_alpha, grid_lw)  # right

        elif self.lattice_type == "SphericalLattice":
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = self.radius * np.outer(np.cos(u), np.sin(v)) + self.center[0]
            y = self.radius * np.outer(np.sin(u), np.sin(v)) + self.center[1]
            z = self.radius * np.outer(np.ones_like(u), np.cos(v)) + self.center[2]

            if skeleton_alpha > 0:
                ax.plot_wireframe(
                    x,
                    y,
                    z,
                    color=skeleton_color,
                    alpha=skeleton_alpha,
                    rstride=2,
                    cstride=2,
                    linewidth=1.5,
                )

            if show_grid:
                ax.plot_wireframe(
                    x,
                    y,
                    z,
                    color=grid_color,
                    alpha=grid_alpha,
                    rstride=1,
                    cstride=1,
                    linewidth=grid_lw,
                )

            if fill_lattice:
                ax.plot_surface(
                    x,
                    y,
                    z,
                    color=fill_color,
                    alpha=fill_alpha,
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                    zsort="average",
                )

        if equal_aspect:
            min_vals = self.lattice_min
            max_vals = self.lattice_max
            max_range = (max_vals - min_vals).max() / 2.0
            mids = (max_vals + min_vals) / 2.0
            ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
            ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
            ax.set_zlim(mids[2] - max_range, mids[2] + max_range)

    def plot_bubbles_centers(
        self,
        fig,
        ax,
        tmax=None,
        show_bubbles_exterior=False,
        marker_size=20,
        alpha=1.0,
        cmap="viridis",
        color_by_time=True,
        equal_aspect=True,
        title=None,
        show_labels=False,
        label_fontsize=8,
        label_color="black",
        **kwargs_scatter,
    ):
        """
        Plot only the centers of bubbles (as points).

        Returns
        -------
        list of PathCollection
            Scatter artists for each group (interior/exterior).
        """
        to_plot = [("I", self.interior)]
        if show_bubbles_exterior and self.exterior.shape[0] > 0:
            to_plot.append(("E", self.exterior))

        all_x, all_y, all_z = [], [], []
        scatters = []

        scatter_kwargs = kwargs_scatter.copy()
        scatter_kwargs.setdefault("s", marker_size)
        scatter_kwargs.setdefault("alpha", alpha)
        scatter_kwargs.setdefault("depthshade", True)

        for prefix, bubbles in to_plot:
            if bubbles.shape[0] == 0:
                continue

            mask = (
                bubbles[:, 0] < tmax
                if tmax is not None
                else np.ones(bubbles.shape[0], dtype=bool)
            )
            filtered = bubbles[mask]
            if filtered.shape[0] == 0:
                continue

            x, y, z = filtered[:, 1], filtered[:, 2], filtered[:, 3]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

            if color_by_time:
                sc = ax.scatter(x, y, z, c=filtered[:, 0], cmap=cmap, **scatter_kwargs)
            else:
                color = "tab:orange" if prefix == "I" else "tab:blue"
                sc_kwargs = scatter_kwargs.copy()
                sc_kwargs.setdefault("c", color)
                sc = ax.scatter(x, y, z, label=f"{prefix} bubbles", **sc_kwargs)

            scatters.append(sc)

            if show_labels:
                start_idx = 0 if prefix == "I" else len(self.interior)
                for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
                    global_idx = start_idx + np.nonzero(mask)[0][i]
                    ax.text(
                        xi,
                        yi,
                        zi,
                        f"{prefix}({global_idx})",
                        fontsize=label_fontsize,
                        color=label_color,
                        ha="center",
                        va="center",
                    )

        if not scatters:
            print("No bubble centers to plot with current filters.")
            ax.set_title(title or "Bubble centers (none found)")
            return []

        if color_by_time and "c" not in kwargs_scatter:
            fig.colorbar(
                scatters[-1],
                ax=ax,
                shrink=0.6,
                aspect=20,
                label="Nucleation time $t_b$",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if title is None:
            # automatic title logic unchanged for brevity
            ax.set_title("Bubble centers")
        else:
            ax.set_title(title)

        if equal_aspect and all_x:
            coords = np.array([all_x, all_y, all_z])
            max_range = (coords.max(1) - coords.min(1)).max() / 2.0
            mids = coords.mean(1)
            ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
            ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
            ax.set_zlim(mids[2] - max_range, mids[2] + max_range)

        if len(scatters) > 1:
            ax.legend()

        return scatters

    def plot_bubbles(
        self,
        fig,
        ax,
        t,
        show_bubbles_exterior=False,
        alpha=0.3,
        interior_color="tab:orange",
        exterior_color="tab:blue",
        equal_aspect=True,
        title=None,
        show_labels=False,
        label_fontsize=8,
        label_color="black",
    ):
        """
        Draw actual 3D spheres for bubbles that have nucleated by time t,
        with radius = t - t_bubble (only for t > t_bubble).

        Returns
        -------
        list of artists for the drawn spheres.
        """
        to_plot = [("I", self.interior, interior_color)]
        if show_bubbles_exterior and self.exterior.shape[0] > 0:
            to_plot.append(("E", self.exterior, exterior_color))

        artists = []
        centers_list = []

        for prefix, bubbles, color in to_plot:
            if bubbles.shape[0] == 0:
                continue

            mask = bubbles[:, 0] <= t
            active = bubbles[mask]
            if active.shape[0] == 0:
                continue

            tb = active[:, 0]
            centers = active[:, 1:4]
            radii = t - tb

            for center, r in zip(centers, radii):
                if r <= 0:
                    continue

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
                y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
                z = r * np.outer(np.ones_like(u), np.cos(v)) + center[2]

                surf = ax.plot_surface(
                    x,
                    y,
                    z,
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                )
                artists.append(surf)
                centers_list.append(center)

            if show_labels:
                start_idx = 0 if prefix == "I" else len(self.interior)
                orig_indices = np.nonzero(mask)[0]
                for orig_idx, center in zip(orig_indices, centers):
                    global_idx = start_idx + orig_idx
                    ax.text(
                        center[0],
                        center[1],
                        center[2],
                        f"{prefix}({global_idx})",
                        fontsize=label_fontsize,
                        color=label_color,
                        ha="center",
                        va="center",
                    )

        if not artists:
            print(f"No bubbles visible at t = {t}")
            ax.set_title(title or f"Bubbles at t = {t:.2f} (none visible)")
            return []

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title or f"Bubbles at t = {t:.2f}")

        if equal_aspect and centers_list:
            centers_arr = np.array(centers_list)
            # find largest current radius
            min_tb = np.inf
            if (self.interior[:, 0] <= t).any():
                min_tb = min(min_tb, self.interior[self.interior[:, 0] <= t, 0].min())
            if show_bubbles_exterior and (self.exterior[:, 0] <= t).any():
                min_tb = min(min_tb, self.exterior[self.exterior[:, 0] <= t, 0].min())
            max_r = t - min_tb if np.isfinite(min_tb) else 0

            overall_min = centers_arr.min(axis=0) - max_r
            overall_max = centers_arr.max(axis=0) + max_r
            max_range = (overall_max - overall_min).max() / 2.0
            mid = (overall_max + overall_min) / 2.0
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        if len(to_plot) > 1:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Interior bubbles",
                    markerfacecolor=interior_color,
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Exterior bubbles",
                    markerfacecolor=exterior_color,
                    markersize=10,
                ),
            ]
            ax.legend(handles=legend_elements)

        return artists


# %%
L = 2.0
nucleation_strategy = many_bubbles.SpontaneousNucleation(
    n_bubbles=4,
    seed=0,
)
lattice_bubbles_cartesian = nucleation_strategy.nucleate(
    lattice=many_bubbles.CartesianLattice(
        origin=[-L / 2, -L / 2, -L / 2],
        basis=[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
    ),
    boundary_condition="periodic",
)

# %%
# Assuming you have a lattice_bubbles object
visualizer_cartesian = LatticeBubblesVisualizer(lattice_bubbles_cartesian)

# Plot centers
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

visualizer_cartesian.plot_bubbles_centers(
    fig,
    ax,
    alpha=0.5,
    cmap="jet",
    s=500,
    show_bubbles_exterior=False,
    color_by_time=True,
    show_labels=True,
    label_fontsize=7,
    marker="o",
    edgecolor="black",
    linewidths=0.5,
)
visualizer_cartesian.plot_lattice(
    fig, ax, skeleton_alpha=0.1, fill_alpha=0.1, show_grid=True
)
ax.set_xlim(-1.5 * L, 1.5 * L)
ax.set_ylim(-1.5 * L, 1.5 * L)
ax.set_zlim(-1.5 * L, 1.5 * L)
fig.savefig(
    f"./figures/many_bubbles/many_bubbles_centers_cartesian.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
visualizer_cartesian.plot_bubbles_centers(
    fig,
    ax,
    alpha=1,
    cmap="jet",
    s=5,
    show_bubbles_exterior=True,
    color_by_time=False,
    show_labels=False,
    label_fontsize=7,
    marker="o",
    edgecolor="black",
    linewidths=0.5,
)
visualizer_cartesian.plot_lattice(
    fig, ax, skeleton_alpha=0.2, fill_alpha=0.1, show_grid=True
)
visualizer_cartesian.plot_bubbles(fig, ax, t=0.4, show_bubbles_exterior=True, alpha=0.4)
ax.set_xlim(-1.5 * L, 1.5 * L)
ax.set_ylim(-1.5 * L, 1.5 * L)
ax.set_zlim(-1.5 * L, 1.5 * L)
fig.savefig(
    f"./figures/many_bubbles/many_bubbles_at_time_cartesian.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

# %%
nucleation_strategy = many_bubbles.FixedNucleationRate(
    beta=1,
    gamma0=1,
    t0=0.0,
    d_p0=0.01,
    seed=0,
    # Method to compute lattice volume that is outside of all bubbles. "approximation" means that volume_remaining = volume_lattice - sum_n(volume_bubble_n)
    method="approximation",
)
lattice_bubbles_spherical = nucleation_strategy.nucleate(
    lattice=many_bubbles.SphericalLattice(center=[0.0, 0.0, 0.0], radius=L),
    boundary_condition="reflection",
)

# %%
lattice_bubbles_spherical.bubbles_interior.shape

# %%
# Assuming you have a lattice_bubbles object
visualizer_spherical = LatticeBubblesVisualizer(lattice_bubbles_spherical)

# Plot centers
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

visualizer_spherical.plot_bubbles_centers(
    fig,
    ax,
    alpha=0.5,
    cmap="jet",
    s=500,
    show_bubbles_exterior=False,
    color_by_time=True,
    show_labels=True,
    label_fontsize=7,
    marker="o",
    edgecolor="black",
    linewidths=0.5,
)
visualizer_spherical.plot_lattice(
    fig, ax, skeleton_alpha=0.1, fill_alpha=0.1, show_grid=True
)
fig.savefig(
    f"./figures/many_bubbles/many_bubbles_centers_spherical.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
visualizer_spherical.plot_bubbles_centers(
    fig,
    ax,
    alpha=1,
    cmap="jet",
    s=5,
    show_bubbles_exterior=True,
    color_by_time=False,
    show_labels=False,
    label_fontsize=7,
    marker="o",
    edgecolor="black",
    linewidths=0.5,
)
visualizer_spherical.plot_bubbles(fig, ax, t=0.7, show_bubbles_exterior=True, alpha=0.4)
visualizer_spherical.plot_lattice(
    fig, ax, skeleton_alpha=0.1, fill_alpha=0.1, show_grid=True
)
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-2 * L, 2 * L)
ax.set_ylim(-2 * L, 2 * L)
ax.set_zlim(-2 * L, 2 * L)
fig.savefig(
    f"./figures/many_bubbles/many_bubbles_at_time_spherical.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

# %%
L = 1.0
nucleation_strategy_apprx = many_bubbles.FixedNucleationRate(
    beta=1,
    gamma0=1.0e-3,
    t0=0.0,
    d_p0=0.01,
    seed=0,
    method="approximation",
    volume_remaining_fraction_cutoff=1e-3,
    max_time_steps=1_000_000,
)
lattice_bubbles_apprx = nucleation_strategy_apprx.nucleate(
    lattice=many_bubbles.SphericalLattice(center=[0.0, 0.0, 0.0], radius=2 * L),
    boundary_condition="reflection",
)

nucleation_strategy_montecarlo = many_bubbles.FixedNucleationRate(
    beta=1,
    gamma0=1.0e-3,
    t0=0.0,
    d_p0=0.01,
    seed=0,
    method="montecarlo",
    n_points=10000,
    volume_remaining_fraction_cutoff=1e-3,
    max_time_steps=1_000_000,
)
lattice_bubbles_montecarlo = nucleation_strategy_montecarlo.nucleate(
    lattice=many_bubbles.SphericalLattice(center=[0.0, 0.0, 0.0], radius=2 * L),
    boundary_condition="reflection",
)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    nucleation_strategy_apprx.time_history,
    nucleation_strategy_apprx.volume_remaining_history / lattice_bubbles_apprx.volume,
    color="tab:orange",
    label=r"Approximation: $f_\text{FV} = 1 - \dfrac{4 \pi}{3 V_\text{lattice}} \displaystyle\sum_n R_n^3$",
)
ax.plot(
    nucleation_strategy_montecarlo.time_history,
    nucleation_strategy_montecarlo.volume_remaining_history
    / lattice_bubbles_montecarlo.volume,
    color="tab:blue",
    label=rf"Monte-Carlo, $n_\text{{points}} = {nucleation_strategy_montecarlo.n_points}$",
)
ax.set_xlabel(r"$t$", fontsize=18)
ax.set_ylabel(r"$f_{\text{FV}} = V_{\text{FV}} / V_{\text{lattice}}$", fontsize=12)
ax.grid(True)
ax.set_title(
    r"Volume fraction of False Vacuum (i.e volume of lattice that are outside all existing bubbles at time $t$)",
    fontsize=14,
)
ax.set_ylim(bottom=0.0)
ax.legend()
fig.savefig(
    f"./figures/many_bubbles/f_FV.png",
    bbox_inches="tight",
    facecolor="white",
)
fig

# %% [markdown]
# ## Distributions over different nucleation histories

# %%
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

# Parameters for lattice/nucleation strategies
beta = 0.05
w_peak = 1 / beta
L_beta = 20
L_box = L_beta / beta
V_box = L_box**3
d_p0 = 1e-2
gamma0 = 1e-3
volume_remaining_fraction_cutoff = 1e-3
n_points = 20000


def run_single_realization(_):
    """Run a single realization with independent random seed.

    Args:
        _: Dummy argument (ignored), used to trigger multiple calls via executor.map

    Returns:
        tuple: (n_bubbles, time_history, volume_remaining_history)
    """
    nucleation_strategy = many_bubbles.FixedNucleationRate(
        beta=beta,
        gamma0=gamma0,
        t0=0.0,
        d_p0=d_p0,
        seed=None,  # Each process gets independent random seed via getrandom
        method="montecarlo",
        n_points=n_points,
        volume_remaining_fraction_cutoff=volume_remaining_fraction_cutoff,
        max_time_steps=1_000_000,
    )

    lattice_bubbles = nucleation_strategy.nucleate(
        lattice=many_bubbles.CartesianLattice(
            origin=[0.0, 0.0, 0.0],
            basis=[
                [L_box, 0.0, 0.0],
                [0.0, L_box, 0.0],
                [0.0, 0.0, L_box],
            ],
        ),
        boundary_condition="periodic",
    )

    # Extract histories before nucleation_strategy goes out of scope
    time_history = nucleation_strategy.time_history
    volume_remaining_history = nucleation_strategy.volume_remaining_history
    n_bubbles = len(lattice_bubbles.bubbles_interior)

    return n_bubbles, time_history, volume_remaining_history


def interpolate_histories_to_common_grid(
    time_histories, volume_histories, lattice_volume, n_points=1000
):
    """
    Interpolate all histories onto a common time grid.

    Args:
        time_histories: List of numpy arrays, each containing time points for one realization
        volume_histories: List of numpy arrays, each containing volume values for one realization
        lattice_volume: Total volume of the lattice
        n_points: Number of points in the common time grid

    Returns:
        t_common: Common time grid
        f_interp: Array of shape (n_histories, n_points) with interpolated fractions
    """
    # Find global time range across all histories
    t_min = min(th[0] for th in time_histories)
    t_max = max(th[-1] for th in time_histories)

    # Create common time grid
    t_common = np.linspace(t_min, t_max, n_points)

    # Interpolate each history onto common grid
    n_histories = len(time_histories)
    f_interp = np.zeros((n_histories, n_points))

    for i, (t_hist, v_hist) in enumerate(zip(time_histories, volume_histories)):
        # Convert to fraction
        f_hist = np.array(v_hist) / lattice_volume

        # Interpolate onto common grid
        f_interp[i, :] = np.interp(t_common, t_hist, f_hist)

    return t_common, f_interp


# %%
# ============================================================================
# Run simulations
# ============================================================================

n_histories = 1024

print(f"Running {n_histories} realizations...")
with ProcessPoolExecutor() as executor:
    results = list(
        executor.map(
            run_single_realization,
            range(n_histories),
        )
    )

# Unpack results
bubbles_len, time_histories, volume_histories = zip(*results)
bubbles_len = np.array(bubbles_len)

# Extract first and last bubble times from time_histories
t_first_bubble = np.array([th[0] for th in time_histories])
t_last_bubble = np.array([th[-1] for th in time_histories])

# %%
# ============================================================================
# Plot 1: Distribution plots
# ============================================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Left plot: Total number of bubbles
ax[0].hist(bubbles_len, bins=20)
y_min, y_max = ax[0].get_ylim()
ax[0].vlines(
    bubbles_len.mean(),
    y_min,
    y_max,
    color="lime",
    label=f"Mean(Code) = {bubbles_len.mean():.2f}",
)
ax[0].vlines(
    316.8,
    y_min,
    y_max,
    color="red",
    linestyles="--",
    label=r"Mean(Paper) $\sim 316.8$",
)
ax[0].set_ylim(y_min, y_max)
ax[0].set_xlabel(r"Total $\#$ bubbles")
ax[0].set_ylabel(r"$\#$ of configurations")
ax[0].legend()

# Right plot: Duration
duration = (t_last_bubble - t_first_bubble) * beta
ax[1].hist(duration, bins=20)
y_min, y_max = ax[1].get_ylim()
ax[1].vlines(
    duration.mean(), y_min, y_max, color="lime", label=f"Mean = {duration.mean():.2f}"
)
ax[1].vlines(
    7.5,
    y_min,
    y_max,
    color="red",
    linestyles="--",
    label=r"Mean(Paper) $\sim 7.5$",
)
ax[1].set_xlabel(r"$\beta (t_\text{last bubble} - t_\text{first bubble})$")

for i in range(2):
    ax[i].legend()
    ax[i].grid(True, alpha=0.5)

fig.suptitle(f"Distribution over {len(bubbles_len)} histories")
fig.savefig(
    f"./figures/many_bubbles/distributions_over_histories.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

# %%
# ============================================================================
# Plot 2: Volume fraction vs time with error band
# ============================================================================

# Interpolate all histories to common grid
lattice_volume = V_box
t_common, f_interp = interpolate_histories_to_common_grid(
    time_histories, volume_histories, lattice_volume, n_points=1000
)

# Calculate statistics at each time point
f_mean = np.mean(f_interp, axis=0)
f_min = np.min(f_interp, axis=0)
f_max = np.max(f_interp, axis=0)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot error band (min-max)
ax.fill_between(t_common, f_min, f_max, alpha=0.3, color="blue", label="Min-Max range")

# Plot mean
ax.plot(
    t_common,
    f_mean,
    color="blue",
    linewidth=2,
    label=f"Mean over {n_histories} histories",
)

ax.set_xlim(t_common.min(), t_common.max())
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r"$t$", fontsize=12)
ax.set_ylabel(r"$f_{\text{FV}} = V_{\text{FV}} / V_{\text{lattice}}$", fontsize=12)
ax.set_title("Volume Remaining Fraction vs Time", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

fig.savefig(
    f"./figures/many_bubbles/V_FV_over_histories.png",
    bbox_inches="tight",
    facecolor="white",
)
fig.show()

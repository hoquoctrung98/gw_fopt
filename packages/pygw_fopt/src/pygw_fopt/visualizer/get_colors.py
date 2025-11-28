import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Union, Tuple

# ------------------------------------------------------------------
# Fully automatic detection using Matplotlib's own registry
# ------------------------------------------------------------------
def _is_qualitative_cmap(cmap_input) -> bool:
    """
    Return True if the colormap is qualitative/categorical (e.g. tab10, Set1).
    Uses Matplotlib's official categorization — no manual lists!
    """
    # Get the name if it's a Colormap object
    if hasattr(cmap_input, 'name'):
        name = cmap_input.name
    elif isinstance(cmap_input, str):
        name = cmap_input
    else:
        return False

    # Matplotlib 3.5+ has plt.colormaps() with category info
    try:
        categories = plt.colormaps.categories
        category = categories.get(name.lower(), "")
        return "qualitative" in category.lower() or "categorical" in category.lower()
    except Exception:
        # Fallback: very small number of colors → likely qualitative
        try:
            cmap_obj = plt.get_cmap(name)
            return hasattr(cmap_obj, 'colors') and len(cmap_obj.colors) <= 32
        except:
            return False


# ------------------------------------------------------------------
# Clean, readable, and 100% correct main function
# ------------------------------------------------------------------
def get_colors_from_cmap(cmap: Union[str, mcolors.Colormap], n_colors: int) -> List[Tuple[float, float, float, float]]:
    """
    Return n distinct colors from a colormap.

    - Qualitative colormaps (tab10, Set1, etc.) → use and cycle their exact colors
    - All other colormaps (viridis, plasma, turbo, jet, etc.) → sample uniformly from 0 to 1
    """
    if n_colors < 1:
        raise ValueError("n_colors must be >= 1")

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # Qualitative / discrete colormaps
    if _is_qualitative_cmap(cmap_obj):
        colors = getattr(cmap_obj, "colors", None)
        if colors is None or len(colors) == 0:
            N = cmap_obj.N
            colors = [cmap_obj(i) for i in range(N)]
        else:
            colors = [tuple(c) + (1.0,) if len(c) == 3 else tuple(c) for c in colors]

        return colors[:n_colors] if n_colors <= len(colors) else [colors[i % len(colors)] for i in range(n_colors)]

    # Continuous / sequential colormaps
    else:
        if n_colors == 1:
            return [cmap_obj(0.0)]
        positions = np.linspace(0.0, 1.0, n_colors)
        return [cmap_obj(p) for p in positions]

if __name__ == "__main__":
    test_cmaps = ["tab10", "Set1", "viridis", "plasma", "turbo", "cividis", "jet"]
    n = 20

    fig, axes = plt.subplots(len(test_cmaps), 1, figsize=(10, 1.5 * len(test_cmaps)),
                             constrained_layout=True)

    for ax, name in zip(axes, test_cmaps):
        colors = get_colors_from_cmap(name, n)
        for i, col in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=col, ec="k", lw=0.5))
        kind = "Discrete" if _is_qualitative_cmap(name) else "Continuous"
        ax.set_title(f"{name} – {n} colors [{kind}]")
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.show()

"""Rust-accelerated gravitational wave calculations."""

# Import the Rust extension
from gw_fopt.bubble_gw import _py_bubble_gw

# Re-export submodules for convenience
from gw_fopt.bubble_gw._py_bubble_gw import (
    two_bubbles,
    many_bubbles,
    utils,
)

# You can also re-export specific classes/functions if desired
# from gw_fopt.bubble_gw._py_bubble_gw.utils import sample, sample_arr
# from gw_fopt.bubble_gw._py_bubble_gw.many_bubbles import PyLatticeBubbles

__all__ = [
    "two_bubbles",
    "many_bubbles",
    "utils",
    # Add specific exports here if you want
]

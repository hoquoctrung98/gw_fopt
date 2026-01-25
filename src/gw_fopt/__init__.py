"""GW-FOPT: Gravitational Wave background from First Order Phase Transitions."""

# Import subpackages to make them visible
from gw_fopt import bubble_gw
from gw_fopt import bubble_dynamics

__version__ = "0.1.0"

# Expose subpackages in __all__
__all__ = [
    "bubble_gw",
    "bubble_dynamics",
]

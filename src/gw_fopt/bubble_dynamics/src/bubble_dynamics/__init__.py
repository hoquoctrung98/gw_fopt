# Forward to namespace package
from gw_fopt.bubble_dynamics import *
# from gw_fopt.bubble_dynamics import __all__  # if defined

# Optional: warn or note
import warnings
warnings.warn(
    "Importing from 'bubble_dynamics' is deprecated. Use 'gw_fopt.bubble_dynamics'.",
    FutureWarning,
    stacklevel=2
)

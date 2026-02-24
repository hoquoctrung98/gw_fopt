from enum import Enum


class MeanErrorType(Enum):
    """Type of error bar to display in averaged plots."""

    SEM = 0  # Standard Error of the Mean
    STD = 1  # Standard Deviation
    ABS = 3  # Absolute range (min–max envelope)

"""
This module provides a function for generating dark current noise.
"""

import numpy
import hologen.utils as utils


def add_dark_current_noise(
    field: numpy.ndarray[float], dark_current_mean: float
) -> numpy.ndarray[float]:
    """Add dark current noise to a given field."""
    dark_signal = utils.RandomNumberGenerator().poisson(dark_current_mean, field.shape)
    field += dark_signal
    return field

"""
This module provides a function for generating dark current noise.

Heat frees electrons in the sensor even without light. They accumulate during the
exposure at the dark current rate mu_I (e-/pixel/s), so the mean number of dark
electrons is mu_I * t_exp, and like photo-electrons they are Poisson distributed
(EMVA Standard 1288, Release 4.0 Linear, 2021, Sec. 3.1, Eqs. 29-30). The dark
current roughly doubles with every T_d degrees of temperature (Sec. 3.2, Eq. 31).
"""

import numpy

import hologen.utils as utils


def add_dark_current_noise(
    field: numpy.ndarray[float], dark_current_mean: float
) -> numpy.ndarray[float]:
    """
    Add Poisson-distributed dark electrons to every pixel.

    Args:
        field (numpy.ndarray[float]): Electrons per pixel; it is not modified.
        dark_current_mean (float): Mean dark electrons per pixel in one exposure,
            i.e. dark current (e-/s) times exposure time (s).

    Returns:
        numpy.ndarray[float]: A new array with the dark electrons added.
    """
    dark_signal: numpy.ndarray[float] = utils.RandomNumberGenerator().poisson(
        dark_current_mean, field.shape
    )
    return field + dark_signal

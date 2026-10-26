"""
This module provides a function for generating read noise.

The amplifier and readout electronics that measure a pixel's charge add a
signal-independent, normally distributed error with standard deviation sigma_d
(EMVA Standard 1288, Release 4.0 Linear, 2021, Sec. 2.4). It is given in
electrons, before the analog-to-digital conversion.
"""

import numpy

import hologen.utils as utils


def add_read_noise(field: numpy.ndarray[float], sigma: float) -> numpy.ndarray[float]:
    """
    Add Gaussian read noise to every pixel.

    Args:
        field (numpy.ndarray[float]): Electrons per pixel; it is not modified.
        sigma (float): Read noise standard deviation (in electrons).

    Returns:
        numpy.ndarray[float]: A new array with the read noise added.
    """
    read_noise: numpy.ndarray[float] = utils.RandomNumberGenerator().normal(
        0.0, sigma, field.shape
    )
    return field + read_noise

"""
This module exposes the mathmetical operations for easy-to-use.
"""

import numpy
import math

from .domains import FrequencyGrid, create_frequency_grid


def get_zernike_polynomial(rho: numpy.ndarray, theta: numpy.ndarray, m, n):
    """Calculates Zernike polynomial Z_n^m on the current grid."""
    # Radial polynomial R_n^m(rho)
    r = numpy.zeros_like(rho)
    if (n - abs(m)) % 2 == 0:
        for k in range((n - abs(m)) // 2 + 1):
            c = ((-1) ** k * math.factorial(n - k)) / (
                math.factorial(k)
                * math.factorial((n + abs(m)) // 2 - k)
                * math.factorial((n - abs(m)) // 2 - k)
            )
            r += c * rho ** (n - 2 * k)

    # Zernike definition
    mask = rho <= 1.0
    z = numpy.zeros_like(rho)

    if m >= 0:
        z[mask] = r[mask] * numpy.cos(m * theta[mask])
    else:
        z[mask] = r[mask] * numpy.sin(-m * theta[mask])
    return z


def get_angular_spectrum_transfer_function(
    resolution: int,
    pixel_size: float,
    wavelength: float,
    z_distance: float,
    is_forward: bool,
) -> numpy.ndarray:
    grid: FrequencyGrid = create_frequency_grid(resolution, pixel_size)
    sq_arg: numpy.ndarray = (
        1 - (wavelength * grid.FX) ** 2 - (wavelength * grid.FY) ** 2
    )
    sq_arg: numpy.ndarray = numpy.maximum(sq_arg, 0)
    root: numpy.ndarray = numpy.sqrt(sq_arg)

    k: float = 2 * numpy.pi / wavelength
    if is_forward:
        return numpy.exp(1j * k * z_distance * root)
    return numpy.exp(-1j * k * z_distance * root)

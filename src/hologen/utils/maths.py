"""Zernike polynomials and the angular spectrum transfer function."""

import numpy
import math

from .domains import FrequencyGrid, create_frequency_grid


def get_zernike_polynomial(
    rho: numpy.ndarray, theta: numpy.ndarray, m: int, n: int
) -> numpy.ndarray:
    """
    Evaluate the Zernike polynomial Z_n^m on a polar grid.

    Args:
        rho (numpy.ndarray): Radius normalised to 1 at the pupil edge.
        theta (numpy.ndarray): Angle in radians, same shape as rho.
        m (int): Azimuthal order; negative values give the sine terms.
        n (int): Radial order, at least abs(m).

    Returns:
        numpy.ndarray: Z_n^m inside the unit circle and 0 outside, same shape
            as rho.
    """
    # Radial polynomial R_n^m(rho)
    r: numpy.ndarray = numpy.zeros_like(rho)
    if (n - abs(m)) % 2 == 0:
        for k in range((n - abs(m)) // 2 + 1):
            c: float = ((-1) ** k * math.factorial(n - k)) / (
                math.factorial(k)
                * math.factorial((n + abs(m)) // 2 - k)
                * math.factorial((n - abs(m)) // 2 - k)
            )
            r += c * rho ** (n - 2 * k)

    # Zernike definition
    mask: numpy.ndarray = rho <= 1.0
    z: numpy.ndarray = numpy.zeros_like(rho)

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
    """
    Compute the angular spectrum transfer function of free-space propagation.

    Evanescent frequencies, beyond 1 / wavelength, get a root of 0, so they
    pass with a phase of 0 instead of decaying.

    Args:
        resolution (int): Number of pixels along each side.
        pixel_size (float): Pixel size, in um.
        wavelength (float): Wavelength, in um.
        z_distance (float): Propagation distance, in um.
        is_forward (bool): Whether to propagate forwards (towards the detector)
            or backwards.

    Returns:
        numpy.ndarray: Complex transfer function on the centred frequency grid,
            shape (resolution, resolution).
    """
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

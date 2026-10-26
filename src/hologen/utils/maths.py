"""Zernike polynomials and the angular spectrum transfer function."""

import math

import numpy

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

    Plane waves with a spatial frequency beyond 1 / wavelength are evanescent.
    They decay as exp(-k z sqrt((wavelength f)^2 - 1)) instead of travelling
    (Goodman, Introduction to Fourier Optics, Sec. 3.10), and the transfer
    function is 0 there. Forwards this matches the physical decay to below 1e-12
    once f is 0.02 cycles per um past 1 / wavelength at 20 um. Only a thin ring
    of frequencies closer to the cutoff keeps more, and an objective pupil
    removes it anyway. Backwards zero is the only stable choice, because the
    exact inverse would amplify these waves by up to exp(k z sqrt(...)).
    Evanescent frequencies only exist on grids with pixels smaller than
    wavelength / 2.

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
    propagating: numpy.ndarray = sq_arg >= 0
    root: numpy.ndarray = numpy.sqrt(numpy.maximum(sq_arg, 0))

    k: float = 2 * numpy.pi / wavelength
    sign: float = 1.0 if is_forward else -1.0
    return numpy.where(propagating, numpy.exp(sign * 1j * k * z_distance * root), 0)

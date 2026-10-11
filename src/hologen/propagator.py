"""
This module provides propagation methods.
"""

import numpy
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import hologen.utils as utils


def backwards_propagate(
    complex_field_hologram: numpy.ndarray[numpy.complex128],
    z_distance: float,
    wavelength: float,
    pixel_size: float,
    resolution: int,
) -> numpy.ndarray[numpy.complex128]:
    """
    Back-propagate a complex field from hologram plane to object plane using Angular Spectrum Method.

    Args:
        complex_field_hologram: Complex field at hologram plane
        z_distance: Propagation distance (in microns)
        wavelength: Wavelength (in microns)
        pixel_size: Pixel size (in microns)
        resolution: Resolution of the grid

    Returns:
        Complex field at object plane
    """
    # Transform to frequency domain.
    U_f = fftshift(fft2(fftshift(complex_field_hologram)))
    # Calculate propagation kernel (transfer function)
    H_back = utils.get_angular_spectrum_transfer_function(
        resolution, pixel_size, wavelength, z_distance, is_forward=False
    )
    # Apply back-propagation.
    U_obj_f = U_f * H_back
    U_obj = ifftshift(ifft2(ifftshift(U_obj_f)))
    return U_obj


def forwards_propagate(
    object_field: numpy.ndarray[numpy.complex128],
    z_distance: float,
    wavelength: float,
    pixel_size: float,
    resolution: int,
    numerical_aperture: float | None = None,
) -> numpy.ndarray[numpy.complex128]:
    """
    Propagate a complex field of object plane to hologram plane using Angular Spectrum Method.

    Args:
        object_field: Complex field at object plane
        z_distance: Propagation distance (in microns)
        wavelength: Wavelength (in microns)
        pixel_size: Pixel size (in microns)
        resolution: Resolution of the grid
        numerical_aperture: NA of the imaging objective in a magnified setup. Its
            coherent transfer function passes spatial frequencies up to NA /
            wavelength (Goodman, Introduction to Fourier Optics, Sec. 6.2). None
            for a lensless setup without a pupil.

    Returns:
        Complex field at hologram plane
    """
    # Transform to frequency domain.
    U_f = fftshift(fft2(fftshift(object_field)))
    # Calculate propagation kernel (transfer function)
    H = utils.get_angular_spectrum_transfer_function(
        resolution, pixel_size, wavelength, z_distance, is_forward=True
    )
    # Apply propagation.
    U_z_f = U_f * H
    if numerical_aperture is not None:
        U_z_f = U_z_f * objective_pupil(
            resolution, pixel_size, wavelength, numerical_aperture
        )
    U_z = ifftshift(ifft2(ifftshift(U_z_f)))
    return U_z


def objective_pupil(
    resolution: int, pixel_size: float, wavelength: float, numerical_aperture: float
) -> numpy.ndarray[bool]:
    """
    Coherent transfer function of an objective: a circular pupil that passes
    spatial frequencies up to NA / wavelength, on the same centred frequency grid
    as the angular spectrum transfer function.
    """
    if not 0 < numerical_aperture <= 1.5:
        raise ValueError(
            f"numerical_aperture must be in (0, 1.5], got {numerical_aperture}."
        )
    grid: utils.FrequencyGrid = utils.create_frequency_grid(resolution, pixel_size)
    cutoff: float = numerical_aperture / wavelength
    return grid.FX**2 + grid.FY**2 <= cutoff**2

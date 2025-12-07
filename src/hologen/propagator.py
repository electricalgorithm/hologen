"""
This module provides propagation methods.
"""
import numpy
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import hologen.utils as utils


def backwards_propagate(
    complex_field_hologram: numpy.ndarray[numpy.complex128], z_distance: float, wavelength: float, pixel_size: float, resolution: int
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
    H_back = utils.get_angular_spectrum_transfer_function(resolution, pixel_size, wavelength, z_distance, is_forward=False)
    # Apply back-propagation.
    U_obj_f = U_f * H_back
    U_obj = ifftshift(ifft2(ifftshift(U_obj_f)))
    return U_obj

def forwards_propagate(
    object_field: numpy.ndarray[numpy.complex128], z_distance: float, wavelength: float, pixel_size: float, resolution: int
) -> numpy.ndarray[numpy.complex128]:
    """
    Propagate a complex field of object plane to hologram plane using Angular Spectrum Method.

    Args:
        object_field: Complex field at object plane
        z_distance: Propagation distance (in microns)
        wavelength: Wavelength (in microns)
        pixel_size: Pixel size (in microns)
        resolution: Resolution of the grid

    Returns:
        Complex field at hologram plane
    """
    # Transform to frequency domain.
    U_f = fftshift(fft2(fftshift(object_field)))
    # Calculate propagation kernel (transfer function)
    H = utils.get_angular_spectrum_transfer_function(resolution, pixel_size, wavelength, z_distance, is_forward=True)
    # Apply propagation.
    U_z_f = U_f * H
    U_z = ifftshift(ifft2(ifftshift(U_z_f)))
    return U_z
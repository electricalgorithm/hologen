"""Free-space propagation with the angular spectrum method."""

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
    Back-propagate a field from the hologram plane to the object plane.

    The propagation uses the angular spectrum method.

    Args:
        complex_field_hologram (numpy.ndarray[numpy.complex128]): Complex field
            at the hologram plane, shape (resolution, resolution).
        z_distance (float): Propagation distance, in um.
        wavelength (float): Wavelength, in um.
        pixel_size (float): Pixel size, in um.
        resolution (int): Number of pixels along each side.

    Returns:
        numpy.ndarray[numpy.complex128]: Complex field at the object plane,
            same shape as the input.
    """
    # Transform to frequency domain.
    U_f: numpy.ndarray = fftshift(fft2(fftshift(complex_field_hologram)))
    # Calculate propagation kernel (transfer function)
    H_back: numpy.ndarray = utils.get_angular_spectrum_transfer_function(
        resolution, pixel_size, wavelength, z_distance, is_forward=False
    )
    # Apply back-propagation.
    U_obj_f: numpy.ndarray = U_f * H_back
    U_obj: numpy.ndarray = ifftshift(ifft2(ifftshift(U_obj_f)))
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
    Propagate a field from the object plane to the hologram plane.

    The propagation uses the angular spectrum method.

    Args:
        object_field (numpy.ndarray[numpy.complex128]): Complex field at the
            object plane, shape (resolution, resolution).
        z_distance (float): Propagation distance, in um.
        wavelength (float): Wavelength, in um.
        pixel_size (float): Pixel size, in um.
        resolution (int): Number of pixels along each side.
        numerical_aperture (float | None): NA of the imaging objective in a
            magnified setup. Its coherent transfer function passes spatial
            frequencies up to NA / wavelength (Goodman, Introduction to Fourier
            Optics, Sec. 6.2). None for a lensless setup without a pupil.

    Returns:
        numpy.ndarray[numpy.complex128]: Complex field at the hologram plane,
            same shape as the input.
    """
    # Transform to frequency domain.
    U_f: numpy.ndarray = fftshift(fft2(fftshift(object_field)))
    # Calculate propagation kernel (transfer function)
    H: numpy.ndarray = utils.get_angular_spectrum_transfer_function(
        resolution, pixel_size, wavelength, z_distance, is_forward=True
    )
    # Apply propagation.
    U_z_f: numpy.ndarray = U_f * H
    if numerical_aperture is not None:
        U_z_f = U_z_f * objective_pupil(
            resolution, pixel_size, wavelength, numerical_aperture
        )
    U_z: numpy.ndarray = ifftshift(ifft2(ifftshift(U_z_f)))
    return U_z


def objective_pupil(
    resolution: int, pixel_size: float, wavelength: float, numerical_aperture: float
) -> numpy.ndarray[bool]:
    """
    Compute the coherent transfer function of an objective.

    It is a circular pupil that passes spatial frequencies up to NA / wavelength,
    on the same centred frequency grid as the angular spectrum transfer function.

    Args:
        resolution (int): Number of pixels along each side.
        pixel_size (float): Pixel size, in um.
        wavelength (float): Wavelength, in um.
        numerical_aperture (float): NA of the objective, in (0, 1.5].

    Returns:
        numpy.ndarray[bool]: True inside the pupil, shape (resolution,
            resolution).

    Raises:
        ValueError: If the numerical aperture is outside (0, 1.5].
    """
    if not 0 < numerical_aperture <= 1.5:
        raise ValueError(
            f"numerical_aperture must be in (0, 1.5], got {numerical_aperture}."
        )
    grid: utils.FrequencyGrid = utils.create_frequency_grid(resolution, pixel_size)
    cutoff: float = numerical_aperture / wavelength
    return grid.FX**2 + grid.FY**2 <= cutoff**2

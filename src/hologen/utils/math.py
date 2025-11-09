"""Mathematical utilities supporting holography computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hologen.types import ArrayFloat, GridSpec


@dataclass(slots=True)
class FourierGrid:
    """Frequency-domain sampling constructed for a spatial grid.

    Args:
        fx: Two-dimensional array of spatial frequencies along the x axis.
        fy: Two-dimensional array of spatial frequencies along the y axis.
    """

    fx: NDArray[np.float64]
    fy: NDArray[np.float64]


def make_fourier_grid(grid: GridSpec) -> FourierGrid:
    """Create Fourier-domain sampling coordinates for a spatial grid.

    Args:
        grid: Spatial grid specification defining the sampling resolution.

    Returns:
        FourierGrid containing spatial frequency meshes along both axes.
    """

    fx = np.fft.fftfreq(grid.width, d=grid.pixel_pitch)
    fy = np.fft.fftfreq(grid.height, d=grid.pixel_pitch)
    fx_mesh, fy_mesh = np.meshgrid(fx, fy, indexing="xy")
    return FourierGrid(fx=fx_mesh.astype(np.float64), fy=fy_mesh.astype(np.float64))


def normalize_image(image: ArrayFloat) -> ArrayFloat:
    """Normalize an image to the range ``[0.0, 1.0]``.

    Args:
        image: Arbitrary floating-point image.

    Returns:
        Normalized image or zeros when the input is constant.
    """

    value_min = np.min(image)
    value_max = np.max(image)
    if value_max <= value_min:
        # Ensure the returned zero array is float64 for consistency.
        return np.zeros_like(image, dtype=np.float64)
    normalized = (image - value_min) / (value_max - value_min)
    return normalized.astype(np.float64)
    return normalized.astype(np.float64)


def _convolve_axis(
    data: ArrayFloat, kernel: NDArray[np.float64], axis: int
) -> ArrayFloat:
    """Convolve data with a one-dimensional kernel along a specific axis.

    Args:
        data: Input array to convolve.
        kernel: One-dimensional convolution kernel.
        axis: Axis along which convolution is applied.

    Returns:
        Array with the same shape as ``data`` after convolution.
    """

    padding = kernel.size // 2
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (padding, padding)
    padded = np.pad(data, pad_width=pad_width, mode="edge")

    def convolve_line(line: ArrayFloat) -> ArrayFloat:
        line_convolved = np.convolve(line, kernel, mode="valid")
        return line_convolved.astype(np.float64)

    convolved = np.apply_along_axis(convolve_line, axis, padded)
    return convolved.astype(np.float64)


def gaussian_blur(image: ArrayFloat, sigma: float) -> ArrayFloat:
    """Apply an isotropic Gaussian blur to a two-dimensional image.

    Args:
        image: Input image to filter.
        sigma: Standard deviation of the Gaussian kernel in pixel units.

    Returns:
        Blurred image with identical shape to the input.
    """

    if sigma <= 0.0:
        return image.copy()

    radius = max(int(3.0 * sigma), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)

    blurred = _convolve_axis(image, kernel, axis=0)
    blurred = _convolve_axis(blurred, kernel, axis=1)
    return blurred.astype(np.float64)

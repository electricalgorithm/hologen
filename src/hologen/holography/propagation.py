"""Wave propagation utilities for holography simulations."""

from __future__ import annotations

import numpy as np

from hologen.types import ArrayComplex, GridSpec, OpticalConfig


def angular_spectrum_propagate(
    field: ArrayComplex,
    grid: GridSpec,
    optics: OpticalConfig,
    distance: float,
) -> ArrayComplex:
    """Propagate a complex optical field using the angular spectrum method.

    The angular spectrum method models propagation by decomposing the source
    field into plane waves (its spatial frequency spectrum), applying the
    appropriate phase (or evanescent decay) for a given propagation distance,
    and recomposing the field in the observation plane.

    This implementation assumes:
    - A uniformly sampled square/rectangular grid described by GridSpec.
    - Monochromatic illumination described by OpticalConfig.wavelength.
    - No additional aperturing or windowing: the entire sampled field is used.

    Args:
        field: Complex field distribution sampled in the source plane with
            shape (grid.height, grid.width).
        grid: Spatial sampling specification for the field. Must provide
            .height, .width and .pixel_pitch (in meters).
        optics: Optical parameters describing the illumination. Must provide
            .wavelength (in meters).
        distance: Propagation distance along the optical axis in meters. Positive
            values advance the field, negative values back-propagate.

    Returns:
        Complex field after propagation over the requested distance, sampled on
        the same grid. The returned array has dtype numpy.complex128.

    Raises:
        ValueError: If the supplied field shape is incompatible with the grid.
    """

    if field.shape != (grid.height, grid.width):
        raise ValueError(
            "Field shape must match grid dimensions: "
            f"expected {(grid.height, grid.width)}, received {field.shape}."
        )

    # Short-circuit no-op propagation.
    if distance == 0.0:
        return field

    # Spatial frequency coordinates for the sampled grid (cycles per meter).
    fy = np.fft.fftfreq(grid.height, d=grid.pixel_pitch)
    fx = np.fft.fftfreq(grid.width, d=grid.pixel_pitch)
    fx_mesh, fy_mesh = np.meshgrid(fx, fy, indexing="xy")

    wavelength = optics.wavelength
    k = 2.0 * np.pi / wavelength

    # Compute the squared longitudinal component argument:
    # argument = 1 - (lambda * fx)^2 - (lambda * fy)^2
    # positive values correspond to propagating plane waves (real kz),
    # negative values correspond to evanescent components (imaginary kz).
    argument = 1.0 - (wavelength * fx_mesh) ** 2 - (wavelength * fy_mesh) ** 2
    positive_mask = argument >= 0.0
    propagation_kernel = np.zeros_like(argument, dtype=np.complex128)

    # Propagating components: phase delay exp(i k_z z)
    if np.any(positive_mask):
        kz = np.sqrt(argument, where=positive_mask, out=np.zeros_like(argument))
        propagation_kernel[positive_mask] = np.exp(
            1j * k * distance * kz[positive_mask]
        )

    # Evanescent components: exponential decay with distance magnitude
    if np.any(~positive_mask):
        decay = np.sqrt(-argument, where=~positive_mask, out=np.zeros_like(argument))
        propagation_kernel[~positive_mask] = np.exp(
            -k * np.abs(distance) * decay[~positive_mask]
        )

    # Apply kernel in the spatial-frequency domain
    spectrum = np.fft.fft2(field)
    propagated_spectrum = spectrum * propagation_kernel
    propagated_field = np.fft.ifft2(propagated_spectrum)

    return propagated_field.astype(np.complex128)

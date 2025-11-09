"""Inline holography strategy implementation."""

from __future__ import annotations

import numpy as np

from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import ArrayComplex, ArrayFloat, HolographyConfig, HolographyStrategy


def _object_to_complex(object_field: ArrayFloat) -> ArrayComplex:
    """Convert a real amplitude field into a complex representation."""

    return object_field.astype(np.complex128)


def _field_to_intensity(field: ArrayComplex) -> ArrayFloat:
    """Convert a complex field to its intensity distribution."""

    return np.abs(field) ** 2


class InlineHolographyStrategy(HolographyStrategy):
    """Implement inline hologram generation and reconstruction."""

    def create_hologram(
        self, object_field: ArrayFloat, config: HolographyConfig
    ) -> ArrayFloat:
        """Generate an inline hologram from an object-domain amplitude field."""

        complex_object = _object_to_complex(object_field)
        propagated = angular_spectrum_propagate(
            field=complex_object,
            grid=config.grid,
            optics=config.optics,
            distance=config.optics.propagation_distance,
        )
        hologram = _field_to_intensity(propagated)
        return hologram.astype(np.float64)

    def reconstruct(self, hologram: ArrayFloat, config: HolographyConfig) -> ArrayFloat:
        """Reconstruct the object domain from an inline hologram."""

        field = np.sqrt(np.maximum(hologram, 0.0)).astype(np.complex128)
        reconstructed = angular_spectrum_propagate(
            field=field,
            grid=config.grid,
            optics=config.optics,
            distance=-config.optics.propagation_distance,
        )
        return np.abs(reconstructed).astype(np.float64)

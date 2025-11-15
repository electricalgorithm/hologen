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
        self, object_field: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Generate an inline hologram from an object-domain complex field."""

        propagated = angular_spectrum_propagate(
            field=object_field,
            grid=config.grid,
            optics=config.optics,
            distance=config.optics.propagation_distance,
        )
        return propagated

    def reconstruct(
        self, hologram: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct the object domain from an inline hologram."""

        reconstructed = angular_spectrum_propagate(
            field=hologram,
            grid=config.grid,
            optics=config.optics,
            distance=-config.optics.propagation_distance,
        )
        return reconstructed

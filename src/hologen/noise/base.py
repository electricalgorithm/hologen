"""Base noise model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from hologen.types import ArrayFloat, HolographyConfig, NoiseModel


@dataclass(slots=True)
class BaseNoiseModel(NoiseModel):
    """Abstract base for noise model implementations.

    Args:
        name: Identifier for the noise model.
    """

    name: str

    def apply(
        self, hologram: ArrayFloat, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        """Apply noise to a hologram.

        Args:
            hologram: Perfect hologram intensity distribution.
            config: Holography configuration containing grid and optical parameters.
            rng: Random number generator for stochastic noise.

        Returns:
            Noisy hologram intensity distribution.

        Raises:
            NotImplementedError: If the subclass does not override the method.
        """

        raise NotImplementedError

    def _ensure_positive(self, hologram: ArrayFloat) -> ArrayFloat:
        """Ensure hologram values are non-negative.

        Args:
            hologram: Hologram array to clamp.

        Returns:
            Hologram with non-negative values.
        """

        return np.maximum(hologram, 0.0)

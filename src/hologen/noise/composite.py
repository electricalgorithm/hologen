"""Composite noise model implementation."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from hologen.noise.base import BaseNoiseModel
from hologen.types import ArrayFloat, HolographyConfig, NoiseModel


@dataclass(slots=True)
class CompositeNoiseModel(BaseNoiseModel):
    """Combine multiple noise models in sequence.

    Args:
        models: Tuple of noise models to apply in order.
    """

    models: tuple[NoiseModel, ...]

    def apply(
        self, hologram: ArrayFloat, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        noisy = hologram
        for model in self.models:
            noisy = model.apply(noisy, config, rng)
        return noisy

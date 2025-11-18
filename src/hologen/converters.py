"""Holography conversion pipeline components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.random import Generator

from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.noise import (
    AberrationNoiseModel,
    CompositeNoiseModel,
    SensorNoiseModel,
    SpeckleNoiseModel,
)
from hologen.phase import PhaseGenerationConfig
from hologen.shapes import available_generators
from hologen.types import (
    ArrayComplex,
    ComplexHologramSample,
    ComplexObjectSample,
    DatasetGenerator,
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    HolographyStrategy,
    NoiseConfig,
    NoiseModel,
    ObjectShapeGenerator,
)


@dataclass(slots=True)
class ObjectDomainProducer:
    """Generate object-domain samples using registered shape generators.

    Args:
        shape_generators: Tuple of shape generator implementations to sample from.
    """

    phase_config: PhaseGenerationConfig

    def generate_complex(
        self,
        grid: GridSpec,
        rng: Generator,
        wavelength: float = 632.8e-9,
    ) -> ComplexObjectSample:
        """Produce a new complex object-domain sample.

        Args:
            grid: Grid specification describing the required output resolution.
            rng: Random number generator providing stochastic parameters.
            phase_shift: Phase modulation in radians for phase-only objects.
            mode: Generation mode - "amplitude" or "phase".
            wavelength: Illumination wavelength in meters (for physics-based phase).

        Returns:
            ComplexObjectSample containing the generated complex field.
        """
        generators: tuple[ObjectShapeGenerator, ...] = tuple(available_generators())
        generator = cast(ObjectShapeGenerator, rng.choice(generators))
        field = generator.generate_complex(
            grid, rng, self.phase_config, wavelength 
        )
        return ComplexObjectSample(
            name=generator.name,
            field=field,
        )


@dataclass(slots=True)
class ObjectToHologramConverter:
    """Convert object-domain amplitudes into hologram representations.

    Args:
        strategy_mapping: Mapping from holography methods to strategy implementations.
        noise_model: Optional noise model to apply to generated holograms.
        output_config: Configuration for output field representations.
    """

    config: HolographyConfig
    noise_model: NoiseModel | None = None

    def create_hologram(
        self,
        sample: ComplexObjectSample,
        rng: Generator | None = None,
    ) -> ArrayComplex:
        """Generate a hologram for the provided object sample.

        Args:
            sample: Object-domain sample to transform (ComplexObjectSample).
            rng: Random number generator for noise application (optional for backward compatibility).

        Returns:
            Hologram field.
        """

        strategy = self._resolve_strategy(self.config.method)
        complex_field = sample.field
        hologram_field = strategy.create_hologram(complex_field, self.config)

        # Apply noise model to intensity representation while preserving phase
        if self.noise_model is not None and rng is not None:
            # Extract intensity and phase
            intensity = np.abs(hologram_field) ** 2
            phase = np.angle(hologram_field)

            # Apply noise to intensity
            noisy_intensity = self.noise_model.apply(intensity, self.config, rng)

            # Reconstruct complex field with noisy amplitude and original phase
            amplitude = np.sqrt(np.maximum(noisy_intensity, 0.0))
            hologram_field = amplitude * np.exp(1j * phase)
        return hologram_field

    def reconstruct(
        self, hologram: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct an object-domain field from a hologram.

        Args:
            hologram: Hologram field (intensity for legacy, complex for new).
            config: Holography configuration specifying physical parameters.

        Returns:
            Reconstructed field (amplitude for legacy, complex for new).
        """

        strategy = self._resolve_strategy(config.method)
        return strategy.reconstruct(hologram, config)

    def _resolve_strategy(self, method: HolographyMethod) -> HolographyStrategy:
        """Resolve a holography strategy for the requested method.

        Args:
            method: Holography method identifier.

        Returns:
            Strategy capable of performing the requested conversions.

        Raises:
            KeyError: If the strategy mapping does not contain the method.
        """
        strategy_mapping: dict[HolographyMethod, HolographyStrategy] = {
            HolographyMethod.INLINE: InlineHolographyStrategy(),
            HolographyMethod.OFF_AXIS: OffAxisHolographyStrategy(),
        }
        if method not in strategy_mapping:
            raise KeyError(f"Unknown holography method: {method}.")
        return strategy_mapping[method]


@dataclass(slots=True)
class HologramDatasetGenerator(DatasetGenerator):
    """Generate full hologram samples from object-domain sources.

    Args:
        object_producer: Producer responsible for creating object samples.
        converter: Converter performing hologram generation and reconstruction.
    """

    object_producer: ObjectDomainProducer
    converter: ObjectToHologramConverter

    def generate(
        self,
        count: int,
        rng: Generator,
    ) -> Iterable[ComplexHologramSample]:
        """Yield hologram samples as an iterable sequence.

        Args:
            count: Number of samples to generate.
            rng: Random number generator used throughout the pipeline.

        Yields:
            Sequential hologram samples containing object, hologram, and reconstruction data.
        """

        # Extract wavelength from HolographyConfig
        wavelength = self.converter.config.optics.wavelength

        for _ in range(count):
            # Generate complex object sample
            object_sample = self.object_producer.generate_complex(
                self.converter.config.grid, rng, wavelength
            )

            # Override representation with output_config
            object_sample = ComplexObjectSample(
                name=object_sample.name,
                field=object_sample.field,
            )

            # Create hologram and reconstruction
            hologram_field = self.converter.create_hologram(
                object_sample, rng
            )
            reconstruction_field = self.converter.reconstruct(
                hologram_field, self.converter.config
            )

            yield ComplexHologramSample(
                object_sample=object_sample,
                hologram_field=hologram_field,
                reconstruction_field=reconstruction_field,
            )


def create_noise_model(config: NoiseConfig) -> NoiseModel | None:
    """Create a composite noise model from configuration.

    Args:
        config: Noise configuration specifying all noise parameters.

    Returns:
        Composite noise model or None if all noise is disabled.
    """
    models: list[NoiseModel] = []

    has_aberration = (
        config.aberration_defocus != 0.0
        or config.aberration_astigmatism_x != 0.0
        or config.aberration_astigmatism_y != 0.0
        or config.aberration_coma_x != 0.0
        or config.aberration_coma_y != 0.0
    )
    if has_aberration:
        models.append(
            AberrationNoiseModel(
                name="aberration",
                defocus=config.aberration_defocus,
                astigmatism_x=config.aberration_astigmatism_x,
                astigmatism_y=config.aberration_astigmatism_y,
                coma_x=config.aberration_coma_x,
                coma_y=config.aberration_coma_y,
            )
        )

    if config.speckle_contrast > 0.0:
        models.append(
            SpeckleNoiseModel(
                name="speckle",
                contrast=config.speckle_contrast,
                correlation_length=config.speckle_correlation_length,
            )
        )

    has_sensor = (
        config.sensor_read_noise > 0.0
        or config.sensor_shot_noise
        or config.sensor_dark_current > 0.0
        or config.sensor_bit_depth is not None
    )
    if has_sensor:
        models.append(
            SensorNoiseModel(
                name="sensor",
                read_noise=config.sensor_read_noise,
                shot_noise=config.sensor_shot_noise,
                dark_current=config.sensor_dark_current,
                bit_depth=config.sensor_bit_depth,
            )
        )

    if not models:
        return None

    if len(models) == 1:
        return models[0]

    return CompositeNoiseModel(name="composite", models=tuple(models))

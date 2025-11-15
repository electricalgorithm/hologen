"""Holography conversion pipeline components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.random import Generator

from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import available_generators
from hologen.types import (
    ArrayComplex,
    ArrayFloat,
    ComplexHologramSample,
    ComplexObjectSample,
    DatasetGenerator,
    DatasetWriter,
    FieldRepresentation,
    GridSpec,
    HologramSample,
    HolographyConfig,
    HolographyMethod,
    HolographyStrategy,
    NoiseModel,
    ObjectSample,
    ObjectShapeGenerator,
    OutputConfig,
)
from hologen.noise import (
    AberrationNoiseModel,
    CompositeNoiseModel,
    SensorNoiseModel,
    SpeckleNoiseModel,
)
from hologen.utils.math import normalize_image


@dataclass(slots=True)
class ObjectDomainProducer:
    """Generate object-domain samples using registered shape generators.

    Args:
        shape_generators: Tuple of shape generator implementations to sample from.
    """

    shape_generators: tuple[ObjectShapeGenerator, ...]

    def generate(self, grid: GridSpec, rng: Generator) -> ObjectSample:
        """Produce a new object-domain sample.

        Args:
            grid: Grid specification describing the required output resolution.
            rng: Random number generator providing stochastic parameters.

        Returns:
            ObjectSample containing the generated amplitude image.
        """

        generator = cast(ObjectShapeGenerator, rng.choice(self.shape_generators))
        pixels = generator.generate(grid, rng)
        normalized = normalize_image(pixels)
        return ObjectSample(name=generator.name, pixels=normalized)

    def generate_complex(
        self,
        grid: GridSpec,
        rng: Generator,
        phase_shift: float = 0.0,
        mode: str = "amplitude",
    ) -> ComplexObjectSample:
        """Produce a new complex object-domain sample.

        Args:
            grid: Grid specification describing the required output resolution.
            rng: Random number generator providing stochastic parameters.
            phase_shift: Phase modulation in radians for phase-only objects.
            mode: Generation mode - "amplitude" or "phase".

        Returns:
            ComplexObjectSample containing the generated complex field.
        """

        generator = cast(ObjectShapeGenerator, rng.choice(self.shape_generators))
        field = generator.generate_complex(grid, rng, phase_shift, mode)
        
        # Determine representation based on mode
        if mode == "amplitude":
            representation = FieldRepresentation.AMPLITUDE
        elif mode == "phase":
            representation = FieldRepresentation.PHASE
        else:
            representation = FieldRepresentation.COMPLEX
        
        return ComplexObjectSample(
            name=generator.name,
            field=field,
            representation=representation,
        )


@dataclass(slots=True)
class ObjectToHologramConverter:
    """Convert object-domain amplitudes into hologram representations.

    Args:
        strategy_mapping: Mapping from holography methods to strategy implementations.
        noise_model: Optional noise model to apply to generated holograms.
        output_config: Configuration for output field representations.
    """

    strategy_mapping: dict[HolographyMethod, HolographyStrategy]
    noise_model: NoiseModel | None = None
    output_config: OutputConfig = field(default_factory=OutputConfig)

    def create_hologram(
        self,
        sample: ObjectSample | ComplexObjectSample,
        config: HolographyConfig,
        rng: Generator | None = None,
    ) -> ArrayFloat | ArrayComplex:
        """Generate a hologram for the provided object sample.

        Args:
            sample: Object-domain sample to transform (legacy ObjectSample or ComplexObjectSample).
            config: Holography configuration specifying physical parameters.
            rng: Random number generator for noise application (optional for backward compatibility).

        Returns:
            Hologram field (intensity for legacy ObjectSample, complex for ComplexObjectSample).
        """

        strategy = self._resolve_strategy(config.method)
        
        # Handle legacy ObjectSample by converting to complex field
        if isinstance(sample, ObjectSample):
            complex_field = sample.pixels.astype(np.complex128)
            is_legacy = True
        else:
            complex_field = sample.field
            is_legacy = False
        
        hologram_field = strategy.create_hologram(complex_field, config)
        
        # Apply noise model to intensity representation while preserving phase
        if self.noise_model is not None and rng is not None:
            # Extract intensity and phase
            intensity = np.abs(hologram_field) ** 2
            phase = np.angle(hologram_field)
            
            # Apply noise to intensity
            noisy_intensity = self.noise_model.apply(intensity, config, rng)
            
            # Reconstruct complex field with noisy amplitude and original phase
            amplitude = np.sqrt(np.maximum(noisy_intensity, 0.0))
            hologram_field = amplitude * np.exp(1j * phase)
        
        # Return intensity for legacy samples, complex for new samples
        if is_legacy:
            return np.abs(hologram_field) ** 2
        return hologram_field

    def reconstruct(
        self, hologram: ArrayFloat | ArrayComplex, config: HolographyConfig
    ) -> ArrayFloat | ArrayComplex:
        """Reconstruct an object-domain field from a hologram.

        Args:
            hologram: Hologram field (intensity for legacy, complex for new).
            config: Holography configuration specifying physical parameters.

        Returns:
            Reconstructed field (amplitude for legacy, complex for new).
        """

        strategy = self._resolve_strategy(config.method)
        
        # Detect if this is legacy intensity data (real-valued)
        is_legacy = not np.iscomplexobj(hologram)
        
        if is_legacy:
            # Convert intensity to complex field (assume amplitude from sqrt)
            hologram_complex = np.sqrt(np.maximum(hologram, 0.0)).astype(np.complex128)
        else:
            hologram_complex = hologram
        
        reconstruction = strategy.reconstruct(hologram_complex, config)
        
        # Return amplitude for legacy, complex for new
        if is_legacy:
            return np.abs(reconstruction)
        return reconstruction

    def _resolve_strategy(self, method: HolographyMethod) -> HolographyStrategy:
        """Resolve a holography strategy for the requested method.

        Args:
            method: Holography method identifier.

        Returns:
            Strategy capable of performing the requested conversions.

        Raises:
            KeyError: If the strategy mapping does not contain the method.
        """

        if method not in self.strategy_mapping:
            raise KeyError(f"Unknown holography method: {method}.")
        return self.strategy_mapping[method]


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
        config: HolographyConfig,
        rng: Generator,
        phase_shift: float = 0.0,
        mode: str = "amplitude",
        use_complex: bool = False,
    ) -> Iterable[HologramSample | ComplexHologramSample]:
        """Yield hologram samples as an iterable sequence.

        Args:
            count: Number of samples to generate.
            config: Holography configuration applied to all samples.
            rng: Random number generator used throughout the pipeline.
            phase_shift: Phase modulation in radians for phase-only objects.
            mode: Generation mode - "amplitude" or "phase".
            use_complex: If True, generate ComplexHologramSample; if False, generate legacy HologramSample.

        Yields:
            Sequential hologram samples containing object, hologram, and reconstruction data.
        """

        for _ in range(count):
            if use_complex:
                # Generate complex object sample
                object_sample = self.object_producer.generate_complex(
                    config.grid, rng, phase_shift, mode
                )
                
                # Override representation with output_config
                output_config = self.converter.output_config
                object_sample = ComplexObjectSample(
                    name=object_sample.name,
                    field=object_sample.field,
                    representation=output_config.object_representation,
                )
                
                # Create hologram and reconstruction
                hologram_field = self.converter.create_hologram(object_sample, config, rng)
                reconstruction_field = self.converter.reconstruct(hologram_field, config)
                
                yield ComplexHologramSample(
                    object_sample=object_sample,
                    hologram_field=hologram_field,
                    hologram_representation=output_config.hologram_representation,
                    reconstruction_field=reconstruction_field,
                    reconstruction_representation=output_config.reconstruction_representation,
                )
            else:
                # Legacy path: generate ObjectSample
                object_sample = self.object_producer.generate(config.grid, rng)
                hologram = self.converter.create_hologram(object_sample, config, rng)
                reconstruction = self.converter.reconstruct(hologram, config)
                
                yield HologramSample(
                    object_sample=object_sample,
                    hologram=hologram,
                    reconstruction=reconstruction,
                )


def default_object_producer() -> ObjectDomainProducer:
    """Create the default object domain producer with built-in shapes."""

    generators = tuple(available_generators())
    return ObjectDomainProducer(shape_generators=generators)


def default_converter(noise_model: NoiseModel | None = None) -> ObjectToHologramConverter:
    """Create the default converter with inline and off-axis strategies."""

    strategies: dict[HolographyMethod, HolographyStrategy] = {
        HolographyMethod.INLINE: InlineHolographyStrategy(),
        HolographyMethod.OFF_AXIS: OffAxisHolographyStrategy(),
    }
    return ObjectToHologramConverter(strategy_mapping=strategies, noise_model=noise_model)


def generate_dataset(
    count: int,
    config: HolographyConfig,
    rng: Generator,
    writer: DatasetWriter,
    generator: HologramDatasetGenerator | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate and persist a holography dataset using the pipeline.

    Args:
        count: Number of samples to produce.
        config: Holography configuration applied to all samples.
        rng: Random number generator used for stochastic steps.
        writer: Dataset writer responsible for persisting results.
        generator: Optional pre-configured generator to reuse.
        output_dir: Optional output directory override for writer.
    """

    if generator is None:
        generator = HologramDatasetGenerator(
            object_producer=default_object_producer(),
            converter=default_converter(),
        )

    samples = list(generator.generate(count=count, config=config, rng=rng))
    target_dir = output_dir if output_dir is not None else Path("dataset")
    writer.save(samples=samples, output_dir=target_dir)


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

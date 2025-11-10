"""Holography conversion pipeline components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from numpy.random import Generator

from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import available_generators
from hologen.types import (
    ArrayFloat,
    DatasetGenerator,
    DatasetWriter,
    GridSpec,
    HologramSample,
    HolographyConfig,
    HolographyMethod,
    HolographyStrategy,
    NoiseModel,
    ObjectSample,
    ObjectShapeGenerator,
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


@dataclass(slots=True)
class ObjectToHologramConverter:
    """Convert object-domain amplitudes into hologram representations.

    Args:
        strategy_mapping: Mapping from holography methods to strategy implementations.
        noise_model: Optional noise model to apply to generated holograms.
    """

    strategy_mapping: dict[HolographyMethod, HolographyStrategy]
    noise_model: NoiseModel | None = None

    def create_hologram(
        self, sample: ObjectSample, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        """Generate a hologram for the provided object sample.

        Args:
            sample: Object-domain sample to transform.
            config: Holography configuration specifying physical parameters.

        Returns:
            Intensity hologram representation of the sample.
        """

        strategy = self._resolve_strategy(config.method)
        hologram = strategy.create_hologram(sample.pixels, config)
        
        # Apply the noise model if given.
        if self.noise_model is not None and rng is not None:
            hologram = self.noise_model.apply(hologram, config, rng)
        return hologram

    def reconstruct(self, hologram: ArrayFloat, config: HolographyConfig) -> ArrayFloat:
        """Reconstruct an object-domain field from a hologram.

        Args:
            hologram: Intensity hologram generated from an object sample.
            config: Holography configuration specifying physical parameters.

        Returns:
            Reconstructed amplitude distribution of the object domain.
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
        self, count: int, config: HolographyConfig, rng: Generator
    ) -> Iterable[HologramSample]:
        """Yield hologram samples as an iterable sequence.

        Args:
            count: Number of samples to generate.
            config: Holography configuration applied to all samples.
            rng: Random number generator used throughout the pipeline.

        Yields:
            Sequential hologram samples containing object, hologram, and reconstruction data.
        """

        for _ in range(count):
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

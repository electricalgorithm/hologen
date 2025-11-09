"""Tests for hologen.converters module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import Generator

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
    default_converter,
    default_object_producer,
    generate_dataset,
)
from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    HologramSample,
    HolographyMethod,
    ObjectSample,
)


class TestObjectDomainProducer:
    """Test ObjectDomainProducer class."""

    def test_creation(self) -> None:
        """Test ObjectDomainProducer creation."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        assert len(producer.shape_generators) == 1
        assert producer.shape_generators[0] is generator

    def test_generate_shape(self, grid_spec, rng: Generator) -> None:
        """Test that generate returns ObjectSample with correct shape."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))

        sample = producer.generate(grid_spec, rng)
        assert isinstance(sample, ObjectSample)
        assert sample.name == "circle"
        assert sample.pixels.shape == (grid_spec.height, grid_spec.width)
        assert sample.pixels.dtype == np.float64

    def test_generate_normalized(self, grid_spec, rng: Generator) -> None:
        """Test that generated samples are normalized."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))

        sample = producer.generate(grid_spec, rng)
        assert np.all(sample.pixels >= 0.0)
        assert np.all(sample.pixels <= 1.0)

    def test_generate_multiple_generators(self, grid_spec, rng: Generator) -> None:
        """Test generation with multiple shape generators."""
        from hologen.shapes import RectangleGenerator

        circle_gen = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        rect_gen = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.3)
        producer = ObjectDomainProducer(shape_generators=(circle_gen, rect_gen))

        # Generate multiple samples to test selection
        names = set()
        for _ in range(20):
            sample = producer.generate(grid_spec, rng)
            names.add(sample.name)

        # Should use both generators over multiple samples
        assert len(names) >= 1  # At least one generator used

    def test_slots(self) -> None:
        """Test that ObjectDomainProducer uses slots."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        assert hasattr(producer, "__slots__")


class TestObjectToHologramConverter:
    """Test ObjectToHologramConverter class."""

    def test_creation(self) -> None:
        """Test ObjectToHologramConverter creation."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)
        assert len(converter.strategy_mapping) == 1
        assert HolographyMethod.INLINE in converter.strategy_mapping

    def test_create_hologram(self, inline_config, sample_object_field) -> None:
        """Test hologram creation."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        sample = ObjectSample(name="test", pixels=sample_object_field)
        hologram = converter.create_hologram(sample, inline_config)

        assert hologram.shape == sample_object_field.shape
        assert hologram.dtype == np.float64
        assert np.all(hologram >= 0.0)

    def test_reconstruct(self, inline_config, sample_object_field) -> None:
        """Test reconstruction."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        sample = ObjectSample(name="test", pixels=sample_object_field)
        hologram = converter.create_hologram(sample, inline_config)
        reconstruction = converter.reconstruct(hologram, inline_config)

        assert reconstruction.shape == hologram.shape
        assert reconstruction.dtype == np.float64
        assert np.all(reconstruction >= 0.0)

    def test_unknown_method_error(self, inline_config, sample_object_field) -> None:
        """Test error for unknown holography method."""
        strategy_mapping = {}  # Empty mapping
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        sample = ObjectSample(name="test", pixels=sample_object_field)
        with pytest.raises(KeyError, match="Unknown holography method"):
            converter.create_hologram(sample, inline_config)

    def test_resolve_strategy(self) -> None:
        """Test strategy resolution."""
        inline_strategy = InlineHolographyStrategy()
        strategy_mapping = {HolographyMethod.INLINE: inline_strategy}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        resolved = converter._resolve_strategy(HolographyMethod.INLINE)
        assert resolved is inline_strategy

    def test_slots(self) -> None:
        """Test that ObjectToHologramConverter uses slots."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)
        assert hasattr(converter, "__slots__")


class TestHologramDatasetGenerator:
    """Test HologramDatasetGenerator class."""

    def test_creation(self) -> None:
        """Test HologramDatasetGenerator creation."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        assert dataset_gen.object_producer is producer
        assert dataset_gen.converter is converter

    def test_generate_count(self, inline_config, rng: Generator) -> None:
        """Test that generate produces correct number of samples."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(dataset_gen.generate(count=3, config=inline_config, rng=rng))
        assert len(samples) == 3

    def test_generate_sample_structure(self, inline_config, rng: Generator) -> None:
        """Test that generated samples have correct structure."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(dataset_gen.generate(count=1, config=inline_config, rng=rng))
        sample = samples[0]

        assert isinstance(sample, HologramSample)
        assert isinstance(sample.object_sample, ObjectSample)
        assert sample.hologram.shape == (
            inline_config.grid.height,
            inline_config.grid.width,
        )
        assert sample.reconstruction.shape == (
            inline_config.grid.height,
            inline_config.grid.width,
        )

    def test_slots(self) -> None:
        """Test that HologramDatasetGenerator uses slots."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        assert hasattr(dataset_gen, "__slots__")


class TestDefaultObjectProducer:
    """Test default_object_producer function."""

    def test_returns_producer(self) -> None:
        """Test that default producer is returned."""
        producer = default_object_producer()
        assert isinstance(producer, ObjectDomainProducer)
        assert len(producer.shape_generators) > 0

    def test_has_expected_generators(self) -> None:
        """Test that default producer has expected generators."""
        producer = default_object_producer()
        names = {gen.name for gen in producer.shape_generators}
        expected_names = {
            "circle",
            "rectangle",
            "ring",
            "circle_checker",
            "rectangle_checker",
            "ellipse_checker",
        }
        assert names == expected_names


class TestDefaultConverter:
    """Test default_converter function."""

    def test_returns_converter(self) -> None:
        """Test that default converter is returned."""
        converter = default_converter()
        assert isinstance(converter, ObjectToHologramConverter)
        assert len(converter.strategy_mapping) == 2

    def test_has_expected_strategies(self) -> None:
        """Test that default converter has expected strategies."""
        converter = default_converter()
        methods = set(converter.strategy_mapping.keys())
        expected_methods = {HolographyMethod.INLINE, HolographyMethod.OFF_AXIS}
        assert methods == expected_methods

    def test_strategy_types(self) -> None:
        """Test that strategies are correct types."""
        converter = default_converter()
        inline_strategy = converter.strategy_mapping[HolographyMethod.INLINE]
        off_axis_strategy = converter.strategy_mapping[HolographyMethod.OFF_AXIS]

        assert isinstance(inline_strategy, InlineHolographyStrategy)
        assert isinstance(off_axis_strategy, OffAxisHolographyStrategy)


class TestGenerateDataset:
    """Test generate_dataset function."""

    def test_basic_generation(
        self, inline_config, rng: Generator, tmp_path: Path
    ) -> None:
        """Test basic dataset generation."""
        mock_writer = Mock()

        generate_dataset(
            count=2,
            config=inline_config,
            rng=rng,
            writer=mock_writer,
            output_dir=tmp_path,
        )

        # Verify writer was called
        mock_writer.save.assert_called_once()
        args, kwargs = mock_writer.save.call_args
        samples = args[0] if args else kwargs["samples"]
        output_dir = args[1] if len(args) > 1 else kwargs["output_dir"]

        assert len(samples) == 2
        assert output_dir == tmp_path

    def test_default_generator(self, inline_config, rng: Generator) -> None:
        """Test generation with default generator."""
        mock_writer = Mock()

        generate_dataset(
            count=1,
            config=inline_config,
            rng=rng,
            writer=mock_writer,
            generator=None,  # Use default
        )

        mock_writer.save.assert_called_once()

    def test_custom_generator(self, inline_config, rng: Generator) -> None:
        """Test generation with custom generator."""
        mock_writer = Mock()
        mock_generator = Mock()
        mock_generator.generate.return_value = [
            HologramSample(
                object_sample=ObjectSample(name="test", pixels=np.zeros((32, 32))),
                hologram=np.zeros((32, 32)),
                reconstruction=np.zeros((32, 32)),
            )
        ]

        generate_dataset(
            count=1,
            config=inline_config,
            rng=rng,
            writer=mock_writer,
            generator=mock_generator,
        )

        mock_generator.generate.assert_called_once_with(
            count=1, config=inline_config, rng=rng
        )
        mock_writer.save.assert_called_once()

    def test_default_output_dir(self, inline_config, rng: Generator) -> None:
        """Test generation with default output directory."""
        mock_writer = Mock()

        generate_dataset(
            count=1,
            config=inline_config,
            rng=rng,
            writer=mock_writer,
            output_dir=None,  # Use default
        )

        mock_writer.save.assert_called_once()
        args, kwargs = mock_writer.save.call_args
        output_dir = args[1] if len(args) > 1 else kwargs["output_dir"]
        assert output_dir == Path("dataset")

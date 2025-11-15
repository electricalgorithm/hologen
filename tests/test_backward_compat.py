"""Tests for backward compatibility with legacy intensity-only workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.random import Generator

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
    default_converter,
    generate_dataset,
)
from hologen.holography.inline import InlineHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    FieldRepresentation,
    HologramSample,
    HolographyConfig,
    HolographyMethod,
    ObjectSample,
    OutputConfig,
)
from hologen.utils.io import NumpyDatasetWriter, load_complex_sample


class TestLegacyIntensityWorkflow:
    """Test that intensity-only workflow produces expected results."""

    def test_legacy_object_sample_workflow(
        self, inline_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test that legacy ObjectSample workflow still works."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        # Generate with use_complex=False (legacy mode)
        samples = list(
            dataset_gen.generate(
                count=2, config=inline_config, rng=rng, use_complex=False
            )
        )

        assert len(samples) == 2

        for sample in samples:
            assert isinstance(sample, HologramSample)
            assert isinstance(sample.object_sample, ObjectSample)
            assert not np.iscomplexobj(sample.hologram)
            assert not np.iscomplexobj(sample.reconstruction)
            assert sample.hologram.dtype == np.float64
            assert sample.reconstruction.dtype == np.float64

    def test_legacy_numpy_writer(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test that NumpyDatasetWriter still works with legacy samples."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(
            dataset_gen.generate(
                count=1, config=inline_config, rng=rng, use_complex=False
            )
        )

        writer = NumpyDatasetWriter(save_preview=True)
        writer.save(samples, tmp_path)

        # Check files exist
        assert (tmp_path / "sample_00000_circle.npz").exists()
        assert (tmp_path / "sample_00000_circle_object.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram.png").exists()
        assert (tmp_path / "sample_00000_circle_reconstruction.png").exists()

    def test_default_output_config_is_intensity(self) -> None:
        """Test that default OutputConfig uses intensity representation."""
        config = OutputConfig()

        assert config.object_representation == FieldRepresentation.INTENSITY
        assert config.hologram_representation == FieldRepresentation.INTENSITY
        assert config.reconstruction_representation == FieldRepresentation.INTENSITY

    def test_default_converter_has_intensity_config(self) -> None:
        """Test that default converter uses intensity-only output config."""
        converter = default_converter()

        assert (
            converter.output_config.object_representation
            == FieldRepresentation.INTENSITY
        )
        assert (
            converter.output_config.hologram_representation
            == FieldRepresentation.INTENSITY
        )
        assert (
            converter.output_config.reconstruction_representation
            == FieldRepresentation.INTENSITY
        )


class TestLegacyFileLoading:
    """Test loading legacy .npz files with new code."""

    def test_load_legacy_npz_format(self, tmp_path: Path) -> None:
        """Test that legacy .npz files can be loaded."""
        # Create legacy format .npz file
        pixels = np.random.rand(64, 64)
        npz_path = tmp_path / "legacy_sample.npz"
        np.savez(
            npz_path,
            object=pixels,
            hologram=pixels * 0.5,
            reconstruction=pixels * 0.8,
        )

        # Load using new loader
        sample = load_complex_sample(npz_path)

        assert isinstance(sample, ObjectSample)
        assert np.allclose(sample.pixels, pixels)
        assert sample.name == "legacy_sample"

    def test_legacy_npz_without_representation_key(self, tmp_path: Path) -> None:
        """Test loading legacy files that don't have 'representation' key."""
        pixels = np.ones((32, 32), dtype=np.float64)
        npz_path = tmp_path / "old_format.npz"
        np.savez(npz_path, object=pixels)

        sample = load_complex_sample(npz_path)

        assert isinstance(sample, ObjectSample)
        assert np.allclose(sample.pixels, pixels)


class TestBackwardCompatibleDefaults:
    """Test that default arguments maintain backward compatibility."""

    def test_generate_dataset_default_is_intensity(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test that generate_dataset without arguments produces intensity output."""
        writer = NumpyDatasetWriter(save_preview=False)

        # Call with minimal arguments (should default to intensity)
        generate_dataset(
            count=1,
            config=inline_config,
            rng=rng,
            writer=writer,
            output_dir=tmp_path,
        )

        # Load the generated file
        npz_path = tmp_path / "sample_00000_circle.npz"
        assert npz_path.exists()

        data = np.load(npz_path)

        # Should have legacy keys
        assert "object" in data
        assert "hologram" in data
        assert "reconstruction" in data

    def test_converter_without_output_config(
        self, inline_config: HolographyConfig
    ) -> None:
        """Test that converter without output_config defaults to intensity."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        # Should have default intensity config
        assert (
            converter.output_config.object_representation
            == FieldRepresentation.INTENSITY
        )


class TestLegacyObjectSampleStillWorks:
    """Test that legacy ObjectSample and HologramSample still work."""

    def test_object_sample_creation(self) -> None:
        """Test that ObjectSample can still be created."""
        pixels = np.ones((64, 64), dtype=np.float64)
        sample = ObjectSample(name="test", pixels=pixels)

        assert sample.name == "test"
        assert np.array_equal(sample.pixels, pixels)

    def test_hologram_sample_creation(self) -> None:
        """Test that HologramSample can still be created."""
        pixels = np.ones((64, 64), dtype=np.float64)
        object_sample = ObjectSample(name="test", pixels=pixels)

        hologram_sample = HologramSample(
            object_sample=object_sample,
            hologram=pixels * 0.5,
            reconstruction=pixels * 0.8,
        )

        assert hologram_sample.object_sample is object_sample
        assert hologram_sample.hologram.shape == (64, 64)
        assert hologram_sample.reconstruction.shape == (64, 64)

    def test_legacy_sample_with_converter(
        self, inline_config: HolographyConfig
    ) -> None:
        """Test that legacy ObjectSample works with converter."""
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        # Create legacy sample
        pixels = np.ones(
            (inline_config.grid.height, inline_config.grid.width), dtype=np.float64
        )
        legacy_sample = ObjectSample(name="test", pixels=pixels)

        # Should work with converter
        hologram = converter.create_hologram(legacy_sample, inline_config)

        assert isinstance(hologram, np.ndarray)
        assert hologram.dtype == np.float64
        assert not np.iscomplexobj(hologram)


class TestNoRegressions:
    """Test that existing functionality hasn't regressed."""

    def test_intensity_values_in_valid_range(
        self, inline_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test that intensity values are non-negative."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(strategy_mapping=strategy_mapping)

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(
            dataset_gen.generate(
                count=1, config=inline_config, rng=rng, use_complex=False
            )
        )

        sample = samples[0]

        # All intensity values should be non-negative
        assert np.all(sample.object_sample.pixels >= 0.0)
        assert np.all(sample.hologram >= 0.0)
        assert np.all(sample.reconstruction >= 0.0)

    def test_shape_generators_produce_binary_output(
        self, grid_spec, rng: Generator
    ) -> None:
        """Test that shape generators still produce binary (0 or 1) output."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)

        field = generator.generate(grid_spec, rng)

        # Should be binary values
        unique_values = np.unique(field)
        assert len(unique_values) <= 2
        assert np.all((field == 0.0) | (field == 1.0))

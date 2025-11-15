"""Integration tests for end-to-end complex field pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
)
from hologen.holography.inline import InlineHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    ComplexHologramSample,
    FieldRepresentation,
    HolographyConfig,
    HolographyMethod,
    OutputConfig,
)
from hologen.utils.io import ComplexFieldWriter


class TestEndToEndPipeline:
    """Test full pipeline from complex object to hologram to reconstruction."""

    def test_complex_object_to_hologram_to_reconstruction(
        self, inline_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test complete pipeline with complex objects."""
        # Setup pipeline
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.AMPLITUDE,
                hologram_representation=FieldRepresentation.COMPLEX,
                reconstruction_representation=FieldRepresentation.COMPLEX,
            ),
        )
        
        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        
        # Generate samples
        samples = list(
            dataset_gen.generate(count=2, config=inline_config, rng=rng, use_complex=True)
        )
        
        assert len(samples) == 2
        
        for sample in samples:
            assert isinstance(sample, ComplexHologramSample)
            assert np.iscomplexobj(sample.hologram_field)
            assert np.iscomplexobj(sample.reconstruction_field)
            assert sample.hologram_field.shape == (
                inline_config.grid.height,
                inline_config.grid.width,
            )

    def test_phase_only_object_pipeline(
        self, inline_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test pipeline with phase-only objects."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.PHASE,
                hologram_representation=FieldRepresentation.COMPLEX,
                reconstruction_representation=FieldRepresentation.COMPLEX,
            ),
        )
        
        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        
        samples = list(
            dataset_gen.generate(
                count=1,
                config=inline_config,
                rng=rng,
                use_complex=True,
                mode="phase",
                phase_shift=np.pi / 2,
            )
        )
        
        sample = samples[0]
        
        # Object should have uniform amplitude
        object_amplitude = np.abs(sample.object_sample.field)
        assert np.allclose(object_amplitude, 1.0)


class TestOutputFileNaming:
    """Test that output files are created with correct naming conventions."""

    def test_complex_output_file_naming(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test that complex field output files have correct names."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.COMPLEX,
                hologram_representation=FieldRepresentation.COMPLEX,
                reconstruction_representation=FieldRepresentation.COMPLEX,
            ),
        )
        
        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        
        samples = list(
            dataset_gen.generate(count=1, config=inline_config, rng=rng, use_complex=True)
        )
        
        writer = ComplexFieldWriter(save_preview=True, phase_colormap=None)
        writer.save(samples, tmp_path)
        
        # Check .npz files
        assert (tmp_path / "sample_00000_circle_object.npz").exists()
        assert (tmp_path / "sample_00000_circle_hologram.npz").exists()
        assert (tmp_path / "sample_00000_circle_reconstruction.npz").exists()
        
        # Check PNG files (amplitude and phase for complex representation)
        assert (tmp_path / "sample_00000_circle_object_amplitude.png").exists()
        assert (tmp_path / "sample_00000_circle_object_phase.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram_amplitude.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram_phase.png").exists()

    def test_amplitude_output_file_naming(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test that amplitude output files have correct names."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.AMPLITUDE,
                hologram_representation=FieldRepresentation.AMPLITUDE,
                reconstruction_representation=FieldRepresentation.AMPLITUDE,
            ),
        )
        
        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        
        samples = list(
            dataset_gen.generate(count=1, config=inline_config, rng=rng, use_complex=True)
        )
        
        writer = ComplexFieldWriter(save_preview=True)
        writer.save(samples, tmp_path)
        
        # Check PNG files (single file for amplitude representation)
        assert (tmp_path / "sample_00000_circle_object.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram.png").exists()
        assert (tmp_path / "sample_00000_circle_reconstruction.png").exists()


class TestNoiseModelWithComplexFields:
    """Test noise model application to complex fields."""

    def test_noise_preserves_phase(
        self, inline_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test that noise model preserves phase information."""
        from hologen.noise.sensor import SensorNoiseModel
        
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        
        # Add noise model
        noise_model = SensorNoiseModel(
            name="sensor",
            read_noise=0.01,
            shot_noise=False,
            dark_current=0.0,
            bit_depth=None,
        )
        
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            noise_model=noise_model,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.AMPLITUDE,
                hologram_representation=FieldRepresentation.COMPLEX,
                reconstruction_representation=FieldRepresentation.COMPLEX,
            ),
        )
        
        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )
        
        samples = list(
            dataset_gen.generate(count=1, config=inline_config, rng=rng, use_complex=True)
        )
        
        sample = samples[0]
        
        # Hologram should still be complex after noise application
        assert np.iscomplexobj(sample.hologram_field)
        
        # Phase should still have variation
        phase_var = np.var(np.angle(sample.hologram_field))
        assert phase_var > 0

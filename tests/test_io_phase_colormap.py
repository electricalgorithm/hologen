"""Tests for phase colormap functionality in ComplexFieldWriter."""

from __future__ import annotations

from pathlib import Path

from numpy.random import Generator

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
)
from hologen.holography.inline import InlineHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    FieldRepresentation,
    HolographyConfig,
    HolographyMethod,
    OutputConfig,
)
from hologen.utils.io import ComplexFieldWriter


class TestPhaseColormap:
    """Test phase colormap functionality."""

    def test_phase_colormap_with_matplotlib(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test phase colormap with matplotlib installed."""
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
                count=1, config=inline_config, rng=rng, use_complex=True, mode="phase"
            )
        )

        # Test with twilight colormap (default)
        writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
        writer.save(samples, tmp_path)

        # Check that phase PNG was created
        assert (tmp_path / "sample_00000_circle_object.png").exists()

    def test_phase_only_representation(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test saving phase-only representation."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.PHASE,
                hologram_representation=FieldRepresentation.PHASE,
                reconstruction_representation=FieldRepresentation.PHASE,
            ),
        )

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(
            dataset_gen.generate(
                count=1, config=inline_config, rng=rng, use_complex=True, mode="phase"
            )
        )

        writer = ComplexFieldWriter(save_preview=True, phase_colormap="hsv")
        writer.save(samples, tmp_path)

        # Check that phase PNGs were created
        assert (tmp_path / "sample_00000_circle_object.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram.png").exists()
        assert (tmp_path / "sample_00000_circle_reconstruction.png").exists()

    def test_intensity_representation(
        self, inline_config: HolographyConfig, rng: Generator, tmp_path: Path
    ) -> None:
        """Test saving intensity representation."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        strategy_mapping = {HolographyMethod.INLINE: InlineHolographyStrategy()}
        converter = ObjectToHologramConverter(
            strategy_mapping=strategy_mapping,
            output_config=OutputConfig(
                object_representation=FieldRepresentation.INTENSITY,
                hologram_representation=FieldRepresentation.INTENSITY,
                reconstruction_representation=FieldRepresentation.INTENSITY,
            ),
        )

        dataset_gen = HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        samples = list(
            dataset_gen.generate(
                count=1, config=inline_config, rng=rng, use_complex=True
            )
        )

        writer = ComplexFieldWriter(save_preview=True)
        writer.save(samples, tmp_path)

        # Check that intensity PNGs were created
        assert (tmp_path / "sample_00000_circle_object.png").exists()
        assert (tmp_path / "sample_00000_circle_hologram.png").exists()
        assert (tmp_path / "sample_00000_circle_reconstruction.png").exists()

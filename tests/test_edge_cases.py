"""Tests for edge cases to achieve 100% coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator

from hologen.converters import ObjectDomainProducer
from hologen.holography.propagation import angular_spectrum_propagate
from hologen.shapes import (
    CircleCheckerGenerator,
    EllipseCheckerGenerator,
    TriangleCheckerGenerator,
)
from hologen.types import GridSpec, OpticalConfig
from hologen.utils.io import ComplexFieldWriter


class TestConvertersEdgeCases:
    """Test edge cases in converters module."""

    def test_generate_complex_with_invalid_mode(self, rng: Generator) -> None:
        """Test generate_complex with mode that's neither amplitude nor phase."""
        from hologen.shapes import CircleGenerator

        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        producer = ObjectDomainProducer(shape_generators=(generator,))
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)

        # The else branch in converters.py line 97 is for when mode is neither
        # "amplitude" nor "phase", but shapes.py validates this and raises ValueError
        # So we test that the validation works
        with pytest.raises(ValueError, match="Invalid mode"):
            producer.generate_complex(grid, rng, phase_shift=0.0, mode="invalid")


class TestPropagationEdgeCases:
    """Test edge cases in propagation module."""

    def test_nyquist_criterion_violation(self) -> None:
        """Test that Nyquist criterion violation raises ValueError."""
        field = np.ones((64, 64), dtype=np.complex128)

        # Create a grid with pixel pitch that violates Nyquist criterion
        # The criterion is: max_spatial_freq * wavelength < 1.0
        # where max_spatial_freq = 1 / (2 * pixel_pitch)
        # So: wavelength / (2 * pixel_pitch) >= 1.0 violates
        # Or: pixel_pitch >= wavelength / 2
        # For wavelength 532nm, we need pixel_pitch >= 266nm to violate
        # Use exactly wavelength/2 to trigger the >= condition
        wavelength = 532e-9
        grid = GridSpec(height=64, width=64, pixel_pitch=wavelength / 2)
        optics = OpticalConfig(wavelength=wavelength, propagation_distance=0.01)

        with pytest.raises(ValueError, match="Nyquist criterion violated"):
            angular_spectrum_propagate(field, grid, optics, distance=0.01)


class TestShapesEdgeCases:
    """Test edge cases in shapes module."""

    def test_circle_checker_empty_mask(self, rng: Generator) -> None:
        """Test CircleCheckerGenerator with parameters that create empty mask."""
        generator = CircleCheckerGenerator(
            name="circle_checker", min_radius=0.0, max_radius=0.001, checker_size=8
        )
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)

        # Generate with very small radius that might create empty mask
        result = generator.generate(grid, rng)
        assert result.shape == (64, 64)
        assert np.all((result == 0.0) | (result == 1.0))

    def test_ellipse_checker_empty_mask(self, rng: Generator) -> None:
        """Test EllipseCheckerGenerator with parameters that create empty mask."""
        generator = EllipseCheckerGenerator(
            name="ellipse_checker",
            min_radius_y=0.0,
            max_radius_y=0.001,
            min_radius_x=0.0,
            max_radius_x=0.001,
            checker_size=8,
        )
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)

        # Generate with very small radii that might create empty mask
        result = generator.generate(grid, rng)
        assert result.shape == (64, 64)
        assert np.all((result == 0.0) | (result == 1.0))

    def test_triangle_checker_empty_mask(self, rng: Generator) -> None:
        """Test TriangleCheckerGenerator with parameters that create empty mask."""
        generator = TriangleCheckerGenerator(
            name="triangle_checker", min_scale=0.0, max_scale=0.001, checker_size=8
        )
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)

        # Generate with very small scale that might create empty mask
        result = generator.generate(grid, rng)
        assert result.shape == (64, 64)
        assert np.all((result == 0.0) | (result == 1.0))

    def test_triangle_checker_clipped_vertices(self, rng: Generator) -> None:
        """Test TriangleCheckerGenerator with vertices that get clipped."""
        generator = TriangleCheckerGenerator(
            name="triangle_checker", min_scale=0.5, max_scale=0.8, checker_size=4
        )
        grid = GridSpec(height=32, width=32, pixel_pitch=5e-6)

        # Generate multiple times to increase chance of hitting edge cases
        for _ in range(10):
            result = generator.generate(grid, rng)
            assert result.shape == (32, 32)
            assert np.all((result == 0.0) | (result == 1.0))

    def test_rectangle_checker_invalid_bounds(self, rng: Generator) -> None:
        """Test RectangleCheckerGenerator with invalid bounds."""
        from hologen.shapes import RectangleCheckerGenerator

        generator = RectangleCheckerGenerator(
            name="rectangle_checker", min_scale=0.0, max_scale=0.001, checker_size=8
        )
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)

        # Generate with very small scale that creates invalid bounds
        result = generator.generate(grid, rng)
        assert result.shape == (64, 64)
        assert np.all((result == 0.0) | (result == 1.0))


class TestIOEdgeCases:
    """Test edge cases in IO module."""

    def test_phase_colormap_without_matplotlib(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test phase colormap fallback when matplotlib is not available."""
        import builtins

        from hologen.types import ComplexObjectSample, FieldRepresentation

        # Save original import
        original_import = builtins.__import__

        # Create a mock that raises ImportError for matplotlib
        def mock_import(name, *args, **kwargs):
            if "matplotlib" in name:
                raise ImportError("matplotlib not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Create a sample with phase representation
        field = np.exp(1j * np.random.uniform(-np.pi, np.pi, (32, 32)))
        sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.PHASE
        )

        # Create writer with colormap (should fall back to grayscale)
        writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")

        # Save should work even without matplotlib
        from hologen.types import ComplexHologramSample

        hologram_sample = ComplexHologramSample(
            object_sample=sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.PHASE,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.PHASE,
        )

        writer.save([hologram_sample], tmp_path)

        # Check that files were created (using grayscale fallback)
        assert (tmp_path / "sample_00000_test_object.png").exists()

    def test_phase_colormap_with_different_colormaps(
        self, inline_config, rng: Generator, tmp_path: Path
    ) -> None:
        """Test phase colormap with various matplotlib colormaps."""
        from hologen.converters import (
            HologramDatasetGenerator,
            ObjectDomainProducer,
            ObjectToHologramConverter,
        )
        from hologen.holography.inline import InlineHolographyStrategy
        from hologen.shapes import CircleGenerator
        from hologen.types import (
            FieldRepresentation,
            HolographyMethod,
            OutputConfig,
        )

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

        # Test with different colormaps
        for colormap in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            output_dir = tmp_path / colormap
            writer = ComplexFieldWriter(save_preview=True, phase_colormap=colormap)
            writer.save(samples, output_dir)

            # Verify files were created
            assert (output_dir / "sample_00000_circle_object.png").exists()

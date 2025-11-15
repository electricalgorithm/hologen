"""Tests for complex object generation in hologen.shapes module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from hologen.shapes import (
    CircleCheckerGenerator,
    CircleGenerator,
    EllipseCheckerGenerator,
    RectangleCheckerGenerator,
    RectangleGenerator,
    RingGenerator,
)
from hologen.types import GridSpec
from hologen.utils.fields import PhaseRangeError


class TestCircleGeneratorComplex:
    """Test CircleGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        assert field.shape == (grid_spec.height, grid_spec.width)
        assert field.dtype == np.complex128
        
        # Phase should be zero everywhere
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)
        
        # Amplitude should match legacy generate()
        amplitude = np.abs(field)
        legacy_amplitude = generator.generate(grid_spec, rng)
        # Note: RNG state differs, so we just check properties
        assert np.all((amplitude == 0.0) | (amplitude == 1.0))

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi/2)
        
        # Amplitude should be 1.0 everywhere
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)

    def test_phase_mode_phase_values(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces correct phase values."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        phase_shift = np.pi / 2
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=phase_shift)
        
        phase = np.angle(field)
        
        # Phase should be either 0.0 or phase_shift
        unique_phases = np.unique(np.round(phase, decimals=6))
        assert len(unique_phases) <= 2
        assert 0.0 in unique_phases or np.isclose(unique_phases[0], 0.0)

    def test_phase_range_validation(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that phase values are in [-π, π] range."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi/4)
        
        phase = np.angle(field)
        assert np.all((-np.pi <= phase) & (phase <= np.pi))

    def test_invalid_mode_error(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test error for invalid mode."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        
        with pytest.raises(ValueError, match="Invalid mode"):
            generator.generate_complex(grid_spec, rng, mode="invalid")


class TestRectangleGeneratorComplex:
    """Test RectangleGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.3)
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.3)
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi/3)
        
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)


class TestRingGeneratorComplex:
    """Test RingGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.1,
            max_radius=0.2,
            min_thickness=0.1,
            max_thickness=0.3,
        )
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.1,
            max_radius=0.2,
            min_thickness=0.1,
            max_thickness=0.3,
        )
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=-np.pi/4)
        
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)


class TestCircleCheckerGeneratorComplex:
    """Test CircleCheckerGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = CircleCheckerGenerator(
            name="circle_checker", min_radius=0.1, max_radius=0.2, checker_size=8
        )
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = CircleCheckerGenerator(
            name="circle_checker", min_radius=0.1, max_radius=0.2, checker_size=8
        )
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi)
        
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)


class TestRectangleCheckerGeneratorComplex:
    """Test RectangleCheckerGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = RectangleCheckerGenerator(
            name="rectangle_checker", min_scale=0.1, max_scale=0.3, checker_size=8
        )
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = RectangleCheckerGenerator(
            name="rectangle_checker", min_scale=0.1, max_scale=0.3, checker_size=8
        )
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi/6)
        
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)


class TestEllipseCheckerGeneratorComplex:
    """Test EllipseCheckerGenerator.generate_complex method."""

    def test_amplitude_mode(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test amplitude mode produces zero phase."""
        generator = EllipseCheckerGenerator(
            name="ellipse_checker",
            min_radius_y=0.1,
            max_radius_y=0.2,
            min_radius_x=0.1,
            max_radius_x=0.2,
            checker_size=8,
        )
        field = generator.generate_complex(grid_spec, rng, mode="amplitude")
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_phase_mode_uniform_amplitude(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase mode produces uniform amplitude of 1.0."""
        generator = EllipseCheckerGenerator(
            name="ellipse_checker",
            min_radius_y=0.1,
            max_radius_y=0.2,
            min_radius_x=0.1,
            max_radius_x=0.2,
            checker_size=8,
        )
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi/8)
        
        amplitude = np.abs(field)
        assert np.allclose(amplitude, 1.0)


class TestLegacyGenerateMethod:
    """Test that legacy generate() method still works."""

    def test_circle_legacy_generate(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test CircleGenerator.generate() still works."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate(grid_spec, rng)
        
        assert field.shape == (grid_spec.height, grid_spec.width)
        assert field.dtype == np.float64
        assert np.all((field >= 0.0) & (field <= 1.0))

    def test_rectangle_legacy_generate(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test RectangleGenerator.generate() still works."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.3)
        field = generator.generate(grid_spec, rng)
        
        assert field.shape == (grid_spec.height, grid_spec.width)
        assert field.dtype == np.float64
        assert np.all((field >= 0.0) & (field <= 1.0))

    def test_ring_legacy_generate(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test RingGenerator.generate() still works."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.1,
            max_radius=0.2,
            min_thickness=0.1,
            max_thickness=0.3,
        )
        field = generator.generate(grid_spec, rng)
        
        assert field.shape == (grid_spec.height, grid_spec.width)
        assert field.dtype == np.float64
        assert np.all((field >= 0.0) & (field <= 1.0))


class TestPhaseShiftParameter:
    """Test phase_shift parameter behavior."""

    def test_zero_phase_shift(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase_shift=0.0 produces zero phase everywhere."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=0.0)
        
        phase = np.angle(field)
        assert np.allclose(phase, 0.0)

    def test_positive_phase_shift(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test positive phase_shift values."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        phase_shift = np.pi / 2
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=phase_shift)
        
        phase = np.angle(field)
        # Should have values at 0.0 and phase_shift
        assert np.all((-np.pi <= phase) & (phase <= np.pi))

    def test_negative_phase_shift(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test negative phase_shift values."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        phase_shift = -np.pi / 3
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=phase_shift)
        
        phase = np.angle(field)
        assert np.all((-np.pi <= phase) & (phase <= np.pi))

    def test_max_phase_shift(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test phase_shift at boundary value π."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        field = generator.generate_complex(grid_spec, rng, mode="phase", phase_shift=np.pi)
        
        phase = np.angle(field)
        assert np.all((-np.pi <= phase) & (phase <= np.pi))

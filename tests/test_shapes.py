"""Tests for hologen.shapes module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from hologen.shapes import (
    BaseShapeGenerator,
    CircleGenerator,
    RectangleGenerator,
    RingGenerator,
    available_generators,
)
from hologen.types import GridSpec


class TestBaseShapeGenerator:
    """Test BaseShapeGenerator abstract base class."""

    def test_abstract_generate(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that generate method raises NotImplementedError."""
        generator = BaseShapeGenerator(name="test")
        with pytest.raises(NotImplementedError):
            generator.generate(grid_spec, rng)

    def test_name_property(self) -> None:
        """Test name property."""
        generator = BaseShapeGenerator(name="test_shape")
        assert generator.name == "test_shape"

    def test_empty_canvas(self, grid_spec: GridSpec) -> None:
        """Test _empty_canvas method."""
        generator = BaseShapeGenerator(name="test")
        canvas = generator._empty_canvas(grid_spec)
        assert canvas.shape == (grid_spec.height, grid_spec.width)
        assert canvas.dtype == np.float64
        assert np.all(canvas == 0.0)

    def test_clamp_method(self) -> None:
        """Test _clamp method."""
        generator = BaseShapeGenerator(name="test")
        canvas = np.array([[-1.0, 0.5], [1.5, 2.0]], dtype=np.float64)
        clamped = generator._clamp(canvas)
        expected = np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float64)
        np.testing.assert_array_equal(clamped, expected)
        assert clamped is canvas  # Should modify in-place

    def test_slots(self) -> None:
        """Test that BaseShapeGenerator uses slots."""
        generator = BaseShapeGenerator(name="test")
        assert hasattr(generator, "__slots__")


class TestCircleGenerator:
    """Test CircleGenerator class."""

    def test_creation(self) -> None:
        """Test CircleGenerator creation."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.3)
        assert generator.name == "circle"
        assert generator.min_radius == pytest.approx(0.1)
        assert generator.max_radius == pytest.approx(0.3)

    def test_generate_shape(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test circle generation produces correct shape."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        result = generator.generate(grid_spec, rng)
        assert result.shape == (grid_spec.height, grid_spec.width)
        assert result.dtype == np.float64

    def test_generate_binary_values(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that generated circle has binary values."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        result = generator.generate(grid_spec, rng)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert 0.0 in unique_values
        if len(unique_values) == 2:
            assert 1.0 in unique_values

    def test_generate_deterministic(self, grid_spec: GridSpec) -> None:
        """Test that same seed produces same result."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)
        result1 = generator.generate(grid_spec, rng1)
        result2 = generator.generate(grid_spec, rng2)
        np.testing.assert_array_equal(result1, result2)

    def test_radius_bounds(self, grid_spec: GridSpec) -> None:
        """Test that radius is within specified bounds."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.1)
        rng = np.random.default_rng(seed=42)
        result = generator.generate(grid_spec, rng)
        # With fixed radius, should produce consistent results
        assert np.any(result == 1.0)  # Should have some filled pixels

    def test_slots(self) -> None:
        """Test that CircleGenerator uses slots."""
        generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
        assert hasattr(generator, "__slots__")


class TestRectangleGenerator:
    """Test RectangleGenerator class."""

    def test_creation(self) -> None:
        """Test RectangleGenerator creation."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.4)
        assert generator.name == "rectangle"
        assert generator.min_scale == pytest.approx(0.1)
        assert generator.max_scale == pytest.approx(0.4)

    def test_generate_shape(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test rectangle generation produces correct shape."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.2, max_scale=0.3)
        result = generator.generate(grid_spec, rng)
        assert result.shape == (grid_spec.height, grid_spec.width)
        assert result.dtype == np.float64

    def test_generate_binary_values(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that generated rectangle has binary values."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.2, max_scale=0.3)
        result = generator.generate(grid_spec, rng)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert 0.0 in unique_values
        if len(unique_values) == 2:
            assert 1.0 in unique_values

    def test_generate_deterministic(self, grid_spec: GridSpec) -> None:
        """Test that same seed produces same result."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.2, max_scale=0.3)
        rng1 = np.random.default_rng(seed=456)
        rng2 = np.random.default_rng(seed=456)
        result1 = generator.generate(grid_spec, rng1)
        result2 = generator.generate(grid_spec, rng2)
        np.testing.assert_array_equal(result1, result2)

    def test_slots(self) -> None:
        """Test that RectangleGenerator uses slots."""
        generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.3)
        assert hasattr(generator, "__slots__")


class TestRingGenerator:
    """Test RingGenerator class."""

    def test_creation(self) -> None:
        """Test RingGenerator creation."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.1,
            max_radius=0.3,
            min_thickness=0.1,
            max_thickness=0.2,
        )
        assert generator.name == "ring"
        assert generator.min_radius == pytest.approx(0.1)
        assert generator.max_radius == pytest.approx(0.3)
        assert generator.min_thickness == pytest.approx(0.1)
        assert generator.max_thickness == pytest.approx(0.2)

    def test_generate_shape(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test ring generation produces correct shape."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.15,
            max_radius=0.25,
            min_thickness=0.1,
            max_thickness=0.2,
        )
        result = generator.generate(grid_spec, rng)
        assert result.shape == (grid_spec.height, grid_spec.width)
        assert result.dtype == np.float64

    def test_generate_binary_values(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that generated ring has binary values."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.15,
            max_radius=0.25,
            min_thickness=0.1,
            max_thickness=0.2,
        )
        result = generator.generate(grid_spec, rng)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert 0.0 in unique_values
        if len(unique_values) == 2:
            assert 1.0 in unique_values

    def test_generate_deterministic(self, grid_spec: GridSpec) -> None:
        """Test that same seed produces same result."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.15,
            max_radius=0.25,
            min_thickness=0.1,
            max_thickness=0.2,
        )
        rng1 = np.random.default_rng(seed=789)
        rng2 = np.random.default_rng(seed=789)
        result1 = generator.generate(grid_spec, rng1)
        result2 = generator.generate(grid_spec, rng2)
        np.testing.assert_array_equal(result1, result2)

    def test_inner_radius_constraint(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that inner radius is constrained to minimum value."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.01,
            max_radius=0.02,
            min_thickness=0.9,
            max_thickness=1.0,
        )
        result = generator.generate(grid_spec, rng)
        # Should still produce valid output even with extreme thickness
        assert result.shape == (grid_spec.height, grid_spec.width)

    def test_slots(self) -> None:
        """Test that RingGenerator uses slots."""
        generator = RingGenerator(
            name="ring",
            min_radius=0.1,
            max_radius=0.2,
            min_thickness=0.1,
            max_thickness=0.2,
        )
        assert hasattr(generator, "__slots__")


class TestAvailableGenerators:
    """Test available_generators function."""

    def test_returns_generators(self) -> None:
        """Test that available_generators returns generator instances."""
        generators = list(available_generators())
        assert len(generators) == 3

        names = [gen.name for gen in generators]
        assert "circle" in names
        assert "rectangle" in names
        assert "ring" in names

    def test_generator_types(self) -> None:
        """Test that returned generators are correct types."""
        generators = list(available_generators())

        circle_gen = next(gen for gen in generators if gen.name == "circle")
        rectangle_gen = next(gen for gen in generators if gen.name == "rectangle")
        ring_gen = next(gen for gen in generators if gen.name == "ring")

        assert isinstance(circle_gen, CircleGenerator)
        assert isinstance(rectangle_gen, RectangleGenerator)
        assert isinstance(ring_gen, RingGenerator)

    def test_generator_parameters(self) -> None:
        """Test that generators have reasonable parameters."""
        generators = list(available_generators())

        for gen in generators:
            if isinstance(gen, CircleGenerator):
                assert 0 < gen.min_radius < gen.max_radius < 1
            elif isinstance(gen, RectangleGenerator):
                assert 0 < gen.min_scale < gen.max_scale < 1
            elif isinstance(gen, RingGenerator):
                assert 0 < gen.min_radius < gen.max_radius < 1
                assert 0 < gen.min_thickness < gen.max_thickness < 1

    def test_generators_functional(self, grid_spec: GridSpec, rng: Generator) -> None:
        """Test that all generators can produce valid output."""
        generators = list(available_generators())

        for gen in generators:
            result = gen.generate(grid_spec, rng)
            assert result.shape == (grid_spec.height, grid_spec.width)
            assert result.dtype == np.float64
            assert np.all((result >= 0.0) & (result <= 1.0))

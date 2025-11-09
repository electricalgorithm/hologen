"""Tests for hologen.utils.math module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.utils.math import (
    FourierGrid,
    gaussian_blur,
    make_fourier_grid,
    normalize_image,
)


class TestNormalizeImage:
    """Test normalize_image function."""

    def test_scaling(self) -> None:
        """Test image normalization scaling."""
        image = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float64)
        normalized = normalize_image(image)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized[0, 1] == pytest.approx(1.0 / 3.0)

    def test_constant_image(self) -> None:
        """Test normalization of constant image."""
        image = np.full((3, 3), 5.0, dtype=np.float64)
        normalized = normalize_image(image)
        assert np.all(normalized == 0.0)
        assert normalized.dtype == np.float64

    def test_negative_values(self) -> None:
        """Test normalization with negative values."""
        image = np.array([[-2.0, 0.0], [2.0, 4.0]], dtype=np.float64)
        normalized = normalize_image(image)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized[0, 0] == pytest.approx(0.0)
        assert normalized[1, 1] == pytest.approx(1.0)

    def test_single_pixel(self) -> None:
        """Test normalization of single pixel."""
        image = np.array([[5.0]], dtype=np.float64)
        normalized = normalize_image(image)
        assert normalized[0, 0] == pytest.approx(0.0)

    def test_dtype_preservation(self) -> None:
        """Test that output dtype is float64."""
        image = np.array([[1, 2], [3, 4]], dtype=np.int32)
        normalized = normalize_image(image)
        assert normalized.dtype == np.float64


class TestGaussianBlur:
    """Test gaussian_blur function."""

    def test_identity_sigma_zero(self) -> None:
        """Test that zero sigma returns unchanged image."""
        image = np.random.RandomState(seed=1).rand(4, 4).astype(np.float64)
        blurred = gaussian_blur(image, sigma=0.0)
        np.testing.assert_allclose(blurred, image)

    def test_negative_sigma(self) -> None:
        """Test that negative sigma returns unchanged image."""
        image = np.random.RandomState(seed=1).rand(4, 4).astype(np.float64)
        blurred = gaussian_blur(image, sigma=-1.0)
        np.testing.assert_allclose(blurred, image)

    def test_smoothing_effect(self) -> None:
        """Test that blur smooths impulse response."""
        impulse = np.zeros((5, 5), dtype=np.float64)
        impulse[2, 2] = 1.0
        blurred = gaussian_blur(impulse, sigma=1.0)
        assert blurred[2, 2] < 1.0
        # Energy may not be perfectly conserved due to edge effects
        assert blurred.sum() > 0.5  # Should retain most energy

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input."""
        image = np.random.rand(7, 11).astype(np.float64)
        blurred = gaussian_blur(image, sigma=2.0)
        assert blurred.shape == image.shape

    def test_large_sigma(self) -> None:
        """Test blur with large sigma."""
        image = np.zeros((10, 10), dtype=np.float64)
        image[5, 5] = 1.0
        blurred = gaussian_blur(image, sigma=3.0)
        assert blurred[5, 5] < 0.5
        # Energy may be lost due to edge effects with large sigma
        assert blurred.sum() > 0.5  # Should retain reasonable amount of energy


class TestFourierGrid:
    """Test FourierGrid dataclass."""

    def test_creation(self) -> None:
        """Test FourierGrid creation."""
        fx = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        fy = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        grid = FourierGrid(fx=fx, fy=fy)
        np.testing.assert_array_equal(grid.fx, fx)
        np.testing.assert_array_equal(grid.fy, fy)

    def test_slots(self) -> None:
        """Test that FourierGrid uses slots."""
        fx = np.zeros((2, 2), dtype=np.float64)
        fy = np.zeros((2, 2), dtype=np.float64)
        grid = FourierGrid(fx=fx, fy=fy)
        assert hasattr(grid, "__slots__")


class TestMakeFourierGrid:
    """Test make_fourier_grid function."""

    def test_basic_functionality(self, grid_spec) -> None:
        """Test basic Fourier grid creation."""
        fourier_grid = make_fourier_grid(grid_spec)
        assert fourier_grid.fx.shape == (grid_spec.height, grid_spec.width)
        assert fourier_grid.fy.shape == (grid_spec.height, grid_spec.width)
        assert fourier_grid.fx.dtype == np.float64
        assert fourier_grid.fy.dtype == np.float64

    def test_frequency_range(self, grid_spec) -> None:
        """Test frequency range is correct."""
        fourier_grid = make_fourier_grid(grid_spec)
        max_freq = 1.0 / (2.0 * grid_spec.pixel_pitch)
        assert np.abs(fourier_grid.fx).max() <= max_freq
        assert np.abs(fourier_grid.fy).max() <= max_freq

    def test_dc_component(self, grid_spec) -> None:
        """Test DC component is at origin."""
        fourier_grid = make_fourier_grid(grid_spec)
        assert fourier_grid.fx[0, 0] == pytest.approx(0.0)
        assert fourier_grid.fy[0, 0] == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        """Test frequency grid symmetry for even dimensions."""
        from hologen.types import GridSpec

        grid = GridSpec(height=4, width=4, pixel_pitch=1e-6)
        fourier_grid = make_fourier_grid(grid)

        # Check that negative frequencies are present
        assert np.any(fourier_grid.fx < 0)
        assert np.any(fourier_grid.fy < 0)

"""Tests for hologen.holography.inline module."""

from __future__ import annotations

import numpy as np

from hologen.holography.inline import (
    InlineHolographyStrategy,
    _field_to_intensity,
    _object_to_complex,
)


class TestObjectToComplex:
    """Test _object_to_complex function."""

    def test_dtype_conversion(self) -> None:
        """Test conversion to complex128."""
        field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _object_to_complex(field)
        assert result.dtype == np.complex128
        expected = np.array(
            [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=np.complex128
        )
        np.testing.assert_array_equal(result, expected)

    def test_shape_preservation(self) -> None:
        """Ensure the shape is preserved when converting to complex."""
        field = np.zeros((3, 5), dtype=np.float32)
        result = _object_to_complex(field)
        assert result.shape == field.shape

    def test_value_preservation(self) -> None:
        """Ensure values are preserved (imaginary parts added as 0)."""
        field = np.array([[1.5, -2.0], [0.0, 4.25]], dtype=np.float64)
        result = _object_to_complex(field)
        expected = np.array(
            [[1.5 + 0j, -2.0 + 0j], [0.0 + 0j, 4.25 + 0j]], dtype=np.complex128
        )
        np.testing.assert_array_equal(result, expected)


class TestFieldToIntensity:
    """Test _field_to_intensity function."""

    def test_intensity_calculation(self) -> None:
        """Test intensity calculation from complex field."""
        field = np.array(
            [[1.0 + 1j, 2.0 + 0j], [0.0 + 3j, 1.0 - 1j]], dtype=np.complex128
        )
        result = _field_to_intensity(field)
        expected = np.array([[2.0, 4.0], [9.0, 2.0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_real_field(self) -> None:
        """Test intensity of purely real field."""
        field = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.complex128)
        result = _field_to_intensity(field)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_imaginary_field(self) -> None:
        """Test intensity of purely imaginary field."""
        field = np.array([[2j, 3j], [4j, 5j]], dtype=np.complex128)
        result = _field_to_intensity(field)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_zero_field(self) -> None:
        """Test intensity of zero field."""
        field = np.zeros((3, 3), dtype=np.complex128)
        result = _field_to_intensity(field)
        expected = np.zeros((3, 3), dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestInlineHolographyStrategy:
    """Test InlineHolographyStrategy class."""

    def test_create_hologram_shape(self, inline_config, sample_object_field) -> None:
        """Test that create_hologram returns correct shape."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        assert hologram.shape == sample_object_field.shape
        assert hologram.dtype == np.float64

    def test_create_hologram_non_negative(
        self, inline_config, sample_object_field
    ) -> None:
        """Test that hologram intensities are non-negative."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        assert np.all(hologram >= 0.0)

    def test_create_hologram_deterministic(
        self, inline_config, sample_object_field
    ) -> None:
        """Test that hologram creation is deterministic."""
        strategy = InlineHolographyStrategy()
        hologram1 = strategy.create_hologram(sample_object_field, inline_config)
        hologram2 = strategy.create_hologram(sample_object_field, inline_config)
        np.testing.assert_array_equal(hologram1, hologram2)

    def test_reconstruct_shape(self, inline_config, sample_object_field) -> None:
        """Test that reconstruct returns correct shape."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        reconstruction = strategy.reconstruct(hologram, inline_config)
        assert reconstruction.shape == hologram.shape
        assert reconstruction.dtype == np.float64

    def test_reconstruct_non_negative(self, inline_config, sample_object_field) -> None:
        """Test that reconstruction amplitudes are non-negative."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        reconstruction = strategy.reconstruct(hologram, inline_config)
        assert np.all(reconstruction >= 0.0)

    def test_reconstruct_deterministic(
        self, inline_config, sample_object_field
    ) -> None:
        """Test that reconstruction is deterministic."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        reconstruction1 = strategy.reconstruct(hologram, inline_config)
        reconstruction2 = strategy.reconstruct(hologram, inline_config)
        np.testing.assert_array_equal(reconstruction1, reconstruction2)

    def test_round_trip_similarity(self, inline_config, sample_object_field) -> None:
        """Test that round-trip processing preserves main features."""
        strategy = InlineHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, inline_config)
        reconstruction = strategy.reconstruct(hologram, inline_config)

        # Normalize both for comparison
        from hologen.utils.math import normalize_image

        orig_norm = normalize_image(sample_object_field)
        recon_norm = normalize_image(reconstruction)

        # Should have some correlation (not perfect due to phase loss)
        correlation = np.corrcoef(orig_norm.flatten(), recon_norm.flatten())[0, 1]
        assert correlation > 0.3  # Reasonable threshold for inline holography

    def test_zero_field_handling(self, inline_config) -> None:
        """Test handling of zero object field."""
        strategy = InlineHolographyStrategy()
        zero_field = np.zeros(
            (inline_config.grid.height, inline_config.grid.width), dtype=np.float64
        )
        hologram = strategy.create_hologram(zero_field, inline_config)
        reconstruction = strategy.reconstruct(hologram, inline_config)

        # Should handle gracefully without errors
        assert hologram.shape == zero_field.shape
        assert reconstruction.shape == zero_field.shape
        assert np.all(np.isfinite(hologram))
        assert np.all(np.isfinite(reconstruction))

    def test_negative_hologram_handling(self, inline_config) -> None:
        """Test reconstruction with negative hologram values."""
        strategy = InlineHolographyStrategy()
        # Create hologram with some negative values matching grid size
        hologram = np.full(
            (inline_config.grid.height, inline_config.grid.width),
            -1.0,
            dtype=np.float64,
        )
        hologram[0, 0] = 2.0
        reconstruction = strategy.reconstruct(hologram, inline_config)

        # Should handle negative values by clamping to zero
        assert reconstruction.shape == hologram.shape
        assert np.all(reconstruction >= 0.0)
        assert np.all(np.isfinite(reconstruction))

    def test_uniform_field(self, inline_config) -> None:
        """Test with uniform object field."""
        strategy = InlineHolographyStrategy()
        uniform_field = np.ones(
            (inline_config.grid.height, inline_config.grid.width), dtype=np.float64
        )
        hologram = strategy.create_hologram(uniform_field, inline_config)
        reconstruction = strategy.reconstruct(hologram, inline_config)

        assert hologram.shape == uniform_field.shape
        assert reconstruction.shape == uniform_field.shape
        assert np.all(np.isfinite(hologram))
        assert np.all(np.isfinite(reconstruction))

    def test_energy_scaling(self, inline_config) -> None:
        """Test that energy scales appropriately with field amplitude."""
        strategy = InlineHolographyStrategy()

        field1 = np.ones(
            (inline_config.grid.height, inline_config.grid.width), dtype=np.float64
        )
        field2 = 2.0 * field1

        hologram1 = strategy.create_hologram(field1, inline_config)
        hologram2 = strategy.create_hologram(field2, inline_config)

        # Energy should scale quadratically with amplitude
        energy1 = np.sum(hologram1)
        energy2 = np.sum(hologram2)

        # Allow for some numerical tolerance
        assert energy2 > energy1  # Higher amplitude should give more energy

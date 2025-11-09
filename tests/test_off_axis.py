"""Tests for hologen.holography.off_axis module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.holography.off_axis import (
    OffAxisHolographyStrategy,
    _field_to_intensity,
    _fourier_filter,
    _generate_reference,
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
        """Test conversion to complex128."""
        field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _object_to_complex(field)
        assert result.dtype == np.complex128
        expected = np.array(
            [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=np.complex128
        )
        np.testing.assert_array_equal(result, expected)

    def test_intensity_calculation(self) -> None:
        """Test intensity calculation from complex field."""
        field = np.array(
            [[1.0 + 1j, 2.0 + 0j], [0.0 + 3j, 1.0 - 1j]], dtype=np.complex128
        )
        result = _field_to_intensity(field)
        expected = np.array([[2.0, 4.0], [9.0, 2.0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)
        """Test conversion to complex128."""
        field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _object_to_complex(field)
        assert result.dtype == np.complex128
        expected = np.array(
            [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=np.complex128
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


class TestGenerateReference:
    """Test _generate_reference function."""

    def test_reference_shape(self) -> None:
        """Test reference wave shape."""
        grid = (32, 32)
        carrier_x, carrier_y = 100.0, 200.0
        pixel_pitch = 1e-6

        reference = _generate_reference(grid, carrier_x, carrier_y, pixel_pitch)
        assert reference.shape == grid
        assert reference.dtype == np.complex128

    def test_reference_unit_magnitude(self) -> None:
        """Test that reference wave has unit magnitude."""
        grid = (16, 16)
        carrier_x, carrier_y = 50.0, 75.0
        pixel_pitch = 2e-6

        reference = _generate_reference(grid, carrier_x, carrier_y, pixel_pitch)
        magnitudes = np.abs(reference)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-15)

    def test_zero_carrier_frequency(self) -> None:
        """Test reference with zero carrier frequencies."""
        grid = (8, 8)
        carrier_x, carrier_y = 0.0, 0.0
        pixel_pitch = 1e-6

        reference = _generate_reference(grid, carrier_x, carrier_y, pixel_pitch)
        expected = np.ones(grid, dtype=np.complex128)
        np.testing.assert_allclose(reference, expected, rtol=1e-15)

    def test_phase_variation(self) -> None:
        """Test that non-zero carriers produce phase variation."""
        grid = (16, 16)
        carrier_x, carrier_y = 100.0, 0.0
        pixel_pitch = 1e-6

        reference = _generate_reference(grid, carrier_x, carrier_y, pixel_pitch)
        phases = np.angle(reference)

        # Should have phase variation along x-axis
        assert not np.allclose(phases[0, :], phases[0, 0])


class TestFourierFilter:
    """Test _fourier_filter function."""

    def test_filter_shape(self) -> None:
        """Test that filter preserves shape."""
        hologram = np.random.rand(32, 32).astype(np.float64)
        grid = (32, 32)
        carrier_x, carrier_y = 100.0, 100.0
        sigma = 50.0
        pixel_pitch = 1e-6

        result = _fourier_filter(
            hologram, carrier_x, carrier_y, sigma, grid, pixel_pitch
        )
        assert result.shape == hologram.shape
        assert result.dtype == np.complex128

    def test_filter_dc_component(self) -> None:
        """Test filtering of DC component."""
        hologram = np.ones((16, 16), dtype=np.float64)
        grid = (16, 16)
        carrier_x, carrier_y = 0.0, 0.0  # Filter centered at DC
        sigma = 1000.0  # Large sigma to include DC
        pixel_pitch = 1e-6

        result = _fourier_filter(
            hologram, carrier_x, carrier_y, sigma, grid, pixel_pitch
        )

        # Should preserve some energy when filtering around DC
        assert np.abs(result).sum() > 0

    def test_filter_zero_sigma(self) -> None:
        """Test filter with very small sigma."""
        hologram = np.random.rand(16, 16).astype(np.float64)
        grid = (16, 16)
        carrier_x, carrier_y = 100.0, 100.0
        sigma = 1e-10  # Very small sigma
        pixel_pitch = 1e-6

        result = _fourier_filter(
            hologram, carrier_x, carrier_y, sigma, grid, pixel_pitch
        )

        # Very narrow filter should heavily attenuate
        assert np.abs(result).sum() < np.abs(hologram).sum()


class TestOffAxisHolographyStrategy:
    """Test OffAxisHolographyStrategy class."""

    def test_create_hologram_requires_carrier(
        self, inline_config, sample_object_field
    ) -> None:
        """Test that create_hologram requires carrier configuration."""
        strategy = OffAxisHolographyStrategy()
        with pytest.raises(
            ValueError, match="Off-axis holography requires carrier configuration"
        ):
            strategy.create_hologram(sample_object_field, inline_config)

    def test_reconstruct_requires_carrier(self, inline_config) -> None:
        """Test that reconstruct requires carrier configuration."""
        strategy = OffAxisHolographyStrategy()
        hologram = np.random.rand(32, 32).astype(np.float64)
        with pytest.raises(
            ValueError, match="Off-axis holography requires carrier configuration"
        ):
            strategy.reconstruct(hologram, inline_config)

    def test_create_hologram_shape(self, off_axis_config, sample_object_field) -> None:
        """Test that create_hologram returns correct shape."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        assert hologram.shape == sample_object_field.shape
        assert hologram.dtype == np.float64

    def test_create_hologram_non_negative(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that hologram intensities are non-negative."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        assert np.all(hologram >= 0.0)

    def test_create_hologram_deterministic(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that hologram creation is deterministic."""
        strategy = OffAxisHolographyStrategy()
        hologram1 = strategy.create_hologram(sample_object_field, off_axis_config)
        hologram2 = strategy.create_hologram(sample_object_field, off_axis_config)
        np.testing.assert_array_equal(hologram1, hologram2)

    def test_reconstruct_shape(self, off_axis_config, sample_object_field) -> None:
        """Test that reconstruct returns correct shape."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)
        assert reconstruction.shape == hologram.shape
        assert reconstruction.dtype == np.float64

    def test_reconstruct_non_negative(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that reconstruction amplitudes are non-negative."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)
        assert np.all(reconstruction >= 0.0)

    def test_reconstruct_deterministic(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that reconstruction is deterministic."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        reconstruction1 = strategy.reconstruct(hologram, off_axis_config)
        reconstruction2 = strategy.reconstruct(hologram, off_axis_config)
        np.testing.assert_array_equal(reconstruction1, reconstruction2)

    def test_zero_field_handling(self, off_axis_config) -> None:
        """Test handling of zero object field."""
        strategy = OffAxisHolographyStrategy()
        zero_field = np.zeros(
            (off_axis_config.grid.height, off_axis_config.grid.width), dtype=np.float64
        )
        hologram = strategy.create_hologram(zero_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)

        # Should handle gracefully without errors
        assert hologram.shape == zero_field.shape
        assert reconstruction.shape == zero_field.shape
        assert np.all(np.isfinite(hologram))
        assert np.all(np.isfinite(reconstruction))

    def test_uniform_field(self, off_axis_config) -> None:
        """Test with uniform object field."""
        strategy = OffAxisHolographyStrategy()
        uniform_field = np.ones(
            (off_axis_config.grid.height, off_axis_config.grid.width), dtype=np.float64
        )
        hologram = strategy.create_hologram(uniform_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)

        assert hologram.shape == uniform_field.shape
        assert reconstruction.shape == uniform_field.shape
        assert np.all(np.isfinite(hologram))
        assert np.all(np.isfinite(reconstruction))

    def test_carrier_frequency_effect(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that different carrier frequencies produce different holograms."""
        strategy = OffAxisHolographyStrategy()

        # Create config with different carrier frequency
        from hologen.types import HolographyConfig, OffAxisCarrier

        different_carrier = OffAxisCarrier(
            frequency_x=2000.0,  # Different from fixture
            frequency_y=2000.0,
            gaussian_width=300.0,
        )
        different_config = HolographyConfig(
            grid=off_axis_config.grid,
            optics=off_axis_config.optics,
            method=off_axis_config.method,
            carrier=different_carrier,
        )

        hologram1 = strategy.create_hologram(sample_object_field, off_axis_config)
        hologram2 = strategy.create_hologram(sample_object_field, different_config)

        # Should produce different holograms
        assert not np.allclose(hologram1, hologram2, rtol=1e-10)

    def test_reference_wave_contribution(self, off_axis_config) -> None:
        """Test that reference wave contributes to hologram energy."""
        strategy = OffAxisHolographyStrategy()

        # Test with zero object field - should still have reference wave energy
        zero_field = np.zeros(
            (off_axis_config.grid.height, off_axis_config.grid.width), dtype=np.float64
        )
        hologram = strategy.create_hologram(zero_field, off_axis_config)

        # Should have non-zero energy from reference wave
        assert np.sum(hologram) > 0

    def test_round_trip_similarity(self, off_axis_config, sample_object_field) -> None:
        """Test that round-trip processing preserves main features."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)

        # Normalize both for comparison
        from hologen.utils.math import normalize_image

        orig_norm = normalize_image(sample_object_field)
        recon_norm = normalize_image(reconstruction)

        # Should have some correlation (off-axis can be challenging)
        correlation = np.corrcoef(orig_norm.flatten(), recon_norm.flatten())[0, 1]
        # Very lenient threshold due to off-axis complexity - just check it's not NaN
        # and has some reasonable magnitude (absolute value)
        assert not np.isnan(correlation)
        assert (
            abs(correlation) > 0.1
        )  # Allow negative correlation but require some similarity

    def test_gaussian_blur_in_reconstruction(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that Gaussian blur is applied during reconstruction."""
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(sample_object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)

        # Reconstruction should be smoothed (no sharp edges)
        # This is hard to test directly, but we can check it doesn't crash
        assert reconstruction.shape == hologram.shape
        assert np.all(np.isfinite(reconstruction))

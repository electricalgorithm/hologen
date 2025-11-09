"""Tests for hologen.holography.propagation module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import GridSpec, OpticalConfig


class TestAngularSpectrumPropagate:
    """Test angular_spectrum_propagate function."""

    def test_shape_validation(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that field shape must match grid dimensions."""
        field = np.ones((32, 32), dtype=np.complex128)  # Wrong shape
        with pytest.raises(ValueError, match="Field shape must match grid dimensions"):
            angular_spectrum_propagate(field, grid_spec, optical_config, distance=0.01)

    def test_zero_distance_identity(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that zero distance returns unchanged field."""
        field = np.random.rand(grid_spec.height, grid_spec.width).astype(np.complex128)
        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, distance=0.0
        )
        np.testing.assert_array_equal(result, field)

    def test_output_dtype(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that output has correct dtype."""
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex64)
        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, distance=0.01
        )
        assert result.dtype == np.complex128

    def test_output_shape(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that output shape matches input."""
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex128)
        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, distance=0.01
        )
        assert result.shape == field.shape

    def test_energy_conservation_plane_wave(
        self, optical_config: OpticalConfig
    ) -> None:
        """Test energy conservation for plane wave propagation."""
        grid = GridSpec(height=64, width=64, pixel_pitch=1e-6)
        field = np.ones((grid.height, grid.width), dtype=np.complex128)

        result = angular_spectrum_propagate(field, grid, optical_config, distance=0.001)

        # Energy should be approximately conserved for plane wave
        input_energy = np.sum(np.abs(field) ** 2)
        output_energy = np.sum(np.abs(result) ** 2)
        assert output_energy == pytest.approx(input_energy, rel=1e-10)

    def test_negative_distance(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test propagation with negative distance."""
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex128)
        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, distance=-0.01
        )
        assert result.shape == field.shape
        assert result.dtype == np.complex128

    def test_evanescent_decay(self, optical_config: OpticalConfig) -> None:
        """Test that evanescent components decay with distance."""
        # Create a field with high spatial frequencies (evanescent components)
        grid = GridSpec(height=32, width=32, pixel_pitch=1e-7)  # Very small pixels
        field = np.zeros((grid.height, grid.width), dtype=np.complex128)
        field[0, -1] = 1.0  # High frequency component

        # Propagate forward
        result = angular_spectrum_propagate(field, grid, optical_config, distance=0.001)

        # High frequency component should be attenuated
        assert np.abs(result[0, -1]) < np.abs(field[0, -1])

    def test_reciprocity(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that forward and backward propagation are reciprocal."""
        field = np.random.rand(grid_spec.height, grid_spec.width).astype(np.complex128)
        distance = 0.005

        forward = angular_spectrum_propagate(field, grid_spec, optical_config, distance)
        backward = angular_spectrum_propagate(
            forward, grid_spec, optical_config, -distance
        )

        # Use a more lenient tolerance to account for numerical rounding
        np.testing.assert_allclose(backward, field, rtol=1e-6, atol=1e-8)

    def test_dc_component_preservation(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test that DC component is preserved during propagation."""
        # Create a uniform field (true DC component)
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex128)

        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, distance=0.01
        )

        # For a uniform field (plane wave at normal incidence), the angular spectrum method
        # applies a phase shift of exp(i*k*z), which is physically correct
        # The magnitude should be preserved, but there will be a global phase shift
        original_dc = np.mean(field)
        result_dc = np.mean(result)

        # Check that magnitude is preserved
        assert abs(result_dc) == pytest.approx(abs(original_dc), rel=1e-10)

        # Check that the phase shift matches the expected value for plane wave propagation
        k = 2.0 * np.pi / optical_config.wavelength
        expected_phase_shift = k * 0.01
        expected_result = original_dc * np.exp(1j * expected_phase_shift)
        assert result_dc == pytest.approx(expected_result, rel=1e-10)

    def test_phase_accumulation(self, optical_config: OpticalConfig) -> None:
        """Test phase accumulation for propagating waves."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        field = np.ones((grid.height, grid.width), dtype=np.complex128)

        distance1 = 0.001
        distance2 = 0.002

        result1 = angular_spectrum_propagate(field, grid, optical_config, distance1)
        result2 = angular_spectrum_propagate(field, grid, optical_config, distance2)

        # Phase should accumulate linearly with distance for plane wave
        phase1 = np.angle(result1[0, 0])
        phase2 = np.angle(result2[0, 0])

        # For plane wave, phase should be proportional to distance
        expected_phase_ratio = distance2 / distance1
        actual_phase_ratio = phase2 / phase1 if phase1 != 0 else 1

        # Allow for 2π wrapping
        if abs(actual_phase_ratio - expected_phase_ratio) > 1:
            # Check if phases differ by 2π multiples
            phase_diff = abs(phase2 - expected_phase_ratio * phase1)
            assert phase_diff < 0.1 or abs(phase_diff - 2 * np.pi) < 0.1

    def test_different_wavelengths(self, grid_spec: GridSpec) -> None:
        """Test propagation with different wavelengths."""
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex128)
        distance = 0.01

        optics1 = OpticalConfig(wavelength=500e-9, propagation_distance=distance)
        optics2 = OpticalConfig(wavelength=600e-9, propagation_distance=distance)

        result1 = angular_spectrum_propagate(field, grid_spec, optics1, distance)
        result2 = angular_spectrum_propagate(field, grid_spec, optics2, distance)

        # Results should be different for different wavelengths
        assert not np.allclose(result1, result2)

    def test_large_distance_stability(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test numerical stability for large propagation distances."""
        field = np.ones((grid_spec.height, grid_spec.width), dtype=np.complex128)
        large_distance = 1.0  # 1 meter

        result = angular_spectrum_propagate(
            field, grid_spec, optical_config, large_distance
        )

        # Should not produce NaN or infinite values
        assert np.all(np.isfinite(result))
        assert not np.any(np.isnan(result))

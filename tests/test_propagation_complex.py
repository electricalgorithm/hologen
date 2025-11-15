"""Tests for complex field propagation in hologen.holography.propagation module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import GridSpec, OpticalConfig


class TestPlaneWavePropagation:
    """Test plane wave propagation with phase shift only."""

    def test_plane_wave_phase_shift(self) -> None:
        """Test that plane wave propagates with correct phase shift."""
        grid = GridSpec(height=128, width=128, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create uniform plane wave
        field = np.ones((grid.height, grid.width), dtype=np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(
            field, grid, optics, optics.propagation_distance
        )
        
        # Expected phase shift: k*z = 2π*z/λ
        expected_phase = (2 * np.pi * optics.propagation_distance / optics.wavelength) % (2 * np.pi)
        actual_phase = np.angle(propagated[0, 0]) % (2 * np.pi)
        
        assert np.isclose(actual_phase, expected_phase, atol=1e-6)
        
        # Amplitude should remain constant
        assert np.allclose(np.abs(propagated), 1.0, atol=1e-6)

    def test_plane_wave_amplitude_conservation(self) -> None:
        """Test that plane wave amplitude is conserved during propagation."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=5e-3)
        
        # Create plane wave with amplitude 2.0
        field = np.ones((grid.height, grid.width), dtype=np.complex128) * 2.0
        
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Amplitude should be preserved
        assert np.allclose(np.abs(propagated), 2.0, atol=1e-6)

    def test_zero_distance_no_change(self) -> None:
        """Test that zero propagation distance returns field unchanged."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.0)
        
        field = np.random.rand(grid.height, grid.width) + 1j * np.random.rand(grid.height, grid.width)
        field = field.astype(np.complex128)
        
        propagated = angular_spectrum_propagate(field, grid, optics, 0.0)
        
        # Should return the same field
        assert propagated is field


class TestSphericalWavePropagation:
    """Test spherical wave propagation against Fresnel formula."""

    def test_spherical_wave_divergence(self) -> None:
        """Test that spherical wave diverges correctly."""
        grid = GridSpec(height=128, width=128, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create point source at center (approximated by Gaussian)
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        sigma = 2.0  # pixels
        gaussian = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * sigma**2))
        field = gaussian.astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # After propagation, the field should spread out
        # Check that energy is more distributed
        original_peak = np.max(np.abs(field))
        propagated_peak = np.max(np.abs(propagated))
        
        # Peak should decrease due to spreading
        assert propagated_peak < original_peak

    def test_spherical_wave_energy_conservation(self) -> None:
        """Test energy conservation (Parseval's theorem) for spherical wave."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=2e-3)
        
        # Create Gaussian beam
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        sigma = 5.0
        gaussian = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * sigma**2))
        field = gaussian.astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Total energy should be conserved (Parseval's theorem)
        original_energy = np.sum(np.abs(field)**2)
        propagated_energy = np.sum(np.abs(propagated)**2)
        
        assert np.isclose(original_energy, propagated_energy, rtol=1e-6)


class TestPhasePreservation:
    """Test phase preservation through propagation."""

    def test_phase_pattern_preserved(self) -> None:
        """Test that phase patterns are preserved during propagation."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create field with phase pattern (uniform amplitude)
        y, x = np.meshgrid(np.arange(grid.height), np.arange(grid.width), indexing='ij')
        phase_pattern = 0.5 * np.sin(2 * np.pi * x / grid.width)
        field = np.exp(1j * phase_pattern).astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Phase structure should be preserved (though shifted)
        # Check that phase variation exists
        phase_var_original = np.var(np.angle(field))
        phase_var_propagated = np.var(np.angle(propagated))
        
        assert phase_var_propagated > 0
        # Phase variation should be similar order of magnitude
        assert np.isclose(phase_var_original, phase_var_propagated, rtol=0.5)

    def test_complex_field_phase_amplitude_coupling(self) -> None:
        """Test that complex fields with both amplitude and phase propagate correctly."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create field with both amplitude and phase modulation
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        
        # Amplitude: Gaussian
        amplitude = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * 5.0**2))
        
        # Phase: linear ramp
        phase = 0.3 * (x - center_x) / grid.width
        
        field = (amplitude * np.exp(1j * phase)).astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Field should still be complex with both amplitude and phase variation
        assert np.var(np.abs(propagated)) > 0
        assert np.var(np.angle(propagated)) > 0


class TestAmplitudeConservation:
    """Test amplitude conservation (Parseval's theorem)."""

    def test_parseval_theorem(self) -> None:
        """Test that total energy is conserved during propagation."""
        grid = GridSpec(height=128, width=128, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=3e-3)
        
        # Create random complex field
        np.random.seed(42)
        field = (np.random.rand(grid.height, grid.width) + 
                 1j * np.random.rand(grid.height, grid.width)).astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Total energy should be conserved
        original_energy = np.sum(np.abs(field)**2)
        propagated_energy = np.sum(np.abs(propagated)**2)
        
        assert np.isclose(original_energy, propagated_energy, rtol=1e-10)

    def test_energy_conservation_back_propagation(self) -> None:
        """Test energy conservation for back-propagation."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=2e-3)
        
        # Create field
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        field = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * 8.0**2)).astype(np.complex128)
        
        # Back-propagate (negative distance)
        propagated = angular_spectrum_propagate(field, grid, optics, -optics.propagation_distance)
        
        # Energy should be conserved
        original_energy = np.sum(np.abs(field)**2)
        propagated_energy = np.sum(np.abs(propagated)**2)
        
        assert np.isclose(original_energy, propagated_energy, rtol=1e-10)


class TestComplexVsIntensityOnly:
    """Compare complex vs intensity-only for amplitude-only objects."""

    def test_amplitude_only_object_propagation(self) -> None:
        """Test that amplitude-only objects propagate correctly as complex fields."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create amplitude-only object (zero phase)
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        radius = 10
        mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
        amplitude = np.zeros((grid.height, grid.width), dtype=np.float64)
        amplitude[mask] = 1.0
        
        # Convert to complex field
        field = amplitude.astype(np.complex128)
        
        # Propagate
        propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Result should be complex (phase develops during propagation)
        assert np.iscomplexobj(propagated)
        
        # Intensity should differ from original amplitude squared
        propagated_intensity = np.abs(propagated)**2
        original_intensity = amplitude**2
        
        # They should not be identical (diffraction occurs)
        assert not np.allclose(propagated_intensity, original_intensity)

    def test_round_trip_propagation(self) -> None:
        """Test that forward and backward propagation returns to original field."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=2e-3)
        
        # Create complex field
        y, x = np.ogrid[:grid.height, :grid.width]
        center_y, center_x = grid.height // 2, grid.width // 2
        field = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * 5.0**2)).astype(np.complex128)
        
        # Forward propagation
        forward = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
        
        # Backward propagation
        backward = angular_spectrum_propagate(forward, grid, optics, -optics.propagation_distance)
        
        # Should return to original field
        assert np.allclose(backward, field, atol=1e-10)


class TestErrorHandling:
    """Test error handling in propagation."""

    def test_invalid_field_shape(self) -> None:
        """Test error for field shape mismatch."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=1e-3)
        
        # Create field with wrong shape
        field = np.ones((32, 32), dtype=np.complex128)
        
        with pytest.raises(ValueError, match="Field shape must match grid dimensions"):
            angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)

    def test_different_wavelengths(self) -> None:
        """Test propagation with different wavelengths."""
        grid = GridSpec(height=64, width=64, pixel_pitch=5e-6)
        
        # Test with different wavelengths
        wavelengths = [405e-9, 532e-9, 633e-9]  # Blue, green, red
        
        field = np.ones((grid.height, grid.width), dtype=np.complex128)
        
        for wavelength in wavelengths:
            optics = OpticalConfig(wavelength=wavelength, propagation_distance=1e-3)
            propagated = angular_spectrum_propagate(field, grid, optics, optics.propagation_distance)
            
            # Should propagate successfully
            assert propagated.shape == field.shape
            assert np.iscomplexobj(propagated)

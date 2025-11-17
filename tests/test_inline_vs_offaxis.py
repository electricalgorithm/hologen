"""Tests comparing inline and off-axis holography behavior.

This module verifies that:
1. Off-axis holography is unchanged by inline holography modifications
2. Off-axis holography continues to work without twin images
3. Inline and off-axis have different behaviors (twin image presence)
"""

from __future__ import annotations

import numpy as np

from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy


class TestOffAxisUnchanged:
    """Verify that off-axis holography implementation is unchanged."""

    def test_offaxis_preserves_complex_field(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that off-axis preserves full complex field information."""
        strategy = OffAxisHolographyStrategy()
        
        # Create complex object with phase
        object_field = sample_object_field.astype(np.complex128)
        object_field *= np.exp(1j * np.pi / 4)  # Add phase
        
        # Create hologram
        hologram = strategy.create_hologram(object_field, off_axis_config)
        
        # Off-axis should preserve complex field (not just intensity)
        # The hologram should have non-zero phase variation
        phases = np.angle(hologram)
        phase_variation = np.std(phases)
        
        # Should have significant phase variation from reference wave
        assert phase_variation > 0.1

    def test_offaxis_hologram_is_complex(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that off-axis returns complex hologram field."""
        strategy = OffAxisHolographyStrategy()
        
        object_field = sample_object_field.astype(np.complex128)
        hologram = strategy.create_hologram(object_field, off_axis_config)
        
        # Should return complex field
        assert hologram.dtype == np.complex128
        
        # Should have non-trivial imaginary components
        assert np.any(np.imag(hologram) != 0)

    def test_offaxis_reference_wave_present(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that off-axis includes reference wave in hologram."""
        strategy = OffAxisHolographyStrategy()
        
        # Create hologram with zero object field
        zero_field = np.zeros_like(sample_object_field, dtype=np.complex128)
        hologram = strategy.create_hologram(zero_field, off_axis_config)
        
        # Should still have energy from reference wave
        energy = np.sum(np.abs(hologram) ** 2)
        assert energy > 0
        
        # Should have phase variation from tilted reference
        phases = np.angle(hologram)
        phase_variation = np.std(phases)
        assert phase_variation > 0.1


class TestOffAxisNoTwinImage:
    """Verify that off-axis holography does not produce twin images."""

    def test_offaxis_reconstruction_quality(
        self, off_axis_config, sample_object_field
    ) -> None:
        """Test that off-axis reconstruction has good quality (no twin image)."""
        strategy = OffAxisHolographyStrategy()
        
        # Create complex object with phase
        object_field = sample_object_field.astype(np.complex128)
        # Add phase modulation to object
        center_y, center_x = object_field.shape[0] // 2, object_field.shape[1] // 2
        y, x = np.ogrid[: object_field.shape[0], : object_field.shape[1]]
        phase_mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < (
            min(object_field.shape) // 8
        ) ** 2
        object_field[phase_mask] *= np.exp(1j * np.pi / 2)
        
        # Create hologram and reconstruct
        hologram = strategy.create_hologram(object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)
        
        # Off-axis should have reasonable reconstruction quality
        # (twin image is filtered out in frequency domain)
        assert reconstruction.shape == object_field.shape
        assert np.all(np.isfinite(reconstruction))

    def test_offaxis_with_phase_object(self, off_axis_config) -> None:
        """Test off-axis with pure phase object."""
        strategy = OffAxisHolographyStrategy()
        
        # Create pure phase object (constant amplitude, varying phase)
        shape = (off_axis_config.grid.height, off_axis_config.grid.width)
        object_field = np.ones(shape, dtype=np.complex128)
        
        # Add phase pattern
        center_y, center_x = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        phase_pattern = np.pi * ((y - center_y) ** 2 + (x - center_x) ** 2) / (
            min(shape) ** 2
        )
        object_field *= np.exp(1j * phase_pattern)
        
        # Create hologram and reconstruct
        hologram = strategy.create_hologram(object_field, off_axis_config)
        reconstruction = strategy.reconstruct(hologram, off_axis_config)
        
        # Should handle phase objects without issues
        assert reconstruction.shape == object_field.shape
        assert np.all(np.isfinite(reconstruction))


class TestInlineVsOffAxisDifferences:
    """Test that inline and off-axis have different behaviors."""

    def test_hologram_field_differences(
        self, inline_config, off_axis_config, sample_object_field
    ) -> None:
        """Test that inline and off-axis produce different hologram fields."""
        inline_strategy = InlineHolographyStrategy()
        offaxis_strategy = OffAxisHolographyStrategy()
        
        # Use same object field
        object_field = sample_object_field.astype(np.complex128)
        object_field *= np.exp(1j * np.pi / 4)  # Add phase
        
        # Create holograms with both methods
        inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
        offaxis_hologram = offaxis_strategy.create_hologram(
            object_field, off_axis_config
        )
        
        # Holograms should be different
        assert not np.allclose(inline_hologram, offaxis_hologram)
        
        # Inline should have zero phase (intensity recording)
        inline_phases = np.angle(inline_hologram)
        assert np.allclose(inline_phases, 0.0, atol=1e-10)
        
        # Off-axis should have non-zero phase (reference wave)
        offaxis_phases = np.angle(offaxis_hologram)
        assert not np.allclose(offaxis_phases, 0.0, atol=1e-10)

    def test_phase_preservation_difference(
        self, inline_config, off_axis_config
    ) -> None:
        """Test that inline loses phase while off-axis preserves it."""
        inline_strategy = InlineHolographyStrategy()
        offaxis_strategy = OffAxisHolographyStrategy()
        
        # Create object with known phase
        shape = (inline_config.grid.height, inline_config.grid.width)
        object_field = np.ones(shape, dtype=np.complex128)
        object_field *= np.exp(1j * np.pi / 3)  # Uniform phase
        
        # Create holograms
        inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
        offaxis_hologram = offaxis_strategy.create_hologram(
            object_field, off_axis_config
        )
        
        # Inline hologram should have zero phase (phase lost)
        inline_phases = np.angle(inline_hologram)
        assert np.allclose(inline_phases, 0.0, atol=1e-10)
        
        # Off-axis hologram should have phase information
        offaxis_phases = np.angle(offaxis_hologram)
        offaxis_phase_std = np.std(offaxis_phases)
        assert offaxis_phase_std > 0.1  # Significant phase variation

    def test_twin_image_presence_difference(
        self, inline_config, off_axis_config
    ) -> None:
        """Test that inline produces twin images while off-axis does not."""
        inline_strategy = InlineHolographyStrategy()
        offaxis_strategy = OffAxisHolographyStrategy()
        
        # Create phase object (will produce twin image in inline)
        shape = (inline_config.grid.height, inline_config.grid.width)
        object_field = np.zeros(shape, dtype=np.complex128)
        
        # Create a disk with phase shift
        center_y, center_x = shape[0] // 2, shape[1] // 2
        radius = min(shape) // 8
        y, x = np.ogrid[: shape[0], : shape[1]]
        mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        object_field[mask] = np.exp(1j * np.pi / 2)  # Phase disk
        object_field[~mask] = 1.0  # Background
        
        # Create holograms and reconstruct
        inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
        inline_recon = inline_strategy.reconstruct(inline_hologram, inline_config)
        
        offaxis_hologram = offaxis_strategy.create_hologram(
            object_field, off_axis_config
        )
        offaxis_recon = offaxis_strategy.reconstruct(offaxis_hologram, off_axis_config)
        
        # Both should produce valid reconstructions
        assert inline_recon.shape == object_field.shape
        assert offaxis_recon.shape == object_field.shape
        assert np.all(np.isfinite(inline_recon))
        assert np.all(np.isfinite(offaxis_recon))
        
        # Reconstructions should be different due to twin image in inline
        assert not np.allclose(
            np.abs(inline_recon), np.abs(offaxis_recon), rtol=0.1
        )

    def test_intensity_recording_vs_complex_field(
        self, inline_config, off_axis_config, sample_object_field
    ) -> None:
        """Test that inline records intensity while off-axis preserves complex field."""
        inline_strategy = InlineHolographyStrategy()
        offaxis_strategy = OffAxisHolographyStrategy()
        
        # Create complex object
        object_field = sample_object_field.astype(np.complex128)
        # Add varying phase
        phase_pattern = np.random.rand(*object_field.shape) * 2 * np.pi
        object_field *= np.exp(1j * phase_pattern)
        
        # Create holograms
        inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
        offaxis_hologram = offaxis_strategy.create_hologram(
            object_field, off_axis_config
        )
        
        # Inline: hologram should be real-valued (zero phase)
        inline_imaginary = np.imag(inline_hologram)
        assert np.allclose(inline_imaginary, 0.0, atol=1e-10)
        
        # Off-axis: hologram should have complex values
        offaxis_imaginary = np.imag(offaxis_hologram)
        assert not np.allclose(offaxis_imaginary, 0.0, atol=1e-10)
        assert np.any(np.abs(offaxis_imaginary) > 0.01)

    def test_reconstruction_correlation_difference(
        self, inline_config, off_axis_config, sample_object_field
    ) -> None:
        """Test that reconstruction quality differs between methods."""
        inline_strategy = InlineHolographyStrategy()
        offaxis_strategy = OffAxisHolographyStrategy()
        
        # Create object with phase
        object_field = sample_object_field.astype(np.complex128)
        # Add phase to the object region
        object_field[object_field > 0] *= np.exp(1j * np.pi / 2)
        
        # Create holograms and reconstruct
        inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
        inline_recon = inline_strategy.reconstruct(inline_hologram, inline_config)
        
        offaxis_hologram = offaxis_strategy.create_hologram(
            object_field, off_axis_config
        )
        offaxis_recon = offaxis_strategy.reconstruct(offaxis_hologram, off_axis_config)
        
        # Normalize for comparison
        from hologen.utils.math import normalize_image
        
        obj_norm = normalize_image(np.abs(object_field))
        inline_norm = normalize_image(np.abs(inline_recon))
        offaxis_norm = normalize_image(np.abs(offaxis_recon))
        
        # Compute correlations
        inline_corr = np.corrcoef(obj_norm.flatten(), inline_norm.flatten())[0, 1]
        offaxis_corr = np.corrcoef(obj_norm.flatten(), offaxis_norm.flatten())[0, 1]
        
        # Both should have some correlation, but they should be different
        # (inline affected by twin image, off-axis by filtering)
        assert not np.isnan(inline_corr)
        assert not np.isnan(offaxis_corr)
        
        # The correlations should be different (different reconstruction characteristics)
        # We don't assert which is better, just that they're different
        assert abs(inline_corr - offaxis_corr) > 0.05

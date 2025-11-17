"""Tests for PhaseGenerator class."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from hologen.phase import PhaseGenerationConfig, PhaseGenerator
from hologen.types import GridSpec


class TestPhaseGeneratorEquation:
    """Test phase equation φ = (2π/λ) × (n - n₀) × d."""

    def test_phase_equation_with_known_inputs(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase equation with known inputs produces expected output."""
        # Create a simple uniform mask
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Use uniform mode for predictable refractive index and thickness
        # Use narrow range to get consistent values
        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="constant",
            ambient_refractive_index=1.0,
            refractive_index_range=(1.49, 1.51),  # Narrow range around 1.5
            thickness_range=(4.9e-6, 5.1e-6),  # Narrow range around 5e-6
        )

        wavelength = 632.8e-9  # Red laser
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Expected phase should be approximately: φ = (2π/λ) × (n - n₀) × d
        # With n ≈ 1.5 and d ≈ 5e-6
        expected_phase_approx = (2 * np.pi / wavelength) * (1.5 - 1.0) * 5e-6
        expected_phase_approx = np.angle(np.exp(1j * expected_phase_approx))

        # All values inside mask should be close to expected phase
        # Use looser tolerance since we're sampling from a range
        assert np.allclose(phase[mask > 0.5], expected_phase_approx, rtol=0.05)

    def test_phase_equation_various_wavelengths(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase equation with various wavelengths."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="constant",
            ambient_refractive_index=1.0,
            refractive_index_range=(1.39, 1.41),
            thickness_range=(2.9e-6, 3.1e-6),
        )

        generator = PhaseGenerator()

        # Test with different wavelengths
        wavelengths = [405e-9, 532e-9, 632.8e-9, 780e-9]  # Violet, green, red, NIR

        for wavelength in wavelengths:
            phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

            # Expected phase (approximate)
            expected_phase = (2 * np.pi / wavelength) * (1.4 - 1.0) * 3e-6
            expected_phase = np.angle(np.exp(1j * expected_phase))

            assert np.allclose(phase[mask > 0.5], expected_phase, rtol=0.05)

    def test_phase_equation_various_refractive_indices(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase equation with various refractive indices."""
        mask = np.ones((grid_spec.height, grid_spec.width))
        wavelength = 532e-9

        generator = PhaseGenerator()

        # Test with different refractive indices
        refractive_indices = [1.33, 1.45, 1.55, 1.7]  # Water, glass, etc.

        for n in refractive_indices:
            config = PhaseGenerationConfig(
                enabled=True,
                refractive_index_mode="uniform",
                thickness_mode="constant",
                ambient_refractive_index=1.0,
                refractive_index_range=(n - 0.01, n + 0.01),
                thickness_range=(3.9e-6, 4.1e-6),
            )

            phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

            # Expected phase (approximate)
            expected_phase = (2 * np.pi / wavelength) * (n - 1.0) * 4e-6
            expected_phase = np.angle(np.exp(1j * expected_phase))

            assert np.allclose(phase[mask > 0.5], expected_phase, rtol=0.05)

    def test_phase_equation_various_thicknesses(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase equation with various thicknesses."""
        mask = np.ones((grid_spec.height, grid_spec.width))
        wavelength = 632.8e-9

        generator = PhaseGenerator()

        # Test with different thicknesses
        thicknesses = [1e-6, 5e-6, 10e-6, 20e-6]

        for d in thicknesses:
            config = PhaseGenerationConfig(
                enabled=True,
                refractive_index_mode="uniform",
                thickness_mode="constant",
                ambient_refractive_index=1.0,
                refractive_index_range=(1.49, 1.51),
                thickness_range=(d * 0.98, d * 1.02),
            )

            phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

            # Expected phase (approximate)
            expected_phase = (2 * np.pi / wavelength) * (1.5 - 1.0) * d
            expected_phase = np.angle(np.exp(1j * expected_phase))

            assert np.allclose(phase[mask > 0.5], expected_phase, rtol=0.05)

    def test_phase_wrapping_to_minus_pi_to_pi(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that phase values are wrapped to [-π, π] range."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Use parameters that will produce phase > 2π before wrapping
        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="constant",
            ambient_refractive_index=1.0,
            refractive_index_range=(1.99, 2.01),
            thickness_range=(9.9e-6, 10.1e-6),
        )

        wavelength = 400e-9  # Short wavelength for larger phase
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # All phase values should be in [-π, π]
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_phase_outside_mask_is_zero(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that phase is zero outside the object mask."""
        # Create a circular mask
        mask = np.zeros((grid_spec.height, grid_spec.width))
        center_y, center_x = grid_spec.height // 2, grid_spec.width // 2
        radius = 15
        y, x = np.ogrid[: grid_spec.height, : grid_spec.width]
        circle_mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        mask[circle_mask] = 1.0

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="constant",
            ambient_refractive_index=1.0,
            refractive_index_range=(1.49, 1.51),
            thickness_range=(4.9e-6, 5.1e-6),
        )

        wavelength = 632.8e-9
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Phase outside mask should be zero (since thickness is zero outside)
        outside_phase = phase[mask <= 0.5]
        assert np.allclose(outside_phase, 0.0, atol=1e-10)

    def test_phase_with_different_ambient_refractive_index(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase equation with different ambient refractive index."""
        mask = np.ones((grid_spec.height, grid_spec.width))
        wavelength = 532e-9

        # Test with water as ambient medium (n₀ = 1.33)
        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="constant",
            ambient_refractive_index=1.33,
            refractive_index_range=(1.49, 1.51),
            thickness_range=(4.9e-6, 5.1e-6),
        )

        generator = PhaseGenerator()
        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Expected phase (approximate): φ = (2π/λ) × (1.5 - 1.33) × 5e-6
        expected_phase = (2 * np.pi / wavelength) * (1.5 - 1.33) * 5e-6
        expected_phase = np.angle(np.exp(1j * expected_phase))

        assert np.allclose(phase[mask > 0.5], expected_phase, rtol=0.05)


class TestPhaseGeneratorCorrelatedAmplitude:
    """Test generate_correlated_amplitude method."""

    def test_positive_correlation_increases_amplitude_with_refractive_index(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that positive correlation increases amplitude with refractive index."""
        # Create a mask
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Create a refractive index field with spatial variation
        n_field = np.ones((grid_spec.height, grid_spec.width))
        # Left half has lower refractive index
        n_field[:, : grid_spec.width // 2] = 1.3
        # Right half has higher refractive index
        n_field[:, grid_spec.width // 2 :] = 1.5

        config = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.5),
            correlation_coefficient=0.9,  # Strong positive correlation
        )

        generator = PhaseGenerator()
        amplitude = generator.generate_correlated_amplitude(mask, n_field, rng, config)

        # Right half (higher n) should have higher amplitude on average
        left_amplitude = amplitude[:, : grid_spec.width // 2]
        right_amplitude = amplitude[:, grid_spec.width // 2 :]

        assert np.mean(right_amplitude) > np.mean(left_amplitude)

    def test_negative_correlation_decreases_amplitude_with_refractive_index(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that negative correlation decreases amplitude with refractive index."""
        # Create a mask
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Create a refractive index field with spatial variation
        n_field = np.ones((grid_spec.height, grid_spec.width))
        # Left half has lower refractive index
        n_field[:, : grid_spec.width // 2] = 1.3
        # Right half has higher refractive index
        n_field[:, grid_spec.width // 2 :] = 1.5

        config = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.5),
            correlation_coefficient=-0.9,  # Strong negative correlation
        )

        generator = PhaseGenerator()
        amplitude = generator.generate_correlated_amplitude(mask, n_field, rng, config)

        # Right half (higher n) should have lower amplitude on average
        left_amplitude = amplitude[:, : grid_spec.width // 2]
        right_amplitude = amplitude[:, grid_spec.width // 2 :]

        assert np.mean(left_amplitude) > np.mean(right_amplitude)

    def test_zero_correlation_produces_uncorrelated_amplitude(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that zero correlation produces amplitude uncorrelated with n."""
        # Create a mask
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Create a refractive index field with strong spatial variation
        n_field = np.ones((grid_spec.height, grid_spec.width))
        n_field[:, : grid_spec.width // 2] = 1.3
        n_field[:, grid_spec.width // 2 :] = 1.5

        config = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.5),
            correlation_coefficient=0.0,  # No correlation
        )

        generator = PhaseGenerator()

        # Generate multiple samples to test statistical independence
        left_amplitudes = []
        right_amplitudes = []

        for _ in range(10):
            amplitude = generator.generate_correlated_amplitude(
                mask, n_field, rng, config
            )
            left_amplitudes.append(np.mean(amplitude[:, : grid_spec.width // 2]))
            right_amplitudes.append(np.mean(amplitude[:, grid_spec.width // 2 :]))

        # With zero correlation, the difference should be small and inconsistent
        # (sometimes left > right, sometimes right > left)
        differences = np.array(left_amplitudes) - np.array(right_amplitudes)

        # The mean absolute difference should be small
        assert np.mean(np.abs(differences)) < 0.1

    def test_amplitude_values_in_valid_range(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that amplitude values are clipped to [0, 1]."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Create extreme refractive index field
        n_field = np.random.uniform(1.3, 2.5, size=(grid_spec.height, grid_spec.width))

        config = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 2.5),
            correlation_coefficient=0.8,
        )

        generator = PhaseGenerator()
        amplitude = generator.generate_correlated_amplitude(mask, n_field, rng, config)

        # All amplitude values should be in [0, 1]
        assert np.all(amplitude >= 0.0)
        assert np.all(amplitude <= 1.0)

    def test_amplitude_zero_outside_mask(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test that amplitude is zero outside mask."""
        # Create a circular mask
        mask = np.zeros((grid_spec.height, grid_spec.width))
        center_y, center_x = grid_spec.height // 2, grid_spec.width // 2
        radius = 15
        y, x = np.ogrid[: grid_spec.height, : grid_spec.width]
        circle_mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        mask[circle_mask] = 1.0

        # Create refractive index field
        n_field = np.full((grid_spec.height, grid_spec.width), 1.5)

        config = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.7),
            correlation_coefficient=0.5,
        )

        generator = PhaseGenerator()
        amplitude = generator.generate_correlated_amplitude(mask, n_field, rng, config)

        # Amplitude outside mask should be zero
        outside_amplitude = amplitude[mask <= 0.5]
        assert np.allclose(outside_amplitude, 0.0)

    def test_correlation_coefficient_boundary_values(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test correlation coefficient at boundary values -1.0 and 1.0."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        # Create refractive index field with variation
        n_field = np.linspace(1.3, 1.5, grid_spec.width)
        n_field = np.tile(n_field, (grid_spec.height, 1))

        generator = PhaseGenerator()

        # Test with correlation = 1.0 (perfect positive correlation)
        config_pos = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.5),
            correlation_coefficient=1.0,
        )
        amplitude_pos = generator.generate_correlated_amplitude(
            mask, n_field, rng, config_pos
        )

        # Test with correlation = -1.0 (perfect negative correlation)
        config_neg = PhaseGenerationConfig(
            ambient_refractive_index=1.0,
            refractive_index_range=(1.3, 1.5),
            correlation_coefficient=-1.0,
        )
        amplitude_neg = generator.generate_correlated_amplitude(
            mask, n_field, rng, config_neg
        )

        # Both should produce valid amplitudes
        assert np.all(amplitude_pos >= 0.0) and np.all(amplitude_pos <= 1.0)
        assert np.all(amplitude_neg >= 0.0) and np.all(amplitude_neg <= 1.0)

        # Positive correlation should increase with n, negative should decrease
        # Check left vs right side
        left_pos = np.mean(amplitude_pos[:, : grid_spec.width // 4])
        right_pos = np.mean(amplitude_pos[:, -grid_spec.width // 4 :])
        left_neg = np.mean(amplitude_neg[:, : grid_spec.width // 4])
        right_neg = np.mean(amplitude_neg[:, -grid_spec.width // 4 :])

        assert right_pos > left_pos  # Positive correlation
        assert left_neg > right_neg  # Negative correlation


class TestPhaseGeneratorIntegration:
    """Integration tests for PhaseGenerator with different modes."""

    def test_gaussian_blobs_mode_integration(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase generation with gaussian_blobs refractive index mode."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="gaussian_blobs",
            thickness_mode="constant",
            gaussian_blob_count=3,
        )

        wavelength = 632.8e-9
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Phase should be in valid range
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

        # Phase should have spatial variation (not uniform)
        assert np.std(phase) > 0

    def test_perlin_noise_mode_integration(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase generation with perlin_noise refractive index mode."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="perlin_noise",
            thickness_mode="constant",
            perlin_noise_scale=30.0,
            perlin_noise_octaves=3,
        )

        wavelength = 532e-9
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Phase should be in valid range
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

        # Phase should have spatial variation
        assert np.std(phase) > 0

    def test_gradient_thickness_mode_integration(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase generation with gradient thickness mode."""
        mask = np.ones((grid_spec.height, grid_spec.width))

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="gradient",
            gradient_direction=0.0,
            gradient_magnitude=1.0,
        )

        wavelength = 632.8e-9
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Phase should be in valid range
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

        # Phase should have gradient (left side different from right side)
        left_phase = phase[:, : grid_spec.width // 4]
        right_phase = phase[:, -grid_spec.width // 4 :]
        assert not np.allclose(np.mean(left_phase), np.mean(right_phase))

    def test_shape_based_thickness_mode_integration(
        self, grid_spec: GridSpec, rng: Generator
    ) -> None:
        """Test phase generation with shape_based thickness mode."""
        # Create a circular mask
        mask = np.zeros((grid_spec.height, grid_spec.width))
        center_y, center_x = grid_spec.height // 2, grid_spec.width // 2
        radius = 20
        y, x = np.ogrid[: grid_spec.height, : grid_spec.width]
        circle_mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        mask[circle_mask] = 1.0

        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="uniform",
            thickness_mode="shape_based",
        )

        wavelength = 532e-9
        generator = PhaseGenerator()

        phase = generator.generate_phase(grid_spec, mask, wavelength, rng, config)

        # Phase should be in valid range
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

        # Center should have different phase than edges
        center_phase = phase[center_y, center_x]
        edge_phase = phase[center_y, center_x + radius - 2]
        assert not np.isclose(center_phase, edge_phase)

"""Tests for PhaseGenerationConfig validation."""

from __future__ import annotations

import pytest

from hologen.phase import PhaseGenerationConfig


class TestPhaseGenerationConfig:
    """Test PhaseGenerationConfig dataclass validation."""

    def test_valid_default_config(self) -> None:
        """Test that default configuration is valid."""
        config = PhaseGenerationConfig()
        assert config.enabled is False
        assert config.refractive_index_mode == "uniform"
        assert config.thickness_mode == "constant"
        assert config.ambient_refractive_index == 1.0
        assert config.refractive_index_range == (1.33, 1.55)
        assert config.thickness_range == (1e-6, 10e-6)
        assert config.correlation_coefficient == 0.0

    def test_valid_custom_config(self) -> None:
        """Test creation with valid custom parameters."""
        config = PhaseGenerationConfig(
            enabled=True,
            refractive_index_mode="gaussian_blobs",
            thickness_mode="gradient",
            ambient_refractive_index=1.0,
            refractive_index_range=(1.4, 1.6),
            thickness_range=(2e-6, 8e-6),
            correlation_coefficient=0.5,
        )
        assert config.enabled is True
        assert config.refractive_index_mode == "gaussian_blobs"
        assert config.thickness_mode == "gradient"
        assert config.correlation_coefficient == 0.5

    def test_invalid_refractive_index_mode(self) -> None:
        """Test that invalid refractive index mode raises error."""
        with pytest.raises(ValueError, match="Invalid refractive_index_mode"):
            PhaseGenerationConfig(refractive_index_mode="invalid_mode")

    def test_invalid_thickness_mode(self) -> None:
        """Test that invalid thickness mode raises error."""
        with pytest.raises(ValueError, match="Invalid thickness_mode"):
            PhaseGenerationConfig(thickness_mode="invalid_mode")

    def test_refractive_index_range_wrong_length(self) -> None:
        """Test that refractive index range with wrong length raises error."""
        with pytest.raises(ValueError, match="must be a tuple of 2 values"):
            PhaseGenerationConfig(refractive_index_range=(1.0, 1.5, 2.0))  # type: ignore

    def test_refractive_index_range_min_greater_than_max(self) -> None:
        """Test that refractive index range with min >= max raises error."""
        with pytest.raises(ValueError, match="must be less than max"):
            PhaseGenerationConfig(refractive_index_range=(1.6, 1.4))

    def test_refractive_index_range_below_physical_limit(self) -> None:
        """Test that refractive index below 1.0 raises error."""
        with pytest.raises(ValueError, match="must be >= 1.0"):
            PhaseGenerationConfig(refractive_index_range=(0.8, 1.5))

    def test_refractive_index_range_above_physical_limit(self) -> None:
        """Test that refractive index above 3.0 raises error."""
        with pytest.raises(ValueError, match="must be <= 3.0"):
            PhaseGenerationConfig(refractive_index_range=(1.3, 3.5))

    def test_thickness_range_wrong_length(self) -> None:
        """Test that thickness range with wrong length raises error."""
        with pytest.raises(ValueError, match="must be a tuple of 2 values"):
            PhaseGenerationConfig(thickness_range=(1e-6,))  # type: ignore

    def test_thickness_range_min_greater_than_max(self) -> None:
        """Test that thickness range with min >= max raises error."""
        with pytest.raises(ValueError, match="must be less than max"):
            PhaseGenerationConfig(thickness_range=(10e-6, 1e-6))

    def test_thickness_range_negative(self) -> None:
        """Test that negative thickness raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            PhaseGenerationConfig(thickness_range=(-1e-6, 10e-6))

    def test_correlation_coefficient_above_range(self) -> None:
        """Test that correlation coefficient above 1.0 raises error."""
        with pytest.raises(ValueError, match="must be in range"):
            PhaseGenerationConfig(correlation_coefficient=1.5)

    def test_correlation_coefficient_below_range(self) -> None:
        """Test that correlation coefficient below -1.0 raises error."""
        with pytest.raises(ValueError, match="must be in range"):
            PhaseGenerationConfig(correlation_coefficient=-1.5)

    def test_correlation_coefficient_boundary_values(self) -> None:
        """Test that boundary correlation coefficient values are valid."""
        config_min = PhaseGenerationConfig(correlation_coefficient=-1.0)
        assert config_min.correlation_coefficient == -1.0

        config_max = PhaseGenerationConfig(correlation_coefficient=1.0)
        assert config_max.correlation_coefficient == 1.0

    def test_gaussian_blob_count_zero(self) -> None:
        """Test that zero blob count raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            PhaseGenerationConfig(gaussian_blob_count=0)

    def test_gaussian_blob_count_negative(self) -> None:
        """Test that negative blob count raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            PhaseGenerationConfig(gaussian_blob_count=-5)

    def test_gaussian_blob_size_range_wrong_length(self) -> None:
        """Test that blob size range with wrong length raises error."""
        with pytest.raises(ValueError, match="must be a tuple of 2 values"):
            PhaseGenerationConfig(gaussian_blob_size_range=(10.0, 20.0, 30.0))  # type: ignore

    def test_gaussian_blob_size_range_min_greater_than_max(self) -> None:
        """Test that blob size range with min >= max raises error."""
        with pytest.raises(ValueError, match="must be less than max"):
            PhaseGenerationConfig(gaussian_blob_size_range=(50.0, 10.0))

    def test_gaussian_blob_size_range_non_positive(self) -> None:
        """Test that non-positive blob size raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            PhaseGenerationConfig(gaussian_blob_size_range=(0.0, 50.0))

        with pytest.raises(ValueError, match="must be > 0"):
            PhaseGenerationConfig(gaussian_blob_size_range=(-10.0, 50.0))

    def test_perlin_noise_scale_zero(self) -> None:
        """Test that zero Perlin noise scale raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            PhaseGenerationConfig(perlin_noise_scale=0.0)

    def test_perlin_noise_scale_negative(self) -> None:
        """Test that negative Perlin noise scale raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            PhaseGenerationConfig(perlin_noise_scale=-10.0)

    def test_perlin_noise_octaves_zero(self) -> None:
        """Test that zero Perlin noise octaves raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            PhaseGenerationConfig(perlin_noise_octaves=0)

    def test_perlin_noise_octaves_negative(self) -> None:
        """Test that negative Perlin noise octaves raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            PhaseGenerationConfig(perlin_noise_octaves=-3)

    def test_gradient_magnitude_above_range(self) -> None:
        """Test that gradient magnitude above 1.0 raises error."""
        with pytest.raises(ValueError, match="must be in range"):
            PhaseGenerationConfig(gradient_magnitude=1.5)

    def test_gradient_magnitude_below_range(self) -> None:
        """Test that gradient magnitude below 0.0 raises error."""
        with pytest.raises(ValueError, match="must be in range"):
            PhaseGenerationConfig(gradient_magnitude=-0.5)

    def test_gradient_magnitude_boundary_values(self) -> None:
        """Test that boundary gradient magnitude values are valid."""
        config_min = PhaseGenerationConfig(gradient_magnitude=0.0)
        assert config_min.gradient_magnitude == 0.0

        config_max = PhaseGenerationConfig(gradient_magnitude=1.0)
        assert config_max.gradient_magnitude == 1.0

    def test_all_refractive_index_modes_valid(self) -> None:
        """Test that all documented refractive index modes are valid."""
        modes = ["uniform", "gaussian_blobs", "perlin_noise"]
        for mode in modes:
            config = PhaseGenerationConfig(refractive_index_mode=mode)
            assert config.refractive_index_mode == mode

    def test_all_thickness_modes_valid(self) -> None:
        """Test that all documented thickness modes are valid."""
        modes = ["constant", "gradient", "shape_based"]
        for mode in modes:
            config = PhaseGenerationConfig(thickness_mode=mode)
            assert config.thickness_mode == mode

    def test_slots(self) -> None:
        """Test that PhaseGenerationConfig uses slots."""
        config = PhaseGenerationConfig()
        assert hasattr(config, "__slots__")

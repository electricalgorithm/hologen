"""Tests for noise model implementations."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from hologen.noise.aberrations import AberrationNoiseModel
from hologen.noise.composite import CompositeNoiseModel
from hologen.noise.sensor import SensorNoiseModel
from hologen.noise.speckle import SpeckleNoiseModel
from hologen.types import GridSpec, HolographyConfig, HolographyMethod, OpticalConfig


@pytest.fixture
def test_hologram(rng: Generator) -> np.ndarray:
    """Create a test hologram."""
    return rng.uniform(0.0, 1.0, size=(64, 64))


@pytest.fixture
def test_config() -> HolographyConfig:
    """Create a test holography configuration."""
    return HolographyConfig(
        grid=GridSpec(height=64, width=64, pixel_pitch=5e-6),
        optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.01),
        method=HolographyMethod.INLINE,
    )


class TestAberrationNoiseModel:
    """Test aberration noise model."""

    def test_zero_aberration_returns_unchanged(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test that zero aberration returns unchanged hologram."""
        model = AberrationNoiseModel(name="aberration")
        result = model.apply(test_hologram, test_config, rng)
        np.testing.assert_array_equal(result, test_hologram)

    def test_defocus_aberration(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test defocus aberration application."""
        model = AberrationNoiseModel(name="aberration", defocus=0.5)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)
        assert np.all(result >= 0)

    def test_astigmatism_x_aberration(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test astigmatism x aberration application."""
        model = AberrationNoiseModel(name="aberration", astigmatism_x=0.3)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_astigmatism_y_aberration(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test astigmatism y aberration application."""
        model = AberrationNoiseModel(name="aberration", astigmatism_y=0.3)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_coma_x_aberration(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test coma x aberration application."""
        model = AberrationNoiseModel(name="aberration", coma_x=0.2)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_coma_y_aberration(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test coma y aberration application."""
        model = AberrationNoiseModel(name="aberration", coma_y=0.2)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_combined_aberrations(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test combined aberrations."""
        model = AberrationNoiseModel(
            name="aberration",
            defocus=0.5,
            astigmatism_x=0.3,
            astigmatism_y=0.2,
            coma_x=0.1,
            coma_y=0.1,
        )
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)
        assert np.all(result >= 0)


class TestSpeckleNoiseModel:
    """Test speckle noise model."""

    def test_zero_contrast_returns_unchanged(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test that zero contrast returns unchanged hologram."""
        model = SpeckleNoiseModel(name="speckle", contrast=0.0)
        result = model.apply(test_hologram, test_config, rng)
        np.testing.assert_array_equal(result, test_hologram)

    def test_speckle_application(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test speckle noise application."""
        model = SpeckleNoiseModel(name="speckle", contrast=0.5, correlation_length=2.0)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)
        assert np.all(result >= 0)

    def test_speckle_with_zero_correlation(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test speckle with zero correlation length."""
        model = SpeckleNoiseModel(name="speckle", contrast=0.5, correlation_length=0.0)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_high_contrast_speckle(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test high contrast speckle."""
        model = SpeckleNoiseModel(name="speckle", contrast=1.0, correlation_length=3.0)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert np.all(result >= 0)


class TestSensorNoiseModel:
    """Test sensor noise model."""

    def test_dark_current_noise(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test dark current noise application."""
        model = SensorNoiseModel(name="sensor", dark_current=0.1)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        # Dark current adds positive values
        assert np.mean(result) >= np.mean(test_hologram)

    def test_shot_noise(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test shot noise application."""
        model = SensorNoiseModel(name="sensor", shot_noise=True)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_read_noise(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test read noise application."""
        model = SensorNoiseModel(name="sensor", read_noise=0.05)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_quantization(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test quantization application."""
        model = SensorNoiseModel(name="sensor", bit_depth=8)
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        # Check that values are quantized
        max_val = np.max(result)
        if max_val > 0:
            normalized = result / max_val
            unique_values = len(np.unique(normalized))
            assert unique_values <= 256

    def test_combined_sensor_noise(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test combined sensor noise."""
        model = SensorNoiseModel(
            name="sensor",
            read_noise=0.02,
            shot_noise=True,
            dark_current=0.05,
            bit_depth=10,
        )
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert np.all(result >= 0)

    def test_quantization_with_zero_max(
        self, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test quantization with zero maximum value."""
        zero_hologram = np.zeros((64, 64))
        model = SensorNoiseModel(name="sensor", bit_depth=8)
        result = model.apply(zero_hologram, test_config, rng)
        np.testing.assert_array_equal(result, zero_hologram)

    def test_shot_noise_with_zero_max(
        self, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test shot noise with zero maximum value."""
        zero_hologram = np.zeros((64, 64))
        model = SensorNoiseModel(name="sensor", shot_noise=True)
        result = model.apply(zero_hologram, test_config, rng)
        # Should handle zero max gracefully
        assert result.shape == zero_hologram.shape


class TestCompositeNoiseModel:
    """Test composite noise model."""

    def test_empty_models(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test composite with no models."""
        model = CompositeNoiseModel(name="composite", models=())
        result = model.apply(test_hologram, test_config, rng)
        np.testing.assert_array_equal(result, test_hologram)

    def test_single_model(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test composite with single model."""
        sensor = SensorNoiseModel(name="sensor", read_noise=0.05)
        model = CompositeNoiseModel(name="composite", models=(sensor,))
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_multiple_models(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test composite with multiple models."""
        speckle = SpeckleNoiseModel(name="speckle", contrast=0.3)
        sensor = SensorNoiseModel(name="sensor", read_noise=0.02)
        model = CompositeNoiseModel(name="composite", models=(speckle, sensor))
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert not np.array_equal(result, test_hologram)

    def test_all_noise_types(
        self, test_hologram: np.ndarray, test_config: HolographyConfig, rng: Generator
    ) -> None:
        """Test composite with all noise types."""
        aberration = AberrationNoiseModel(name="aberration", defocus=0.2)
        speckle = SpeckleNoiseModel(name="speckle", contrast=0.3)
        sensor = SensorNoiseModel(name="sensor", read_noise=0.02, shot_noise=True)
        model = CompositeNoiseModel(
            name="composite", models=(aberration, speckle, sensor)
        )
        result = model.apply(test_hologram, test_config, rng)
        assert result.shape == test_hologram.shape
        assert np.all(result >= 0)

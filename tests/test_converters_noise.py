"""Tests for noise model creation in converters."""

from __future__ import annotations

from hologen.converters import create_noise_model
from hologen.noise.aberrations import AberrationNoiseModel
from hologen.noise.composite import CompositeNoiseModel
from hologen.noise.sensor import SensorNoiseModel
from hologen.noise.speckle import SpeckleNoiseModel
from hologen.types import NoiseConfig


class TestCreateNoiseModel:
    """Test create_noise_model function."""

    def test_no_noise_returns_none(self) -> None:
        """Test that no noise configuration returns None."""
        config = NoiseConfig()
        result = create_noise_model(config)
        assert result is None

    def test_aberration_only(self) -> None:
        """Test creating aberration noise model only."""
        config = NoiseConfig(aberration_defocus=0.5)
        result = create_noise_model(config)
        assert isinstance(result, AberrationNoiseModel)
        assert result.defocus == 0.5

    def test_aberration_astigmatism_x(self) -> None:
        """Test creating aberration with astigmatism x."""
        config = NoiseConfig(aberration_astigmatism_x=0.3)
        result = create_noise_model(config)
        assert isinstance(result, AberrationNoiseModel)
        assert result.astigmatism_x == 0.3

    def test_aberration_astigmatism_y(self) -> None:
        """Test creating aberration with astigmatism y."""
        config = NoiseConfig(aberration_astigmatism_y=0.3)
        result = create_noise_model(config)
        assert isinstance(result, AberrationNoiseModel)
        assert result.astigmatism_y == 0.3

    def test_aberration_coma_x(self) -> None:
        """Test creating aberration with coma x."""
        config = NoiseConfig(aberration_coma_x=0.2)
        result = create_noise_model(config)
        assert isinstance(result, AberrationNoiseModel)
        assert result.coma_x == 0.2

    def test_aberration_coma_y(self) -> None:
        """Test creating aberration with coma y."""
        config = NoiseConfig(aberration_coma_y=0.2)
        result = create_noise_model(config)
        assert isinstance(result, AberrationNoiseModel)
        assert result.coma_y == 0.2

    def test_speckle_only(self) -> None:
        """Test creating speckle noise model only."""
        config = NoiseConfig(speckle_contrast=0.5, speckle_correlation_length=2.0)
        result = create_noise_model(config)
        assert isinstance(result, SpeckleNoiseModel)
        assert result.contrast == 0.5
        assert result.correlation_length == 2.0

    def test_sensor_read_noise_only(self) -> None:
        """Test creating sensor noise model with read noise."""
        config = NoiseConfig(sensor_read_noise=0.05)
        result = create_noise_model(config)
        assert isinstance(result, SensorNoiseModel)
        assert result.read_noise == 0.05

    def test_sensor_shot_noise_only(self) -> None:
        """Test creating sensor noise model with shot noise."""
        config = NoiseConfig(sensor_shot_noise=True)
        result = create_noise_model(config)
        assert isinstance(result, SensorNoiseModel)
        assert result.shot_noise is True

    def test_sensor_dark_current_only(self) -> None:
        """Test creating sensor noise model with dark current."""
        config = NoiseConfig(sensor_dark_current=0.1)
        result = create_noise_model(config)
        assert isinstance(result, SensorNoiseModel)
        assert result.dark_current == 0.1

    def test_sensor_bit_depth_only(self) -> None:
        """Test creating sensor noise model with bit depth."""
        config = NoiseConfig(sensor_bit_depth=8)
        result = create_noise_model(config)
        assert isinstance(result, SensorNoiseModel)
        assert result.bit_depth == 8

    def test_two_noise_types_creates_composite(self) -> None:
        """Test that two noise types create a composite model."""
        config = NoiseConfig(speckle_contrast=0.5, sensor_read_noise=0.05)
        result = create_noise_model(config)
        assert isinstance(result, CompositeNoiseModel)
        assert len(result.models) == 2

    def test_all_noise_types_creates_composite(self) -> None:
        """Test that all noise types create a composite model."""
        config = NoiseConfig(
            aberration_defocus=0.5,
            speckle_contrast=0.3,
            sensor_read_noise=0.02,
        )
        result = create_noise_model(config)
        assert isinstance(result, CompositeNoiseModel)
        assert len(result.models) == 3
        assert isinstance(result.models[0], AberrationNoiseModel)
        assert isinstance(result.models[1], SpeckleNoiseModel)
        assert isinstance(result.models[2], SensorNoiseModel)

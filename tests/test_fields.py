"""Tests for hologen.utils.fields module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.types import FieldRepresentation
from hologen.utils.fields import (
    FieldRepresentationError,
    PhaseRangeError,
    amplitude_phase_to_complex,
    complex_to_representation,
    validate_phase_range,
)


class TestComplexToRepresentation:
    """Test complex_to_representation function."""

    def test_intensity_representation(self) -> None:
        """Test conversion to intensity representation."""
        field = np.array([[1 + 1j, 2 + 0j, 0 + 2j]], dtype=np.complex128)
        intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)

        expected = np.array([[2.0, 4.0, 4.0]], dtype=np.float64)
        assert np.allclose(intensity, expected)
        assert intensity.dtype == np.float64

    def test_amplitude_representation(self) -> None:
        """Test conversion to amplitude representation."""
        field = np.array([[1 + 1j, 2 + 0j, 0 + 2j]], dtype=np.complex128)
        amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)

        expected = np.array([[np.sqrt(2), 2.0, 2.0]], dtype=np.float64)
        assert np.allclose(amplitude, expected)
        assert amplitude.dtype == np.float64

    def test_phase_representation(self) -> None:
        """Test conversion to phase representation."""
        field = np.array([[1 + 0j, 0 + 1j, -1 + 0j]], dtype=np.complex128)
        phase = complex_to_representation(field, FieldRepresentation.PHASE)

        expected = np.array([[0.0, np.pi / 2, np.pi]], dtype=np.float64)
        assert np.allclose(phase, expected)
        assert phase.dtype == np.float64

    def test_complex_representation(self) -> None:
        """Test that complex representation returns field unchanged."""
        field = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)
        result = complex_to_representation(field, FieldRepresentation.COMPLEX)

        assert result is field
        assert np.array_equal(result, field)

    def test_invalid_representation(self) -> None:
        """Test error for invalid representation type."""
        field = np.array([[1 + 1j]], dtype=np.complex128)

        with pytest.raises(
            FieldRepresentationError, match="Invalid field representation"
        ):
            complex_to_representation(field, "invalid")  # type: ignore

    def test_zero_amplitude(self) -> None:
        """Test conversion with zero amplitude."""
        field = np.array([[0 + 0j, 1 + 0j]], dtype=np.complex128)

        intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
        assert np.allclose(intensity, [[0.0, 1.0]])

        amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
        assert np.allclose(amplitude, [[0.0, 1.0]])

        phase = complex_to_representation(field, FieldRepresentation.PHASE)
        assert np.isfinite(phase).all()

    def test_uniform_phase(self) -> None:
        """Test conversion with uniform phase."""
        phase_value = np.pi / 4
        field = np.array([[1, 2, 3]], dtype=np.float64) * np.exp(1j * phase_value)

        phase = complex_to_representation(field, FieldRepresentation.PHASE)
        assert np.allclose(phase, phase_value)

    def test_2d_array(self) -> None:
        """Test conversion with 2D array."""
        field = np.ones((64, 64), dtype=np.complex128) * (1 + 1j)

        intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
        assert intensity.shape == (64, 64)
        assert np.allclose(intensity, 2.0)


class TestAmplitudePhaseToComplex:
    """Test amplitude_phase_to_complex function."""

    def test_basic_conversion(self) -> None:
        """Test basic amplitude and phase to complex conversion."""
        amplitude = np.array([[1.0, 2.0]], dtype=np.float64)
        phase = np.array([[0.0, np.pi / 2]], dtype=np.float64)

        field = amplitude_phase_to_complex(amplitude, phase)

        expected = np.array([[1 + 0j, 0 + 2j]], dtype=np.complex128)
        assert np.allclose(field, expected)
        assert field.dtype == np.complex128

    def test_round_trip_conversion(self) -> None:
        """Test round-trip conversion: complex -> amplitude/phase -> complex."""
        original_field = np.array([[1 + 1j, 2 + 0j, 0 + 3j]], dtype=np.complex128)

        amplitude = np.abs(original_field)
        phase = np.angle(original_field)

        reconstructed_field = amplitude_phase_to_complex(amplitude, phase)

        assert np.allclose(reconstructed_field, original_field)

    def test_zero_amplitude(self) -> None:
        """Test conversion with zero amplitude."""
        amplitude = np.array([[0.0, 1.0]], dtype=np.float64)
        phase = np.array([[np.pi, 0.0]], dtype=np.float64)

        field = amplitude_phase_to_complex(amplitude, phase)

        assert np.allclose(np.abs(field), amplitude)
        assert field[0, 0] == 0 + 0j

    def test_uniform_amplitude(self) -> None:
        """Test conversion with uniform amplitude."""
        amplitude = np.ones((32, 32), dtype=np.float64)
        phase = np.random.uniform(-np.pi, np.pi, (32, 32))

        field = amplitude_phase_to_complex(amplitude, phase)

        assert np.allclose(np.abs(field), 1.0)
        assert np.allclose(np.angle(field), phase)

    def test_2d_arrays(self) -> None:
        """Test conversion with 2D arrays."""
        amplitude = np.ones((64, 64), dtype=np.float64) * 2.0
        phase = np.zeros((64, 64), dtype=np.float64)

        field = amplitude_phase_to_complex(amplitude, phase)

        assert field.shape == (64, 64)
        assert np.allclose(field, 2.0 + 0j)


class TestValidatePhaseRange:
    """Test validate_phase_range function."""

    def test_valid_phase_range(self) -> None:
        """Test validation passes for valid phase range."""
        phase = np.array(
            [[0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi]], dtype=np.float64
        )

        # Should not raise
        validate_phase_range(phase)

    def test_phase_above_range(self) -> None:
        """Test validation fails for phase above π."""
        phase = np.array([[0.0, 3.5]], dtype=np.float64)

        with pytest.raises(PhaseRangeError, match="Phase values must be in the range"):
            validate_phase_range(phase)

    def test_phase_below_range(self) -> None:
        """Test validation fails for phase below -π."""
        phase = np.array([[0.0, -3.5]], dtype=np.float64)

        with pytest.raises(PhaseRangeError, match="Phase values must be in the range"):
            validate_phase_range(phase)

    def test_nan_values(self) -> None:
        """Test validation fails for NaN values."""
        phase = np.array([[0.0, np.nan]], dtype=np.float64)

        with pytest.raises(PhaseRangeError, match="non-finite values"):
            validate_phase_range(phase)

    def test_inf_values(self) -> None:
        """Test validation fails for infinite values."""
        phase = np.array([[0.0, np.inf]], dtype=np.float64)

        with pytest.raises(PhaseRangeError, match="non-finite values"):
            validate_phase_range(phase)

    def test_boundary_values(self) -> None:
        """Test validation passes for boundary values."""
        phase = np.array([[np.pi, -np.pi]], dtype=np.float64)

        # Should not raise
        validate_phase_range(phase)

    def test_2d_array(self) -> None:
        """Test validation with 2D array."""
        phase = np.random.uniform(-np.pi, np.pi, (64, 64))

        # Should not raise
        validate_phase_range(phase)

    def test_uniform_phase(self) -> None:
        """Test validation with uniform phase."""
        phase = np.ones((32, 32), dtype=np.float64) * (np.pi / 4)

        # Should not raise
        validate_phase_range(phase)

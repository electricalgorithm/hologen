"""Field representation utilities for complex-valued optical fields."""

from __future__ import annotations

import numpy as np

from hologen.types import ArrayComplex, ArrayFloat, FieldRepresentation


class FieldRepresentationError(ValueError):
    """Raised when field representation is invalid or incompatible."""


class PhaseRangeError(ValueError):
    """Raised when phase values are outside the valid [-π, π] range."""


def complex_to_representation(
    field: ArrayComplex, representation: FieldRepresentation
) -> ArrayFloat | ArrayComplex:
    """Convert a complex field to the requested representation.
    
    Args:
        field: Complex-valued optical field.
        representation: Target representation type.
        
    Returns:
        Field in the requested representation. Returns ArrayFloat for intensity,
        amplitude, and phase representations. Returns ArrayComplex for complex
        representation (no conversion).
        
    Raises:
        FieldRepresentationError: If the representation type is invalid.
        
    Examples:
        >>> field = np.array([[1+1j, 2+0j]], dtype=np.complex128)
        >>> intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
        >>> amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
        >>> phase = complex_to_representation(field, FieldRepresentation.PHASE)
    """
    if representation == FieldRepresentation.INTENSITY:
        return np.abs(field) ** 2
    elif representation == FieldRepresentation.AMPLITUDE:
        return np.abs(field)
    elif representation == FieldRepresentation.PHASE:
        return np.angle(field)
    elif representation == FieldRepresentation.COMPLEX:
        return field
    else:
        raise FieldRepresentationError(
            f"Invalid field representation: {representation}. "
            f"Valid options are: {', '.join(r.value for r in FieldRepresentation)}"
        )


def amplitude_phase_to_complex(
    amplitude: ArrayFloat, phase: ArrayFloat
) -> ArrayComplex:
    """Construct a complex field from separate amplitude and phase arrays.
    
    Uses the formula: field = amplitude * exp(i * phase)
    
    Args:
        amplitude: Amplitude values (non-negative).
        phase: Phase values in radians.
        
    Returns:
        Complex field with the specified amplitude and phase.
        
    Examples:
        >>> amplitude = np.array([[1.0, 2.0]], dtype=np.float64)
        >>> phase = np.array([[0.0, np.pi/2]], dtype=np.float64)
        >>> field = amplitude_phase_to_complex(amplitude, phase)
    """
    return amplitude * np.exp(1j * phase)


def validate_phase_range(phase: ArrayFloat) -> None:
    """Validate that all phase values are within the valid [-π, π] range.
    
    Args:
        phase: Phase array in radians.
        
    Raises:
        PhaseRangeError: If any phase values are outside [-π, π] or non-finite.
        
    Examples:
        >>> phase = np.array([[0.0, np.pi/2, -np.pi/2]], dtype=np.float64)
        >>> validate_phase_range(phase)  # No error
        >>> invalid_phase = np.array([[4.0]], dtype=np.float64)
        >>> validate_phase_range(invalid_phase)  # Raises PhaseRangeError
    """
    if not np.isfinite(phase).all():
        raise PhaseRangeError(
            "Phase array contains non-finite values (NaN or Inf). "
            "All phase values must be finite numbers."
        )
    
    if not np.all((-np.pi <= phase) & (phase <= np.pi)):
        min_phase = np.min(phase)
        max_phase = np.max(phase)
        raise PhaseRangeError(
            f"Phase values must be in the range [-π, π] radians. "
            f"Found values in range [{min_phase:.4f}, {max_phase:.4f}]."
        )

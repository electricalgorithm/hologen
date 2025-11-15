"""Input/output utilities for holography datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hologen.types import (
    ArrayComplex,
    ComplexHologramSample,
    ComplexObjectSample,
    FieldRepresentation,
    HologramSample,
    ObjectSample,
)
from hologen.utils.fields import complex_to_representation
from hologen.utils.math import normalize_image


@dataclass(slots=True)
class NumpyDatasetWriter:
    """Persist holography samples in NumPy archives and optional PNG previews.

    Args:
        save_preview: Whether to generate PNG previews for each domain.
    """

    save_preview: bool = True

    def save(self, samples: Iterable[HologramSample], output_dir: Path) -> None:
        """Write hologram samples to disk.

        Args:
            samples: Iterable of hologram samples produced by the pipeline.
            output_dir: Target directory for serialized dataset artifacts.

        Raises:
            IOError: If the dataset cannot be written to the storage path.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            prefix = f"sample_{index:05d}_{sample.object_sample.name}"
            base_path = output_dir / prefix
            np.savez(
                base_path.with_suffix(".npz"),
                object=sample.object_sample.pixels,
                hologram=sample.hologram,
                reconstruction=sample.reconstruction,
            )

            if not self.save_preview:
                continue

            object_image = normalize_image(sample.object_sample.pixels)
            hologram_image = normalize_image(sample.hologram)
            reconstruction_image = normalize_image(sample.reconstruction)

            self._write_png(base_path.with_name(prefix + "_object.png"), object_image)
            self._write_png(
                base_path.with_name(prefix + "_hologram.png"), hologram_image
            )
            self._write_png(
                base_path.with_name(prefix + "_reconstruction.png"),
                reconstruction_image,
            )

    def _write_png(self, path: Path, image: np.ndarray) -> None:
        """Persist a single-channel PNG image.

        Args:
            path: Destination path for the PNG file.
            image: Normalized floating-point image in ``[0.0, 1.0]``.

        Raises:
            IOError: If the image cannot be written to disk.
        """

        pil_image = Image.fromarray((image * 255.0).astype(np.uint8), mode="L")
        pil_image.save(path)


@dataclass(slots=True)
class ComplexFieldWriter:
    """Persist complex holography samples in NumPy archives and optional PNG previews.

    Args:
        save_preview: Whether to generate PNG previews for each domain.
        phase_colormap: Matplotlib colormap name for phase visualization.
    """

    save_preview: bool = True
    phase_colormap: str = "twilight"

    def save(
        self, samples: Iterable[ComplexHologramSample], output_dir: Path
    ) -> None:
        """Write complex hologram samples to disk.

        Args:
            samples: Iterable of complex hologram samples produced by the pipeline.
            output_dir: Target directory for serialized dataset artifacts.

        Raises:
            IOError: If the dataset cannot be written to the storage path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            prefix = f"sample_{index:05d}_{sample.object_sample.name}"
            base_path = output_dir / prefix

            # Save object domain
            self._save_npz_complex(
                base_path.with_name(prefix + "_object.npz"),
                sample.object_sample.field,
                sample.object_sample.representation,
                {},
            )

            # Save hologram
            self._save_npz_complex(
                base_path.with_name(prefix + "_hologram.npz"),
                sample.hologram_field,
                sample.hologram_representation,
                {},
            )

            # Save reconstruction
            self._save_npz_complex(
                base_path.with_name(prefix + "_reconstruction.npz"),
                sample.reconstruction_field,
                sample.reconstruction_representation,
                {},
            )

            if not self.save_preview:
                continue

            # Save PNG previews
            self._save_png_complex(
                base_path.with_name(prefix + "_object"),
                sample.object_sample.field,
                sample.object_sample.representation,
            )
            self._save_png_complex(
                base_path.with_name(prefix + "_hologram"),
                sample.hologram_field,
                sample.hologram_representation,
            )
            self._save_png_complex(
                base_path.with_name(prefix + "_reconstruction"),
                sample.reconstruction_field,
                sample.reconstruction_representation,
            )

    def _save_npz_complex(
        self,
        path: Path,
        field: ArrayComplex,
        representation: FieldRepresentation,
        metadata: dict,
    ) -> None:
        """Save complex field to NumPy archive with metadata.

        Args:
            path: Destination path for the .npz file.
            field: Complex field to save.
            representation: Field representation type.
            metadata: Additional metadata to include in the archive.

        Raises:
            IOError: If the file cannot be written to disk.
        """
        save_dict = {"representation": str(representation.value)}
        save_dict.update(metadata)

        if representation == FieldRepresentation.COMPLEX:
            save_dict["real"] = field.real
            save_dict["imag"] = field.imag
        elif representation == FieldRepresentation.AMPLITUDE:
            save_dict["amplitude"] = np.abs(field)
        elif representation == FieldRepresentation.PHASE:
            save_dict["phase"] = np.angle(field)
        elif representation == FieldRepresentation.INTENSITY:
            save_dict["intensity"] = np.abs(field) ** 2

        np.savez(path, **save_dict)

    def _save_png_complex(
        self,
        base_path: Path,
        field: ArrayComplex,
        representation: FieldRepresentation,
    ) -> None:
        """Save complex field as PNG preview image(s).

        For COMPLEX representation, generates two separate PNG files with
        '_amplitude' and '_phase' suffixes. For other representations,
        generates a single PNG file.

        Args:
            base_path: Base path for PNG file(s) (without extension).
            field: Complex field to visualize.
            representation: Field representation type.

        Raises:
            IOError: If the PNG file(s) cannot be written to disk.
        """
        if representation == FieldRepresentation.COMPLEX:
            # Save amplitude and phase as separate PNGs
            amplitude_path = base_path.with_name(base_path.name + "_amplitude.png")
            phase_path = base_path.with_name(base_path.name + "_phase.png")
            self._write_png_amplitude(amplitude_path, np.abs(field))
            self._write_png_phase(phase_path, np.angle(field))
        elif representation == FieldRepresentation.AMPLITUDE:
            amplitude_path = base_path.with_suffix(".png")
            self._write_png_amplitude(amplitude_path, np.abs(field))
        elif representation == FieldRepresentation.PHASE:
            phase_path = base_path.with_suffix(".png")
            self._write_png_phase(phase_path, np.angle(field))
        elif representation == FieldRepresentation.INTENSITY:
            intensity_path = base_path.with_suffix(".png")
            intensity = np.abs(field) ** 2
            normalized = normalize_image(intensity)
            self._write_png(intensity_path, normalized)

    def _write_png_amplitude(self, path: Path, amplitude: np.ndarray) -> None:
        """Write amplitude values as a grayscale PNG.

        Args:
            path: Destination path for the PNG file.
            amplitude: Amplitude values (non-negative).

        Raises:
            IOError: If the image cannot be written to disk.
        """
        normalized = normalize_image(amplitude)
        self._write_png(path, normalized)

    def _write_png_phase(self, path: Path, phase: np.ndarray) -> None:
        """Write phase values as a PNG with optional colormap.

        Maps phase values from [-π, π] to [0, 255] for 8-bit encoding.
        If phase_colormap is specified, applies the colormap for visualization.

        Args:
            path: Destination path for the PNG file.
            phase: Phase values in radians [-π, π].

        Raises:
            IOError: If the image cannot be written to disk.
        """
        # Map phase from [-π, π] to [0, 1]
        normalized_phase = (phase + np.pi) / (2 * np.pi)

        if self.phase_colormap:
            # Apply colormap using matplotlib
            try:
                import matplotlib.pyplot as plt

                cmap = plt.get_cmap(self.phase_colormap)
                colored = cmap(normalized_phase)
                # Convert RGBA to RGB (drop alpha channel)
                rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
                pil_image = Image.fromarray(rgb_image, mode="RGB")
                pil_image.save(path)
            except ImportError:
                # Fallback to grayscale if matplotlib is not available
                self._write_png(path, normalized_phase)
        else:
            # Save as grayscale
            self._write_png(path, normalized_phase)

    def _write_png(self, path: Path, image: np.ndarray) -> None:
        """Persist a single-channel PNG image.

        Args:
            path: Destination path for the PNG file.
            image: Normalized floating-point image in ``[0.0, 1.0]``.

        Raises:
            IOError: If the image cannot be written to disk.
        """
        pil_image = Image.fromarray((image * 255.0).astype(np.uint8), mode="L")
        pil_image.save(path)


def load_complex_sample(path: Path) -> ComplexObjectSample | ObjectSample:
    """Load a sample from a NumPy archive with automatic format detection.

    Detects whether the file contains complex field data (new format with
    'real'/'imag' keys) or legacy intensity data ('object' key) and returns
    the appropriate sample type.

    Args:
        path: Path to the .npz file.

    Returns:
        ComplexObjectSample if the file contains complex field data,
        ObjectSample if it contains legacy intensity data.

    Raises:
        ValueError: If the file format is not recognized.
        IOError: If the file cannot be read.

    Examples:
        >>> # Load complex field data
        >>> sample = load_complex_sample(Path("complex_object.npz"))
        >>> isinstance(sample, ComplexObjectSample)
        True

        >>> # Load legacy intensity data
        >>> sample = load_complex_sample(Path("legacy_object.npz"))
        >>> isinstance(sample, ObjectSample)
        True
    """
    data = np.load(path)

    # Detect format based on keys present
    if "real" in data and "imag" in data:
        # New complex format
        field = data["real"] + 1j * data["imag"]
        representation = FieldRepresentation(data.get("representation", "complex"))
        return ComplexObjectSample(
            name=path.stem, field=field, representation=representation
        )
    elif "amplitude" in data:
        # Amplitude-only format
        amplitude = data["amplitude"]
        field = amplitude.astype(np.complex128)
        return ComplexObjectSample(
            name=path.stem,
            field=field,
            representation=FieldRepresentation.AMPLITUDE,
        )
    elif "phase" in data:
        # Phase-only format
        phase = data["phase"]
        field = np.exp(1j * phase)
        return ComplexObjectSample(
            name=path.stem, field=field, representation=FieldRepresentation.PHASE
        )
    elif "intensity" in data:
        # Intensity format (new style)
        intensity = data["intensity"]
        field = np.sqrt(intensity).astype(np.complex128)
        return ComplexObjectSample(
            name=path.stem,
            field=field,
            representation=FieldRepresentation.INTENSITY,
        )
    elif "object" in data:
        # Legacy intensity format
        pixels = data["object"]
        return ObjectSample(name=path.stem, pixels=pixels)
    else:
        raise ValueError(
            f"Unknown .npz format in {path}. Expected keys: "
            "'real'/'imag' (complex), 'amplitude', 'phase', 'intensity', or 'object' (legacy)."
        )

"""Input/output utilities for holography datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hologen.types import HologramSample
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

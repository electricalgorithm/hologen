from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from PIL import Image

from hologen.converters import (
    HologramDatasetGenerator,
    NoiseConfig,
    ObjectDomainProducer,
    ObjectToHologramConverter,
    create_noise_model,
)
from hologen.phase import PhaseGenerationConfig
from hologen.types import (
    ArrayFloat,
    ComplexHologramSample,
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OpticalConfig,
)
from hologen.utils.math import normalize_image

# Dataset Configurations
SAMPLES: int = 2
OUTPUT_DIR: Path = Path("dataset")
RANDOM_SEED: int = 42

# Experiment Configurations
GRID_HEIGHT: int = 256  # pixels
GRID_WIDTH: int = 256  # pixels
DETECTOR_PIXEL_PITCH: float = 3.45e-6  # meters
SOURCE_WAVELENGTH: float = 632.8e-9  # meters
PROPAGATION_DISTANCE: float = 1e-2  # meters
SIGNAL_TO_NOISE_RATIO: float = 15.0

# Phase Generation Configurations
REFRACTIVE_INDEX_MODE: str = "gaussian_blobs"
REFRACTIVE_INDEX_RANGE_MIN: float = 1.36
REFRACTIVE_INDEX_RANGE_MAX: float = 1.40
GAUSSIAN_BLOB_COUNT: int = 2
GAUSSIAN_BLOB_SIZE_RANGE_MIN: float = 40.0
GAUSSIAN_BLOB_SIZE_RANGE_MAX: float = 70.0
AMBIENT_REFRACTIVE_INDEX: float = 1.0
THICKNESS_MODE: str = "shape_based"
THICKNESS_RANGE_MIN: float = 4e-6
THICKNESS_RANGE_MAX: float = 7e-6
CORRELATION_COEFFICIENT: float = 0.7

# Noise Configurations
SPECKLE_CORRELATION_LENGTH: float = 1.0
SPECKLE_CONTRAST_MIN: float = 0.1
SPECKLE_CONTRAST_MAX: float = 0.9
SENSOR_READ_NOISE_MEAN: float = 5.0
SENSOR_READ_NOISE_STD: float = 5.0
SENSOR_BIT_DEPTH: int = 8
SENSOR_DARK_CURRENT_MIN: float = 0.2
SENSOR_DARK_CURRENT_MAX: float = 1.0
DEFOCUS_UM_MIN: float = -30.0
DEFOCUS_UM_MAX: float = 30.0
ASTIGMATISM_X_MIN: float = -0.5
ASTIGMATISM_X_MAX: float = 0.5
ASTIGMATISM_Y_MIN: float = -0.5
ASTIGMATISM_Y_MAX: float = 0.5
COMA_X_MIN: float = -0.5
COMA_X_MAX: float = 0.5
COMA_Y_MIN: float = -0.5
COMA_Y_MAX: float = 0.5


def save_sample(
    output_dir: Path,
    index: int,
    sample: ComplexHologramSample,
    metadata: dict[str, Any],
) -> None:
    """Save a single sample with all data and visualizations.

    Args:
        output_dir: Output directory.
        index: Sample index.
        sample: ComplexHologramSample
        metadata: Metadata dictionary.
    """
    prefix = f"sample_{index:05d}_{sample.object_sample.name}"

    object_intensity = np.abs(sample.object_sample.field) ** 2
    object_phase = np.angle(sample.object_sample.field)
    hologram = np.abs(sample.hologram_field) ** 2
    reconstruction_intensity = np.abs(sample.reconstruction_field) ** 2
    reconstruction_phase = np.angle(sample.reconstruction_field)

    # Save NPZ with all data
    np.savez(
        output_dir / f"{prefix}.npz",
        object_intensity=object_intensity,
        object_phase=object_phase,
        hologram=hologram,
        reconstruction_intensity=reconstruction_intensity,
        reconstruction_phase=reconstruction_phase,
        metadata=metadata,
    )

    # Save PNG visualizations
    def save_png(path: Path, image: ArrayFloat, is_phase: bool = False):
        if is_phase:
            # Map phase to [0, 1] and apply colormap
            phase_range = image.max() - image.min()
            if phase_range > 1e-6:
                normalized = (image - image.min()) / phase_range
            else:
                normalized = np.zeros_like(image)
            try:
                cmap = plt.get_cmap("twilight")
                colored = cmap(normalized)
                rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
                pil_image = Image.fromarray(rgb_image, mode="RGB")
                pil_image.save(path)
            except ImportError:
                normalized_img = normalize_image(normalized)
                pil_image = Image.fromarray(
                    (normalized_img * 255).astype(np.uint8), mode="L"
                )
                pil_image.save(path)
        else:
            normalized = normalize_image(image)
            pil_image = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
            pil_image.save(path)

    save_png(output_dir / f"{prefix}_object_intensity.png", object_intensity)
    save_png(output_dir / f"{prefix}_object_phase.png", object_phase, is_phase=True)
    save_png(output_dir / f"{prefix}_hologram.png", hologram)
    save_png(
        output_dir / f"{prefix}_reconstruction_intensity.png", reconstruction_intensity
    )
    save_png(
        output_dir / f"{prefix}_reconstruction_phase.png",
        reconstruction_phase,
        is_phase=True,
    )


def main():
    """The main entrypoint of the script."""
    # Create a random generator to use in all subsystems.
    randomizer = default_rng(RANDOM_SEED)

    # Choose random noise levels.
    speckle_contrast: float = float(
        randomizer.uniform(SPECKLE_CONTRAST_MIN, SPECKLE_CONTRAST_MAX)
    )
    sensor_read_noise: float = float(
        max(0.0, randomizer.normal(SENSOR_READ_NOISE_MEAN, SENSOR_READ_NOISE_STD))
    )
    sensor_dark_current: float = float(
        randomizer.uniform(SENSOR_DARK_CURRENT_MIN, SENSOR_DARK_CURRENT_MAX)
    )
    defocus: float = float(
        randomizer.uniform(DEFOCUS_UM_MIN, DEFOCUS_UM_MAX)
    )  # micrometers
    astigmatism_x: float = float(
        randomizer.uniform(ASTIGMATISM_X_MIN, ASTIGMATISM_X_MAX)
    )
    astigmatism_y: float = float(
        randomizer.uniform(ASTIGMATISM_Y_MIN, ASTIGMATISM_Y_MAX)
    )
    coma_x = float(randomizer.uniform(COMA_X_MIN, COMA_X_MAX))
    coma_y = float(randomizer.uniform(COMA_Y_MIN, COMA_Y_MAX))

    # Create dataset generator
    generator = HologramDatasetGenerator(
        object_producer=ObjectDomainProducer(
            phase_config=PhaseGenerationConfig(
                refractive_index_mode=REFRACTIVE_INDEX_MODE,
                thickness_mode=THICKNESS_MODE,
                ambient_refractive_index=AMBIENT_REFRACTIVE_INDEX,
                refractive_index_range=(
                    REFRACTIVE_INDEX_RANGE_MIN,
                    REFRACTIVE_INDEX_RANGE_MAX,
                ),  # Narrow range for smooth phase
                thickness_range=(
                    THICKNESS_RANGE_MIN,
                    THICKNESS_RANGE_MAX,
                ),  # Moderate thickness
                correlation_coefficient=CORRELATION_COEFFICIENT,  # Strong correlation for smoothness
                gaussian_blob_count=GAUSSIAN_BLOB_COUNT,  # Few blobs for smooth variation
                gaussian_blob_size_range=(
                    GAUSSIAN_BLOB_SIZE_RANGE_MIN,
                    GAUSSIAN_BLOB_SIZE_RANGE_MAX,
                ),  # Large blobs
            )
        ),
        converter=ObjectToHologramConverter(
            config=HolographyConfig(
                method=HolographyMethod.INLINE,
                grid=GridSpec(
                    height=GRID_HEIGHT,
                    width=GRID_WIDTH,
                    pixel_pitch=DETECTOR_PIXEL_PITCH,
                ),
                optics=OpticalConfig(
                    wavelength=SOURCE_WAVELENGTH,
                    propagation_distance=PROPAGATION_DISTANCE,
                ),
            ),
            noise_model=create_noise_model(
                NoiseConfig(
                    sensor_read_noise=sensor_read_noise,
                    sensor_shot_noise=True,
                    sensor_dark_current=sensor_dark_current,
                    sensor_bit_depth=SENSOR_BIT_DEPTH,
                    speckle_contrast=speckle_contrast,
                    speckle_correlation_length=SPECKLE_CORRELATION_LENGTH,
                    aberration_defocus=defocus,
                    aberration_astigmatism_x=astigmatism_x,
                    aberration_astigmatism_y=astigmatism_y,
                    aberration_coma_x=coma_x,
                    aberration_coma_y=coma_y,
                )
            ),
        ),
    )

    # Create the output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(generator.generate(count=SAMPLES, rng=randomizer)):
        save_sample(
            OUTPUT_DIR,
            index,
            sample,
            metadata={
                "shape_name": sample.object_sample.name,
                "grid_height": GRID_HEIGHT,
                "grid_width": GRID_WIDTH,
                "pixel_pitch": DETECTOR_PIXEL_PITCH,
                "wavelength": SOURCE_WAVELENGTH,
                "propagation_distance": PROPAGATION_DISTANCE,
                "snr_db": SIGNAL_TO_NOISE_RATIO,
                "holography_method": str(HolographyMethod.INLINE),
                "refractive_index_mode": REFRACTIVE_INDEX_MODE,
                "thickness_mode": THICKNESS_MODE,
                "refractive_index_min": REFRACTIVE_INDEX_RANGE_MIN,
                "refractive_index_max": REFRACTIVE_INDEX_RANGE_MAX,
                "thickness_min": THICKNESS_RANGE_MIN,
                "thickness_max": THICKNESS_RANGE_MAX,
                "correlation_coefficient": CORRELATION_COEFFICIENT,
            },
        )


if __name__ == "__main__":
    main()

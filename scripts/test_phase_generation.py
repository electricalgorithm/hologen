#!/usr/bin/env python
"""Test script to generate dataset with physics-based phase generation."""

from pathlib import Path

from numpy.random import default_rng

from hologen import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OpticalConfig,
    PhaseGenerationConfig,
)
from hologen.converters import (
    HologramDatasetGenerator,
    default_converter,
    default_object_producer,
)
from hologen.types import FieldRepresentation, OutputConfig
from hologen.utils.io import ComplexFieldWriter

# Configuration
SAMPLES = 10
OUTPUT_DIR = Path("phase_test_dataset")
SEED = 42

# Create holography configuration
grid = GridSpec(height=256, width=256, pixel_pitch=4.65e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.INLINE,
    carrier=None,
)

# Create phase generation configurations to test different modes
phase_configs = [
    # Uniform refractive index with constant thickness
    PhaseGenerationConfig(
        enabled=True,
        refractive_index_mode="uniform",
        thickness_mode="constant",
        refractive_index_range=(1.33, 1.55),
        thickness_range=(1e-6, 10e-6),
    ),
    # Gaussian blobs with shape-based thickness
    PhaseGenerationConfig(
        enabled=True,
        refractive_index_mode="gaussian_blobs",
        thickness_mode="shape_based",
        gaussian_blob_count=5,
        gaussian_blob_size_range=(10.0, 50.0),
        refractive_index_range=(1.33, 1.55),
        thickness_range=(1e-6, 10e-6),
    ),
    # Perlin noise with gradient thickness
    PhaseGenerationConfig(
        enabled=True,
        refractive_index_mode="perlin_noise",
        thickness_mode="gradient",
        perlin_noise_scale=50.0,
        perlin_noise_octaves=4,
        gradient_direction=0.785,  # 45 degrees
        gradient_magnitude=0.5,
        refractive_index_range=(1.33, 1.55),
        thickness_range=(1e-6, 10e-6),
    ),
]

# Generate datasets for each configuration
for idx, phase_config in enumerate(phase_configs):
    print(f"\n{'=' * 60}")
    print(f"Generating dataset {idx + 1}/{len(phase_configs)}")
    print(f"Refractive index mode: {phase_config.refractive_index_mode}")
    print(f"Thickness mode: {phase_config.thickness_mode}")
    print(f"{'=' * 60}\n")

    # Create output directory for this configuration
    output_dir = (
        OUTPUT_DIR
        / f"config_{idx + 1}_{phase_config.refractive_index_mode}_{phase_config.thickness_mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG
    rng = default_rng(SEED + idx)

    # Create producer and converter
    producer = default_object_producer()
    converter = default_converter(noise_model=None)

    # Configure output to save complex fields (amplitude + phase)
    converter.output_config = OutputConfig(
        object_representation=FieldRepresentation.COMPLEX,
        hologram_representation=FieldRepresentation.COMPLEX,
        reconstruction_representation=FieldRepresentation.COMPLEX,
    )

    # Create dataset generator
    generator = HologramDatasetGenerator(
        object_producer=producer,
        converter=converter,
    )

    # Create writer with PNG preview enabled and phase colormap
    writer = ComplexFieldWriter(save_preview=True, phase_colormap="hsv")

    # Generate samples with phase configuration
    samples = generator.generate(
        count=SAMPLES,
        config=config,
        rng=rng,
        mode="complex",  # Use complex mode to enable phase generation
        use_complex=True,
        phase_config=phase_config,
    )

    # Save samples
    writer.save(samples=samples, output_dir=output_dir)

    print(f"✓ Saved {SAMPLES} samples to {output_dir}")

print(f"\n{'=' * 60}")
print("All datasets generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'=' * 60}\n")

"""Advanced dataset generation with per-sample randomization and metadata logging."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectToHologramConverter,
    create_noise_model,
    default_object_producer,
)
from hologen.types import (
    FieldRepresentation,
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    NoiseConfig,
    OpticalConfig,
    OutputConfig,
)
from hologen.utils.io import ComplexFieldWriter


def randomize_sample_config(
    rng: Generator,
    base_grid: GridSpec,
    wavelengths: list[float],
) -> tuple[HolographyConfig, NoiseConfig, dict]:
    """Generate randomized configuration for a single sample.

    Args:
        rng: Random number generator.
        base_grid: Base grid specification (height, width, pixel_pitch).
        wavelengths: List of wavelengths to sample from.

    Returns:
        Tuple of (HolographyConfig, NoiseConfig, metadata_dict).
    """
    # Randomize optical parameters
    wavelength = float(rng.choice(wavelengths))
    propagation_distance = float(rng.uniform(0.03, 0.15))

    # Only use complex objects (amplitude + phase modulation)
    object_type = "complex"

    # Randomize phase shift for phase objects (must be in [-π, π])
    phase_shift = float(rng.uniform(-np.pi, np.pi))

    # Randomize noise parameters
    speckle_contrast = float(rng.uniform(0.1, 0.9))
    sensor_read_noise_mean = 5.0
    sensor_read_noise_std = 5.0
    sensor_read_noise = float(
        max(0.0, rng.normal(sensor_read_noise_mean, sensor_read_noise_std))
    )
    sensor_dark_current = float(rng.uniform(0.2, 1.0))
    sensor_bit_depth = int(rng.integers(8, 17))  # 8 to 16 bits

    # Randomize aberrations (in micrometers for defocus, dimensionless for others)
    aberration_defocus = float(rng.uniform(-30.0, 30.0))  # micrometers
    aberration_astigmatism_x = float(rng.uniform(-0.5, 0.5))
    aberration_astigmatism_y = float(rng.uniform(-0.5, 0.5))
    aberration_coma_x = float(rng.uniform(-0.5, 0.5))
    aberration_coma_y = float(rng.uniform(-0.5, 0.5))

    # Create optical config
    optics = OpticalConfig(
        wavelength=wavelength,
        propagation_distance=propagation_distance,
    )

    # Use inline holography
    config = HolographyConfig(
        grid=base_grid,
        optics=optics,
        method=HolographyMethod.INLINE,
        carrier=None,
    )

    # Create noise config
    noise_config = NoiseConfig(
        sensor_read_noise=sensor_read_noise,
        sensor_shot_noise=True,  # Always enable Poisson shot noise
        sensor_dark_current=sensor_dark_current,
        sensor_bit_depth=sensor_bit_depth,
        speckle_contrast=speckle_contrast,
        speckle_correlation_length=1.0,
        aberration_defocus=aberration_defocus,
        aberration_astigmatism_x=aberration_astigmatism_x,
        aberration_astigmatism_y=aberration_astigmatism_y,
        aberration_coma_x=aberration_coma_x,
        aberration_coma_y=aberration_coma_y,
    )

    # Create metadata dictionary
    metadata = {
        "wavelength": wavelength,
        "propagation_distance": propagation_distance,
        "object_type": object_type,
        "phase_shift": phase_shift,
        "speckle_contrast": speckle_contrast,
        "sensor_read_noise": sensor_read_noise,
        "sensor_shot_noise": True,
        "sensor_dark_current": sensor_dark_current,
        "sensor_bit_depth": sensor_bit_depth,
        "aberration_defocus_um": aberration_defocus,
        "aberration_astigmatism_x": aberration_astigmatism_x,
        "aberration_astigmatism_y": aberration_astigmatism_y,
        "aberration_coma_x": aberration_coma_x,
        "aberration_coma_y": aberration_coma_y,
        "grid_height": base_grid.height,
        "grid_width": base_grid.width,
        "pixel_pitch": base_grid.pixel_pitch,
    }

    return config, noise_config, metadata, object_type, phase_shift


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate advanced holography dataset with per-sample randomization."
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_advanced"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--height", type=int, default=512, help="Image height in pixels."
    )
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels.")
    parser.add_argument(
        "--pixel-pitch", type=float, default=6.4e-6, help="Pixel pitch in meters."
    )
    parser.add_argument(
        "--wavelengths",
        type=str,
        default="532e-9",
        help="Comma-separated list of wavelengths in meters (e.g., '532e-9,633e-9').",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable PNG preview generation."
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for advanced dataset generation."""
    args = parse_args()
    rng = default_rng(args.seed)

    # Parse wavelengths
    wavelengths = [float(w.strip()) for w in args.wavelengths.split(",")]

    # Create base grid specification
    base_grid = GridSpec(
        height=args.height,
        width=args.width,
        pixel_pitch=args.pixel_pitch,
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create object producer
    producer = default_object_producer()

    print(f"Generating {args.samples} samples with randomized parameters...")
    print(f"Output directory: {args.output}")
    print(f"Image size: {args.height}x{args.width}")
    print(f"Pixel pitch: {args.pixel_pitch} m")
    print(f"Wavelengths: {wavelengths}")
    print()

    # Generate samples one by one with randomized parameters
    for sample_idx in range(args.samples):
        # Randomize configuration for this sample
        config, noise_config, metadata, object_type, phase_shift = (
            randomize_sample_config(rng, base_grid, wavelengths)
        )

        # Create noise model for this sample
        noise_model = create_noise_model(noise_config)

        # Create output config (complex objects always use COMPLEX representation)
        output_config = OutputConfig(
            object_representation=FieldRepresentation.COMPLEX,
            hologram_representation=FieldRepresentation.INTENSITY,
            reconstruction_representation=FieldRepresentation.INTENSITY,
        )

        # Create converter with noise model
        converter = ObjectToHologramConverter(
            strategy_mapping={
                HolographyMethod.INLINE: __import__(
                    "hologen.holography.inline", fromlist=["InlineHolographyStrategy"]
                ).InlineHolographyStrategy(),
            },
            noise_model=noise_model,
            output_config=output_config,
        )

        # Create generator
        HologramDatasetGenerator(
            object_producer=producer,
            converter=converter,
        )

        # For complex objects, we need to generate both amplitude and phase
        # First generate amplitude pattern
        object_sample_amp = producer.generate_complex(
            config.grid, rng, phase_shift=0.0, mode="amplitude"
        )

        # Then generate phase pattern (using a different shape)
        object_sample_phase = producer.generate_complex(
            config.grid, rng, phase_shift=phase_shift, mode="phase"
        )

        # Combine: amplitude from first, phase from second
        amplitude = np.abs(object_sample_amp.field)
        phase = np.angle(object_sample_phase.field)

        # Create complex field with both amplitude and phase modulation
        from hologen.types import ComplexHologramSample, ComplexObjectSample
        from hologen.utils.fields import amplitude_phase_to_complex

        complex_field = amplitude_phase_to_complex(amplitude, phase)

        # Create complex object sample
        object_sample = ComplexObjectSample(
            name=object_sample_amp.name + "_" + object_sample_phase.name,
            field=complex_field,
            representation=output_config.object_representation,
        )

        # Create hologram and reconstruction
        hologram_field = converter.create_hologram(object_sample, config, rng)
        reconstruction_field = converter.reconstruct(hologram_field, config)

        # Create sample
        sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=hologram_field,
            hologram_representation=output_config.hologram_representation,
            reconstruction_field=reconstruction_field,
            reconstruction_representation=output_config.reconstruction_representation,
        )

        samples = [sample]

        # Save sample with metadata
        sample = samples[0]
        prefix = f"sample_{sample_idx:05d}_{sample.object_sample.name}"
        base_path = args.output / prefix

        # Save with metadata
        save_dict_object = {
            "representation": str(sample.object_sample.representation.value),
        }
        save_dict_object.update(metadata)

        if sample.object_sample.representation == FieldRepresentation.COMPLEX:
            save_dict_object["real"] = sample.object_sample.field.real
            save_dict_object["imag"] = sample.object_sample.field.imag
        elif sample.object_sample.representation == FieldRepresentation.AMPLITUDE:
            save_dict_object["amplitude"] = np.abs(sample.object_sample.field)
        elif sample.object_sample.representation == FieldRepresentation.PHASE:
            save_dict_object["phase"] = np.angle(sample.object_sample.field)
        elif sample.object_sample.representation == FieldRepresentation.INTENSITY:
            save_dict_object["intensity"] = np.abs(sample.object_sample.field) ** 2

        np.savez(base_path.with_name(prefix + "_object.npz"), **save_dict_object)

        # Save hologram with metadata
        save_dict_hologram = {"representation": "intensity"}
        save_dict_hologram.update(metadata)
        save_dict_hologram["intensity"] = np.abs(sample.hologram_field) ** 2

        np.savez(base_path.with_name(prefix + "_hologram.npz"), **save_dict_hologram)

        # Save reconstruction with metadata
        save_dict_reconstruction = {"representation": "intensity"}
        save_dict_reconstruction.update(metadata)
        save_dict_reconstruction["intensity"] = np.abs(sample.reconstruction_field) ** 2

        np.savez(
            base_path.with_name(prefix + "_reconstruction.npz"),
            **save_dict_reconstruction,
        )

        # Save PNG previews if enabled
        if not args.no_preview:
            writer = ComplexFieldWriter(save_preview=True)
            writer._save_png_complex(
                base_path.with_name(prefix + "_object"),
                sample.object_sample.field,
                sample.object_sample.representation,
            )
            writer._save_png_complex(
                base_path.with_name(prefix + "_hologram"),
                sample.hologram_field,
                sample.hologram_representation,
            )
            writer._save_png_complex(
                base_path.with_name(prefix + "_reconstruction"),
                sample.reconstruction_field,
                sample.reconstruction_representation,
            )

        if (sample_idx + 1) % 10 == 0:
            print(f"Generated {sample_idx + 1}/{args.samples} samples...")

    print(f"\nDataset generation complete! Saved to {args.output}")


if __name__ == "__main__":
    main()

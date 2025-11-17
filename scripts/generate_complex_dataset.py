"""Generate dataset of complex objects with phase, holograms, and reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from PIL import Image

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
    create_noise_model,
    default_converter,
    default_object_producer,
)
from hologen.phase import PhaseGenerationConfig
from hologen.types import (
    ComplexHologramSample,
    FieldRepresentation,
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    NoiseConfig,
    NoiseModel,
    OffAxisCarrier,
    OpticalConfig,
    OutputConfig,
)
from hologen.utils.math import normalize_image


def save_png(path: Path, image: np.ndarray, is_phase: bool = False) -> None:
    """Save array as PNG image.
    
    Args:
        path: Output path for PNG file.
        image: Image array to save.
        is_phase: If True, map phase from [-π, π] to [0, 255] with colormap.
    """
    if is_phase:
        # Map phase from [-π, π] to [0, 1]
        normalized = (image + np.pi) / (2 * np.pi)
        # Apply twilight colormap for phase visualization
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('twilight')
            colored = cmap(normalized)
            rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
            pil_image = Image.fromarray(rgb_image, mode='RGB')
            pil_image.save(path)
        except ImportError:
            # Fallback to grayscale if matplotlib not available
            normalized_img = normalize_image(normalized)
            pil_image = Image.fromarray((normalized_img * 255).astype(np.uint8), mode='L')
            pil_image.save(path)
    else:
        # Regular intensity image
        normalized = normalize_image(image)
        pil_image = Image.fromarray((normalized * 255).astype(np.uint8), mode='L')
        pil_image.save(path)


def save_complex_dataset_with_metadata(
    samples: list[ComplexHologramSample],
    output_dir: Path,
    config: HolographyConfig,
    noise_config: NoiseConfig | None,
    phase_config: PhaseGenerationConfig,
) -> None:
    """Save complex dataset with full metadata.

    Args:
        samples: List of complex hologram samples.
        output_dir: Output directory for dataset.
        config: Holography configuration.
        noise_config: Noise configuration (if any).
        phase_config: Phase generation configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each sample
    for index, sample in enumerate(samples):
        prefix = f"sample_{index:05d}_{sample.object_sample.name}"

        # Extract fields
        obj_field = sample.object_sample.field
        holo_field = sample.hologram_field
        recon_field = sample.reconstruction_field

        # Compute intensity and phase for each
        obj_intensity = np.abs(obj_field) ** 2
        obj_phase = np.angle(obj_field)
        holo_intensity = np.abs(holo_field) ** 2
        recon_intensity = np.abs(recon_field) ** 2
        recon_phase = np.angle(recon_field)

        # Build metadata dictionary
        metadata = {
            # Grid configuration
            "grid_height": config.grid.height,
            "grid_width": config.grid.width,
            "pixel_pitch": config.grid.pixel_pitch,
            # Optical configuration
            "wavelength": config.optics.wavelength,
            "propagation_distance": config.optics.propagation_distance,
            "holography_method": config.method.value,
            # Phase generation configuration
            "phase_enabled": phase_config.enabled,
            "refractive_index_mode": phase_config.refractive_index_mode,
            "thickness_mode": phase_config.thickness_mode,
            "ambient_refractive_index": phase_config.ambient_refractive_index,
            "refractive_index_min": phase_config.refractive_index_range[0],
            "refractive_index_max": phase_config.refractive_index_range[1],
            "thickness_min": phase_config.thickness_range[0],
            "thickness_max": phase_config.thickness_range[1],
            "correlation_coefficient": phase_config.correlation_coefficient,
        }

        # Add noise configuration if present
        if noise_config is not None:
            metadata.update(
                {
                    "noise_sensor_read_noise": noise_config.sensor_read_noise,
                    "noise_sensor_shot_noise": noise_config.sensor_shot_noise,
                    "noise_sensor_dark_current": noise_config.sensor_dark_current,
                    "noise_sensor_bit_depth": (
                        noise_config.sensor_bit_depth
                        if noise_config.sensor_bit_depth is not None
                        else -1
                    ),
                    "noise_speckle_contrast": noise_config.speckle_contrast,
                    "noise_speckle_correlation_length": noise_config.speckle_correlation_length,
                    "noise_aberration_defocus": noise_config.aberration_defocus,
                    "noise_aberration_astigmatism_x": noise_config.aberration_astigmatism_x,
                    "noise_aberration_astigmatism_y": noise_config.aberration_astigmatism_y,
                    "noise_aberration_coma_x": noise_config.aberration_coma_x,
                    "noise_aberration_coma_y": noise_config.aberration_coma_y,
                }
            )

        # Add off-axis carrier if present
        if config.carrier is not None:
            metadata.update(
                {
                    "carrier_frequency_x": config.carrier.frequency_x,
                    "carrier_frequency_y": config.carrier.frequency_y,
                    "carrier_gaussian_width": config.carrier.gaussian_width,
                }
            )

        # Save complete dataset in single .npz file
        np.savez(
            output_dir / f"{prefix}.npz",
            # Object domain
            object_intensity=obj_intensity,
            object_phase=obj_phase,
            # Hologram
            hologram_intensity=holo_intensity,
            # Reconstruction
            reconstruction_intensity=recon_intensity,
            reconstruction_phase=recon_phase,
            # Metadata
            **metadata,
        )

        # Save PNG visualizations
        save_png(output_dir / f"{prefix}_object_intensity.png", obj_intensity)
        save_png(output_dir / f"{prefix}_object_phase.png", obj_phase, is_phase=True)
        save_png(output_dir / f"{prefix}_hologram_intensity.png", holo_intensity)
        save_png(output_dir / f"{prefix}_reconstruction_intensity.png", recon_intensity)
        save_png(output_dir / f"{prefix}_reconstruction_phase.png", recon_phase, is_phase=True)

    print(f"Saved {len(samples)} samples to {output_dir}")


def main() -> None:
    """Generate complex hologram dataset with metadata."""
    parser = argparse.ArgumentParser(
        description="Generate complex hologram dataset with phase and metadata."
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("complex_dataset"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--height", type=int, default=256, help="Grid height in pixels"
    )
    parser.add_argument("--width", type=int, default=256, help="Grid width in pixels")
    parser.add_argument(
        "--pixel-pitch", type=float, default=4.65e-6, help="Pixel pitch in meters"
    )
    parser.add_argument(
        "--wavelength", type=float, default=532e-9, help="Wavelength in meters"
    )
    parser.add_argument(
        "--distance", type=float, default=0.02, help="Propagation distance in meters"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="inline",
        choices=["inline", "off_axis"],
        help="Holography method",
    )

    args = parser.parse_args()

    # Setup RNG
    rng = default_rng(args.seed)

    # Build holography configuration
    grid = GridSpec(
        height=args.height, width=args.width, pixel_pitch=args.pixel_pitch
    )
    optics = OpticalConfig(
        wavelength=args.wavelength, propagation_distance=args.distance
    )
    method = HolographyMethod(args.method)

    # Add carrier for off-axis
    carrier = None
    if method == HolographyMethod.OFF_AXIS:
        carrier = OffAxisCarrier(
            frequency_x=1600.0, frequency_y=1600.0, gaussian_width=400.0
        )

    config = HolographyConfig(grid=grid, optics=optics, method=method, carrier=carrier)

    # Configure realistic but mild noise
    noise_config = NoiseConfig(
        sensor_read_noise=0.5,  # Low read noise
        sensor_shot_noise=True,  # Poisson shot noise (scales with signal)
        sensor_dark_current=0.1,  # Very small dark current
        sensor_bit_depth=14,  # 14-bit ADC (higher precision)
        speckle_contrast=0.05,  # Mild speckle
        speckle_correlation_length=2.0,  # 2-pixel correlation
        aberration_defocus=0.01,  # Very small defocus
        aberration_astigmatism_x=0.005,  # Very small astigmatism
        aberration_astigmatism_y=0.005,
        aberration_coma_x=0.002,  # Minimal coma
        aberration_coma_y=0.002,
    )

    # Configure physics-based phase generation with smoother variations
    phase_config = PhaseGenerationConfig(
        enabled=True,
        refractive_index_mode="gaussian_blobs",  # Spatially varying refractive index
        thickness_mode="shape_based",  # Dome-like thickness profile
        ambient_refractive_index=1.0,  # Air
        refractive_index_range=(1.33, 1.45),  # Narrower range for less extreme phase
        thickness_range=(2e-6, 8e-6),  # 2-8 microns (moderate thickness)
        correlation_coefficient=0.5,  # Higher amplitude-phase correlation
        gaussian_blob_count=3,  # Fewer blobs for smoother variation
        gaussian_blob_size_range=(20.0, 60.0),  # Larger blobs for smoother features
    )

    # Create noise model
    noise_model = create_noise_model(noise_config)

    # Create output configuration for complex fields
    output_config = OutputConfig(
        object_representation=FieldRepresentation.COMPLEX,
        hologram_representation=FieldRepresentation.COMPLEX,
        reconstruction_representation=FieldRepresentation.COMPLEX,
    )

    # Setup pipeline
    producer = default_object_producer()
    converter = default_converter(noise_model)
    converter.output_config = output_config

    generator = HologramDatasetGenerator(
        object_producer=producer, converter=converter
    )

    # Generate samples
    print(f"Generating {args.samples} complex samples...")
    samples = list(
        generator.generate(
            count=args.samples,
            config=config,
            rng=rng,
            mode="complex",
            use_complex=True,
            phase_config=phase_config,
        )
    )

    # Save with metadata
    save_complex_dataset_with_metadata(
        samples, args.output, config, noise_config, phase_config
    )

    print(f"Dataset generation complete!")
    print(f"  Samples: {args.samples}")
    print(f"  Method: {method.value}")
    print(f"  Grid: {args.height}x{args.width} @ {args.pixel_pitch*1e6:.2f} µm/pixel")
    print(f"  Wavelength: {args.wavelength*1e9:.1f} nm")
    print(f"  Distance: {args.distance*1e3:.1f} mm")
    print(f"  Phase: {phase_config.refractive_index_mode} + {phase_config.thickness_mode}")
    print(f"  Noise: enabled (read={noise_config.sensor_read_noise}, speckle={noise_config.speckle_contrast})")


if __name__ == "__main__":
    main()

"""Command-line utility for holography dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from numpy.random import default_rng

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
    default_converter,
    default_object_producer,
    create_noise_model
)
from hologen.types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    NoiseConfig,
    NoiseModel,
    OffAxisCarrier,
    OpticalConfig,
)
from hologen.utils.io import NumpyDatasetWriter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""

    parser = argparse.ArgumentParser(description="Generate holography datasets.")
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to generate."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset"), help="Output directory."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="inline",
        choices=[m.value for m in HolographyMethod],
        help="Holography method to use.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--height", type=int, default=256, help="Object and sensor height in pixels."
    )
    parser.add_argument(
        "--width", type=int, default=256, help="Object and sensor width in pixels."
    )
    parser.add_argument(
        "--pixel-pitch", type=float, default=4.65e-6, help="Pixel pitch in meters."
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Illumination wavelength in meters.",
    )
    parser.add_argument(
        "--distance", type=float, default=0.02, help="Propagation distance in meters."
    )
    parser.add_argument(
        "--carrier-x",
        type=float,
        default=1600.0,
        help="Off-axis carrier frequency along x in cycles per meter.",
    )
    parser.add_argument(
        "--carrier-y",
        type=float,
        default=1600.0,
        help="Off-axis carrier frequency along y in cycles per meter.",
    )
    parser.add_argument(
        "--carrier-sigma",
        type=float,
        default=400.0,
        help="Gaussian filter sigma in cycles per meter for off-axis reconstruction.",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable PNG preview generation."
    )
    noise_group = parser.add_argument_group("noise", "Noise simulation parameters")
    noise_group.add_argument(
        "--sensor-read-noise",
        type=float,
        default=0.0,
        help="Sensor read noise standard deviation.",
    )
    noise_group.add_argument(
        "--sensor-shot-noise",
        action="store_true",
        help="Enable Poisson shot noise.",
    )
    noise_group.add_argument(
        "--sensor-dark-current",
        type=float,
        default=0.0,
        help="Sensor dark current mean value.",
    )
    noise_group.add_argument(
        "--sensor-bit-depth",
        type=int,
        default=None,
        help="ADC bit depth for quantization.",
    )
    noise_group.add_argument(
        "--speckle-contrast",
        type=float,
        default=0.0,
        help="Speckle contrast ratio (0.0 to 1.0).",
    )
    noise_group.add_argument(
        "--speckle-correlation",
        type=float,
        default=1.0,
        help="Speckle correlation length in pixels.",
    )
    noise_group.add_argument(
        "--aberration-defocus",
        type=float,
        default=0.0,
        help="Defocus aberration coefficient.",
    )
    noise_group.add_argument(
        "--aberration-astigmatism-x",
        type=float,
        default=0.0,
        help="Astigmatism x coefficient.",
    )
    noise_group.add_argument(
        "--aberration-astigmatism-y",
        type=float,
        default=0.0,
        help="Astigmatism y coefficient.",
    )
    noise_group.add_argument(
        "--aberration-coma-x",
        type=float,
        default=0.0,
        help="Coma x coefficient.",
    )
    noise_group.add_argument(
        "--aberration-coma-y",
        type=float,
        default=0.0,
        help="Coma y coefficient.",
    )

    return parser.parse_args()

def build_noise_config(args: argparse.Namespace) -> NoiseConfig | None:
    """Construct noise configuration from command-line arguments."""

    config = NoiseConfig(
        sensor_read_noise=args.sensor_read_noise,
        sensor_shot_noise=args.sensor_shot_noise,
        sensor_dark_current=args.sensor_dark_current,
        sensor_bit_depth=args.sensor_bit_depth,
        speckle_contrast=args.speckle_contrast,
        speckle_correlation_length=args.speckle_correlation,
        aberration_defocus=args.aberration_defocus,
        aberration_astigmatism_x=args.aberration_astigmatism_x,
        aberration_astigmatism_y=args.aberration_astigmatism_y,
        aberration_coma_x=args.aberration_coma_x,
        aberration_coma_y=args.aberration_coma_y,
    )
    
    has_any_noise = (
        config.sensor_read_noise > 0.0
        or config.sensor_shot_noise
        or config.sensor_dark_current > 0.0
        or config.sensor_bit_depth is not None
        or config.speckle_contrast > 0.0
        or config.aberration_defocus != 0.0
        or config.aberration_astigmatism_x != 0.0
        or config.aberration_astigmatism_y != 0.0
        or config.aberration_coma_x != 0.0
        or config.aberration_coma_y != 0.0
    )
    
    return config if has_any_noise else None


def build_config(args: argparse.Namespace) -> HolographyConfig:
    """Construct a holography configuration from command-line arguments."""

    grid = GridSpec(height=args.height, width=args.width, pixel_pitch=args.pixel_pitch)
    optics = OpticalConfig(
        wavelength=args.wavelength, propagation_distance=args.distance
    )
    method = HolographyMethod(args.method)
    carrier = None
    if method is HolographyMethod.OFF_AXIS:
        carrier = OffAxisCarrier(
            frequency_x=args.carrier_x,
            frequency_y=args.carrier_y,
            gaussian_width=args.carrier_sigma,
        )
    return HolographyConfig(grid=grid, optics=optics, method=method, carrier=carrier)


def main() -> None:
    """Entry point for the dataset generation command-line interface."""

    args: argparse.Namespace = parse_args()
    rng = default_rng(args.seed)
    config: HolographyConfig = build_config(args)
    noise_config: NoiseConfig = build_noise_config(args)
    producer: ObjectDomainProducer = default_object_producer()
    
    # Set the converter based on noise selection.
    noise_model: NoiseModel | None = None
    if noise_config is not None:
        noise_model: NoiseModel = create_noise_model(noise_config)
    
    # Set the Holography converter (forward-propagation).
    converter: ObjectToHologramConverter = default_converter(noise_model)

    # Create a dataset generator.
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    writer = NumpyDatasetWriter(save_preview=not args.no_preview)
    writer.save(
        samples=generator.generate(count=args.samples, config=config, rng=rng),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

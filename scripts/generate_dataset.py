"""Command-line utility for holography dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from numpy.random import default_rng

from hologen.converters import (
    HologramDatasetGenerator,
    default_converter,
    default_object_producer,
)
from hologen.types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
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
    return parser.parse_args()


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

    args = parse_args()
    rng = default_rng(args.seed)
    config = build_config(args)

    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    writer = NumpyDatasetWriter(save_preview=not args.no_preview)

    writer.save(
        samples=generator.generate(count=args.samples, config=config, rng=rng),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

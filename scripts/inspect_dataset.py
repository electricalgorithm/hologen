"""Inspect and visualize samples from the generated dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect_sample(sample_path: Path) -> None:
    """Display detailed information about a sample.

    Args:
        sample_path: Path to the sample .npz file.
    """
    data = np.load(sample_path)

    print(f"\n{'=' * 60}")
    print(f"Sample: {sample_path.name}")
    print(f"{'=' * 60}\n")

    # Display field information
    print("Field Data:")
    for key in ["intensity", "amplitude", "phase", "real", "imag"]:
        if key in data:
            field = data[key]
            print(f"  {key}: shape={field.shape}, dtype={field.dtype}")
            print(
                f"    min={field.min():.6f}, max={field.max():.6f}, mean={field.mean():.6f}"
            )

    # Display metadata
    print("\nMetadata:")
    metadata_keys = [
        "representation",
        "object_type",
        "wavelength",
        "propagation_distance",
        "phase_shift",
        "speckle_contrast",
        "sensor_read_noise",
        "sensor_shot_noise",
        "sensor_dark_current",
        "sensor_bit_depth",
        "aberration_defocus_um",
        "aberration_astigmatism_x",
        "aberration_astigmatism_y",
        "aberration_coma_x",
        "aberration_coma_y",
        "grid_height",
        "grid_width",
        "pixel_pitch",
    ]

    for key in metadata_keys:
        if key in data:
            value = data[key]
            if key == "wavelength":
                print(f"  {key}: {value * 1e9:.1f} nm")
            elif key == "propagation_distance":
                print(f"  {key}: {value * 1000:.2f} mm")
            elif key == "pixel_pitch":
                print(f"  {key}: {value * 1e6:.2f} μm")
            elif key == "phase_shift":
                print(f"  {key}: {value:.4f} rad ({np.degrees(value):.2f}°)")
            elif isinstance(value, int | np.integer):
                print(f"  {key}: {value}")
            elif isinstance(value, float | np.floating):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def list_samples(dataset_dir: Path, pattern: str = "*_hologram.npz") -> list[Path]:
    """List all samples in the dataset directory.

    Args:
        dataset_dir: Path to the dataset directory.
        pattern: Glob pattern to match sample files.

    Returns:
        List of sample file paths.
    """
    return sorted(dataset_dir.glob(pattern))


def main() -> None:
    """Entry point for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect holography dataset samples.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset directory.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to inspect (default: 0).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all samples in the dataset."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="hologram",
        choices=["object", "hologram", "reconstruction"],
        help="Sample type to inspect (default: hologram).",
    )

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist.")
        return

    # List samples
    pattern = f"*_{args.type}.npz"
    samples = list_samples(args.dataset_dir, pattern)

    if not samples:
        print(f"No samples found in '{args.dataset_dir}' matching pattern '{pattern}'.")
        return

    if args.list:
        print(f"\nFound {len(samples)} samples in '{args.dataset_dir}':\n")
        for i, sample in enumerate(samples):
            print(f"  [{i:3d}] {sample.name}")
        return

    # Inspect specific sample
    if args.sample < 0 or args.sample >= len(samples):
        print(
            f"Error: Sample index {args.sample} out of range [0, {len(samples) - 1}]."
        )
        return

    inspect_sample(samples[args.sample])


if __name__ == "__main__":
    main()

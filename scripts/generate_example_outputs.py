"""Generate example outputs for EXAMPLES.md documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    default_converter,
    default_object_producer,
)
from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OpticalConfig,
)


def create_output_directory():
    """Create directory for example outputs."""
    output_dir = Path("docs/examples/examples_doc")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_basic_visualization(output_dir: Path):
    """Generate visualization for Example 2: Load and Visualize."""
    print("Generating basic visualization example...")

    # Generate a sample
    grid = GridSpec(height=256, width=256, pixel_pitch=4.65e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

    rng = np.random.default_rng(42)
    samples = list(generator.generate(count=1, config=config, rng=rng))
    sample = samples[0]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample.object_sample.pixels, cmap="gray")
    axes[0].set_title("Object Domain", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample.hologram, cmap="gray")
    axes[1].set_title("Hologram", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(sample.reconstruction, cmap="gray")
    axes[2].set_title("Reconstruction", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "basic_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: basic_visualization.png")


def generate_parameter_study_plot(output_dir: Path):
    """Generate plot for Example 12: Parameter Studies."""
    print("Generating parameter study example...")

    grid = GridSpec(height=256, width=256, pixel_pitch=6.4e-6)
    wavelength = 532e-9

    # Use single shape for consistency
    circle_gen = CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18)
    producer = ObjectDomainProducer(shape_generators=(circle_gen,))
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

    distances = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    results = []

    for distance in distances:
        optics = OpticalConfig(wavelength=wavelength, propagation_distance=distance)
        config = HolographyConfig(
            grid=grid, optics=optics, method=HolographyMethod.INLINE
        )

        rng = np.random.default_rng(42)
        samples = list(generator.generate(count=10, config=config, rng=rng))

        mse_values = []
        for sample in samples:
            mse = np.mean((sample.object_sample.pixels - sample.reconstruction) ** 2)
            mse_values.append(mse)

        results.append(
            {
                "distance": distance,
                "avg_mse": np.mean(mse_values),
                "std_mse": np.std(mse_values),
            }
        )

    # Plot results
    distances_mm = [r["distance"] * 1000 for r in results]
    avg_mses = [r["avg_mse"] for r in results]
    std_mses = [r["std_mse"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        distances_mm,
        avg_mses,
        yerr=std_mses,
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    plt.xlabel("Propagation Distance (mm)", fontsize=12, fontweight="bold")
    plt.ylabel("Reconstruction MSE", fontsize=12, fontweight="bold")
    plt.title(
        "Effect of Propagation Distance on Reconstruction Quality",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "parameter_study_distance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("  Saved: parameter_study_distance.png")


def generate_ablation_study_plot(output_dir: Path):
    """Generate plot for Example 13: Ablation Studies."""
    print("Generating ablation study example...")

    # Simulate ablation study results
    conditions = [
        "no_noise",
        "sensor_only",
        "speckle_only",
        "aberration_only",
        "all_noise",
    ]
    mse_means = [0.005, 0.012, 0.018, 0.015, 0.025]
    mse_stds = [0.001, 0.003, 0.004, 0.003, 0.005]
    snr_means = [45.2, 38.5, 35.8, 37.2, 32.1]
    snr_stds = [2.1, 3.2, 3.8, 2.9, 4.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    x_pos = np.arange(len(conditions))

    ax1.bar(x_pos, mse_means, yerr=mse_stds, capsize=5, alpha=0.7, color="steelblue")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions, rotation=45, ha="right")
    ax1.set_ylabel("Reconstruction MSE", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Effect of Noise on Reconstruction Quality", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x_pos, snr_means, yerr=snr_stds, capsize=5, alpha=0.7, color="coral")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions, rotation=45, ha="right")
    ax2.set_ylabel("SNR (dB)", fontsize=12, fontweight="bold")
    ax2.set_title("Effect of Noise on Signal Quality", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_study_noise.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: ablation_study_noise.png")


def generate_quality_distribution_plot(output_dir: Path):
    """Generate plot for Recipe 5: Dataset Validation."""
    print("Generating quality distribution example...")

    # Generate samples for quality distribution
    grid = GridSpec(height=256, width=256, pixel_pitch=4.65e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

    rng = np.random.default_rng(42)
    samples = list(generator.generate(count=100, config=config, rng=rng))

    mse_values = []
    for sample in samples:
        mse = np.mean((sample.object_sample.pixels - sample.reconstruction) ** 2)
        mse_values.append(mse)

    plt.figure(figsize=(10, 6))
    plt.hist(mse_values, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
    plt.xlabel("Reconstruction MSE", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("Distribution of Reconstruction Quality", fontsize=14, fontweight="bold")
    plt.axvline(
        np.mean(mse_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(mse_values):.6f}",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: quality_distribution.png")


def generate_augmentation_example(output_dir: Path):
    """Generate augmentation comparison for Recipe 4."""
    print("Generating augmentation example...")

    # Generate a sample
    grid = GridSpec(height=256, width=256, pixel_pitch=4.65e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

    rng = np.random.default_rng(42)
    samples = list(generator.generate(count=1, config=config, rng=rng))
    original_hologram = samples[0].hologram

    # Apply augmentation
    from scipy.ndimage import rotate

    augmented_hologram = np.fliplr(original_hologram)
    augmented_hologram = rotate(augmented_hologram, 15, reshape=False, order=1)
    noise = rng.normal(0, 0.01, augmented_hologram.shape)
    augmented_hologram = np.clip(augmented_hologram + noise, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_hologram, cmap="gray")
    axes[0].set_title("Original Hologram", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(augmented_hologram, cmap="gray")
    axes[1].set_title(
        "Augmented Hologram\n(Flipped, Rotated, Noisy)", fontsize=14, fontweight="bold"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "augmentation_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("  Saved: augmentation_comparison.png")


def main():
    """Generate all example outputs."""
    print("=" * 60)
    print("Generating Example Outputs for EXAMPLES.md")
    print("=" * 60)

    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir}\n")

    generate_basic_visualization(output_dir)
    generate_parameter_study_plot(output_dir)
    generate_ablation_study_plot(output_dir)
    generate_quality_distribution_plot(output_dir)
    generate_augmentation_example(output_dir)

    print("\n" + "=" * 60)
    print("Example output generation complete!")
    print(f"All files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

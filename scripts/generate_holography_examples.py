#!/usr/bin/env python3
"""Generate visual examples for holography methods documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec as MPLGridSpec

from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OffAxisCarrier,
    OpticalConfig,
)


def setup_output_dir() -> Path:
    """Create output directory for holography examples."""
    output_dir = Path("docs/examples/holography_methods")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_inline_vs_offaxis_comparison(output_dir: Path) -> None:
    """Generate side-by-side comparison of inline vs off-axis holography."""
    print("Generating inline vs off-axis comparison...")

    # Configuration
    grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    rng = np.random.default_rng(42)

    # Generate object
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    object_field = generator.generate_complex(
        grid=grid, rng=rng, mode="phase", phase_shift=np.pi / 2
    )

    # Inline holography
    inline_config = HolographyConfig(
        grid=grid, optics=optics, method=HolographyMethod.INLINE
    )
    inline_strategy = InlineHolographyStrategy()
    inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
    inline_reconstruction = inline_strategy.reconstruct(inline_hologram, inline_config)

    # Off-axis holography
    carrier = OffAxisCarrier(frequency_x=1e6, frequency_y=0, gaussian_width=2e5)
    offaxis_config = HolographyConfig(
        grid=grid, optics=optics, method=HolographyMethod.OFF_AXIS, carrier=carrier
    )
    offaxis_strategy = OffAxisHolographyStrategy()
    offaxis_hologram = offaxis_strategy.create_hologram(object_field, offaxis_config)
    offaxis_reconstruction = offaxis_strategy.reconstruct(
        offaxis_hologram, offaxis_config
    )

    # Create comparison figure
    fig = plt.figure(figsize=(15, 10))
    gs = MPLGridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Object
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.abs(object_field), cmap="gray")
    ax.set_title("Object Amplitude", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.angle(object_field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title("Object Phase", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.text(
        0.5,
        0.5,
        "Phase-only\nCircular Object\n(π/2 phase shift)",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.axis("off")

    # Row 2: Inline
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.abs(inline_hologram) ** 2, cmap="gray")
    ax.set_title("Inline Hologram", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.abs(inline_reconstruction), cmap="gray")
    ax.set_title("Inline Reconstruction", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.text(
        0.5,
        0.5,
        "Inline Holography\n\n• Simple setup\n• Twin-image present\n• Overlapping orders",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.axis("off")

    # Row 3: Off-axis
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(np.abs(offaxis_hologram) ** 2, cmap="gray")
    ax.set_title("Off-Axis Hologram", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(np.abs(offaxis_reconstruction), cmap="gray")
    ax.set_title("Off-Axis Reconstruction", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 2])
    ax.text(
        0.5,
        0.5,
        "Off-Axis Holography\n\n• Carrier frequency\n• Clean reconstruction\n• Separated orders",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.axis("off")

    plt.suptitle(
        "Inline vs Off-Axis Holography Comparison", fontsize=14, fontweight="bold"
    )
    plt.savefig(
        output_dir / "inline_vs_offaxis_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {output_dir / 'inline_vs_offaxis_comparison.png'}")


def generate_frequency_domain_visualization(output_dir: Path) -> None:
    """Generate frequency domain visualization for off-axis holography."""
    print("Generating frequency domain visualization...")

    # Configuration
    grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    rng = np.random.default_rng(42)

    # Generate object
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    object_field = generator.generate_complex(
        grid=grid, rng=rng, mode="amplitude", phase_shift=0
    )

    # Inline holography
    inline_config = HolographyConfig(
        grid=grid, optics=optics, method=HolographyMethod.INLINE
    )
    inline_strategy = InlineHolographyStrategy()
    inline_hologram = inline_strategy.create_hologram(object_field, inline_config)
    inline_intensity = np.abs(inline_hologram) ** 2

    # Off-axis holography
    carrier = OffAxisCarrier(frequency_x=1e6, frequency_y=0, gaussian_width=2e5)
    offaxis_config = HolographyConfig(
        grid=grid, optics=optics, method=HolographyMethod.OFF_AXIS, carrier=carrier
    )
    offaxis_strategy = OffAxisHolographyStrategy()
    offaxis_hologram = offaxis_strategy.create_hologram(object_field, offaxis_config)
    offaxis_intensity = np.abs(offaxis_hologram) ** 2

    # Compute Fourier transforms
    inline_fft = np.fft.fftshift(np.fft.fft2(inline_intensity))
    offaxis_fft = np.fft.fftshift(np.fft.fft2(offaxis_intensity))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Inline hologram
    axes[0, 0].imshow(inline_intensity, cmap="gray")
    axes[0, 0].set_title("Inline Hologram (Spatial)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Inline FFT
    axes[0, 1].imshow(np.log(1 + np.abs(inline_fft)), cmap="hot")
    axes[0, 1].set_title(
        "Inline Hologram (Fourier)\nOverlapping Orders", fontsize=12, fontweight="bold"
    )
    axes[0, 1].axis("off")

    # Off-axis hologram
    axes[1, 0].imshow(offaxis_intensity, cmap="gray")
    axes[1, 0].set_title("Off-Axis Hologram (Spatial)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Off-axis FFT
    axes[1, 1].imshow(np.log(1 + np.abs(offaxis_fft)), cmap="hot")
    axes[1, 1].set_title(
        "Off-Axis Hologram (Fourier)\nSeparated Orders",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].axis("off")

    plt.suptitle("Frequency Domain: Inline vs Off-Axis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        output_dir / "frequency_domain_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {output_dir / 'frequency_domain_comparison.png'}")


def generate_carrier_frequency_effects(output_dir: Path) -> None:
    """Generate visualization showing effect of different carrier frequencies."""
    print("Generating carrier frequency effects...")

    # Configuration
    grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    rng = np.random.default_rng(42)

    # Generate object
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    object_field = generator.generate_complex(
        grid=grid, rng=rng, mode="amplitude", phase_shift=0
    )

    # Different carrier frequencies
    carrier_freqs = [5e5, 1e6, 2e6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, freq in enumerate(carrier_freqs):
        carrier = OffAxisCarrier(
            frequency_x=freq, frequency_y=0, gaussian_width=freq * 0.2
        )
        config = HolographyConfig(
            grid=grid, optics=optics, method=HolographyMethod.OFF_AXIS, carrier=carrier
        )
        strategy = OffAxisHolographyStrategy()
        hologram = strategy.create_hologram(object_field, config)
        intensity = np.abs(hologram) ** 2
        fft = np.fft.fftshift(np.fft.fft2(intensity))

        # Spatial domain
        axes[0, idx].imshow(intensity, cmap="gray")
        axes[0, idx].set_title(
            f"Carrier: {freq:.1e} cycles/m", fontsize=11, fontweight="bold"
        )
        axes[0, idx].axis("off")

        # Frequency domain
        axes[1, idx].imshow(np.log(1 + np.abs(fft)), cmap="hot")
        axes[1, idx].set_title("Fourier Domain", fontsize=11, fontweight="bold")
        axes[1, idx].axis("off")

    plt.suptitle("Effect of Carrier Frequency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        output_dir / "carrier_frequency_effects.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {output_dir / 'carrier_frequency_effects.png'}")


def main():
    """Generate all holography method examples."""
    print("Generating holography method visual examples...")
    print("=" * 60)

    output_dir = setup_output_dir()

    generate_inline_vs_offaxis_comparison(output_dir)
    generate_frequency_domain_visualization(output_dir)
    generate_carrier_frequency_effects(output_dir)

    print("=" * 60)
    print(f"All examples generated in: {output_dir}")
    print("Complete!")


if __name__ == "__main__":
    main()

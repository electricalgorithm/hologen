#!/usr/bin/env python3
"""Generate visual examples for complex field documentation.

This script generates sample images showing:
1. Intensity vs complex representations
2. Phase-only vs amplitude-only objects
3. Hologram and reconstruction examples
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec,
    OpticalConfig,
    HolographyConfig,
    HolographyMethod,
    FieldRepresentation,
    OutputConfig,
)
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter
from hologen.holography.inline import InlineHolographyStrategy
from hologen.utils.fields import complex_to_representation


def save_field_comparison(
    field: np.ndarray,
    output_path: Path,
    title: str = "Field Comparison"
) -> None:
    """Save comparison of different field representations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    # Intensity
    intensity = np.abs(field) ** 2
    im0 = axes[0, 0].imshow(intensity, cmap='gray')
    axes[0, 0].set_title('Intensity: |E|²')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Amplitude
    amplitude = np.abs(field)
    im1 = axes[0, 1].imshow(amplitude, cmap='gray')
    axes[0, 1].set_title('Amplitude: |E|')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Phase
    phase = np.angle(field)
    im2 = axes[1, 0].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Phase: arg(E)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='radians')
    
    # Complex (amplitude with phase overlay)
    # Create HSV image: hue=phase, saturation=1, value=amplitude
    phase_normalized = (phase + np.pi) / (2 * np.pi)  # [0, 1]
    amplitude_normalized = amplitude / amplitude.max() if amplitude.max() > 0 else amplitude
    
    hsv = np.zeros((*field.shape, 3))
    hsv[..., 0] = phase_normalized  # Hue from phase
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = amplitude_normalized  # Value from amplitude
    
    rgb = hsv_to_rgb(hsv)
    axes[1, 1].imshow(rgb)
    axes[1, 1].set_title('Complex: Amplitude + Phase')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_object_type_comparison(output_dir: Path) -> None:
    """Generate comparison of amplitude-only vs phase-only objects."""
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    producer = ObjectDomainProducer(shape_generators=(generator,))
    rng = np.random.default_rng(42)
    
    # Generate amplitude-only object
    amp_sample = producer.generate_complex(grid, rng, mode="amplitude")
    amp_field = amp_sample.field
    
    # Generate phase-only object
    phase_sample = producer.generate_complex(grid, rng, phase_shift=np.pi/2, mode="phase")
    phase_field = phase_sample.field
    
    # Save comparisons
    save_field_comparison(
        amp_field,
        output_dir / "amplitude_only_object.png",
        "Amplitude-Only Object (Absorbing Circle)"
    )
    
    save_field_comparison(
        phase_field,
        output_dir / "phase_only_object.png",
        "Phase-Only Object (Transparent Circle with π/2 Phase Shift)"
    )


def save_hologram_reconstruction_example(output_dir: Path) -> None:
    """Generate complete hologram and reconstruction example."""
    # Configuration
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    # Create components
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    producer = ObjectDomainProducer(shape_generators=(generator,))
    strategy = InlineHolographyStrategy()
    converter = ObjectToHologramConverter(
        strategy_mapping={HolographyMethod.INLINE: strategy}
    )
    
    # Generate sample
    rng = np.random.default_rng(42)
    object_sample = producer.generate_complex(config.grid, rng, phase_shift=np.pi/2, mode="phase")
    
    # Create hologram and reconstruction
    hologram = converter.create_hologram(object_sample, config, rng)
    reconstruction = converter.reconstruct(hologram, config)
    
    # Save all three
    save_field_comparison(
        object_sample.field,
        output_dir / "example_object.png",
        "Object Field (Phase-Only Circle)"
    )
    
    save_field_comparison(
        hologram,
        output_dir / "example_hologram.png",
        "Hologram Field (After Propagation)"
    )
    
    save_field_comparison(
        reconstruction,
        output_dir / "example_reconstruction.png",
        "Reconstruction Field (Back-Propagated)"
    )


def save_side_by_side_comparison(output_dir: Path) -> None:
    """Generate side-by-side comparison of intensity vs complex."""
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    generator = CircleGenerator(name="circle", min_radius=0.15, max_radius=0.15)
    producer = ObjectDomainProducer(shape_generators=(generator,))
    strategy = InlineHolographyStrategy()
    converter = ObjectToHologramConverter(
        strategy_mapping={HolographyMethod.INLINE: strategy}
    )
    
    rng = np.random.default_rng(42)
    object_sample = producer.generate_complex(config.grid, rng, phase_shift=np.pi/2, mode="phase")
    hologram = converter.create_hologram(object_sample, config, rng)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Intensity-Only vs Complex Field Representations', fontsize=16)
    
    # Top row: Intensity-only view
    object_intensity = np.abs(object_sample.field) ** 2
    hologram_intensity = np.abs(hologram) ** 2
    
    axes[0, 0].imshow(object_intensity, cmap='gray')
    axes[0, 0].set_title('Object (Intensity Only)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hologram_intensity, cmap='gray')
    axes[0, 1].set_title('Hologram (Intensity Only)')
    axes[0, 1].axis('off')
    
    axes[0, 2].text(0.5, 0.5, 'Phase information\nLOST!\n\nNo visible contrast\nfor phase-only objects',
                    ha='center', va='center', fontsize=14, color='red',
                    transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')
    
    # Bottom row: Complex field view
    object_phase = np.angle(object_sample.field)
    hologram_phase = np.angle(hologram)
    
    axes[1, 0].imshow(object_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Object Phase')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hologram_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Hologram Phase')
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, 'Phase information\nPRESERVED!\n\nFull field information\navailable for ML',
                    ha='center', va='center', fontsize=14, color='green',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "intensity_vs_complex_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'intensity_vs_complex_comparison.png'}")


def main():
    """Generate all visual examples."""
    output_dir = Path("docs/examples/complex_fields")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visual examples for complex field documentation...")
    print(f"Output directory: {output_dir}")
    
    # Generate examples
    print("\n1. Generating object type comparisons...")
    save_object_type_comparison(output_dir)
    
    print("\n2. Generating hologram and reconstruction examples...")
    save_hologram_reconstruction_example(output_dir)
    
    print("\n3. Generating side-by-side comparison...")
    save_side_by_side_comparison(output_dir)
    
    print("\n✓ All visual examples generated successfully!")
    print(f"\nGenerated files in {output_dir}:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()

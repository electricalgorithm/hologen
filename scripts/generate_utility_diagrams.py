#!/usr/bin/env python3
"""Generate visual diagrams for utility functions documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
output_dir = Path("docs/examples/utilities")
output_dir.mkdir(parents=True, exist_ok=True)


def create_conversion_flow_diagram():
    """Create a diagram showing field conversion flows."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Define positions for nodes
    positions = {
        "complex": (5, 8),
        "intensity": (2, 5),
        "amplitude": (5, 5),
        "phase": (8, 5),
        "amp_phase": (5, 2),
    }

    # Define node styles
    node_style = dict(
        boxstyle="round,pad=0.3",
        facecolor="lightblue",
        edgecolor="darkblue",
        linewidth=2,
    )
    func_style = dict(
        boxstyle="round,pad=0.2",
        facecolor="lightyellow",
        edgecolor="orange",
        linewidth=1.5,
    )

    # Draw main nodes
    ax.text(
        positions["complex"][0],
        positions["complex"][1],
        "Complex Field\n(real + imag)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=node_style,
    )

    ax.text(
        positions["intensity"][0],
        positions["intensity"][1],
        "Intensity\n|E|²",
        ha="center",
        va="center",
        fontsize=11,
        bbox=node_style,
    )

    ax.text(
        positions["amplitude"][0],
        positions["amplitude"][1],
        "Amplitude\n|E|",
        ha="center",
        va="center",
        fontsize=11,
        bbox=node_style,
    )

    ax.text(
        positions["phase"][0],
        positions["phase"][1],
        "Phase\narg(E)",
        ha="center",
        va="center",
        fontsize=11,
        bbox=node_style,
    )

    ax.text(
        positions["amp_phase"][0],
        positions["amp_phase"][1],
        "Amplitude + Phase\n(separate arrays)",
        ha="center",
        va="center",
        fontsize=11,
        bbox=node_style,
    )

    # Draw conversion arrows with function labels
    arrow_style = dict(arrowstyle="->", lw=2, color="darkgreen")

    # complex_to_representation arrows
    # Complex to Intensity
    ax.annotate(
        "",
        xy=(positions["intensity"][0] + 0.5, positions["intensity"][1] + 0.3),
        xytext=(positions["complex"][0] - 0.8, positions["complex"][1] - 0.5),
        arrowprops=arrow_style,
    )
    ax.text(
        3,
        6.8,
        "complex_to_representation\n(INTENSITY)",
        fontsize=8,
        ha="center",
        bbox=func_style,
    )

    # Complex to Amplitude
    ax.annotate(
        "",
        xy=(positions["amplitude"][0], positions["amplitude"][1] + 0.5),
        xytext=(positions["complex"][0], positions["complex"][1] - 0.5),
        arrowprops=arrow_style,
    )
    ax.text(
        5,
        6.5,
        "complex_to_representation\n(AMPLITUDE)",
        fontsize=8,
        ha="center",
        bbox=func_style,
    )

    # Complex to Phase
    ax.annotate(
        "",
        xy=(positions["phase"][0] - 0.5, positions["phase"][1] + 0.3),
        xytext=(positions["complex"][0] + 0.8, positions["complex"][1] - 0.5),
        arrowprops=arrow_style,
    )
    ax.text(
        7,
        6.8,
        "complex_to_representation\n(PHASE)",
        fontsize=8,
        ha="center",
        bbox=func_style,
    )

    # amplitude_phase_to_complex arrow
    reverse_arrow_style = dict(arrowstyle="->", lw=2, color="darkred")
    ax.annotate(
        "",
        xy=(positions["complex"][0], positions["complex"][1] - 0.5),
        xytext=(positions["amp_phase"][0], positions["amp_phase"][1] + 0.5),
        arrowprops=reverse_arrow_style,
    )
    ax.text(
        5.8,
        5,
        "amplitude_phase_to_complex",
        fontsize=8,
        ha="center",
        bbox=func_style,
        rotation=90,
    )

    # Add title
    ax.text(
        5,
        9.5,
        "Field Conversion Flow Diagram",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor="lightblue", edgecolor="darkblue", label="Field Representation"
        ),
        mpatches.Patch(
            facecolor="lightyellow", edgecolor="orange", label="Conversion Function"
        ),
        mpatches.FancyArrow(
            0, 0, 1, 0, width=0.1, color="darkgreen", label="Forward Conversion"
        ),
        mpatches.FancyArrow(
            0, 0, 1, 0, width=0.1, color="darkred", label="Reconstruction"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "conversion_flow_diagram.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Created: {output_dir / 'conversion_flow_diagram.png'}")


def create_file_format_examples():
    """Create visual examples of different file formats."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("File Format Examples", fontsize=16, fontweight="bold")

    # Example 1: Legacy intensity format
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Legacy Intensity Format (.npz)", fontsize=12, fontweight="bold")

    format_text = """
    sample_00000_circle.npz
    ├── object: ndarray (512, 512) float64
    ├── hologram: ndarray (512, 512) float64
    └── reconstruction: ndarray (512, 512) float64
    
    Usage:
    data = np.load('sample_00000_circle.npz')
    object_img = data['object']
    hologram_img = data['hologram']
    reconstruction_img = data['reconstruction']
    
    Characteristics:
    • Real-valued intensity images
    • Backward compatible
    • Single .npz file per sample
    • ~2-5 MB per sample (compressed)
    """
    ax.text(
        0.05,
        0.95,
        format_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Example 2: Complex field format
    ax = axes[0, 1]
    ax.axis("off")
    ax.set_title("Complex Field Format (.npz)", fontsize=12, fontweight="bold")

    format_text = """
    sample_00000_object.npz
    ├── representation: str "complex"
    ├── real: ndarray (512, 512) float64
    └── imag: ndarray (512, 512) float64
    
    Usage:
    data = np.load('sample_00000_object.npz')
    representation = str(data['representation'])
    field = data['real'] + 1j * data['imag']
    
    Characteristics:
    • Complex-valued fields
    • Separate files per domain
    • Includes representation metadata
    • ~4-8 MB per file (compressed)
    """
    ax.text(
        0.05,
        0.95,
        format_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Example 3: Amplitude format
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("Amplitude Format (.npz)", fontsize=12, fontweight="bold")

    format_text = """
    sample_00000_object.npz
    ├── representation: str "amplitude"
    └── amplitude: ndarray (512, 512) float64
    
    Usage:
    data = np.load('sample_00000_object.npz')
    representation = str(data['representation'])
    amplitude = data['amplitude']
    
    Characteristics:
    • Real-valued amplitude
    • Phase information lost
    • Compact storage
    • ~2-4 MB per file (compressed)
    """
    ax.text(
        0.05,
        0.95,
        format_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # Example 4: Phase format
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Phase Format (.npz)", fontsize=12, fontweight="bold")

    format_text = """
    sample_00000_object.npz
    ├── representation: str "phase"
    └── phase: ndarray (512, 512) float64
    
    Usage:
    data = np.load('sample_00000_object.npz')
    representation = str(data['representation'])
    phase = data['phase']  # Range: [-π, π]
    
    Characteristics:
    • Real-valued phase in radians
    • Amplitude information lost
    • Compact storage
    • ~2-4 MB per file (compressed)
    """
    ax.text(
        0.05,
        0.95,
        format_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "file_format_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_dir / 'file_format_examples.png'}")


def create_io_workflow_diagram():
    """Create a diagram showing I/O workflow."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "Dataset I/O Workflow", fontsize=16, fontweight="bold", ha="center")

    # Define positions
    positions = {
        "generator": (2, 7),
        "writer": (5, 7),
        "disk": (8, 7),
        "loader": (5, 4),
        "ml_model": (5, 1),
    }

    # Node styles
    process_style = dict(
        boxstyle="round,pad=0.3",
        facecolor="lightgreen",
        edgecolor="darkgreen",
        linewidth=2,
    )
    storage_style = dict(
        boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black", linewidth=2
    )
    model_style = dict(
        boxstyle="round,pad=0.3",
        facecolor="lightcoral",
        edgecolor="darkred",
        linewidth=2,
    )

    # Draw nodes
    ax.text(
        positions["generator"][0],
        positions["generator"][1],
        "Dataset\nGenerator",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=process_style,
    )

    ax.text(
        positions["writer"][0],
        positions["writer"][1],
        "Writer\n(NumpyDatasetWriter\nor ComplexFieldWriter)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=process_style,
    )

    ax.text(
        positions["disk"][0],
        positions["disk"][1],
        "Disk Storage\n(.npz + .png files)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=storage_style,
    )

    ax.text(
        positions["loader"][0],
        positions["loader"][1],
        "Data Loader\n(load_complex_sample\nor custom loader)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=process_style,
    )

    ax.text(
        positions["ml_model"][0],
        positions["ml_model"][1],
        "ML Model\n(PyTorch/TensorFlow)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=model_style,
    )

    # Draw arrows
    arrow_style = dict(arrowstyle="->", lw=2.5, color="darkblue")

    # Generator to Writer
    ax.annotate(
        "",
        xy=(positions["writer"][0] - 0.8, positions["writer"][1]),
        xytext=(positions["generator"][0] + 0.8, positions["generator"][1]),
        arrowprops=arrow_style,
    )
    ax.text(
        3.5,
        7.5,
        "samples",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Writer to Disk
    ax.annotate(
        "",
        xy=(positions["disk"][0] - 0.8, positions["disk"][1]),
        xytext=(positions["writer"][0] + 0.8, positions["writer"][1]),
        arrowprops=arrow_style,
    )
    ax.text(
        6.5,
        7.5,
        "save()",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Disk to Loader
    ax.annotate(
        "",
        xy=(positions["loader"][0], positions["loader"][1] + 0.5),
        xytext=(positions["disk"][0], positions["disk"][1] - 0.5),
        arrowprops=arrow_style,
    )
    ax.text(
        7,
        5.5,
        "load files",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Loader to ML Model
    ax.annotate(
        "",
        xy=(positions["ml_model"][0], positions["ml_model"][1] + 0.5),
        xytext=(positions["loader"][0], positions["loader"][1] - 0.5),
        arrowprops=arrow_style,
    )
    ax.text(
        5.8,
        2.5,
        "batches",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add annotations
    annotation_style = dict(fontsize=8, style="italic", color="gray")
    ax.text(
        2, 6.2, "Generate synthetic\nhologram samples", ha="center", **annotation_style
    )
    ax.text(
        5,
        6.2,
        "Persist to disk with\noptional previews",
        ha="center",
        **annotation_style,
    )
    ax.text(8, 6.2, "NumPy archives +\nPNG previews", ha="center", **annotation_style)
    ax.text(
        5, 3.2, "Load and preprocess\nfor training", ha="center", **annotation_style
    )
    ax.text(
        5,
        0.2,
        "Train neural network\non synthetic data",
        ha="center",
        **annotation_style,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "io_workflow_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_dir / 'io_workflow_diagram.png'}")


def create_normalization_example():
    """Create visual example of image normalization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("normalize_image() Example", fontsize=14, fontweight="bold")

    # Create example image with arbitrary range
    x = np.linspace(-5, 5, 256)
    y = np.linspace(-5, 5, 256)
    X, Y = np.meshgrid(x, y)

    # Original image with arbitrary range [50, 250]
    original = 150 + 50 * np.sin(X) * np.cos(Y)

    # Normalized image [0, 1]
    normalized = (original - original.min()) / (original.max() - original.min())

    # 8-bit image [0, 255]
    uint8_image = (normalized * 255).astype(np.uint8)

    # Plot original
    im1 = axes[0].imshow(original, cmap="gray")
    axes[0].set_title(
        f"Original Image\nRange: [{original.min():.1f}, {original.max():.1f}]",
        fontsize=11,
    )
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot normalized
    im2 = axes[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"After normalize_image()\nRange: [{normalized.min():.1f}, {normalized.max():.1f}]",
        fontsize=11,
    )
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 8-bit
    im3 = axes[2].imshow(uint8_image, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"8-bit PNG\nRange: [{uint8_image.min()}, {uint8_image.max()}]", fontsize=11
    )
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / "normalization_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_dir / 'normalization_example.png'}")


def create_phase_validation_example():
    """Create visual example of phase validation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("validate_phase_range() Example", fontsize=14, fontweight="bold")

    # Valid phase
    x = np.linspace(-np.pi, np.pi, 256)
    y = np.linspace(-np.pi, np.pi, 256)
    X, Y = np.meshgrid(x, y)
    valid_phase = np.arctan2(Y, X)

    # Invalid phase (out of range)
    invalid_phase = valid_phase * 2  # Range: [-2π, 2π]

    # Phase with NaN
    nan_phase = valid_phase.copy()
    nan_phase[100:150, 100:150] = np.nan

    # Plot valid phase
    im1 = axes[0].imshow(valid_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(
        "Valid Phase\nRange: [-π, π]\n✓ Passes validation", fontsize=11, color="green"
    )
    axes[0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_ticks([-np.pi, 0, np.pi])
    cbar1.set_ticklabels(["-π", "0", "π"])

    # Plot invalid phase
    im2 = axes[1].imshow(invalid_phase, cmap="twilight")
    axes[1].set_title(
        "Invalid Phase\nRange: [-2π, 2π]\n✗ Fails validation", fontsize=11, color="red"
    )
    axes[1].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_ticks([-2 * np.pi, 0, 2 * np.pi])
    cbar2.set_ticklabels(["-2π", "0", "2π"])

    # Plot NaN phase
    im3 = axes[2].imshow(nan_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title(
        "Phase with NaN\nContains non-finite values\n✗ Fails validation",
        fontsize=11,
        color="red",
    )
    axes[2].axis("off")
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_ticks([-np.pi, 0, np.pi])
    cbar3.set_ticklabels(["-π", "0", "π"])

    plt.tight_layout()
    plt.savefig(
        output_dir / "phase_validation_example.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Created: {output_dir / 'phase_validation_example.png'}")


if __name__ == "__main__":
    print("Generating utility diagrams...")
    print()

    create_conversion_flow_diagram()
    create_file_format_examples()
    create_io_workflow_diagram()
    create_normalization_example()
    create_phase_validation_example()

    print()
    print(f"All diagrams saved to: {output_dir}")
    print("Done!")

#!/usr/bin/env python3
"""Generate visual examples for all shape generators.

This script creates example images showing:
1. Each shape generator output
2. Parameter effect demonstrations
3. Amplitude vs phase comparison images
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hologen.shapes import (
    CircleCheckerGenerator,
    CircleGenerator,
    EllipseCheckerGenerator,
    RectangleCheckerGenerator,
    RectangleGenerator,
    RingGenerator,
    TriangleCheckerGenerator,
)
from hologen.types import GridSpec


def save_image(array: np.ndarray, path: Path, normalize: bool = True) -> None:
    """Save a numpy array as a PNG image.

    Args:
        array: 2D array to save.
        path: Output file path.
        normalize: Whether to normalize to [0, 255] range.
    """
    if normalize:
        # Normalize to [0, 255]
        array_min = array.min()
        array_max = array.max()
        if array_max > array_min:
            array_norm = 255 * (array - array_min) / (array_max - array_min)
        else:
            array_norm = np.zeros_like(array)
    else:
        array_norm = array * 255

    # Convert to uint8 and save
    img = Image.fromarray(array_norm.astype(np.uint8), mode="L")
    img.save(path)
    print(f"Saved: {path}")


def generate_basic_examples(output_dir: Path) -> None:
    """Generate basic examples for each shape type."""
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    rng = np.random.default_rng(42)

    # Create output directory
    basic_dir = output_dir / "basic"
    basic_dir.mkdir(parents=True, exist_ok=True)

    # Circle
    circle_gen = CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18)
    circle = circle_gen.generate(grid, rng)
    save_image(circle, basic_dir / "circle.png", normalize=False)

    # Rectangle
    rect_gen = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.35)
    rectangle = rect_gen.generate(grid, rng)
    save_image(rectangle, basic_dir / "rectangle.png", normalize=False)

    # Ring
    ring_gen = RingGenerator(
        name="ring",
        min_radius=0.12,
        max_radius=0.25,
        min_thickness=0.1,
        max_thickness=0.3,
    )
    ring = ring_gen.generate(grid, rng)
    save_image(ring, basic_dir / "ring.png", normalize=False)

    # Circle Checker
    circle_checker_gen = CircleCheckerGenerator(
        name="circle_checker", min_radius=0.1, max_radius=0.2, checker_size=16
    )
    circle_checker = circle_checker_gen.generate(grid, rng)
    save_image(circle_checker, basic_dir / "circle_checker.png", normalize=False)

    # Rectangle Checker
    rect_checker_gen = RectangleCheckerGenerator(
        name="rectangle_checker", min_scale=0.1, max_scale=0.35, checker_size=16
    )
    rect_checker = rect_checker_gen.generate(grid, rng)
    save_image(rect_checker, basic_dir / "rectangle_checker.png", normalize=False)

    # Ellipse Checker
    ellipse_checker_gen = EllipseCheckerGenerator(
        name="ellipse_checker",
        min_radius_y=0.1,
        max_radius_y=0.35,
        min_radius_x=0.1,
        max_radius_x=0.35,
        checker_size=16,
    )
    ellipse_checker = ellipse_checker_gen.generate(grid, rng)
    save_image(ellipse_checker, basic_dir / "ellipse_checker.png", normalize=False)

    # Triangle Checker
    triangle_checker_gen = TriangleCheckerGenerator(
        name="triangle_checker", min_scale=0.15, max_scale=0.3, checker_size=16
    )
    triangle_checker = triangle_checker_gen.generate(grid, rng)
    save_image(triangle_checker, basic_dir / "triangle_checker.png", normalize=False)


def generate_parameter_variations(output_dir: Path) -> None:
    """Generate examples showing parameter effects."""
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)

    # Create output directory
    param_dir = output_dir / "parameters"
    param_dir.mkdir(parents=True, exist_ok=True)

    # Circle size variations
    for i, (min_r, max_r) in enumerate([(0.05, 0.1), (0.08, 0.18), (0.15, 0.25)]):
        rng = np.random.default_rng(100 + i)
        gen = CircleGenerator(name="circle", min_radius=min_r, max_radius=max_r)
        circle = gen.generate(grid, rng)
        save_image(circle, param_dir / f"circle_size_{i + 1}.png", normalize=False)

    # Checker size variations
    for i, checker_size in enumerate([8, 16, 32]):
        rng = np.random.default_rng(200 + i)
        gen = CircleCheckerGenerator(
            name="circle_checker",
            min_radius=0.15,
            max_radius=0.2,
            checker_size=checker_size,
        )
        checker = gen.generate(grid, rng)
        save_image(
            checker, param_dir / f"checker_size_{checker_size}.png", normalize=False
        )

    # Ring thickness variations
    for i, (min_t, max_t) in enumerate([(0.1, 0.2), (0.2, 0.4), (0.4, 0.6)]):
        rng = np.random.default_rng(300 + i)
        gen = RingGenerator(
            name="ring",
            min_radius=0.15,
            max_radius=0.2,
            min_thickness=min_t,
            max_thickness=max_t,
        )
        ring = gen.generate(grid, rng)
        save_image(ring, param_dir / f"ring_thickness_{i + 1}.png", normalize=False)


def generate_complex_field_examples(output_dir: Path) -> None:
    """Generate amplitude vs phase comparison images."""
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    rng = np.random.default_rng(42)

    # Create output directory
    complex_dir = output_dir / "complex_fields"
    complex_dir.mkdir(parents=True, exist_ok=True)

    # Use circle generator for demonstration
    gen = CircleGenerator(name="circle", min_radius=0.12, max_radius=0.18)

    # Amplitude-only object
    amplitude_field = gen.generate_complex(grid, rng, mode="amplitude")
    amplitude_amp = np.abs(amplitude_field)
    amplitude_phase = np.angle(amplitude_field)

    save_image(amplitude_amp, complex_dir / "amplitude_mode_amplitude.png")
    save_image(
        (amplitude_phase + np.pi) / (2 * np.pi),
        complex_dir / "amplitude_mode_phase.png",
    )

    # Phase-only object with π/2 shift
    rng = np.random.default_rng(42)  # Reset for same shape
    phase_field = gen.generate_complex(grid, rng, mode="phase", phase_shift=np.pi / 2)
    phase_amp = np.abs(phase_field)
    phase_phase = np.angle(phase_field)

    save_image(phase_amp, complex_dir / "phase_mode_amplitude.png")
    save_image(
        (phase_phase + np.pi) / (2 * np.pi), complex_dir / "phase_mode_phase.png"
    )

    # Phase-only with different phase shifts
    for i, shift in enumerate([np.pi / 4, np.pi / 2, np.pi]):
        rng = np.random.default_rng(42)  # Reset for same shape
        field = gen.generate_complex(grid, rng, mode="phase", phase_shift=shift)
        phase = np.angle(field)
        save_image(
            (phase + np.pi) / (2 * np.pi),
            complex_dir / f"phase_shift_{int(np.degrees(shift))}_deg.png",
        )


def generate_variety_grid(output_dir: Path) -> None:
    """Generate a grid showing variety of all shapes."""
    grid = GridSpec(height=256, width=256, pixel_pitch=6.4e-6)

    # Create 3x3 grid of different shapes
    rows = 3
    cols = 3
    canvas = np.zeros((rows * 256, cols * 256), dtype=np.float64)

    generators = [
        CircleGenerator(name="circle", min_radius=0.15, max_radius=0.2),
        RectangleGenerator(name="rectangle", min_scale=0.15, max_scale=0.25),
        RingGenerator(
            name="ring",
            min_radius=0.15,
            max_radius=0.2,
            min_thickness=0.2,
            max_thickness=0.3,
        ),
        CircleCheckerGenerator(
            name="circle_checker", min_radius=0.15, max_radius=0.2, checker_size=12
        ),
        RectangleCheckerGenerator(
            name="rectangle_checker", min_scale=0.15, max_scale=0.25, checker_size=12
        ),
        EllipseCheckerGenerator(
            name="ellipse_checker",
            min_radius_y=0.15,
            max_radius_y=0.25,
            min_radius_x=0.1,
            max_radius_x=0.2,
            checker_size=12,
        ),
        TriangleCheckerGenerator(
            name="triangle_checker", min_scale=0.2, max_scale=0.25, checker_size=12
        ),
    ]

    # Generate shapes
    for idx, gen in enumerate(generators):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols
        rng = np.random.default_rng(500 + idx)
        shape = gen.generate(grid, rng)
        canvas[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = shape

    # Save grid
    variety_dir = output_dir / "variety"
    variety_dir.mkdir(parents=True, exist_ok=True)
    save_image(canvas, variety_dir / "shape_variety_grid.png", normalize=False)


def main() -> None:
    """Generate all shape examples."""
    output_dir = Path("docs/examples/shapes")

    print("Generating shape examples...")
    print(f"Output directory: {output_dir}")

    # Generate all example types
    generate_basic_examples(output_dir)
    generate_parameter_variations(output_dir)
    generate_complex_field_examples(output_dir)
    generate_variety_grid(output_dir)

    print("\nDone! Generated examples in:")
    print(f"  - {output_dir / 'basic'}")
    print(f"  - {output_dir / 'parameters'}")
    print(f"  - {output_dir / 'complex_fields'}")
    print(f"  - {output_dir / 'variety'}")


if __name__ == "__main__":
    main()

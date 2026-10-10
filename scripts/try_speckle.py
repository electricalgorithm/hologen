"""
Interactively add speckle noise to an image and see the result live.

The image is treated as an intensity (brightness) map: speckle multiplies it by a
random pattern S with mean 1, so the overall brightness is kept and only grainy
fluctuations are added. Use the sliders to change the speckle contrast and grain
size; the same random seed is reused so only the parameters change.

Usage:
    uv run scripts/try_speckle.py                      # simulated hologram
    uv run scripts/try_speckle.py photo.png            # your own image
    uv run scripts/try_speckle.py photo.png --contrast 0.3 --grain-size 30
    uv run scripts/try_speckle.py photo.png --output noisy.png --no-show
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from hologen.noises import add_speckle_noise
from hologen.noises.speckle import generate_speckle_pattern
from hologen.objects import add_random_cells


def load_intensity(path: pathlib.Path) -> numpy.ndarray[float]:
    """Load an image as a square grayscale intensity map in [0, 1]."""
    image: numpy.ndarray = plt.imread(path)
    if numpy.issubdtype(image.dtype, numpy.integer):
        image = image / numpy.iinfo(image.dtype).max
    if image.ndim == 3:
        # Drop alpha, then convert RGB to luminance (ITU-R BT.709 weights).
        image = image[..., :3] @ numpy.array([0.2126, 0.7152, 0.0722])

    # The speckle model works on square sensors, so crop the centre square.
    side: int = min(image.shape)
    top: int = (image.shape[0] - side) // 2
    left: int = (image.shape[1] - side) // 2
    return image[top : top + side, left : left + side].astype(float)


def simulate_hologram(pixel_size: float, seed: int) -> numpy.ndarray[float]:
    """Simulate a clean 224x224 in-line hologram with the dataset geometry."""
    hologen.RandomNumberGenerator.set_seed(seed)
    sim = hologen.InlineHologramSimulator(resolution=224, pixel_size=pixel_size)
    add_random_cells(sim, num_cells=5)
    return sim.generate_hologram(z_distance=20000)


def speckled(
    intensity: numpy.ndarray[float],
    pixel_size: float,
    grain_size: float,
    contrast: float,
    seed: int,
) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Return (pattern S, intensity with speckle).

    Re-seeding before each call makes generate_speckle_pattern() return exactly
    the pattern that add_speckle_noise() multiplies the intensity by.
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    pattern = generate_speckle_pattern(
        intensity.shape[0], pixel_size, grain_size, contrast
    )
    hologen.RandomNumberGenerator.set_seed(seed)
    return pattern, add_speckle_noise(intensity, pixel_size, grain_size, contrast)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("image", nargs="?", type=pathlib.Path, help="input image")
    parser.add_argument("--contrast", type=float, default=0.3, help="std/mean, (0, 1]")
    parser.add_argument("--grain-size", type=float, default=30.0, help="microns")
    parser.add_argument("--pixel-size", type=float, default=4.65, help="microns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, help="save the noisy image")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    if args.image is None:
        intensity = simulate_hologram(args.pixel_size, args.seed)
        title = "Simulated hologram"
    else:
        intensity = load_intensity(args.image)
        title = args.image.name

    pattern, noisy = speckled(
        intensity, args.pixel_size, args.grain_size, args.contrast, args.seed
    )
    if args.output is not None:
        plt.imsave(args.output, noisy, cmap="gray", vmin=0, vmax=intensity.max())
        print(f"Saved {args.output}")
    if args.no_show:
        return

    # Same grey scale for input and output, so brightness changes are comparable.
    vmax: float = float(intensity.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(bottom=0.25)

    axes[0].imshow(intensity, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title(f"Input: {title}")
    pattern_image = axes[1].imshow(pattern)
    axes[1].set_title("Speckle pattern S (mean 1)")
    fig.colorbar(pattern_image, ax=axes[1], fraction=0.046)
    output_image = axes[2].imshow(noisy, cmap="gray", vmin=0, vmax=vmax)
    axes[2].set_title("Output: input x S")
    for ax in axes:
        ax.axis("off")

    contrast_slider = Slider(
        fig.add_axes((0.15, 0.12, 0.7, 0.03)),
        "Contrast C",
        0.01,
        1.0,
        valinit=args.contrast,
    )
    grain_slider = Slider(
        fig.add_axes((0.15, 0.06, 0.7, 0.03)),
        "Grain size (um)",
        2 * args.pixel_size,
        40 * args.pixel_size,
        valinit=args.grain_size,
    )

    def update(_: float) -> None:
        new_pattern, result = speckled(
            intensity,
            args.pixel_size,
            grain_slider.val,
            contrast_slider.val,
            args.seed,
        )
        pattern_image.set_data(new_pattern)
        pattern_image.set_clim(0, numpy.percentile(new_pattern, 99.5))
        output_image.set_data(result)
        fig.canvas.draw_idle()

    contrast_slider.on_changed(update)
    grain_slider.on_changed(update)
    update(0.0)
    plt.show()


if __name__ == "__main__":
    main()

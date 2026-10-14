"""
Interactively add speckle noise to an image and see the result live.

The image is treated as an intensity (brightness) map: speckle multiplies it by a
random pattern S with mean 1, so the overall brightness is kept and only grainy
fluctuations are added. Use the sliders to change the speckle contrast and grain
size; the same random seed is reused so only the parameters change.

Usage:
    hologen explore speckle                         # simulated hologram
    hologen explore speckle photo.png               # your own image
    hologen explore speckle --contrast 0.3 --grain-size 30
    hologen explore speckle --save speckle.png      # save, no window
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from hologen.noises import add_speckle_noise
from hologen.noises.speckle import generate_speckle_pattern
from hologen.explore.common import (
    RECIPE,
    load_intensity,
    show_or_save,
    simulate_intensity,
)


def speckled(
    intensity: numpy.ndarray[float],
    pixel_size: float,
    grain_size: float,
    contrast: float,
    seed: int,
) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Add speckle to an intensity and return the pattern that was used.

    Re-seeding before each call makes generate_speckle_pattern() return exactly
    the pattern that add_speckle_noise() multiplies the intensity by.

    Args:
        intensity (numpy.ndarray[float]): Intensity map, square.
        pixel_size (float): Pixel size, in um.
        grain_size (float): Mean speckle grain size, in um.
        contrast (float): Speckle contrast std/mean, in (0, 1].
        seed (int): Random seed of the pattern.

    Returns:
        tuple[numpy.ndarray[float], numpy.ndarray[float]]: The speckle pattern S
            with mean 1, and the intensity multiplied by it.
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    pattern = generate_speckle_pattern(
        intensity.shape[0], pixel_size, grain_size, contrast
    )
    hologen.RandomNumberGenerator.set_seed(seed)
    return pattern, add_speckle_noise(intensity, pixel_size, grain_size, contrast)


def run(
    image: pathlib.Path | None = None,
    contrast: float = RECIPE.speckle.laser_contrast,
    grain_size: float = RECIPE.speckle.grain_size,
    pixel_size: float = RECIPE.geometry.pixel_size,
    seed: int = 0,
    save: pathlib.Path | None = None,
) -> None:
    """
    Open the explorer.

    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        contrast (float): Speckle contrast std/mean, in (0, 1].
        grain_size (float): Mean speckle grain size, in um.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise pattern.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    if image is None:
        intensity = simulate_intensity(pixel_size, seed)
        title = "Simulated hologram"
    else:
        intensity = load_intensity(image)
        title = image.name

    pattern, noisy = speckled(intensity, pixel_size, grain_size, contrast, seed)

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
        valinit=contrast,
    )
    grain_slider = Slider(
        fig.add_axes((0.15, 0.06, 0.7, 0.03)),
        "Grain size (um)",
        2 * pixel_size,
        40 * pixel_size,
        valinit=grain_size,
    )

    def update(_: float) -> None:
        """
        Redraw the figure after a slider moved.

        Args:
            _ (float): New value of the slider that moved, unused because every
                slider is read again.
        """
        new_pattern, result = speckled(
            intensity,
            pixel_size,
            grain_slider.val,
            contrast_slider.val,
            seed,
        )
        pattern_image.set_data(new_pattern)
        pattern_image.set_clim(0, numpy.percentile(new_pattern, 99.5))
        output_image.set_data(result)
        fig.canvas.draw_idle()

    contrast_slider.on_changed(update)
    grain_slider.on_changed(update)
    update(0.0)
    show_or_save(fig, save)

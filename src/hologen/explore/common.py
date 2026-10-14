"""
Shared helpers for the interactive noise explorers (hologen explore ...).

The values come from the default dataset recipe, so the explorers show exactly
what the dataset uses.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.figure import Figure

import hologen
from hologen.dataset.recipe import Geometry, Recipe, default_recipe
from hologen.objects import add_random_cells

RECIPE: Recipe = default_recipe()
DARK_CURRENT: float = RECIPE.camera.dark_current
EXPOSURE_TIME: float = RECIPE.camera.exposure_time
READ_NOISE_SIGMA: float = RECIPE.camera.read_noise

# Saturation capacity of the modelled sensor; pixels collecting more are clipped.
SATURATION_ELECTRONS: float = RECIPE.camera.saturation_electrons


def load_intensity(path: pathlib.Path) -> numpy.ndarray[float]:
    """
    Load an image as a square grayscale intensity map in [0, 1].

    Args:
        path (pathlib.Path): Image file readable by matplotlib.

    Returns:
        numpy.ndarray[float]: The centre square of the image as luminance, shape
            (side, side).
    """
    image: numpy.ndarray = plt.imread(path)
    if numpy.issubdtype(image.dtype, numpy.integer):
        image = image / numpy.iinfo(image.dtype).max
    if image.ndim == 3:
        # Drop alpha, then convert RGB to luminance (ITU-R BT.709 weights).
        image = image[..., :3] @ numpy.array([0.2126, 0.7152, 0.0722])

    # The noise models work on square sensors, so crop the centre square.
    side: int = min(image.shape)
    top: int = (image.shape[0] - side) // 2
    left: int = (image.shape[1] - side) // 2
    return image[top : top + side, left : left + side].astype(float)


def simulate_intensity(pixel_size: float, seed: int) -> numpy.ndarray[float]:
    """
    Simulate a clean, noise-free in-line hologram with the dataset geometry.

    Args:
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed for the objects.

    Returns:
        numpy.ndarray[float]: The intensity |U|^2, normalised so that the
            background illumination is 1, shape (resolution, resolution).
    """
    geometry: Geometry = RECIPE.geometry
    hologen.RandomNumberGenerator.set_seed(seed)
    sim: hologen.InlineHologramSimulator = hologen.InlineHologramSimulator(
        resolution=geometry.resolution,
        pixel_size=pixel_size,
        wavelength=geometry.wavelength,
    )
    add_random_cells(sim, num_cells=geometry.num_cells)
    return sim.generate_hologram(
        z_distance=geometry.z_distance, background_electrons=1.0
    )


def show_or_save(fig: Figure, save: pathlib.Path | None) -> None:
    """
    Open the interactive window, or save the figure without opening one.

    Args:
        fig (Figure): The explorer figure.
        save (pathlib.Path | None): File to save the figure to, or None to open
            the window.
    """
    if save is None:
        plt.show()
        return
    fig.savefig(save, dpi=100)
    plt.close(fig)
    print(f"Saved {save}")

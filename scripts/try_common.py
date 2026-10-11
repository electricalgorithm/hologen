"""
Shared helpers for the interactive try_*.py scripts.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy

import hologen
from hologen.objects import add_random_cells

# Saturation capacity of the modelled Sony ICX267 CCD (Basler EMVA 1288 data
# overview, scout scA1390-17gm). Pixels collecting more electrons are clipped.
SATURATION_ELECTRONS: float = 11000.0


def load_intensity(path: pathlib.Path) -> numpy.ndarray[float]:
    """Load an image as a square grayscale intensity map in [0, 1]."""
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
    Simulate a clean, noise-free 224x224 in-line hologram with the dataset
    geometry, normalised so that the background illumination is 1 (|U|^2).
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    sim = hologen.InlineHologramSimulator(resolution=224, pixel_size=pixel_size)
    add_random_cells(sim, num_cells=5)
    return sim.generate_hologram(z_distance=20000, background_electrons=1.0)

"""Speckle of the light source and the shot, dark, read and ADC noise of the camera."""

from .bit_depth import add_bit_depth_noise
from .dark_current import add_dark_current_noise
from .read import add_read_noise
from .shot import add_shot_noise
from .speckle import add_speckle_noise

__all__ = [
    "add_speckle_noise",
    "add_dark_current_noise",
    "add_shot_noise",
    "add_read_noise",
    "add_bit_depth_noise",
]

"""
Interactively change sensor temperature and exposure time and watch dark current.

Heat frees electrons in the sensor even without light. Their mean number per
pixel is dark current (e-/s) x exposure time (s), and the dark current roughly
doubles every T_d degrees (EMVA 1288 Release 4.0 Linear, Sec. 3). The sliders
change the two physical knobs:

- temperature: the dark current rate, starting from the dataset's 25 C value
- exposure time: how long the dark electrons accumulate

The illumination is assumed to be adjusted so the background always collects the
same number of photo-electrons, so only the dark electrons change.

Usage:
    hologen explore dark-current                    # simulated hologram
    hologen explore dark-current photo.png          # your own image
    hologen explore dark-current --doubling 6       # faster temperature rise
    hologen explore dark-current --save dark.png    # save, no window
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from hologen.inline_simulator import DEFAULT_BACKGROUND_ELECTRONS
from hologen.noises import add_dark_current_noise, add_shot_noise
from hologen.explore.common import (
    DARK_CURRENT,
    EXPOSURE_TIME,
    READ_NOISE_SIGMA,
    RECIPE,
    SATURATION_ELECTRONS,
    load_intensity,
    show_or_save,
    simulate_intensity,
)

REFERENCE_TEMPERATURE: float = 25.0  # C, temperature of DARK_CURRENT


def dark_current_rate(temperature: float, doubling: float) -> float:
    """Dark current (e-/s) at a temperature (EMVA 1288 Sec. 3.2, Eq. 31)."""
    return DARK_CURRENT * 2 ** ((temperature - REFERENCE_TEMPERATURE) / doubling)


def record(
    intensity: numpy.ndarray[float], electrons: float, dark_mean: float, seed: int
) -> tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[bool]]:
    """
    Record a dark frame and the hologram with shot noise and dark electrons.

    Returns:
        (dark frame in e-, hologram normalised to background = 1, clipped mask)
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    dark_frame = add_dark_current_noise(numpy.zeros_like(intensity), dark_mean)
    hologen.RandomNumberGenerator.set_seed(seed + 1)
    counts = add_dark_current_noise(add_shot_noise(intensity * electrons), dark_mean)
    clipped = counts >= SATURATION_ELECTRONS
    counts = numpy.minimum(counts, SATURATION_ELECTRONS)
    return dark_frame, counts / electrons, clipped


def run(
    image: pathlib.Path | None = None,
    temperature: float = REFERENCE_TEMPERATURE,
    exposure: float = EXPOSURE_TIME,
    doubling: float = 9.0,
    pixel_size: float = RECIPE.geometry.pixel_size,
    seed: int = 0,
    save: pathlib.Path | None = None,
) -> None:
    """
    Open the explorer.

    Args:
        image: Image to use as the intensity; a simulated hologram if None.
        temperature: Sensor temperature (in C).
        exposure: Exposure time (in seconds).
        doubling: Temperature step that doubles the dark current (in C).
        pixel_size: Pixel size (in microns).
        seed: Random seed; the same seed gives the same noise pattern.
        save: Save the figure to this file instead of opening a window.
    """

    electrons: float = DEFAULT_BACKGROUND_ELECTRONS
    if image is None:
        intensity = simulate_intensity(pixel_size, seed)
        title = "Simulated hologram"
    else:
        intensity = load_intensity(image)
        intensity = intensity / intensity.max()
        title = image.name
    vmin, vmax = float(intensity.min()), float(intensity.max())

    fig = plt.figure(figsize=(16, 6.5))
    grid = fig.add_gridspec(1, 3, bottom=0.24, wspace=0.3, width_ratios=(1, 1, 1.2))
    ax_dark = fig.add_subplot(grid[0])
    ax_holo = fig.add_subplot(grid[1])
    ax_budget = fig.add_subplot(grid[2])

    dark_image = ax_dark.imshow(numpy.zeros_like(intensity), cmap="magma")
    fig.colorbar(dark_image, ax=ax_dark, fraction=0.046, label="dark electrons")
    holo_image = ax_holo.imshow(intensity, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = numpy.zeros((*intensity.shape, 4))
    overlay[..., 0] = 1.0  # red
    clipped_image = ax_holo.imshow(overlay)
    for ax in (ax_dark, ax_holo):
        ax.axis("off")

    # Noise budget: every term in electrons per pixel, on one log scale.
    labels = [
        "signal\n(background)",
        "shot noise\n(std)",
        "dark\n(mean)",
        "dark noise\n(std)",
        "read noise\n(std, next step)",
    ]
    colors = ["0.6", "tab:blue", "tab:red", "tab:orange", "0.8"]
    bars = ax_budget.bar(labels, numpy.ones(len(labels)), color=colors)
    bars[-1].set_hatch("//")
    ax_budget.set_yscale("log")
    ax_budget.set_ylim(0.1, 1e6)
    ax_budget.axhline(SATURATION_ELECTRONS, color="k", ls="--", lw=0.8)
    ax_budget.text(
        4.4, SATURATION_ELECTRONS * 1.3, "saturation", ha="right", fontsize=8
    )
    ax_budget.set_ylabel("electrons per pixel")
    ax_budget.tick_params(axis="x", labelsize=8)
    value_texts = [ax_budget.text(i, 1, "", ha="center", va="bottom") for i in range(5)]

    temperature_slider = Slider(
        fig.add_axes((0.15, 0.11, 0.7, 0.03)),
        "Temperature (C)",
        0.0,
        70.0,
        valinit=temperature,
    )
    exposure_slider = Slider(
        fig.add_axes((0.15, 0.05, 0.7, 0.03)),
        "log10(exposure / s)",
        -3.0,
        1.5,
        valinit=numpy.log10(exposure),
    )

    def update(_: float) -> None:
        temperature = temperature_slider.val
        exposure = 10**exposure_slider.val
        rate = dark_current_rate(temperature, doubling)
        dark_mean = rate * exposure
        dark_frame, result, clipped = record(intensity, electrons, dark_mean, seed)

        dark_image.set_data(dark_frame)
        dark_image.set_clim(0, max(float(dark_frame.max()), 1.0))
        ax_dark.set_title(
            f"Dark frame (light off)\n{rate:,.0f} e-/s x {exposure * 1000:,.1f} ms"
            f" = {dark_mean:,.1f} e- mean"
        )
        holo_image.set_data(result)
        overlay[..., 3] = clipped * 0.9
        clipped_image.set_data(overlay)
        ax_holo.set_title(
            f"{title} with shot noise + dark\n"
            f"{clipped.mean() * 100:.1f}% pixels clipped (red)"
        )

        values = [
            electrons,
            numpy.sqrt(electrons),
            dark_mean,
            numpy.sqrt(dark_mean),
            READ_NOISE_SIGMA,
        ]
        for bar, text, value in zip(bars, value_texts, values):
            bar.set_height(max(value, 0.1))
            text.set_position((text.get_position()[0], max(value, 0.1) * 1.15))
            text.set_text(f"{value:,.1f}")
        ax_budget.set_title(
            f"Noise budget at {temperature:.0f} C (T_d = {doubling:g} C)\n"
            f"dataset: {REFERENCE_TEMPERATURE:.0f} C, {EXPOSURE_TIME * 1000:.1f} ms"
        )
        fig.canvas.draw_idle()

    temperature_slider.on_changed(update)
    exposure_slider.on_changed(update)
    update(0.0)
    show_or_save(fig, save)

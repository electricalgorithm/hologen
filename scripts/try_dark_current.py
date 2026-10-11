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
    uv run scripts/try_dark_current.py                 # simulated hologram
    uv run scripts/try_dark_current.py photo.png       # your own image
    uv run scripts/try_dark_current.py --doubling 6    # faster temperature rise
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from create_dataset import DARK_CURRENT, EXPOSURE_TIME, READ_NOISE_SIGMA
from hologen.inline_simulator import DEFAULT_BACKGROUND_ELECTRONS
from hologen.noises import add_dark_current_noise, add_shot_noise
from try_common import SATURATION_ELECTRONS, load_intensity, simulate_intensity

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("image", nargs="?", type=pathlib.Path, help="input image")
    parser.add_argument("--temperature", type=float, default=REFERENCE_TEMPERATURE)
    parser.add_argument("--exposure", type=float, default=EXPOSURE_TIME, help="seconds")
    parser.add_argument(
        "--doubling", type=float, default=9.0, help="T_d: C per doubling (6-9)"
    )
    parser.add_argument("--pixel-size", type=float, default=4.65, help="microns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    electrons: float = DEFAULT_BACKGROUND_ELECTRONS
    if args.image is None:
        intensity = simulate_intensity(args.pixel_size, args.seed)
        title = "Simulated hologram"
    else:
        intensity = load_intensity(args.image)
        intensity = intensity / intensity.max()
        title = args.image.name
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
        valinit=args.temperature,
    )
    exposure_slider = Slider(
        fig.add_axes((0.15, 0.05, 0.7, 0.03)),
        "log10(exposure / s)",
        -3.0,
        1.5,
        valinit=numpy.log10(args.exposure),
    )

    def update(_: float) -> None:
        temperature = temperature_slider.val
        exposure = 10**exposure_slider.val
        rate = dark_current_rate(temperature, args.doubling)
        dark_mean = rate * exposure
        dark_frame, result, clipped = record(intensity, electrons, dark_mean, args.seed)

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
            f"Noise budget at {temperature:.0f} C (T_d = {args.doubling:g} C)\n"
            f"dataset: {REFERENCE_TEMPERATURE:.0f} C, {EXPOSURE_TIME * 1000:.1f} ms"
        )
        fig.canvas.draw_idle()

    temperature_slider.on_changed(update)
    exposure_slider.on_changed(update)
    update(0.0)
    plt.show()


if __name__ == "__main__":
    main()

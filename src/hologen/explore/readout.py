"""
Interactively change read noise and ADC bit depth and watch the camera readout.

After the exposure each pixel holds electrons (signal + dark). The camera reads
them out in two steps (EMVA 1288 Release 4.0 Linear, Sec. 2.4, Eq. 14):

- read noise: the amplifier adds a Gaussian error of sigma_d electrons
- ADC: electrons are multiplied by the gain K = (2^k - 1) / saturation capacity,
  clipped at 2^k - 1 and rounded to whole digital numbers (DN); rounding adds a
  uniform error of 1/sqrt(12) DN, i.e. 1 / (sqrt(12) K) electrons

Shot noise and dark current stay at the dataset's values. Fewer bits mean larger
DN steps: below about 8 bits the steps become visible as a staircase.

Usage:
    hologen explore readout                         # simulated hologram
    hologen explore readout photo.png               # your own image
    hologen explore readout --bits 6 --read-noise 30
    hologen explore readout --save readout.png      # save, no window
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
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
from hologen.inline_simulator import DEFAULT_BACKGROUND_ELECTRONS
from hologen.noises import (
    add_bit_depth_noise,
    add_dark_current_noise,
    add_read_noise,
    add_shot_noise,
)
from hologen.noises.bit_depth import adc_gain

DATASET_BITS: int = RECIPE.camera.bit_depth


def collect(intensity: numpy.ndarray[float], seed: int) -> numpy.ndarray[float]:
    """
    Collect the electrons of one exposure, with shot noise and dark electrons.

    Args:
        intensity (numpy.ndarray[float]): Intensity with the background at 1.
        seed (int): Random seed of the shot noise and dark electrons.

    Returns:
        numpy.ndarray[float]: Electrons per pixel, same shape as intensity.
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    electrons = add_shot_noise(intensity * DEFAULT_BACKGROUND_ELECTRONS)
    return add_dark_current_noise(electrons, DARK_CURRENT * EXPOSURE_TIME)


def read_out(
    electrons: numpy.ndarray[float], read_noise: float, bits: int, seed: int
) -> numpy.ndarray[float]:
    """
    Add read noise and digitise the electrons.

    Args:
        electrons (numpy.ndarray[float]): Electrons per pixel.
        read_noise (float): Read noise, in e- rms.
        bits (int): ADC bit depth.
        seed (int): Random seed; the read noise uses seed + 1.

    Returns:
        numpy.ndarray[float]: The recorded image in digital numbers.
    """
    hologen.RandomNumberGenerator.set_seed(seed + 1)
    return add_bit_depth_noise(
        add_read_noise(electrons, read_noise), bits, SATURATION_ELECTRONS
    )


def run(
    image: pathlib.Path | None = None,
    read_noise: float = READ_NOISE_SIGMA,
    bits: int = DATASET_BITS,
    pixel_size: float = RECIPE.geometry.pixel_size,
    seed: int = 0,
    save: pathlib.Path | None = None,
) -> None:
    """
    Open the explorer.

    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        read_noise (float): Read noise, in e- rms.
        bits (int): ADC bit depth.
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
        intensity = intensity / intensity.max()
        title = image.name
    electrons = collect(intensity, seed)
    dark_mean = DARK_CURRENT * EXPOSURE_TIME

    # Profile: 60 pixels across the row with the strongest fringes.
    row: int = int(numpy.argmax(intensity.std(axis=1)))
    start: int = max(0, int(numpy.argmax(intensity[row])) - 30)
    x = numpy.arange(start, min(start + 60, intensity.shape[1]))

    fig = plt.figure(figsize=(16, 6.5))
    grid = fig.add_gridspec(1, 3, bottom=0.24, wspace=0.3, width_ratios=(1, 1.2, 1.2))
    ax_image = fig.add_subplot(grid[0])
    ax_profile = fig.add_subplot(grid[1])
    ax_budget = fig.add_subplot(grid[2])

    image = ax_image.imshow(intensity, cmap="gray")
    ax_image.axhline(row, color="tab:orange", lw=0.8)
    ax_image.axis("off")

    (ideal_line,) = ax_profile.plot(
        [], [], color="black", lw=1.2, label="ideal (no readout)"
    )
    (step_line,) = ax_profile.step(
        [], [], where="mid", color="tab:orange", lw=1.0, label="recorded DN"
    )
    ax_profile.set_xlabel("pixel (orange line, zoomed)")
    ax_profile.set_ylabel("digital number (DN)")
    ax_profile.legend(loc="upper right")

    labels = [
        "shot\nnoise",
        "dark\nnoise",
        "read\nnoise",
        "quantisation\nnoise",
        "total",
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "0.5"]
    bars = ax_budget.bar(labels, numpy.ones(5), color=colors)
    ax_budget.set_yscale("log")
    ax_budget.set_ylim(0.1, 3000)
    ax_budget.set_ylabel("std in electrons per pixel")
    ax_budget.tick_params(axis="x", labelsize=8)
    value_texts = [ax_budget.text(i, 1, "", ha="center", va="bottom") for i in range(5)]

    read_slider = Slider(
        fig.add_axes((0.15, 0.11, 0.7, 0.03)),
        "Read noise (e- rms)",
        0.0,
        60.0,
        valinit=read_noise,
    )
    bits_slider = Slider(
        fig.add_axes((0.15, 0.05, 0.7, 0.03)),
        "ADC bit depth",
        4,
        16,
        valinit=bits,
        valstep=1,
    )

    def update(_: float) -> None:
        """
        Redraw the figure after a slider moved.

        Args:
            _ (float): New value of the slider that moved, unused because every
                slider is read again.
        """
        read_noise = read_slider.val
        bits = int(bits_slider.val)
        gain = adc_gain(bits, SATURATION_ELECTRONS)
        digital = read_out(electrons, read_noise, bits, seed)

        image.set_data(digital)
        image.set_clim(0, 2**bits - 1)
        ax_image.set_title(
            f"{title}, {bits}-bit\nK = {gain:.4g} DN/e- ({1 / gain:,.1f} e- per DN)"
        )

        ideal = intensity[row, x] * DEFAULT_BACKGROUND_ELECTRONS * gain
        ideal_line.set_data(x, ideal)
        step_line.set_data(x, digital[row, x])
        ax_profile.set_xlim(x[0], x[-1])
        low = min(ideal.min(), digital[row, x].min())
        high = max(ideal.max(), digital[row, x].max())
        margin = max(1.0, 0.1 * (high - low))
        ax_profile.set_ylim(low - margin, high + margin)
        ax_profile.set_title(f"Line profile: one DN step = {1 / gain:,.1f} electrons")

        quantisation = 1 / (numpy.sqrt(12) * gain)
        values = [
            numpy.sqrt(DEFAULT_BACKGROUND_ELECTRONS),
            numpy.sqrt(dark_mean),
            read_noise,
            quantisation,
        ]
        values.append(float(numpy.sqrt(numpy.sum(numpy.square(values)))))
        for bar, text, value in zip(bars, value_texts, values):
            bar.set_height(max(value, 0.1))
            text.set_position((text.get_position()[0], max(value, 0.1) * 1.15))
            text.set_text(f"{value:,.1f}")
        ax_budget.set_title(
            f"Noise at the background ({DEFAULT_BACKGROUND_ELECTRONS:,.0f} e-)\n"
            f"dataset: {READ_NOISE_SIGMA:g} e- read noise, {DATASET_BITS} bits"
        )
        fig.canvas.draw_idle()

    read_slider.on_changed(update)
    bits_slider.on_changed(update)
    update(0.0)
    show_or_save(fig, save)

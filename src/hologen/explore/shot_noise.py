"""
Interactively record an image at different exposures and watch shot noise live.

The image is treated as an intensity map where 1 is the background illumination.
The exposure slider sets how many photo-electrons a pixel collects from that
background (background_electrons). Each pixel then records a Poisson-distributed
count, so the relative noise is 1 / sqrt(electrons): dim exposures are grainy,
bright ones are clean, and too bright ones saturate the sensor (shown in red).

All images are divided by the exposure, so they keep the same brightness and only
the noise changes.

Usage:
    hologen explore shot-noise                      # simulated hologram
    hologen explore shot-noise photo.png            # your own image
    hologen explore shot-noise --electrons 200
    hologen explore shot-noise --save shot.png      # save, no window
"""

import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from hologen.noises import add_shot_noise
from hologen.explore.common import (
    RECIPE,
    SATURATION_ELECTRONS,
    load_intensity,
    show_or_save,
    simulate_intensity,
)


def record(
    intensity: numpy.ndarray[float], electrons: float, seed: int
) -> tuple[numpy.ndarray[float], numpy.ndarray[bool]]:
    """
    Record the intensity at the given exposure.

    Args:
        intensity (numpy.ndarray[float]): Intensity with the background at 1.
        electrons (float): Photo-electrons per pixel at the background.
        seed (int): Random seed of the shot noise.

    Returns:
        tuple[numpy.ndarray[float], numpy.ndarray[bool]]: The recorded intensity
            normalised back so the background is 1, and the mask of clipped
            pixels.
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    counts = add_shot_noise(intensity * electrons)
    clipped = counts >= SATURATION_ELECTRONS
    counts = numpy.minimum(counts, SATURATION_ELECTRONS)
    return counts / electrons, clipped


def run(
    image: pathlib.Path | None = None,
    electrons: float = RECIPE.camera.background_electrons,
    pixel_size: float = RECIPE.geometry.pixel_size,
    seed: int = 0,
    save: pathlib.Path | None = None,
) -> None:
    """
    Open the explorer.

    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        electrons (float): Photo-electrons per pixel at the background.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise pattern.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    if image is None:
        intensity = simulate_intensity(pixel_size, seed)
        title = "Simulated hologram"
    else:
        # For a photo, treat its brightest value as the background illumination.
        intensity = load_intensity(image)
        intensity = intensity / intensity.max()
        title = image.name

    noisy, _ = record(intensity, electrons, seed)
    vmin, vmax = float(intensity.min()), float(intensity.max())

    # The profile goes through the row with the strongest fringes.
    row: int = int(numpy.argmax(intensity.std(axis=1)))

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.4, 1), bottom=0.17, hspace=0.3)
    ax_clean = fig.add_subplot(grid[0, 0])
    ax_noisy = fig.add_subplot(grid[0, 1])
    ax_noise = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[1, :])

    ax_clean.imshow(intensity, cmap="gray", vmin=vmin, vmax=vmax)
    ax_clean.axhline(row, color="tab:orange", lw=0.8)
    ax_clean.set_title(f"Clean: {title}\n(orange line = profile below)")
    noisy_image = ax_noisy.imshow(noisy, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = numpy.zeros((*intensity.shape, 4))
    overlay[..., 0] = 1.0  # red
    clipped_image = ax_noisy.imshow(overlay)
    noise_image = ax_noise.imshow(
        (noisy - intensity) * 100, cmap="RdBu_r", vmin=-15, vmax=15
    )
    ax_noise.set_title("Noise only: recorded - clean\n(% of background)")
    fig.colorbar(noise_image, ax=ax_noise, fraction=0.046)
    for ax in (ax_clean, ax_noisy, ax_noise):
        ax.axis("off")

    x = numpy.arange(intensity.shape[1])
    ax_profile.plot(x, intensity[row], color="black", lw=1.5, label="clean")
    (noisy_line,) = ax_profile.plot(
        x, noisy[row], color="tab:orange", lw=0.9, label="recorded"
    )
    ax_profile.set_xlim(0, x[-1])
    ax_profile.set_xlabel("pixel")
    ax_profile.set_ylabel("intensity (background = 1)")
    ax_profile.legend(loc="upper right")

    # Logarithmic slider: shot noise changes with sqrt(N), so equal steps in
    # log(N) give equal-looking changes in the noise.
    slider = Slider(
        fig.add_axes((0.15, 0.06, 0.7, 0.03)),
        "log10(e- / pixel)",
        1.0,
        numpy.log10(2 * SATURATION_ELECTRONS),
        valinit=numpy.log10(electrons),
    )

    def update(_: float) -> None:
        """
        Redraw the figure after a slider moved.

        Args:
            _ (float): New value of the slider that moved, unused because every
                slider is read again.
        """
        electrons = 10**slider.val
        result, clipped = record(intensity, electrons, seed)
        noisy_image.set_data(result)
        overlay[..., 3] = clipped * 0.9
        clipped_image.set_data(overlay)
        noise_image.set_data((result - intensity) * 100)
        noisy_line.set_ydata(result[row])
        ax_noisy.set_title(
            f"Recorded at {electrons:,.0f} e- per pixel\n"
            f"shot noise {100 / numpy.sqrt(electrons):.1f}% of background, "
            f"{clipped.mean() * 100:.1f}% pixels clipped (red)"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0.0)
    show_or_save(fig, save)

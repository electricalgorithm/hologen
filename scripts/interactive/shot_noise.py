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
    uv run scripts/interactive/shot_noise.py                   # simulated hologram
    uv run scripts/interactive/shot_noise.py photo.png         # your own image
    uv run scripts/interactive/shot_noise.py --electrons 200
    uv run scripts/interactive/shot_noise.py photo.png --output noisy.png --no-show
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

import hologen
from hologen.inline_simulator import DEFAULT_BACKGROUND_ELECTRONS
from hologen.noises import add_shot_noise
from common import SATURATION_ELECTRONS, load_intensity, simulate_intensity


def record(
    intensity: numpy.ndarray[float], electrons: float, seed: int
) -> tuple[numpy.ndarray[float], numpy.ndarray[bool]]:
    """
    Record the intensity at the given exposure.

    Returns:
        (recorded intensity normalised back to the background = 1, clipped mask)
    """
    hologen.RandomNumberGenerator.set_seed(seed)
    counts = add_shot_noise(intensity * electrons)
    clipped = counts >= SATURATION_ELECTRONS
    counts = numpy.minimum(counts, SATURATION_ELECTRONS)
    return counts / electrons, clipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("image", nargs="?", type=pathlib.Path, help="input image")
    parser.add_argument(
        "--electrons",
        type=float,
        default=DEFAULT_BACKGROUND_ELECTRONS,
        help="photo-electrons per pixel at the background (default: hologen's)",
    )
    parser.add_argument("--pixel-size", type=float, default=4.65, help="microns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, help="save the noisy image")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    if args.image is None:
        intensity = simulate_intensity(args.pixel_size, args.seed)
        title = "Simulated hologram"
    else:
        # For a photo, treat its brightest value as the background illumination.
        intensity = load_intensity(args.image)
        intensity = intensity / intensity.max()
        title = args.image.name

    noisy, _ = record(intensity, args.electrons, args.seed)
    vmin, vmax = float(intensity.min()), float(intensity.max())
    if args.output is not None:
        plt.imsave(args.output, noisy, cmap="gray", vmin=vmin, vmax=vmax)
        print(f"Saved {args.output}")
    if args.no_show:
        return

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
        valinit=numpy.log10(args.electrons),
    )

    def update(_: float) -> None:
        electrons = 10**slider.val
        result, clipped = record(intensity, electrons, args.seed)
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
    plt.show()


if __name__ == "__main__":
    main()

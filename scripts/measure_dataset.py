"""
Measure the dataset statistics that the paper reports, from the default recipe.

Every number is reproducible: holograms use fixed seeds (SEED + i) and the default
recipe of hologen.dataset.recipe. Run with:

    uv run scripts/measure_dataset.py [--holograms 50]

It reports, per noise configuration:

- noise: rms deviation of the recorded hologram from the noise-free hologram
  |ASM(O)|^2, both relative to the true background (K * background electrons), in %
- normalised background: what flat-field normalisation by the mean (as in
  holopaswin) or by the median gives at the true background, minus 1, in %;
  positive means the estimator underestimates the background
- brightest pixel: the largest digital number over all holograms, against the
  2^k - 1 full scale
"""

import argparse

import numpy

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.dataset.recipe import CONFIGURATION_NAMES, default_recipe
from hologen.noises.bit_depth import adc_gain
from hologen.objects import add_random_cells

SEED: int = 50000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--holograms", type=int, default=50, help="per configuration")
    args = parser.parse_args()

    recipe = default_recipe()
    config = recipe.hologram_config()
    gain = adc_gain(config.bit_depth, config.saturation_electrons)
    background_dn = gain * config.background_electrons
    full_scale = 2**config.bit_depth - 1

    def simulator(seed: int) -> InlineHologramSimulator:
        RandomNumberGenerator.set_seed(seed)
        sim = InlineHologramSimulator(
            config.resolution, config.pixel_size, config.wavelength
        )
        add_random_cells(sim, num_cells=config.num_cells)
        return sim

    print(
        f"{args.holograms} holograms per configuration, seeds {SEED}-"
        f"{SEED + args.holograms - 1}, true background {background_dn:.1f} DN\n"
    )
    print(
        "| configuration | noise (rms, % of background) | normalised background, "
        "mean (%) | normalised background, median (%) | brightest pixel (DN) |"
    )
    print("|---|---|---|---|---|")
    all_mean, all_median = [], []
    noises = recipe.noise_configurations()
    for name in CONFIGURATION_NAMES:
        noise = noises[name]
        rms, by_mean, by_median, brightest = [], [], [], 0.0
        for i in range(args.holograms):
            seed = SEED + i
            ideal = simulator(seed).generate_hologram(
                config.z_distance, background_electrons=1.0
            )
            recorded = simulator(seed).generate_hologram(
                config.z_distance,
                speckle_grain_size=noise.speckle_grain_size if noise.speckle else None,
                speckle_contrast=noise.speckle_contrast if noise.speckle else 0.0,
                shot_noise=noise.shot,
                dark_noise_mean=noise.dark_noise_mean if noise.dark else 0.0,
                read_noise_sigma=noise.read_noise_sigma if noise.read else 0.0,
                bit_depth=config.bit_depth,
                background_electrons=config.background_electrons,
                saturation_electrons=config.saturation_electrons,
            )
            rms.append(numpy.std(recorded / background_dn - ideal))
            by_mean.append(background_dn / recorded.mean() - 1)
            by_median.append(background_dn / numpy.median(recorded) - 1)
            brightest = max(brightest, float(recorded.max()))
        all_mean += by_mean
        all_median += by_median
        print(
            f"| {name} | {100 * numpy.mean(rms):.2f} | "
            f"{100 * numpy.mean(by_mean):+.2f} (s.d. {100 * numpy.std(by_mean):.2f}) | "
            f"{100 * numpy.mean(by_median):+.2f} (s.d. {100 * numpy.std(by_median):.2f}) | "
            f"{brightest:.0f} / {full_scale} |"
        )
    print(
        f"| all | | {100 * numpy.mean(all_mean):+.2f} "
        f"(s.d. {100 * numpy.std(all_mean):.2f}) | {100 * numpy.mean(all_median):+.2f} "
        f"(s.d. {100 * numpy.std(all_median):.2f}) | |"
    )


if __name__ == "__main__":
    main()

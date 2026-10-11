# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""
This script creates a dataset of inline holograms.

"""

from hologen import DatasetConfig, DatasetGenerator, HologramConfig, NoiseConfig

# Dataset Configurations
BATCH_SIZE = 250
NUM_SAMPLES = 3125  # 25000 / 8 = 3125

# Noise Configurations
# Speckle contrast (std/mean of the speckle intensity) of a laser-illuminated
# in-line hologram: 0.12 for a HeNe laser vs 0.014 for an LED (Hughes,
# arXiv:2006.09296, 2020; order-of-magnitude anchor measured through a fibre
# bundle). C = sigma / <I> as in Bianco et al., Light Sci. Appl. 7:48 (2018).
SPECKLE_CONTRAST = 0.12
# Mean grain size d_s = wavelength * z / D (Goodman, Speckle Phenomena in Optics,
# 2007) for scatterers at the object plane seen over the sensor field of view:
# 0.532 um * 20000 um / (224 * 4.65 um) = 10.2 um, about 2.2 pixels.
SPECKLE_GRAIN_SIZE = 10.2  # microns
# Photo-electrons per pixel from the background illumination (|U|^2 = 1). The
# modelled sensor is the Sony ICX267 CCD (4.65 um pixels), saturation capacity
# 11.0 ke- (Basler EMVA 1288 data overview, scout scA1390-17gm). With speckle the
# brightest fringe reaches 2.53x the background over 300 holograms, so 4000 e-
# keeps it below saturation including 3 sigma of shot noise (2.53 * 4000 + 300 =
# 10.4 ke-), i.e. the exposure is set as bright as possible without clipping.
BACKGROUND_ELECTRONS = 4000.0
READ_NOISE_SIGMA = 10.0
# Dark electrons per exposure = dark current (e-/s) * exposure time (s) (EMVA 1288
# Release 4.0 Linear, Sec. 3.1, Eq. 29). The ICX267 dark signal is at most 8 mV
# against a 380 mV saturation signal at 60 C over 1/15 s (Sony ICX267AL
# datasheet), i.e. 2.1% of the 11.0 ke- saturation capacity = 232 e- in 1/15 s =
# 3474 e-/s. The dark current doubles every T_d degrees (EMVA 1288 Sec. 3.2,
# Eq. 31); T_d = 9 C, the slowest commonly reported doubling for CCDs (6-9 C),
# is the worst case at 25 C: 3474 * 2^((25 - 60) / 9) = 234 e-/s.
DARK_CURRENT = 234.0  # e- per pixel per second, at 25 C (worst case)
# Laser in-line holographic microscopy exposure (Wdowiak et al., arXiv:2508.15419).
EXPOSURE_TIME = 0.0303  # seconds
DARK_NOISE_MEAN: float = DARK_CURRENT * EXPOSURE_TIME  # 7.1 e- per pixel per exposure


def main():
    # Create hologram configuration
    holo_config = HologramConfig(
        resolution=224,
        pixel_size=4.65,  # microns
        wavelength=0.532,  # microns
        z_distance=20000,  # microns
        bit_depth=12,
        num_cells=5,
        background_electrons=BACKGROUND_ELECTRONS,
    )

    # Create dataset configuration
    config = DatasetConfig(
        output_dir="dataset-224",
        base_seed=10000,
        hologram_config=holo_config,
    )

    # Create generator
    generator = DatasetGenerator(config, batch_size=BATCH_SIZE)
    generator.add_config(
        "no_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=False,
            dark=False,
        ),
    )
    generator.add_config(
        "speckle_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=False,
            read=False,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
        ),
    )
    generator.add_config(
        "shot_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=True,
            read=False,
            dark=False,
        ),
    )
    generator.add_config(
        "speckle_shot_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=False,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
        ),
    )
    generator.add_config(
        "read_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=True,
            dark=False,
            read_noise_sigma=READ_NOISE_SIGMA,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )
    generator.add_config(
        "dark_current_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=False,
            dark=True,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )
    generator.add_config(
        "speckle_shot_read_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=True,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
            read_noise_sigma=READ_NOISE_SIGMA,
        ),
    )
    generator.add_config(
        "speckle_shot_read_dark_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=True,
            dark=True,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
            read_noise_sigma=READ_NOISE_SIGMA,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )

    # Generate the dataset
    stats = generator.generate()

    print("Generation complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Configurations: {stats['configurations']}")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()

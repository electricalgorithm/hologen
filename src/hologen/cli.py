"""
Command-line interface of hologen.

    hologen generate [OPTIONS]          generate a dataset from the cited recipe
    hologen recipe [--output FILE]      write the default recipe as TOML
    hologen calibrate [OPTIONS]         calibrate the exposure of each object family
    hologen measure [OPTIONS]           measure noise, normalisation and clipping
    hologen explore <noise> [OPTIONS]   open an interactive noise explorer
"""

import dataclasses
import pathlib
from typing import Annotated

import typer

from hologen.dataset.measure import (
    ExposureCalibration,
    calibrate_families,
    calibration_table,
    exposures_are_current,
    groups_table,
    measure_groups,
    write_calibration,
)
from hologen.dataset.recipe import (
    CONFIGURATION_NAMES,
    DARK_CURRENT_DOUBLING,
    DARK_CURRENT_TEMPERATURE,
    OBJECT_FAMILY_NAMES,
    Geometry,
    Recipe,
    default_recipe,
    load_recipe,
    recipe_to_toml,
)

DEFAULTS: Recipe = default_recipe()

app = typer.Typer(
    help="Generate synthetic in-line holograms with a physical camera model.",
    no_args_is_help=True,
    add_completion=False,
)
explore_app = typer.Typer(
    help="Open an interactive explorer that shows one noise source live.",
    no_args_is_help=True,
)
app.add_typer(explore_app, name="explore")


def build_recipe(
    recipe_file: pathlib.Path | None,
    samples: int | None,
    seed: int | None,
    resolution: int | None,
    configs: list[str] | None,
    batch_size: int | None,
    objects: list[str] | None = None,
    z_error: float | None = None,
    z_offset: float | None = None,
) -> Recipe:
    """
    Read the recipe file, or the defaults, and apply the command-line flags.

    Args:
        recipe_file (pathlib.Path | None): TOML recipe; the default recipe if None.
        samples (int | None): Total number of holograms, or None to keep the recipe's.
        seed (int | None): Base random seed, or None to keep the recipe's.
        resolution (int | None): Hologram size in pixels, or None to keep the
            recipe's.
        configs (list[str] | None): Noise configurations, or None to keep the
            recipe's.
        batch_size (int | None): Holograms per parquet file, or None to keep the
            recipe's.
        objects (list[str] | None): Object families, or None to keep the recipe's.
        z_error (float | None): Half-width of the random recording-distance
            error in um, or None to keep the recipe's.
        z_offset (float | None): Fixed recording-distance offset in um, or None to
            keep the recipe's.

    Returns:
        Recipe: The recipe with the flags applied.

    Raises:
        ValueError: If the recipe file or a flag is invalid.
    """
    recipe: Recipe = load_recipe(recipe_file) if recipe_file else default_recipe()
    dataset_changes: dict[str, int | list[str]] = {
        key: value
        for key, value in {
            "total_samples": samples,
            "seed": seed,
            "configurations": configs,
            "objects": objects,
            "batch_size": batch_size,
        }.items()
        if value is not None
    }
    geometry: Geometry = recipe.geometry
    geometry = dataclasses.replace(
        geometry,
        resolution=geometry.resolution if resolution is None else resolution,
        z_error=geometry.z_error if z_error is None else z_error,
        z_offset=geometry.z_offset if z_offset is None else z_offset,
    )
    return dataclasses.replace(
        recipe,
        dataset=dataclasses.replace(recipe.dataset, **dataset_changes),
        geometry=geometry,
    )


@app.command()
def generate(
    output: Annotated[
        pathlib.Path,
        typer.Option("--output", "-o", help="Folder for the dataset files."),
    ] = pathlib.Path("dataset"),
    recipe: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="TOML recipe with every physical value (see `hologen recipe`). "
            "The options below override it.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    samples: Annotated[
        int | None,
        typer.Option(
            help="Total number of holograms, split equally over the groups (every "
            "configuration with every object family). "
            f"(recipe default: {DEFAULTS.dataset.total_samples})",
            min=1,
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Base random seed; hologram i uses seed + i. "
            f"(recipe default: {DEFAULTS.dataset.seed})",
            min=0,
        ),
    ] = None,
    resolution: Annotated[
        int | None,
        typer.Option(
            help="Hologram size in pixels (square). "
            f"(recipe default: {DEFAULTS.geometry.resolution})",
            min=8,
        ),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(
            "--config",
            "-c",
            help="Noise configuration to include; repeat for several. Choices: "
            + ", ".join(CONFIGURATION_NAMES)
            + ". (recipe default: all)",
        ),
    ] = None,
    objects: Annotated[
        list[str] | None,
        typer.Option(
            "--object",
            "-O",
            help="Object family to include; repeat for several. Choices: "
            + ", ".join(OBJECT_FAMILY_NAMES)
            + ". (recipe default: "
            + ", ".join(DEFAULTS.dataset.objects)
            + ")",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            help="Holograms per parquet file. "
            f"(recipe default: {DEFAULTS.dataset.batch_size})",
            min=1,
        ),
    ] = None,
    z_error: Annotated[
        float | None,
        typer.Option(
            help="Half-width in um of a uniform random error of the recording "
            "distance; holograms are still back-propagated at the nominal "
            f"distance. (recipe default: {DEFAULTS.geometry.z_error})",
            min=0.0,
        ),
    ] = None,
    z_offset: Annotated[
        float | None,
        typer.Option(
            help="Fixed shift in um of the recording distance from the nominal "
            f"one. (recipe default: {DEFAULTS.geometry.z_offset})",
        ),
    ] = None,
    visualization: Annotated[
        bool,
        typer.Option(help="Save an example figure of every configuration."),
    ] = True,
    reconstructed: Annotated[
        bool,
        typer.Option(
            help="Also store the field back-propagated at the nominal distance "
            "(reconstructed_XXXX.parquet). It can be recomputed from the hologram."
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Allow writing into a folder that already has a dataset."),
    ] = False,
) -> None:
    """Generate a hologram dataset and the recipe it was made from.

    The dataset follows a recipe. Without --recipe it is the default recipe with
    the cited physical values, and the options below change parts of it. Every
    noise configuration is combined with every object family, and the holograms
    are split equally over these groups.

    The output folder receives, per batch, hologram_XXXX.parquet (recorded
    intensity as 16-bit camera counts), ground_truth_XXXX.parquet (the complex
    object field as float32) and metadata_XXXX.parquet (seed, object family,
    objects placed, exposure, clipped pixels and distances of each hologram),
    each with one row per hologram. With --reconstructed it also receives
    reconstructed_XXXX.parquet, the field back-propagated at the nominal
    distance. It also receives recipe.toml, the exact recipe used, and
    dataset_examples.png unless --no-visualization is given.

    \f
    Args:
        output (pathlib.Path): Folder for the dataset files.
        recipe (pathlib.Path | None): TOML recipe; the default recipe if None.
        samples (int | None): Total number of holograms, overriding the recipe.
        seed (int | None): Base random seed, overriding the recipe.
        resolution (int | None): Hologram size in pixels, overriding the recipe.
        config (list[str] | None): Noise configurations, overriding the recipe.
        objects (list[str] | None): Object families, overriding the recipe.
        batch_size (int | None): Holograms per parquet file, overriding the recipe.
        z_error (float | None): Random recording-distance error in um, overriding
            the recipe.
        z_offset (float | None): Recording-distance offset in um, overriding the
            recipe.
        visualization (bool): Whether to save dataset_examples.png.
        reconstructed (bool): Whether to store the back-propagated field.
        overwrite (bool): Whether to write into a folder that has a dataset.

    Raises:
        typer.Exit: With code 2 if the recipe or an option is invalid, and with
            code 1 if the output folder already contains a dataset and
            overwrite is False.
    """
    try:
        chosen: Recipe = build_recipe(
            recipe,
            samples,
            seed,
            resolution,
            config,
            batch_size,
            objects,
            z_error,
            z_offset,
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error

    if any(output.glob("*.parquet")) and not overwrite:
        typer.echo(
            f"Error: {output} already contains a dataset. Choose another folder or "
            "pass --overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not exposures_are_current(chosen):
        typer.echo(
            "Warning: the exposures in this recipe were calibrated for different "
            "settings (geometry, speckle, saturation or objects changed). Run "
            "hologen calibrate to recalibrate them.",
            err=True,
        )
    # Imported here so that `hologen --help` stays fast.
    from hologen.dataset import DatasetConfig, DatasetGenerator

    generator = DatasetGenerator(
        DatasetConfig(
            output_dir=output,
            base_seed=chosen.dataset.seed,
            hologram_config=chosen.hologram_config(),
            generate_visualization=visualization,
            write_reconstructed=reconstructed,
        ),
        batch_size=chosen.dataset.batch_size,
    )
    counts: dict[str, int] = chosen.samples_per_group()
    for name, noise, family in chosen.groups():
        generator.add_config(
            name, num_samples=counts[name], noise=noise, objects=family
        )
    generator.generate()

    # Keep the exact recipe next to the data, so the dataset can be reproduced.
    output.mkdir(parents=True, exist_ok=True)
    (output / "recipe.toml").write_text(recipe_to_toml(chosen))
    typer.echo(f"Recipe saved to {output / 'recipe.toml'}")


@app.command("recipe")
def recipe_command(
    output: Annotated[
        pathlib.Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of printing."),
    ] = None,
) -> None:
    """Write the default recipe, with the unit and source of every value.

    \f
    Args:
        output (pathlib.Path | None): File to write to; printed if None.
    """
    text: str = recipe_to_toml(default_recipe())
    if output is None:
        typer.echo(text, nl=False)
        return
    output.write_text(text)
    typer.echo(f"Recipe written to {output}")


RecipeFile = Annotated[
    pathlib.Path | None,
    typer.Option(
        "--recipe",
        help="TOML recipe (see `hologen recipe`); the default recipe if omitted.",
        exists=True,
        dir_okay=False,
    ),
]


def load_or_default(recipe_file: pathlib.Path | None) -> Recipe:
    """
    Load a recipe file, or the default recipe, and turn errors into exit code 2.

    Args:
        recipe_file (pathlib.Path | None): Recipe file, or None for the default.

    Returns:
        Recipe: The loaded recipe.

    Raises:
        typer.Exit: With code 2 if the recipe file is invalid.
    """
    try:
        return load_recipe(recipe_file) if recipe_file else default_recipe()
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command()
def calibrate(
    recipe: RecipeFile = None,
    holograms: Annotated[
        int,
        typer.Option(help="Noise-free holograms per object family.", min=1),
    ] = 400,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed of the first hologram; keep it apart from datasets.", min=0
        ),
    ] = 90000,
    write: Annotated[
        bool,
        typer.Option(
            help="Write the exposures, their provenance and the fingerprint into "
            "the --recipe file."
        ),
    ] = False,
) -> None:
    """Calibrate the background exposure of every object family.

    For each object family, noise-free holograms with laser speckle are simulated
    and their brightest point is measured. The exposure is the largest background
    at which the recipe's exposure percentile of these holograms stays below
    saturation. The results are printed as a Markdown table.

    With --write, the exposures go into the recipe file together with their
    provenance and a fingerprint of the settings they were calibrated for. The
    default recipe keeps its exposures as cited constants in the code, so
    --write needs --recipe.

    \f
    Args:
        recipe (pathlib.Path | None): Recipe file; the default recipe if None.
        holograms (int): Noise-free holograms per object family.
        seed (int): Seed of the first hologram of each family.
        write (bool): Whether to write the results into the recipe file.

    Raises:
        typer.Exit: With code 2 if the recipe is invalid or --write is given
            without --recipe.
    """
    if write and recipe is None:
        typer.echo(
            "Error: --write needs --recipe. The default recipe's exposures are cited "
            "constants in hologen/dataset/recipe.py; update them there.",
            err=True,
        )
        raise typer.Exit(code=2)
    chosen: Recipe = load_or_default(recipe)
    typer.echo(
        f"Exposure calibration: {holograms} noise-free laser-speckle holograms per "
        f"family, seeds {seed}-{seed + holograms - 1}, {chosen.geometry.resolution} "
        f"px, percentile {chosen.objects.exposure_percentile:g}, saturation "
        f"{chosen.camera.saturation_electrons:,.0f} e-\n"
    )
    calibrations: list[ExposureCalibration] = calibrate_families(
        chosen, holograms, seed
    )
    typer.echo(calibration_table(calibrations), nl=False)
    if write:
        write_calibration(recipe, calibrations)
        typer.echo(f"\nExposures written to {recipe}")


@app.command()
def measure(
    recipe: RecipeFile = None,
    holograms: Annotated[int, typer.Option(help="Holograms per group.", min=1)] = 50,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed of the first hologram; keep it apart from datasets.", min=0
        ),
    ] = 50000,
    output: Annotated[
        pathlib.Path | None,
        typer.Option("--output", "-o", help="Also save the table to this file."),
    ] = None,
) -> None:
    """Measure noise, normalisation and clipping of every group of a recipe.

    Every group (noise configuration x object family) is simulated on the same
    seeds and recorded with the exposure of its object family. The Markdown
    table reports the noise relative to the background, how far normalising by
    the mean or the median moves the background, and how often holograms clip.
    Calibrate the exposures first, because the results depend on them.

    \f
    Args:
        recipe (pathlib.Path | None): Recipe file; the default recipe if None.
        holograms (int): Holograms per group.
        seed (int): Seed of the first hologram of each group.
        output (pathlib.Path | None): File to save the table to.

    Raises:
        typer.Exit: With code 2 if the recipe is invalid.
    """
    chosen: Recipe = load_or_default(recipe)
    if not exposures_are_current(chosen):
        typer.echo(
            "Warning: the exposures were calibrated for different settings; run "
            "hologen calibrate first.",
            err=True,
        )
    header: str = (
        f"{holograms} holograms per group, seeds {seed}-{seed + holograms - 1}\n\n"
    )
    table: str = groups_table(measure_groups(chosen, holograms, seed))
    typer.echo(header + table, nl=False)
    if output is not None:
        output.write_text(header + table)
        typer.echo(f"\nTable saved to {output}")


def positive(value: float) -> float:
    """
    Accept only values above zero, for options whose zero has no meaning.

    Args:
        value (float): The option value.

    Returns:
        float: The value unchanged.

    Raises:
        typer.BadParameter: If the value is zero or negative.
    """
    if value <= 0:
        raise typer.BadParameter(f"must be positive (got {value}).")
    return value


Image = Annotated[
    pathlib.Path | None,
    typer.Argument(
        help="Image to use as the intensity; a simulated hologram if omitted.",
        exists=True,
        dir_okay=False,
    ),
]
PixelSize = Annotated[float, typer.Option(help="Pixel size in microns.")]
Seed = Annotated[int, typer.Option(help="Random seed; same seed, same noise.", min=0)]
Save = Annotated[
    pathlib.Path | None,
    typer.Option(help="Save the figure to this file instead of opening a window."),
]


@explore_app.command()
def speckle(
    image: Image = None,
    contrast: Annotated[
        float,
        typer.Option(
            help="Speckle contrast std/mean, in (0, 1].",
            max=1.0,
            callback=positive,
        ),
    ] = DEFAULTS.speckle.laser_contrast,
    grain_size: Annotated[
        float,
        typer.Option(
            help="Mean speckle grain size in microns.",
            callback=positive,
        ),
    ] = DEFAULTS.speckle.grain_size,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Speckle: contrast and grain size of the light source.

    \f
    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        contrast (float): Speckle contrast std/mean, in (0, 1].
        grain_size (float): Mean speckle grain size, in um.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    from hologen.explore import speckle as explorer

    explorer.run(image, contrast, grain_size, pixel_size, seed, save)


@explore_app.command("shot-noise")
def shot_noise(
    image: Image = None,
    electrons: Annotated[
        float, typer.Option(help="Photo-electrons per pixel at the background.")
    ] = DEFAULTS.camera.background_electrons,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Shot noise: exposure in electrons and sensor saturation.

    \f
    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        electrons (float): Photo-electrons per pixel at the background.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    from hologen.explore import shot_noise as explorer

    explorer.run(image, electrons, pixel_size, seed, save)


@explore_app.command("dark-current")
def dark_current(
    image: Image = None,
    temperature: Annotated[
        float, typer.Option(help="Sensor temperature in C.")
    ] = DARK_CURRENT_TEMPERATURE,
    exposure: Annotated[
        float, typer.Option(help="Exposure time in seconds.")
    ] = DEFAULTS.camera.exposure_time,
    doubling: Annotated[
        float, typer.Option(help="Temperature step in C that doubles the dark current.")
    ] = DARK_CURRENT_DOUBLING,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Dark current: sensor temperature and exposure time.

    \f
    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        temperature (float): Sensor temperature, in C.
        exposure (float): Exposure time, in s.
        doubling (float): Temperature step that doubles the dark current, in C.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    from hologen.explore import dark_current as explorer

    explorer.run(image, temperature, exposure, doubling, pixel_size, seed, save)


@explore_app.command()
def readout(
    image: Image = None,
    read_noise: Annotated[
        float, typer.Option(help="Read noise in electrons rms.")
    ] = DEFAULTS.camera.read_noise,
    bits: Annotated[
        int, typer.Option(help="ADC bit depth.", min=1, max=16)
    ] = DEFAULTS.camera.bit_depth,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Readout: read noise and ADC bit depth.

    \f
    Args:
        image (pathlib.Path | None): Image to use as the intensity; a simulated
            hologram if None.
        read_noise (float): Read noise, in e- rms.
        bits (int): ADC bit depth.
        pixel_size (float): Pixel size, in um.
        seed (int): Random seed; the same seed gives the same noise.
        save (pathlib.Path | None): File to save the figure to instead of opening
            a window.
    """
    from hologen.explore import readout as explorer

    explorer.run(image, read_noise, bits, pixel_size, seed, save)

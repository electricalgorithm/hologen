"""
Command-line interface of hologen.

    hologen generate [OPTIONS]          generate a dataset from the cited recipe
    hologen recipe [--output FILE]      write the default recipe as TOML
    hologen explore <noise> [OPTIONS]   open an interactive noise explorer
"""

import dataclasses
import pathlib
from typing import Annotated

import typer

from hologen.dataset.recipe import (
    CONFIGURATION_NAMES,
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
) -> Recipe:
    """Read the recipe file (or the defaults) and apply the command-line flags."""
    recipe = load_recipe(recipe_file) if recipe_file else default_recipe()
    dataset_changes = {
        key: value
        for key, value in {
            "total_samples": samples,
            "seed": seed,
            "configurations": configs,
            "batch_size": batch_size,
        }.items()
        if value is not None
    }
    geometry_changes = {} if resolution is None else {"resolution": resolution}
    return dataclasses.replace(
        recipe,
        dataset=dataclasses.replace(recipe.dataset, **dataset_changes),
        geometry=dataclasses.replace(recipe.geometry, **geometry_changes),
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
            help="Total number of holograms, split equally over the configurations. "
            f"(recipe default: {DEFAULTS.dataset.total_samples})",
            min=1,
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Base random seed; hologram i uses seed + i. "
            f"(recipe default: {DEFAULTS.dataset.seed})"
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
    batch_size: Annotated[
        int | None,
        typer.Option(
            help="Holograms per parquet file. "
            f"(recipe default: {DEFAULTS.dataset.batch_size})",
            min=1,
        ),
    ] = None,
    visualization: Annotated[
        bool,
        typer.Option(help="Save an example figure of every configuration."),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(help="Allow writing into a folder that already has a dataset."),
    ] = False,
) -> None:
    """Generate a hologram dataset and the recipe it was made from."""
    try:
        chosen = build_recipe(recipe, samples, seed, resolution, config, batch_size)
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

    # Imported here so that `hologen --help` stays fast.
    from hologen.dataset import DatasetConfig, DatasetGenerator

    generator = DatasetGenerator(
        DatasetConfig(
            output_dir=output,
            base_seed=chosen.dataset.seed,
            hologram_config=chosen.hologram_config(),
            generate_visualization=visualization,
        ),
        batch_size=chosen.dataset.batch_size,
    )
    counts = chosen.samples_per_configuration()
    for name, noise in chosen.noise_configurations().items():
        generator.add_config(name, num_samples=counts[name], noise=noise)
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
    """Write the default recipe, with the unit and source of every value."""
    text = recipe_to_toml(default_recipe())
    if output is None:
        typer.echo(text, nl=False)
        return
    output.write_text(text)
    typer.echo(f"Recipe written to {output}")


Image = Annotated[
    pathlib.Path | None,
    typer.Argument(
        help="Image to use as the intensity; a simulated hologram if omitted.",
        exists=True,
        dir_okay=False,
    ),
]
PixelSize = Annotated[float, typer.Option(help="Pixel size in microns.")]
Seed = Annotated[int, typer.Option(help="Random seed; same seed, same noise.")]
Save = Annotated[
    pathlib.Path | None,
    typer.Option(help="Save the figure to this file instead of opening a window."),
]


@explore_app.command()
def speckle(
    image: Image = None,
    contrast: Annotated[
        float, typer.Option(help="Speckle contrast std/mean, in (0, 1].")
    ] = DEFAULTS.speckle.laser_contrast,
    grain_size: Annotated[
        float, typer.Option(help="Mean speckle grain size in microns.")
    ] = DEFAULTS.speckle.grain_size,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Speckle: contrast and grain size of the light source."""
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
    """Shot noise: exposure in electrons and sensor saturation."""
    from hologen.explore import shot_noise as explorer

    explorer.run(image, electrons, pixel_size, seed, save)


@explore_app.command("dark-current")
def dark_current(
    image: Image = None,
    temperature: Annotated[float, typer.Option(help="Sensor temperature in C.")] = 25.0,
    exposure: Annotated[
        float, typer.Option(help="Exposure time in seconds.")
    ] = DEFAULTS.camera.exposure_time,
    doubling: Annotated[
        float, typer.Option(help="Temperature step in C that doubles the dark current.")
    ] = 9.0,
    pixel_size: PixelSize = DEFAULTS.geometry.pixel_size,
    seed: Seed = 0,
    save: Save = None,
) -> None:
    """Dark current: sensor temperature and exposure time."""
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
    """Readout: read noise and ADC bit depth."""
    from hologen.explore import readout as explorer

    explorer.run(image, read_noise, bits, pixel_size, seed, save)

"""End-to-end tests of the hologen command-line interface."""

import pathlib
import tomllib

import matplotlib
import matplotlib.image
import numpy
import pandas
import pytest
from typer.testing import CliRunner, Result

from hologen.cli import app
from hologen.dataset.measure import exposures_are_current
from hologen.dataset.recipe import (
    default_recipe,
    load_recipe,
    recipe_from_dict,
    recipe_to_toml,
)
from hologen.dataset.storage import load_samples

matplotlib.use("Agg")
runner = CliRunner()

# A dataset small enough for a test: 8 holograms of 32 x 32 pixels.
TINY = ["--samples", "8", "--resolution", "32", "--no-visualization"]


def test_recipe_prints_the_default_recipe() -> None:
    result = runner.invoke(app, ["recipe"])
    assert result.exit_code == 0
    assert recipe_from_dict(tomllib.loads(result.stdout)) == default_recipe()


def test_recipe_writes_a_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "paper.toml"
    result = runner.invoke(app, ["recipe", "--output", str(path)])
    assert result.exit_code == 0
    assert recipe_from_dict(tomllib.loads(path.read_text())) == default_recipe()


def test_generate_writes_the_dataset_and_its_recipe(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "data"
    result = runner.invoke(app, ["generate", "--output", str(output), *TINY])
    assert result.exit_code == 0, result.stdout

    holograms = load_samples(output, "hologram")
    assert len(holograms) == 8
    saved = tomllib.loads((output / "recipe.toml").read_text())
    assert saved["dataset"]["total_samples"] == 8
    assert saved["geometry"]["resolution"] == 32


def test_generate_uses_only_the_chosen_configurations(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "data"
    args = [
        "generate",
        "-o",
        str(output),
        *TINY,
        "-c",
        "ideal",
        "-c",
        "camera_laser_speckle",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout

    names: pandas.DataFrame = pandas.read_parquet(
        output / "metadata_0000.parquet", columns=["config_name"]
    )
    assert sorted(names["config_name"].unique()) == [
        "camera_laser_speckle/adherent_cells",
        "camera_laser_speckle/red_blood_cells",
        "ideal/adherent_cells",
        "ideal/red_blood_cells",
    ]


def test_generate_uses_only_the_chosen_object_families(
    tmp_path: pathlib.Path,
) -> None:
    output: pathlib.Path = tmp_path / "data"
    args: list[str] = [
        "generate",
        "-o",
        str(output),
        *TINY,
        "-c",
        "camera",
        "-O",
        "ellipses",
    ]
    result: Result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    metadata: pandas.DataFrame = pandas.read_parquet(output / "metadata_0000.parquet")
    assert set(metadata["object_family"]) == {"ellipses"}
    assert len(metadata) == 8


def test_unknown_object_family_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    result: Result = runner.invoke(
        app, ["generate", "-o", str(tmp_path / "d"), *TINY, "-O", "neurons"]
    )
    assert result.exit_code == 2
    assert "Unknown object" in result.output


def test_flags_override_the_recipe_file(tmp_path: pathlib.Path) -> None:
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        "[dataset]\ntotal_samples = 100\nseed = 5\n\n[camera]\nread_noise = 20.0\n"
    )
    output = tmp_path / "data"
    result = runner.invoke(
        app, ["generate", "-o", str(output), "--recipe", str(recipe), *TINY]
    )
    assert result.exit_code == 0, result.stdout

    saved = tomllib.loads((output / "recipe.toml").read_text())
    assert saved["dataset"]["total_samples"] == 8  # flag wins over the file
    assert saved["dataset"]["seed"] == 5  # from the file
    assert saved["camera"]["read_noise"] == 20.0  # from the file


def test_unknown_configuration_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    result = runner.invoke(
        app, ["generate", "-o", str(tmp_path / "d"), *TINY, "-c", "laser"]
    )
    assert result.exit_code == 2
    assert "Unknown configuration" in result.output


def test_existing_dataset_is_not_overwritten_by_accident(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "data"
    assert runner.invoke(app, ["generate", "-o", str(output), *TINY]).exit_code == 0

    again = runner.invoke(app, ["generate", "-o", str(output), *TINY])
    assert again.exit_code == 1
    assert "--overwrite" in again.output
    assert (
        runner.invoke(
            app, ["generate", "-o", str(output), *TINY, "--overwrite"]
        ).exit_code
        == 0
    )


# A recipe small enough to calibrate in a test: red blood cells on 64 x 64 pixels.
SMALL: dict[str, dict[str, object]] = {
    "geometry": {"resolution": 64},
    "dataset": {"objects": ["red_blood_cells"], "configurations": ["ideal", "camera"]},
}


def small_recipe_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Write the small recipe to a file.

    Args:
        tmp_path (pathlib.Path): Folder for the file.

    Returns:
        pathlib.Path: The recipe file.
    """
    path: pathlib.Path = tmp_path / "small.toml"
    path.write_text(recipe_to_toml(recipe_from_dict(SMALL)))
    return path


def test_calibrate_prints_a_table_and_leaves_the_file_alone(
    tmp_path: pathlib.Path,
) -> None:
    path: pathlib.Path = small_recipe_file(tmp_path)
    before: str = path.read_text()
    result: Result = runner.invoke(
        app, ["calibrate", "--recipe", str(path), "--holograms", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "| red_blood_cells |" in result.stdout
    assert "seeds 90000-90004, 64 px" in result.stdout
    assert path.read_text() == before


def test_calibrate_write_makes_the_exposures_current(tmp_path: pathlib.Path) -> None:
    path: pathlib.Path = small_recipe_file(tmp_path)
    assert not exposures_are_current(load_recipe(path))

    result: Result = runner.invoke(
        app, ["calibrate", "--recipe", str(path), "--holograms", "5", "--write"]
    )
    assert result.exit_code == 0, result.output
    assert exposures_are_current(load_recipe(path))

    output: pathlib.Path = tmp_path / "data"
    generated: Result = runner.invoke(
        app,
        [
            "generate",
            "-o",
            str(output),
            "--recipe",
            str(path),
            "--samples",
            "2",
            "--no-visualization",
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert "Warning" not in generated.output


def test_calibrate_write_needs_a_recipe_file() -> None:
    result: Result = runner.invoke(app, ["calibrate", "--write"])
    assert result.exit_code == 2
    assert "--write needs --recipe" in result.output


def test_generate_warns_about_stale_exposures(tmp_path: pathlib.Path) -> None:
    result: Result = runner.invoke(app, ["generate", "-o", str(tmp_path / "d"), *TINY])
    assert result.exit_code == 0, result.output
    assert "hologen calibrate" in result.output  # TINY changes the resolution


def test_measure_prints_and_saves_the_table(tmp_path: pathlib.Path) -> None:
    path: pathlib.Path = small_recipe_file(tmp_path)
    table: pathlib.Path = tmp_path / "groups.md"
    result: Result = runner.invoke(
        app,
        ["measure", "--recipe", str(path), "--holograms", "2", "-o", str(table)],
    )
    assert result.exit_code == 0, result.output
    assert "run hologen calibrate first" in result.output
    assert "| ideal/red_blood_cells |" in result.stdout
    assert "| camera/red_blood_cells |" in table.read_text()


def test_invalid_recipe_file_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    path: pathlib.Path = tmp_path / "bad.toml"
    path.write_text("[camera]\nunknown_key = 1\n")
    for command in ("calibrate", "measure"):
        result: Result = runner.invoke(app, [command, "--recipe", str(path)])
        assert result.exit_code == 2
        assert "Error" in result.output


@pytest.mark.parametrize(
    "content", ["[camera]\nbit_depth = 40\n", '[dataset]\ntotal_samples = "x"\n']
)
def test_bad_recipe_values_stop_generate_before_it_writes(
    content: str, tmp_path: pathlib.Path
) -> None:
    path: pathlib.Path = tmp_path / "bad.toml"
    path.write_text(content)
    output: pathlib.Path = tmp_path / "data"
    result: Result = runner.invoke(
        app, ["generate", "-o", str(output), "--recipe", str(path), *TINY]
    )
    assert result.exit_code == 2
    assert "Error" in result.output
    assert not any(output.glob("*.parquet"))


def test_negative_seed_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    result: Result = runner.invoke(
        app, ["generate", "-o", str(tmp_path / "d"), *TINY, "--seed", "-1"]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "d").exists()


def test_explorers_take_an_image(tmp_path: pathlib.Path) -> None:
    """A wide RGBA image is cropped to its centre square and turned to grey."""
    image: pathlib.Path = tmp_path / "photo.png"
    pixels: numpy.ndarray = numpy.random.default_rng(0).random((40, 60, 4))
    matplotlib.image.imsave(image, pixels)
    figure: pathlib.Path = tmp_path / "speckle.png"
    result: Result = runner.invoke(
        app, ["explore", "speckle", str(image), "--save", str(figure)]
    )
    assert result.exit_code == 0, result.output
    assert figure.stat().st_size > 0


@pytest.mark.parametrize(
    "explorer", ["speckle", "shot-noise", "dark-current", "readout"]
)
def test_explorers_save_their_figure(explorer: str, tmp_path: pathlib.Path) -> None:
    figure = tmp_path / f"{explorer}.png"
    result = runner.invoke(app, ["explore", explorer, "--save", str(figure)])
    assert result.exit_code == 0, result.output
    assert figure.stat().st_size > 10_000


def test_generate_records_at_the_distance_error_and_offset(
    tmp_path: pathlib.Path,
) -> None:
    """
    The distance flags reach the recipe and the metadata.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    output: pathlib.Path = tmp_path / "data"
    result: Result = runner.invoke(
        app,
        ["generate", "-o", str(output), *TINY, "--z-error", "6", "--z-offset", "-2"],
    )
    assert result.exit_code == 0, result.stdout

    saved: dict = tomllib.loads((output / "recipe.toml").read_text())
    assert saved["geometry"]["z_error"] == 6.0
    assert saved["geometry"]["z_offset"] == -2.0
    metadata: pandas.DataFrame = pandas.read_parquet(output / "metadata_0000.parquet")
    assert metadata["z_true"].between(12.0, 24.0).all()
    assert (metadata["z_nominal"] == 20.0).all()


def test_impossible_distance_flags_are_a_clear_error(tmp_path: pathlib.Path) -> None:
    """
    A flag that puts the recording plane before the object is rejected.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    result: Result = runner.invoke(
        app, ["generate", "-o", str(tmp_path / "data"), *TINY, "--z-offset", "-25"]
    )
    assert result.exit_code == 2
    assert "must be positive" in result.output


def test_generate_stores_the_reconstruction_only_on_request(
    tmp_path: pathlib.Path,
) -> None:
    """
    The back-propagated field is written with --reconstructed and not otherwise.

    Args:
        tmp_path (pathlib.Path): Folder for the datasets.
    """
    plain: Result = runner.invoke(app, ["generate", "-o", str(tmp_path / "a"), *TINY])
    with_field: Result = runner.invoke(
        app, ["generate", "-o", str(tmp_path / "b"), *TINY, "--reconstructed"]
    )
    assert plain.exit_code == 0 and with_field.exit_code == 0
    assert not list((tmp_path / "a").glob("reconstructed_*.parquet"))
    assert len(load_samples(tmp_path / "b", "reconstructed")) == 8  # noqa: PLR2004

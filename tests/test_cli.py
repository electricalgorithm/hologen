"""
End-to-end tests of the hologen command-line interface.
"""

import pathlib
import tomllib

import matplotlib
import pytest
from typer.testing import CliRunner

from hologen.cli import app
from hologen.dataset.recipe import default_recipe, recipe_from_dict
from hologen.utils import load_samples_from_parquet

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

    holograms = load_samples_from_parquet(output, "hologram", resolution=32)
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

    import pandas as pd

    names = pd.read_parquet(output / "hologram_0000.parquet", columns=["config_name"])
    assert sorted(names["config_name"].unique()) == ["camera_laser_speckle", "ideal"]


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


@pytest.mark.parametrize(
    "explorer", ["speckle", "shot-noise", "dark-current", "readout"]
)
def test_explorers_save_their_figure(explorer: str, tmp_path: pathlib.Path) -> None:
    figure = tmp_path / f"{explorer}.png"
    result = runner.invoke(app, ["explore", explorer, "--save", str(figure)])
    assert result.exit_code == 0, result.output
    assert figure.stat().st_size > 10_000

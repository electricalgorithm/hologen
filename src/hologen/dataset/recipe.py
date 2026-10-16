"""
The dataset recipe, which holds every physical value used to generate a dataset.

Each value carries its unit and source, and the noise configurations and object
families are built from them. A recipe can be written to and read from a TOML file, so a dataset is reproducible
from one versioned file. Values missing from a file keep their defaults.
"""

import dataclasses
import pathlib
import tomllib
from typing import Any

from hologen.dataset.configs import HologramConfig, NoiseConfig
from hologen.objects.adherent_cells import add_adherent_cells
from hologen.objects.families import ObjectFamily
from hologen.objects.random_cells import add_random_cells
from hologen.objects.red_blood_cells import add_red_blood_cells
from hologen.inline_simulator import (
    DEFAULT_BACKGROUND_ELECTRONS,
    DEFAULT_SATURATION_ELECTRONS,
    InlineHologramSimulator,
)

# Noise configurations, from the noise-free reference to the noisiest recording.
CONFIGURATION_NAMES: tuple[str, ...] = (
    "ideal",
    "camera",
    "camera_led_speckle",
    "camera_laser_speckle",
)

# Kinds of sample. "ellipses" are the first-version objects, kept as a baseline.
OBJECT_FAMILY_NAMES: tuple[str, ...] = (
    "red_blood_cells",
    "adherent_cells",
    "ellipses",
)


def _value(default: Any, unit: str, source: str) -> Any:
    """
    Declare a recipe value together with its unit and source.

    Args:
        default (Any): Default value of the field.
        unit (str): Unit of the value, empty if it has none.
        source (str): Citation or reason for the default.

    Returns:
        Any: A dataclass field that keeps the unit and source as metadata.
    """
    return dataclasses.field(default=default, metadata={"unit": unit, "source": source})


@dataclasses.dataclass
class Geometry:
    """
    Geometry of the magnified in-line (Gabor) holographic microscope.

    The setup is that of Wdowiak et al., arXiv:2508.15419 (2025), where a
    defocused detector sits behind a microscope objective. Distances and pixel
    sizes are given in the object plane.
    """

    resolution: int = _value(
        224, "pixels", "Image size; 54 um field at 0.24 um per pixel"
    )
    sensor_pixel_size: float = _value(
        2.4,
        "um",
        "Sony IMX183 pixel pitch; Wdowiak et al. 2025 report 2.4 um pixels "
        "(5496 x 3672, model not named)",
    )
    magnification: float = _value(
        10.0,
        "",
        "10x objective with a 200 mm tube lens, Wdowiak et al. 2025 (assumes an "
        "objective designed for a 200 mm tube)",
    )
    numerical_aperture: float = _value(
        0.45, "", "Collecting objective NA, Wdowiak et al. 2025"
    )
    wavelength: float = _value(
        0.532, "um", "Single-mode DPSS laser, Wdowiak et al. 2025"
    )
    z_distance: float = _value(
        20.0,
        "um",
        "Object-space defocus: 2 mm detector defocus in Wdowiak et al. 2025 "
        "divided by the axial magnification M^2 = 100",
    )
    num_cells: int = _value(5, "objects per hologram", "Design choice")

    @property
    def pixel_size(self) -> float:
        """
        Pixel size in the object plane.

        Returns:
            float: The sensor pixel size divided by the magnification, in um.
        """
        return self.sensor_pixel_size / self.magnification


@dataclasses.dataclass
class Camera:
    """
    Camera model of the Sony IMX183 CMOS.

    The model follows EMVA 1288 Release 4.0 Linear (2021), Sec. 2.4 Eq. 14, with
    shot noise, dark current, read noise and a k-bit ADC.
    """

    background_electrons: float = _value(
        DEFAULT_BACKGROUND_ELECTRONS,
        "e- per pixel",
        "Brightest exposure without clipping: brightest fringe plus 3 sigma of "
        "shot noise below the saturation capacity, for the first-version "
        "ellipses; the cell families have their own calibrated exposures",
    )
    saturation_electrons: float = _value(
        DEFAULT_SATURATION_ELECTRONS,
        "e- per pixel",
        "IMX183, FLIR BFS-U3-200S6M EMVA 1288 results (20 C); 14,775 e- for "
        "LUCID Triton 20 MP",
    )
    bit_depth: int = _value(
        12, "bits", "IMX183 12-bit ADC (FLIR BFS-U3-200S6M, LUCID Triton 20 MP)"
    )
    read_noise: float = _value(
        3.30,
        "e- rms",
        "IMX183 temporal dark noise, FLIR BFS-U3-200S6M EMVA 1288 results; 3.12 "
        "e- for LUCID Triton 20 MP",
    )
    dark_current: float = _value(
        0.81,
        "e- per pixel per second",
        "IMX183, LUCID Triton 20 MP EMVA 1288 results (conditions not stated)",
    )
    exposure_time: float = _value(
        0.0303,
        "s",
        "Laser in-line holographic microscopy, Wdowiak et al., arXiv:2508.15419",
    )

    @property
    def dark_electrons(self) -> float:
        """
        Mean dark electrons per pixel in one exposure (EMVA 1288 Eq. 29).

        Returns:
            float: Dark current times exposure time, in e- per pixel.
        """
        return self.dark_current * self.exposure_time


@dataclasses.dataclass
class Speckle:
    """Speckle of the light source, Goodman, Speckle Phenomena in Optics (2007)."""

    grain_size: float = _value(
        0.59,
        "um",
        "Imaging speckle set by the objective, d_s = wavelength / (2 NA) = "
        "0.532 / 0.9 um (Goodman 2007, Sec. 4.4); 2.5 pixels",
    )
    led_contrast: float = _value(
        0.014,
        "std/mean",
        "LED illumination, Hughes, arXiv:2006.09296 (2020)",
    )
    laser_contrast: float = _value(
        0.12,
        "std/mean",
        "HeNe laser illumination, Hughes, arXiv:2006.09296 (2020); C = sigma/<I> "
        "as in Bianco et al., Light Sci. Appl. 7:48 (2018)",
    )


@dataclasses.dataclass
class Objects:
    """
    Sample content and the exposure used to record it, per object family.

    Each family has its own exposure, the way an operator sets one per type of
    sample. It is the largest background for which `exposure_percentile`
    percent of the family's holograms stay unsaturated (hologen.dataset.exposure).
    Run `hologen calibrate` whenever the geometry, speckle, saturation capacity
    or object settings change. For a recipe file, `hologen calibrate --write`
    updates the exposures, their provenance and the fingerprint below.
    """

    red_blood_cell_density: float = _value(
        0.00415,
        "cells per um^2",
        "Design choice: 12 cells per 54 x 54 um field, a sparse monolayer",
    )
    adherent_coverage: float = _value(
        0.263,
        "fraction of the field",
        "Measured median field coverage of 7476 adherent cells "
        "(scripts/calibrate_adherent_cells.py, Vicar et al. data)",
    )
    exposure_percentile: float = _value(
        95.0,
        "percent",
        "Percent of holograms that must not saturate; the rest clip at the foci "
        "of the most strongly refracting cells",
    )
    red_blood_cells_background_electrons: float = _value(
        2982.0,
        "e- per pixel",
        "Calibrated with hologen calibrate (see exposure_calibration)",
    )
    adherent_cells_background_electrons: float = _value(
        2593.0,
        "e- per pixel",
        "Calibrated with hologen calibrate (see exposure_calibration)",
    )
    exposure_calibration: str = _value(
        "red_blood_cells p95 peak 4.85x (median 4.26x, max 5.30x); adherent_cells "
        "p95 peak 5.58x (median 2.52x, max 10.92x); 400 holograms per family, "
        "seeds 90000-90399, 224 px",
        "",
        "Provenance of the calibrated exposures, written by hologen calibrate; "
        "recalibrate after the image-size pilot",
    )
    calibration_fingerprint: str = _value(
        "7922139bad49",
        "",
        "Fingerprint of the settings the exposures were calibrated for; hologen "
        "generate warns when it no longer matches the recipe",
    )


@dataclasses.dataclass
class Dataset:
    """Size, randomness and noise configurations of the generated dataset."""

    total_samples: int = _value(
        25000, "holograms", "Split equally over the configurations"
    )
    seed: int = _value(10000, "", "Base seed; hologram i uses seed + i")
    batch_size: int = _value(250, "holograms per file", "Parquet batch size")
    configurations: list[str] = dataclasses.field(
        default_factory=lambda: list(CONFIGURATION_NAMES),
        metadata={
            "unit": "",
            "source": "Any of: " + ", ".join(CONFIGURATION_NAMES),
        },
    )
    objects: list[str] = dataclasses.field(
        default_factory=lambda: ["red_blood_cells", "adherent_cells"],
        metadata={
            "unit": "",
            "source": "Object families, any of: " + ", ".join(OBJECT_FAMILY_NAMES),
        },
    )

    def __post_init__(self) -> None:
        """
        Check the configurations, object families and dataset size.

        Raises:
            ValueError: If a configuration or object family is unknown, missing
                or repeated, or if there are fewer samples than groups.
        """
        # Check if any unknown configuration type is given.
        unknown: list[str] = [
            c for c in self.configurations if c not in CONFIGURATION_NAMES
        ]
        if unknown:
            raise ValueError(
                f"Unknown configuration(s) {unknown}; choose from "
                f"{', '.join(CONFIGURATION_NAMES)}."
            )
        if not self.configurations:
            raise ValueError("At least one configuration is required.")
        if len(set(self.configurations)) != len(self.configurations):
            raise ValueError("Configurations must not repeat.")

        # Check if any unknown object type is given.
        unknown = [o for o in self.objects if o not in OBJECT_FAMILY_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown object famil(y/ies) {unknown}; choose from "
                f"{', '.join(OBJECT_FAMILY_NAMES)}."
            )
        if not self.objects:
            raise ValueError("At least one object family is required.")
        if len(set(self.objects)) != len(self.objects):
            raise ValueError("Object families must not repeat.")

        # Calculate the amount of groups.
        groups: int = len(self.configurations) * len(self.objects)
        if self.total_samples < groups:
            raise ValueError(
                f"total_samples ({self.total_samples}) must be at least the number "
                f"of configuration and object family pairs ({groups})."
            )


@dataclasses.dataclass
class Recipe:
    """A complete, reproducible description of a hologram dataset."""

    dataset: Dataset = dataclasses.field(default_factory=Dataset)
    geometry: Geometry = dataclasses.field(default_factory=Geometry)
    camera: Camera = dataclasses.field(default_factory=Camera)
    speckle: Speckle = dataclasses.field(default_factory=Speckle)
    objects: Objects = dataclasses.field(default_factory=Objects)

    def hologram_config(self) -> HologramConfig:
        """
        Collect the simulator settings of this recipe.

        Returns:
            HologramConfig: Geometry and camera settings for the simulator.
        """
        return HologramConfig(
            resolution=self.geometry.resolution,
            pixel_size=self.geometry.pixel_size,
            wavelength=self.geometry.wavelength,
            numerical_aperture=self.geometry.numerical_aperture,
            z_distance=self.geometry.z_distance,
            bit_depth=self.camera.bit_depth,
            num_cells=self.geometry.num_cells,
            background_electrons=self.camera.background_electrons,
            saturation_electrons=self.camera.saturation_electrons,
        )

    def camera_noise(self, speckle_contrast: float | None = None) -> NoiseConfig:
        """
        Describe the noise of a real recording.

        The camera always adds shot noise, dark current, read noise and
        quantisation, so they cannot be switched off one by one. Only the speckle
        of the light source varies. The illumination is always coherent, because
        the hologram needs it.

        Args:
            speckle_contrast (float | None): Speckle contrast of the light source
                (std/mean), or None for a speckle-free source.

        Returns:
            NoiseConfig: The camera noise with the given speckle.
        """
        speckle: bool = speckle_contrast is not None
        return NoiseConfig(
            speckle=speckle,
            shot=True,
            read=True,
            dark=True,
            speckle_contrast=speckle_contrast,
            speckle_grain_size=self.speckle.grain_size if speckle else None,
            read_noise_sigma=self.camera.read_noise,
            dark_noise_mean=self.camera.dark_electrons,
        )

    def noise_configurations(self) -> dict[str, NoiseConfig]:
        """
        Build the noise configurations selected in the dataset section.

        With the default values and the first-version ellipses, the rms deviation
        from the noise-free hologram is 0.02% of the background for ideal
        (quantisation only), 1.4% for camera, 2.0% for camera_led_speckle and
        12.1% for camera_laser_speckle.

        Returns:
            dict[str, NoiseConfig]: Noise configuration by name, in recipe order.
        """
        available: dict[str, NoiseConfig] = {
            "ideal": NoiseConfig(speckle=False, shot=False, read=False, dark=False),
            "camera": self.camera_noise(),
            "camera_led_speckle": self.camera_noise(self.speckle.led_contrast),
            "camera_laser_speckle": self.camera_noise(self.speckle.laser_contrast),
        }
        return {name: available[name] for name in self.dataset.configurations}

    def object_families(self) -> dict[str, ObjectFamily]:
        """
        Build every object family with its placement and calibrated exposure.

        Returns:
            dict[str, ObjectFamily]: Object family by name, for all known
                families, not only the selected ones.
        """
        objects: Objects = self.objects

        def red_blood_cells(sim: InlineHologramSimulator) -> int:
            """
            Place red blood cells at the recipe density.

            Args:
                sim (InlineHologramSimulator): Simulator to place the cells in.

            Returns:
                int: Number of cells placed.
            """
            area: float = (sim.resolution * sim.pixel_size) ** 2
            count: int = int(round(objects.red_blood_cell_density * area))
            return add_red_blood_cells(sim, count) if count else 0

        def adherent_cells(sim: InlineHologramSimulator) -> int:
            """
            Place adherent cells up to the recipe coverage.

            Args:
                sim (InlineHologramSimulator): Simulator to place the cells in.

            Returns:
                int: Number of cells placed.
            """
            return add_adherent_cells(sim, coverage=objects.adherent_coverage)

        def ellipses(sim: InlineHologramSimulator) -> int:
            """
            Place the first-version ellipses.

            Args:
                sim (InlineHologramSimulator): Simulator to place the ellipses in.

            Returns:
                int: Number of ellipses placed.
            """
            add_random_cells(sim, num_cells=self.geometry.num_cells)
            return self.geometry.num_cells

        return {
            "red_blood_cells": ObjectFamily(
                "red_blood_cells",
                red_blood_cells,
                objects.red_blood_cells_background_electrons,
            ),
            "adherent_cells": ObjectFamily(
                "adherent_cells",
                adherent_cells,
                objects.adherent_cells_background_electrons,
            ),
            "ellipses": ObjectFamily(
                "ellipses", ellipses, self.camera.background_electrons
            ),
        }

    def groups(self) -> list[tuple[str, NoiseConfig, ObjectFamily]]:
        """
        Combine every selected noise configuration with every object family.

        Returns:
            list[tuple[str, NoiseConfig, ObjectFamily]]: One entry per group,
                named "configuration/family", with its noise and object family.
        """
        noises: dict[str, NoiseConfig] = self.noise_configurations()
        families: dict[str, ObjectFamily] = self.object_families()
        return [
            (f"{config}/{family}", noises[config], families[family])
            for config in self.dataset.configurations
            for family in self.dataset.objects
        ]

    def samples_per_group(self) -> dict[str, int]:
        """
        Split the samples equally over the groups.

        Any remainder goes to the first groups, so no sample is lost.

        Returns:
            dict[str, int]: Number of holograms by group name.
        """
        names: list[str] = [name for name, _, _ in self.groups()]
        share, remainder = divmod(self.dataset.total_samples, len(names))
        return {name: share + (i < remainder) for i, name in enumerate(names)}


def default_recipe() -> Recipe:
    """
    Return the recipe of the HoloPASWIN dataset.

    Returns:
        Recipe: The recipe with the cited default values.
    """
    return Recipe()


def load_recipe(path: str | pathlib.Path) -> Recipe:
    """
    Read a recipe from a TOML file.

    Values missing from the file keep their defaults. Unknown sections or keys
    are rejected, which catches typos.

    Args:
        path (str | pathlib.Path): The TOML file.

    Returns:
        Recipe: The recipe in the file.

    Raises:
        ValueError: If the file has an unknown section or key, or an invalid
            dataset section.
    """
    with open(path, "rb") as file:
        data: dict[str, Any] = tomllib.load(file)
    return recipe_from_dict(data)


def recipe_from_dict(data: dict[str, Any]) -> Recipe:
    """
    Build a recipe from nested dictionaries such as a parsed TOML file.

    Args:
        data (dict[str, Any]): Values by section and key. Missing values keep
            their defaults.

    Returns:
        Recipe: The recipe with the given values.

    Raises:
        ValueError: If a section or key is unknown, or the dataset section is
            invalid.
    """
    defaults: Recipe = default_recipe()
    sections: dict[str, Any] = {}
    for section_name, values in data.items():
        if section_name not in {f.name for f in dataclasses.fields(Recipe)}:
            raise ValueError(f"Unknown recipe section [{section_name}].")
        section: Any = getattr(defaults, section_name)
        known: set[str] = {f.name for f in dataclasses.fields(section)}
        unknown: list[str] = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in [{section_name}]; expected one of "
                f"{sorted(known)}."
            )
        sections[section_name] = dataclasses.replace(section, **values)
    return dataclasses.replace(defaults, **sections)


def _toml_value(value: Any) -> str:
    """
    Format a recipe value as a TOML literal.

    Args:
        value (Any): A string, list, number or boolean.

    Returns:
        str: The value as TOML.
    """
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)


def recipe_to_toml(recipe: Recipe) -> str:
    """
    Write a recipe as TOML, with the unit and source of every value.

    Args:
        recipe (Recipe): The recipe to write.

    Returns:
        str: The TOML text, ending with a newline.
    """
    lines: list[str] = [
        "# hologen dataset recipe",
        "# Generate with: hologen generate --recipe <this file>",
    ]
    for section_field in dataclasses.fields(recipe):
        section: Any = getattr(recipe, section_field.name)
        lines += ["", f"[{section_field.name}]"]
        for field in dataclasses.fields(section):
            source: str = field.metadata.get("source", "")
            unit: str = field.metadata.get("unit", "")
            if source:
                lines.append(f"# {source}")
            line: str = f"{field.name} = {_toml_value(getattr(section, field.name))}"
            lines.append(f"{line}  # {unit}" if unit else line)
    return "\n".join(lines) + "\n"

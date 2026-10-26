"""
The dataset recipe, which holds every physical value used to generate a dataset.

Each value carries its unit and source, and the noise configurations and object
families are built from them. A recipe can be written to and read from a TOML
file, so a dataset is reproducible from one versioned file. Values missing from
a file keep their defaults.
"""

import dataclasses
import json
import pathlib
import tomllib
from typing import Any

from hologen.dataset.configs import HologramConfig, NoiseConfig
from hologen.inline_simulator import (
    DEFAULT_BACKGROUND_ELECTRONS,
    DEFAULT_SATURATION_ELECTRONS,
    InlineHologramSimulator,
)
from hologen.objects.adherent_cells import add_adherent_cells
from hologen.objects.families import ObjectFamily
from hologen.objects.random_cells import add_random_cells
from hologen.objects.red_blood_cells import RedBloodCellOptics, add_red_blood_cells

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

# Temperature model of the dark-current explorer, not part of a recipe. The source
# of Camera.dark_current does not state its sensor temperature, so 25 C is assumed.
# The doubling step is typical of CMOS sensors; EMVA 1288 measures it per camera.
DARK_CURRENT_TEMPERATURE: float = 25.0  # C
DARK_CURRENT_DOUBLING: float = 9.0  # C


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


def _check(owner: object, name: str, valid: bool, rule: str) -> None:
    """
    Reject a recipe value that breaks its rule.

    Args:
        owner (object): Recipe section that holds the value.
        name (str): Name of the value.
        valid (bool): Whether the value follows the rule.
        rule (str): The rule, completing "<name> must be ...".

    Raises:
        ValueError: If valid is False.
    """
    if not valid:
        raise ValueError(f"{name} must be {rule} (got {getattr(owner, name)}).")


@dataclasses.dataclass
class Geometry:
    """
    Geometry of the magnified in-line (Gabor) holographic microscope.

    The setup is that of Wdowiak et al., arXiv:2508.15419 (2025), where a
    defocused detector sits behind a microscope objective. Distances and pixel
    sizes are given in the object plane.
    """

    resolution: int = _value(
        1024,
        "pixels",
        "Image size; a 246 um field at 0.24 um per pixel holds adherent cells of "
        "every measured size (at 224 px half of them could not be placed)",
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
    z_offset: float = _value(
        0.0,
        "um",
        "Fixed shift of the recording distance from z_distance, for test sets at a "
        "known distance error",
    )
    z_error: float = _value(
        0.0,
        "um",
        "Half-width of a uniform random error of the recording distance, for "
        "training robust to a wrong distance (6 um is one focus step of "
        "Wdowiak et al. 2025, 0.6 mm at the detector divided by M^2 = 100)",
    )
    num_cells: int = _value(5, "objects per hologram", "Design choice")

    def __post_init__(self) -> None:
        """
        Check the optics, and that every recording distance lies behind the object.

        Raises:
            ValueError: If the image is smaller than 8 pixels, a size, the
                magnification or the wavelength is not positive, the NA is
                outside (0, 1], z_error is negative, or the shortest recording
                distance is not positive.
        """
        _check(self, "resolution", self.resolution >= 8, "at least 8 pixels")
        for name in ("sensor_pixel_size", "magnification", "wavelength"):
            _check(self, name, getattr(self, name) > 0, "positive")
        _check(
            self,
            "numerical_aperture",
            0 < self.numerical_aperture <= 1,
            "in (0, 1] for an objective in air",
        )
        _check(self, "num_cells", self.num_cells >= 0, "zero or more")
        if self.z_error < 0:
            raise ValueError(f"z_error must not be negative (got {self.z_error}).")
        shortest: float = self.z_distance + self.z_offset - self.z_error
        if shortest <= 0:
            raise ValueError(
                f"The shortest recording distance, z_distance + z_offset - z_error = "
                f"{shortest} um, must be positive. A distance of zero or less would "
                "put the recording plane at or before the object."
            )

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

    def __post_init__(self) -> None:
        """
        Check that the camera values describe a working sensor.

        Raises:
            ValueError: If an exposure or the saturation capacity is not
                positive, a noise value or the exposure time is negative, or the
                bit depth does not fit the 16-bit hologram files.
        """
        for name in ("background_electrons", "saturation_electrons"):
            _check(self, name, getattr(self, name) > 0, "positive")
        _check(
            self,
            "bit_depth",
            1 <= self.bit_depth <= 16,
            "between 1 and 16, since holograms are stored as 16-bit counts",
        )
        for name in ("read_noise", "dark_current", "exposure_time"):
            _check(self, name, getattr(self, name) >= 0, "zero or more")

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

    def __post_init__(self) -> None:
        """
        Check the grain size and the speckle contrasts.

        Raises:
            ValueError: If the grain size is not positive or a contrast is
                outside (0, 1]; fully developed speckle has a contrast of 1.
        """
        _check(self, "grain_size", self.grain_size > 0, "positive")
        for name in ("led_contrast", "laser_contrast"):
            _check(self, name, 0 < getattr(self, name) <= 1, "in (0, 1]")


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
        "Design choice: a sparse monolayer, 12 cells per 54 x 54 um "
        "(about 250 per 246 x 246 um field)",
    )
    adherent_coverage: float = _value(
        0.263,
        "fraction of the field",
        "Measured median field coverage of 7476 adherent cells "
        "(scripts/calibrate_adherent_cells.py, Vicar et al. data)",
    )
    opd_scale: float = _value(
        1.0,
        "",
        "Factor on the phase of every cell, 1 for real cells. At 2 (the "
        "out-of-distribution phase test) adherent cells reach about 9.5 rad, as "
        "rounded or dividing cells do; red blood cells would need about twice the "
        "physiological haemoglobin concentration, so they are a stress test",
    )
    adherent_absorption: float = _value(
        0.0,
        "1/um of OPD",
        "Unstained adherent cells are transparent; no measured absorption of "
        "stained adherent cells near 532 nm is available, so every set keeps 0",
    )
    haemoglobin_absorption_scale: float = _value(
        1.0,
        "",
        "Factor on the haemoglobin absorption of red blood cells (Prahl 1998, "
        "532 nm); 3 for the absorption test",
    )
    red_blood_cell_attempts: int = _value(
        1000,
        "",
        "Random positions tried before red blood cell placement stops. Placement "
        "without overlap stalls near half the field covered, so a crowded field "
        "needs more tries; about 10000 reach twice the default density",
    )
    adherent_attempts: int = _value(
        200,
        "",
        "Random positions tried before adherent cell placement stops; about 5000 "
        "reach a coverage of 0.5",
    )
    exposure_percentile: float = _value(
        95.0,
        "percent",
        "Percent of holograms that must not saturate; the rest clip at the foci "
        "of the most strongly refracting cells",
    )
    red_blood_cells_background_electrons: float = _value(
        2735.0,
        "e- per pixel",
        "Calibrated with hologen calibrate (see exposure_calibration)",
    )
    adherent_cells_background_electrons: float = _value(
        1855.0,
        "e- per pixel",
        "Calibrated with hologen calibrate (see exposure_calibration)",
    )
    exposure_calibration: str = _value(
        "red_blood_cells p95 peak 5.29x (median 4.86x, max 5.87x); adherent_cells "
        "p95 peak 7.80x (median 5.28x, max 10.86x); 400 holograms per family, "
        "seeds 90000-90399, 1024 px",
        "",
        "Provenance of the calibrated exposures, written by hologen calibrate",
    )
    calibration_fingerprint: str = _value(
        "ef79cbdc7d12",
        "",
        "Fingerprint of the settings the exposures were calibrated for; hologen "
        "generate warns when it no longer matches the recipe",
    )

    def __post_init__(self) -> None:
        """
        Check that the content, phase, absorption and exposure values are usable.

        Raises:
            ValueError: If the density is negative, the coverage is outside
                [0, 1), opd_scale is not positive, an absorption value is
                negative, a number of placement attempts is below 1, the
                percentile is outside (0, 100] or an exposure is not positive.
        """
        _check(
            self,
            "red_blood_cell_density",
            self.red_blood_cell_density >= 0,
            "zero or more",
        )
        _check(self, "adherent_coverage", 0 <= self.adherent_coverage < 1, "in [0, 1)")
        _check(
            self,
            "exposure_percentile",
            0 < self.exposure_percentile <= 100,
            "in (0, 100]",
        )
        for name in (
            "red_blood_cells_background_electrons",
            "adherent_cells_background_electrons",
        ):
            _check(self, name, getattr(self, name) > 0, "positive")
        if self.opd_scale <= 0:
            raise ValueError(f"opd_scale must be positive (got {self.opd_scale}).")
        if self.adherent_absorption < 0:
            raise ValueError(
                "adherent_absorption must not be negative "
                f"(got {self.adherent_absorption})."
            )
        if self.haemoglobin_absorption_scale < 0:
            raise ValueError(
                "haemoglobin_absorption_scale must not be negative "
                f"(got {self.haemoglobin_absorption_scale})."
            )
        for name in ("red_blood_cell_attempts", "adherent_attempts"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1 (got {getattr(self, name)})."
                )


@dataclasses.dataclass
class Dataset:
    """Size, randomness and noise configurations of the generated dataset."""

    total_samples: int = _value(
        25000,
        "holograms",
        "Split equally over the groups, every configuration with every object family",
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
                or repeated, if there are fewer samples than groups, or if the
                seed is negative or the batch size below 1.
        """
        _check(self, "seed", self.seed >= 0, "zero or more")
        _check(self, "batch_size", self.batch_size >= 1, "at least 1")

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
            z_offset=self.geometry.z_offset,
            z_error=self.geometry.z_error,
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
        12.1% for camera_laser_speckle. The cell families are recorded at their
        lower calibrated exposures, so their shot noise is larger: hologen measure
        gives 1.9% (red blood cells) and 2.3% (adherent cells) for camera, 2.4% and
        2.7% for camera_led_speckle and 12.9% and 12.3% for camera_laser_speckle.

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
        # The scales are applied after every random draw, so a seed keeps its
        # cells and their positions whatever the scales are.
        base: RedBloodCellOptics = RedBloodCellOptics()
        optics: RedBloodCellOptics = dataclasses.replace(
            base,
            delta_n=base.delta_n * objects.opd_scale,
            absorption=base.absorption * objects.haemoglobin_absorption_scale,
        )

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
            if not count:
                return 0
            return add_red_blood_cells(
                sim, count, optics, max_attempts=objects.red_blood_cell_attempts
            )

        def adherent_cells(sim: InlineHologramSimulator) -> int:
            """
            Place adherent cells up to the recipe coverage.

            Args:
                sim (InlineHologramSimulator): Simulator to place the cells in.

            Returns:
                int: Number of cells placed.
            """
            return add_adherent_cells(
                sim,
                coverage=objects.adherent_coverage,
                opd_scale=objects.opd_scale,
                absorption=objects.adherent_absorption,
                max_attempts=objects.adherent_attempts,
            )

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
        ValueError: If a section or key is unknown, a value has the wrong type
            or is out of range, or the dataset section is invalid.
    """
    defaults: Recipe = default_recipe()
    sections: dict[str, Any] = {}
    for section_name, values in data.items():
        if section_name not in {f.name for f in dataclasses.fields(Recipe)}:
            raise ValueError(f"Unknown recipe section [{section_name}].")
        if not isinstance(values, dict):
            raise ValueError(f"[{section_name}] must be a table of values.")
        section: Any = getattr(defaults, section_name)
        known: set[str] = {f.name for f in dataclasses.fields(section)}
        unknown: list[str] = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in [{section_name}]; expected one of "
                f"{sorted(known)}."
            )
        checked: dict[str, Any] = {
            key: _typed(section_name, key, value, getattr(section, key))
            for key, value in values.items()
        }
        sections[section_name] = dataclasses.replace(section, **checked)
    return dataclasses.replace(defaults, **sections)


def _typed(section_name: str, key: str, value: Any, default: Any) -> Any:
    """
    Check that a value from a file has the type of its default.

    A whole number is accepted for a decimal value and turned into a float, so
    `z_distance = 20` means the same as `z_distance = 20.0`, also for the
    exposure fingerprint.

    Args:
        section_name (str): Section of the value, for the error message.
        key (str): Name of the value.
        value (Any): The value read from the file.
        default (Any): The default value, whose type the value must have.

    Returns:
        Any: The value, as a float where the default is a float.

    Raises:
        ValueError: If the value has another type than its default.
    """
    expected: str
    if isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        expected = "a number"
    elif isinstance(default, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        expected = "a whole number"
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
        expected = "a string"
    else:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        expected = "a list of strings"
    raise ValueError(f"{key} in [{section_name}] must be {expected} (got {value!r}).")


def _toml_value(value: Any) -> str:
    """
    Format a recipe value as a TOML literal.

    Args:
        value (Any): A string, list, number or boolean.

    Returns:
        str: The value as TOML.
    """
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string, with quotes and
        # backslashes escaped.
        return json.dumps(value, ensure_ascii=False)
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

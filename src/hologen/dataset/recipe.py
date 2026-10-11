"""
The dataset recipe: every physical value used to generate a dataset, with its unit
and source, and the noise configurations built from them.

A recipe can be written to and read from a TOML file, so a dataset is reproducible
from one versioned file. Values missing from a file keep their defaults.
"""

import dataclasses
import pathlib
import tomllib
from typing import Any

from hologen.dataset.configs import HologramConfig, NoiseConfig
from hologen.inline_simulator import (
    DEFAULT_BACKGROUND_ELECTRONS,
    DEFAULT_SATURATION_ELECTRONS,
)

# Noise configurations, from the noise-free reference to the noisiest recording.
CONFIGURATION_NAMES: tuple[str, ...] = (
    "ideal",
    "camera",
    "camera_led_speckle",
    "camera_laser_speckle",
)


def _value(default: Any, unit: str, source: str) -> Any:
    """Declare a recipe value together with its unit and source."""
    return dataclasses.field(default=default, metadata={"unit": unit, "source": source})


@dataclasses.dataclass
class Geometry:
    """
    Magnified in-line (Gabor) holographic microscope of Wdowiak et al.,
    arXiv:2508.15419 (2025): a defocused detector behind a microscope objective.
    Distances and pixel sizes are given in the object plane.
    """

    resolution: int = _value(
        224, "pixels", "Image size; decided after the HOLO-002 pilot"
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
        """Pixel size in the object plane (sensor pixel / magnification), in um."""
        return self.sensor_pixel_size / self.magnification


@dataclasses.dataclass
class Camera:
    """
    Camera model of the Sony IMX183 CMOS following EMVA 1288 Release 4.0 Linear
    (2021), Sec. 2.4 Eq. 14: shot noise, dark current, read noise and a k-bit ADC.
    """

    background_electrons: float = _value(
        DEFAULT_BACKGROUND_ELECTRONS,
        "e- per pixel",
        "Brightest exposure without clipping: brightest fringe plus 3 sigma of "
        "shot noise below the saturation capacity (PROVISIONAL until the "
        "HOLO-002 cell objects exist)",
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
        """Mean dark electrons per pixel in one exposure (EMVA 1288 Eq. 29)."""
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

    def __post_init__(self) -> None:
        unknown = [c for c in self.configurations if c not in CONFIGURATION_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown configuration(s) {unknown}; choose from "
                f"{', '.join(CONFIGURATION_NAMES)}."
            )
        if not self.configurations:
            raise ValueError("At least one configuration is required.")
        if len(set(self.configurations)) != len(self.configurations):
            raise ValueError("Configurations must not repeat.")
        if self.total_samples < len(self.configurations):
            raise ValueError(
                f"total_samples ({self.total_samples}) must be at least the number "
                f"of configurations ({len(self.configurations)})."
            )


@dataclasses.dataclass
class Recipe:
    """A complete, reproducible description of a hologram dataset."""

    dataset: Dataset = dataclasses.field(default_factory=Dataset)
    geometry: Geometry = dataclasses.field(default_factory=Geometry)
    camera: Camera = dataclasses.field(default_factory=Camera)
    speckle: Speckle = dataclasses.field(default_factory=Speckle)

    def hologram_config(self) -> HologramConfig:
        """The simulator settings of this recipe."""
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
        Noise of a real recording: the camera always adds shot noise, dark current,
        read noise and quantisation, so they cannot be switched off one by one.
        Only the speckle of the light source varies. The illumination is always
        coherent (the hologram needs it); None means a speckle-free source.
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
        The selected noise configurations. With the default values their noise
        (rms deviation from the noise-free hologram, % of the background) is:
        ideal 0.02 (quantisation only), camera 1.4, camera_led_speckle 2.0,
        camera_laser_speckle 12.1.
        """
        available: dict[str, NoiseConfig] = {
            "ideal": NoiseConfig(speckle=False, shot=False, read=False, dark=False),
            "camera": self.camera_noise(),
            "camera_led_speckle": self.camera_noise(self.speckle.led_contrast),
            "camera_laser_speckle": self.camera_noise(self.speckle.laser_contrast),
        }
        return {name: available[name] for name in self.dataset.configurations}

    def samples_per_configuration(self) -> dict[str, int]:
        """Split the samples equally; any remainder goes to the first ones."""
        names = self.dataset.configurations
        share, remainder = divmod(self.dataset.total_samples, len(names))
        return {name: share + (i < remainder) for i, name in enumerate(names)}


def default_recipe() -> Recipe:
    """The recipe of the HoloPASWIN dataset, with cited values."""
    return Recipe()


def load_recipe(path: str | pathlib.Path) -> Recipe:
    """
    Read a recipe from a TOML file. Values missing from the file keep their
    defaults; unknown sections or keys are rejected to catch typos.
    """
    with open(path, "rb") as file:
        data: dict[str, Any] = tomllib.load(file)
    return recipe_from_dict(data)


def recipe_from_dict(data: dict[str, Any]) -> Recipe:
    """Build a recipe from nested dictionaries such as a parsed TOML file."""
    defaults = default_recipe()
    sections: dict[str, Any] = {}
    for section_name, values in data.items():
        if section_name not in {f.name for f in dataclasses.fields(Recipe)}:
            raise ValueError(f"Unknown recipe section [{section_name}].")
        section = getattr(defaults, section_name)
        known = {f.name for f in dataclasses.fields(section)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in [{section_name}]; expected one of "
                f"{sorted(known)}."
            )
        sections[section_name] = dataclasses.replace(section, **values)
    return dataclasses.replace(defaults, **sections)


def _toml_value(value: Any) -> str:
    """Format a recipe value as a TOML literal."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)


def recipe_to_toml(recipe: Recipe) -> str:
    """Write a recipe as TOML, with the unit and source of every value."""
    lines: list[str] = [
        "# hologen dataset recipe",
        "# Generate with: hologen generate --recipe <this file>",
    ]
    for section_field in dataclasses.fields(recipe):
        section = getattr(recipe, section_field.name)
        lines += ["", f"[{section_field.name}]"]
        for field in dataclasses.fields(section):
            source: str = field.metadata.get("source", "")
            unit: str = field.metadata.get("unit", "")
            if source:
                lines.append(f"# {source}")
            line = f"{field.name} = {_toml_value(getattr(section, field.name))}"
            lines.append(f"{line}  # {unit}" if unit else line)
    return "\n".join(lines) + "\n"

"""Unit tests for dark current noise (EMVA 1288, Release 4.0 Linear, Sec. 3)."""

import numpy
import pytest

from hologen import RandomNumberGenerator
from hologen.noises import add_dark_current_noise

SHAPE: tuple[int, int] = (256, 256)


@pytest.fixture(autouse=True)
def seed() -> None:
    """Seed the shared generator so every test is reproducible."""
    RandomNumberGenerator.set_seed(2468)


@pytest.mark.parametrize("dark_mean", [2.0, 7.0, 20.0])
def test_dark_electrons_are_poisson(dark_mean: float) -> None:
    """Dark electrons add their mean and a variance equal to that mean."""
    dark = add_dark_current_noise(numpy.zeros(SHAPE), dark_mean)
    assert dark.mean() == pytest.approx(dark_mean, rel=0.02)
    assert dark.var() == pytest.approx(dark_mean, rel=0.04)


def test_dark_electrons_are_whole_and_non_negative() -> None:
    dark = add_dark_current_noise(numpy.zeros(SHAPE), 3.5)
    numpy.testing.assert_array_equal(dark, numpy.round(dark))
    assert dark.min() >= 0


def test_dark_electrons_add_to_the_signal() -> None:
    """The signal is offset by the dark electrons, not scaled."""
    signal = numpy.full(SHAPE, 4000.0)
    noisy = add_dark_current_noise(signal, 7.0)
    assert (noisy - signal).mean() == pytest.approx(7.0, rel=0.02)
    assert (noisy >= signal).all()


def test_input_is_not_modified() -> None:
    """Regression test: the function used to add to its input in place."""
    signal = numpy.full(SHAPE, 4000.0)
    before = signal.copy()
    add_dark_current_noise(signal, 7.0)
    numpy.testing.assert_array_equal(signal, before)


def test_zero_dark_current_adds_nothing() -> None:
    signal = numpy.full(SHAPE, 4000.0)
    numpy.testing.assert_array_equal(add_dark_current_noise(signal, 0.0), signal)

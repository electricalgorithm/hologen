"""The shared random number generator behind every random draw in hologen."""

import numpy.random


class RandomNumberGenerator:
    """
    One random number generator shared by all of hologen.

    Seeding it once makes every later draw, in every module, reproducible.
    Constructing the class returns the shared numpy generator itself.
    """

    _rng_inst: numpy.random.Generator | None = None

    @classmethod
    def set_seed(cls, seed: int) -> None:
        """
        Replace the shared generator with a freshly seeded one.

        Args:
            seed (int): Seed of the new generator; the same seed gives the same
                draws.
        """
        cls._rng_inst = numpy.random.default_rng(seed)

    @classmethod
    def get_generator(cls) -> numpy.random.Generator:
        """
        Return the shared generator, creating an unseeded one on first use.

        Returns:
            numpy.random.Generator: The shared generator.
        """
        if cls._rng_inst is None:
            cls._rng_inst = numpy.random.default_rng(None)
        return cls._rng_inst

    def __new__(cls) -> numpy.random.Generator:
        """
        Return the shared generator instead of a new instance.

        Returns:
            numpy.random.Generator: The shared generator.
        """
        return cls.get_generator()

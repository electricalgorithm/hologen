"""
This module implements the get random number interface for the internal operations.
"""
import numpy.random


class RandomNumberGenerator:
    """Defines the RNG logic for the internal APIs."""
    _rng_inst: numpy.random.Generator | None = None

    @classmethod
    def set_seed(cls, seed: int) -> None:
        """This method sets the seed of the number generator.
        
        :param seed: The RNG seed for reproduction.
        :returns None:
        """
        cls._rng_inst = numpy.random.default_rng(seed)
    
    @classmethod
    def get_generator(cls) -> numpy.random.Generator:
        """This method returns the random number generator.

        :returns numpy.random.Generator:
        """
        if cls._rng_inst is None:
            cls._rng_inst = numpy.random.default_rng(None)
        return cls._rng_inst

    def __new__(cls) -> numpy.random.Generator:
        """Return the generator if one tried to construct it."""
        return cls.get_generator()

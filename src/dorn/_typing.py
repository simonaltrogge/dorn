from collections.abc import Sequence
from typing import Union

import numpy as np
import numpy.typing as npt
from numpy.random import BitGenerator, Generator, SeedSequence

SequenceLikeBool = Union[bool, Sequence[bool], npt.NDArray[np.bool_]]
SequenceLikeInt = Union[SequenceLikeBool, int, Sequence[int], npt.NDArray[np.int_]]
SequenceLikeFloat = Union[
    SequenceLikeInt, float, Sequence[float], npt.NDArray[np.float64]
]

Seed = Union[Generator, BitGenerator, SeedSequence, Sequence[int], int]

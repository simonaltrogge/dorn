from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import numpy.typing as npt

from dorn._transform import to_neuronpopulation_and_length
from dorn._typing import SequenceLikeFloat
from dorn.neuronpopulation import NeuronPopulation
from dorn.stdp import STDPFunction


class SynapticTraces:
    def __init__(
        self,
        neurons: NeuronPopulation | int,
        presynaptic_time_constants: SequenceLikeFloat,
        postsynaptic_time_constants: SequenceLikeFloat,
        *,
        presynaptic_update_amounts: SequenceLikeFloat | None = None,
        postsynaptic_update_amounts: SequenceLikeFloat | None = None,
        presynaptic_traces: npt.ArrayLike = 0.0,
        postsynaptic_traces: npt.ArrayLike = 0.0,
    ) -> None:
        self.neurons, self.neuron_count = to_neuronpopulation_and_length(neurons)

        self.presynaptic_time_constants = presynaptic_time_constants
        self.postsynaptic_time_constants = postsynaptic_time_constants
        self.presynaptic_update_amounts = presynaptic_update_amounts
        self.postsynaptic_update_amounts = postsynaptic_update_amounts

        self.presynaptic = presynaptic_traces
        self.postsynaptic = postsynaptic_traces

    def decay(self, duration: float) -> None:
        self.presynaptic *= np.exp(-duration / self.presynaptic_time_constants)
        self.postsynaptic *= np.exp(-duration / self.postsynaptic_time_constants)

    def increase(self, neuron: int) -> None:
        self.presynaptic[neuron, :] += self.presynaptic_update_amounts
        self.postsynaptic[neuron, :] += self.postsynaptic_update_amounts

    @property
    def presynaptic(self) -> npt.NDArray[np.float64]:
        return self._presynaptic

    @presynaptic.setter
    def presynaptic(self, value: npt.ArrayLike) -> None:
        self._presynaptic = np.full(
            (self.neuron_count, len(self.presynaptic_time_constants)),
            value,
            dtype=np.float64,
        )

    @property
    def postsynaptic(self) -> npt.NDArray[np.float64]:
        return self._postsynaptic

    @postsynaptic.setter
    def postsynaptic(self, value: npt.ArrayLike) -> None:
        self._postsynaptic = np.full(
            (self.neuron_count, len(self.postsynaptic_time_constants)),
            value,
            dtype=np.float64,
        )

    @property
    def presynaptic_time_constants(self) -> npt.NDArray[np.float64]:
        return self._presynaptic_time_constants

    @presynaptic_time_constants.setter
    def presynaptic_time_constants(self, value: SequenceLikeFloat) -> None:
        self._presynaptic_time_constants = np.atleast_1d(value).astype(
            np.float64, copy=True
        )

    @property
    def postsynaptic_time_constants(self) -> npt.NDArray[np.float64]:
        return self._postsynaptic_time_constants

    @postsynaptic_time_constants.setter
    def postsynaptic_time_constants(self, value: SequenceLikeFloat) -> None:
        self._postsynaptic_time_constants = np.atleast_1d(value).astype(
            np.float64, copy=True
        )

    @property
    def presynaptic_update_amounts(self) -> npt.NDArray[np.float64]:
        return self._presynaptic_update_amounts

    @presynaptic_update_amounts.setter
    def presynaptic_update_amounts(self, value: SequenceLikeFloat | None) -> None:
        if value is None:
            self._presynaptic_update_amounts = 1 / self.presynaptic_time_constants
            return

        self._presynaptic_update_amounts = np.full_like(
            self.presynaptic_time_constants, value, dtype=np.float64
        )

    @property
    def postsynaptic_update_amounts(self) -> npt.NDArray[np.float64]:
        return self._postsynaptic_update_amounts

    @postsynaptic_update_amounts.setter
    def postsynaptic_update_amounts(self, value: SequenceLikeFloat | None) -> None:
        if value is None:
            self._postsynaptic_update_amounts = 1 / self.postsynaptic_time_constants
            return

        self._postsynaptic_update_amounts = np.full_like(
            self.postsynaptic_time_constants, value, dtype=np.float64
        )

    @classmethod
    def from_stdp_function(
        cls,
        neurons: NeuronPopulation | int,
        stdp_function: STDPFunction,
        *,
        presynaptic_traces: npt.ArrayLike = 0.0,
        postsynaptic_traces: npt.ArrayLike = 0.0,
    ) -> Self:
        presynaptic_time_constants, presynaptic_amplitudes = zip(
            *stdp_function.causal_exponentials
        )
        postsynaptic_time_constants, postsynaptic_amplitudes = zip(
            *stdp_function.acausal_exponentials
        )

        return cls(
            neurons,
            presynaptic_time_constants,
            postsynaptic_time_constants,
            presynaptic_update_amounts=presynaptic_amplitudes,
            postsynaptic_update_amounts=postsynaptic_amplitudes,
            presynaptic_traces=presynaptic_traces,
            postsynaptic_traces=postsynaptic_traces,
        )

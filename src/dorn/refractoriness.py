from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from dorn._transform import to_neuronpopulation_and_length
from dorn._typing import Seed, SequenceLikeFloat
from dorn.neuronpopulation import NeuronPopulation


class Refractoriness:
    def __init__(
        self,
        neurons: NeuronPopulation | int,
        periods: SequenceLikeFloat,
        *,
        last_spike_times: SequenceLikeFloat = -np.inf,
    ):
        self.neurons, self.neuron_count = to_neuronpopulation_and_length(neurons)
        self.periods = periods
        self.last_spike_times = last_spike_times

    def spike_is_possible(self, spike: HasTimeAndNeuron) -> bool:
        interspike_interval = spike.time - self.last_spike_times[spike.neuron]

        return interspike_interval > self.periods[spike.neuron]

    def apply_spike(self, spike: HasTimeAndNeuron) -> None:
        self.last_spike_times[spike.neuron] = spike.time

    @property
    def periods(self) -> npt.NDArray[np.float64]:
        return self._periods

    @periods.setter
    def periods(self, value: SequenceLikeFloat) -> None:
        self._periods = np.full(self.neuron_count, value, dtype=np.float64)

    @property
    def last_spike_times(self) -> npt.NDArray[np.float64]:
        return self._last_spike_times

    @last_spike_times.setter
    def last_spike_times(self, value: SequenceLikeFloat) -> None:
        self._last_spike_times = np.full(self.neuron_count, value, dtype=np.float64)


class HasTimeAndNeuron(Protocol):
    time: float
    neuron: int

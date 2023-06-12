from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dorn import poissonprocess
from dorn._typing import Seed, SequenceLikeFloat
from dorn.spikes import NextSpike, SpikeType


class NeuronPopulation:
    def __init__(
        self,
        size: int,
        *,
        time_constant: float = 1.0,
        rates_spontaneous: SequenceLikeFloat = 0.0,
        rates_evoked: SequenceLikeFloat = 0.0,
    ) -> None:
        self.size = size
        self.time_constant = time_constant
        self.rates_spontaneous = rates_spontaneous
        self.rates_evoked = rates_evoked

    def let_evoked_rates_decay(self, duration: float) -> None:
        self.rates_evoked *= np.exp(-duration / self.time_constant)

    def draw_next_spike(self, *, seed: Seed) -> NextSpike:
        rng = np.random.default_rng(seed)

        interspike_interval_spontaneous = self._draw_spontaneous_interspike_interval(
            seed=rng,
        )
        interspike_interval_evoked = self._draw_evoked_interspike_interval(
            seed=rng,
        )

        if interspike_interval_spontaneous < interspike_interval_evoked:
            spike_type = SpikeType.SPONTANEOUS
            interspike_interval = interspike_interval_spontaneous
            spiking_probability_by_neuron = self.rates_spontaneous / np.sum(
                self.rates_spontaneous
            )
        else:
            spike_type = SpikeType.EVOKED
            interspike_interval = interspike_interval_evoked
            spiking_probability_by_neuron = self.rates_evoked / np.sum(
                self.rates_evoked
            )

        spiking_neuron = self._draw_spiking_neuron(
            spiking_probability_by_neuron, seed=rng
        )

        return NextSpike(interspike_interval, spiking_neuron, spike_type)

    def _draw_spontaneous_interspike_interval(self, *, seed: Seed) -> float:
        return poissonprocess.homogeneous.draw_interarrival_time(
            rate=np.sum(self.rates_spontaneous),  # type: ignore (see https://github.com/numpy/numpy/issues/23663)
            seed=seed,
        )

    def _draw_evoked_interspike_interval(self, *, seed: Seed) -> float:
        return poissonprocess.exponentially_decaying.draw_interarrival_time(
            rate=np.sum(self.rates_evoked),  # type: ignore (see https://github.com/numpy/numpy/issues/23663)
            time_constant=self.time_constant,
            seed=seed,
        )

    def _draw_spiking_neuron(
        self, spiking_probability_by_neuron: npt.NDArray[np.float64], *, seed: Seed
    ) -> int:
        rng = np.random.default_rng(seed)
        return rng.choice(self.size, p=spiking_probability_by_neuron)

    @property
    def rates_spontaneous(self) -> npt.NDArray[np.float64]:
        return self._rates_spontaneous

    @rates_spontaneous.setter
    def rates_spontaneous(self, value: SequenceLikeFloat) -> None:
        self._rates_spontaneous = np.full(self.size, value, dtype=np.float64)

    @property
    def rates_evoked(self) -> npt.NDArray[np.float64]:
        return self._rates_evoked

    @rates_evoked.setter
    def rates_evoked(self, value: SequenceLikeFloat) -> None:
        self._rates_evoked = np.full(self.size, value, dtype=np.float64)

    def __len__(self) -> int:
        return self.size

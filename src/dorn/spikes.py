from dataclasses import dataclass
from typing import NamedTuple

from _enums import HiddenValueEnum, OrderedEnum


class SpikeType(OrderedEnum, HiddenValueEnum):
    SPONTANEOUS = "spontaneous"
    EVOKED = "evoked"


@dataclass
class Spike:
    time: float


@dataclass
class NeuronalSpike(Spike):
    neuron: int
    type: SpikeType


class NextSpike(NamedTuple):
    interspike_interval: float
    neuron: int
    type: SpikeType

    def interval_to_time(self, current_time):
        return NeuronalSpike(
            current_time + self.interspike_interval, self.neuron, self.type
        )

from typing import overload

from dorn.neuronpopulation import NeuronPopulation


@overload
def to_neuronpopulation_and_length(
    neurons: NeuronPopulation,
) -> tuple[NeuronPopulation, int]:
    ...


@overload
def to_neuronpopulation_and_length(
    neurons: int,
) -> tuple[None, int]:
    ...


def to_neuronpopulation_and_length(neurons):
    if isinstance(neurons, NeuronPopulation):
        return neurons, len(neurons)

    return None, neurons

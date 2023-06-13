"""A library for event-based simulation of spiking neural networks."""

__version__ = "0.2.0"

from dorn.network import Network
from dorn.neuronpopulation import NeuronPopulation, Spike, SpikeType
from dorn.refractoriness import Refractoriness
from dorn.stdp import Exponential, STDPFunction
from dorn.synapses import Synapses
from dorn.synaptictraces import SynapticTraces

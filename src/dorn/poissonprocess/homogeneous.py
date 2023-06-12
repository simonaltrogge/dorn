import numpy as np

from dorn._typing import Seed


def draw_interarrival_time(rate: float, *, seed: Seed) -> float:
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=1 / rate)

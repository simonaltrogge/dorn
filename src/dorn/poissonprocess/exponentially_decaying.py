import numpy as np

from dorn._typing import Seed


def draw_interarrival_time(rate: float, time_constant: float, *, seed: Seed) -> float:
    rng = np.random.default_rng(seed)
    uniform_sample = rng.random()

    survival_probability = get_survival_probability(rate, time_constant)
    if survival_probability <= uniform_sample:
        return np.inf

    return interarrival_time_quantile_function(
        rate, time_constant, probability=uniform_sample
    )


def get_survival_probability(rate: float, time_constant: float) -> float:
    return 1 - np.exp(-rate * time_constant)


def interarrival_time_quantile_function(
    rate: float, time_constant: float, probability: float
) -> float:
    return -time_constant * np.log(1 + np.log(1 - probability) / (rate * time_constant))


def get_time_evolved_rate(rate: float, time_constant: float, duration: float) -> float:
    propagator = get_rate_propagator(time_constant, duration)
    return propagator * rate


def get_rate_propagator(time_constant: float, duration: float) -> float:
    return np.exp(-duration / time_constant)

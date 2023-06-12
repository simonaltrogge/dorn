from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class STDPFunction(NamedTuple):
    causal_exponentials: Sequence[Exponential]
    acausal_exponentials: Sequence[Exponential]

    def __call__(self, t_post_minus_t_pre):
        time_constants_causal, amplitudes_causal = zip(*self.causal_exponentials)
        time_constants_acausal, amplitudes_acausal = zip(*self.acausal_exponentials)

        t_post_minus_t_pre = np.array(t_post_minus_t_pre)
        time_deltas = np.abs(t_post_minus_t_pre)[..., np.newaxis]

        stdp_positive_delta = np.sum(
            amplitudes_causal * np.exp(-time_deltas / time_constants_causal),
            axis=-1,
        )
        stdp_negative_delta = np.sum(
            amplitudes_acausal * np.exp(-time_deltas / time_constants_acausal),
            axis=-1,
        )

        return (
            np.heaviside(t_post_minus_t_pre, 0) * stdp_positive_delta
            + np.heaviside(-t_post_minus_t_pre, 1) * stdp_negative_delta
        )


class Exponential(NamedTuple):
    time_constant: float
    amplitude: float

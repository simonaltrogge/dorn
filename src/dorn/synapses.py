from __future__ import annotations

from typing import Type, TypeVar, overload

import numpy as np
import numpy.typing as npt

from dorn._transform import to_neuronpopulation_and_length
from dorn.neuronpopulation import NeuronPopulation


class Synapses:
    def __init__(
        self,
        target: NeuronPopulation | int,
        source: NeuronPopulation | int,
        weights: npt.ArrayLike | None = None,
        *,
        connectivity: npt.ArrayLike | None = None,
        lower_bounds: npt.ArrayLike | None = 0.0,
        upper_bounds: npt.ArrayLike | None = None,
    ) -> None:
        self.target, target_length = to_neuronpopulation_and_length(target)
        self.source, source_length = to_neuronpopulation_and_length(source)
        self.shape = (target_length, source_length)
        self.weights = weights
        self.connectivity = connectivity
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def clip_to_bounds(self, ignore_connectivity=False) -> None:
        if not (self.lower_bounds is None and self.upper_bounds is None):
            np.clip(
                self.weights, self.lower_bounds, self.upper_bounds, out=self.weights
            )

        if ignore_connectivity:
            return

        self.weights *= self.connectivity

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self._weights

    @weights.setter
    def weights(self, value: npt.ArrayLike | None) -> None:
        if value is None:
            self._weights = np.zeros(self.shape, dtype=np.float64)
            return

        self._weights = self._to_full_matrix(value, dtype=np.float64)

    @property
    def connectivity(self) -> npt.NDArray[np.bool_]:
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value: npt.ArrayLike | None) -> None:
        if value is None:
            self._connectivity = np.ones(self.shape, dtype=np.bool_)
            np.fill_diagonal(self._connectivity, 0)
            return

        self._connectivity = self._to_full_matrix(value, dtype=np.bool_)

    @property
    def lower_bounds(self) -> npt.NDArray[np.float64] | None:
        return self._lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, value: npt.ArrayLike | None) -> None:
        if value is None:
            self._lower_bounds = None
            return

        self._lower_bounds = self._to_full_matrix(value, dtype=np.float64)

    @property
    def upper_bounds(self) -> npt.NDArray[np.float64] | None:
        return self._upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, value: npt.ArrayLike | None) -> None:
        if value is None:
            self._upper_bounds = None
            return

        self._upper_bounds = self._to_full_matrix(value, dtype=np.float64)

    @overload
    def _to_full_matrix(self, value: npt.ArrayLike, dtype: Type[T]) -> npt.NDArray[T]:
        ...

    @overload
    def _to_full_matrix(self, value: npt.ArrayLike, dtype: None) -> npt.NDArray:
        ...

    def _to_full_matrix(self, value, dtype=None):
        try:
            return np.full(self.shape, value, dtype)
        except ValueError as e:
            raise ValueError(
                f"could not broadcast input from shape {np.shape(value)} into shape"
                f"(target={self.shape[0]}, source={self.shape[1]})"
            ) from e


T = TypeVar("T", bound=np.generic)

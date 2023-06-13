from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Generic, Hashable, Literal, Protocol, TypeVar, Union

if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec
else:
    from typing import Concatenate, ParamSpec

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class HasTime(Protocol):
    time: float


TSpike = TypeVar("TSpike", bound=HasTime)


class Network(ABC, Generic[TSpike]):
    def __init__(self, *, time: float = 0.0, spike_count: int = 0) -> None:
        self.time = time
        self.spike_count = spike_count
        self.event_handlers: defaultdict[Event, list[Callable]] = defaultdict(list)
        self._cached_next_spike = None

    def simulate(self, duration: float) -> None:
        time_end = self.time + duration

        while self.time < time_end:
            next_spike = self._get_next_spike_with_cache(clear_cache=True)

            if next_spike.time > time_end:
                self._cached_next_spike = next_spike

                remaining_time = time_end - self.time
                self._emit_event("pre_time_evolution", remaining_time)
                self._evolve_in_time(remaining_time)
                self._emit_event("post_time_evolution", remaining_time)
                return

            interval = next_spike.time - self.time
            self._emit_event("pre_time_evolution", interval)
            self._evolve_in_time(interval)
            self._emit_event("post_time_evolution", interval)

            if self._is_valid_spike(next_spike):
                self._emit_event("pre_spike", next_spike)
                self._process_spike(next_spike)
                self._emit_event("post_spike", next_spike)

    def simulate_next_spike(self) -> None:
        next_spike = self._get_next_spike_with_cache(clear_cache=True)

        interval = next_spike.time - self.time
        self._emit_event("pre_time_evolution", interval)
        self._evolve_in_time(interval)
        self._emit_event("post_time_evolution", interval)

        if self._is_valid_spike(next_spike):
            self._emit_event("pre_spike", next_spike)
            self._process_spike(next_spike)
            self._emit_event("post_spike", next_spike)
        else:
            self.simulate_next_spike()

    def _get_next_spike_with_cache(self, *, clear_cache=False) -> TSpike:
        if self._cached_next_spike is not None:
            next_spike = self._cached_next_spike

            if clear_cache:
                self.clear_cache()

            return next_spike

        return self._get_next_spike()

    def clear_cache(self) -> None:
        self._cached_next_spike = None

    @abstractmethod
    def _get_next_spike(self) -> TSpike:
        raise NotImplementedError

    def _evolve_in_time(self, duration: float) -> None:
        self.time += duration

    def _is_valid_spike(self, spike: TSpike) -> bool:
        return True

    def _process_spike(self, spike: TSpike) -> None:
        self.spike_count += 1

    def add_event_handler(
        self, event: Event, handler: Callable[Concatenate[Self, P], object]
    ) -> None:
        self.event_handlers[event].append(handler)

        try:
            handler.init(self)
        except (AttributeError, TypeError):
            pass

    def remove_event_handler(
        self, event: Event, handler: Callable[Concatenate[Self, P], object]
    ) -> None:
        try:
            self.event_handlers[event].remove(handler)
        except ValueError:
            pass

    def _emit_event(self, event: Event, *args, **kwargs) -> None:
        for handler in self.event_handlers[event]:
            handler(self, *args, **kwargs)


Event = Union[
    Literal["pre_time_evolution", "post_time_evolution", "pre_spike", "post_spike"],
    Hashable,
]
P = ParamSpec("P")

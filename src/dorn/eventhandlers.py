from __future__ import annotations

import sys
from time import perf_counter
from typing import Callable, Generic, TypeVar

if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec
else:
    from typing import Concatenate, ParamSpec

from dorn.network import Network

TNetwork = TypeVar("TNetwork", bound=Network)
T = TypeVar("T")
P = ParamSpec("P")


def after_n_events(n: int, callback: Callable[P, T]) -> Callable[P, T | None]:
    event_count = 0

    def handler(*args: P.args, **kwargs: P.kwargs) -> T | None:
        nonlocal event_count
        if event_count == n:
            return

        event_count += 1
        if event_count == n:
            return callback(*args, **kwargs)

    return handler


def every_n_events(n: int, callback: Callable[P, T]) -> Callable[P, T | None]:
    event_count = 0

    def handler(*args: P.args, **kwargs: P.kwargs) -> T | None:
        nonlocal event_count
        event_count += 1

        if event_count == n:
            event_count = 0
            return callback(*args, **kwargs)

    return handler


class after_interval(Generic[TNetwork, P, T]):
    def __init__(
        self, interval: float, callback: Callable[Concatenate[TNetwork, P], T]
    ) -> None:
        self.interval = interval
        self.callback = callback
        self.target_time = None

    def __call__(
        self, network: TNetwork, *args: P.args, **kwargs: P.kwargs
    ) -> T | None:
        if self.target_time is None:
            return

        if network.time >= self.target_time:
            self.target_time = None
            return self.callback(network, *args, **kwargs)

    def init(self, network: TNetwork) -> None:
        self.target_time = network.time + self.interval


class every_interval(after_interval[TNetwork, P, T]):
    def __call__(
        self, network: TNetwork, *args: P.args, **kwargs: P.kwargs
    ) -> T | None:
        if self.target_time is None:
            return

        if network.time >= self.target_time:
            self.target_time += self.interval * (
                1 + (network.time - self.target_time) // self.interval
            )
            return self.callback(network, *args, **kwargs)


def after_real_time_interval(
    interval_in_seconds: float, callback: Callable[P, T]
) -> Callable[P, T | None]:
    target_time = perf_counter() + interval_in_seconds
    has_run = False

    def handler(*args: P.args, **kwargs: P.kwargs) -> T | None:
        nonlocal target_time
        nonlocal has_run

        if has_run:
            return

        current_time = perf_counter()
        if current_time >= target_time:
            has_run = True
            return callback(*args, **kwargs)

    return handler


def every_real_time_interval(
    interval_in_seconds: float, callback: Callable[P, T]
) -> Callable[P, T | None]:
    target_time = perf_counter() + interval_in_seconds

    def handler(*args: P.args, **kwargs: P.kwargs) -> T | None:
        nonlocal target_time
        current_time = perf_counter()

        if current_time >= target_time:
            target_time += interval_in_seconds * (
                1 + (current_time - target_time) // interval_in_seconds
            )
            return callback(*args, **kwargs)

    return handler

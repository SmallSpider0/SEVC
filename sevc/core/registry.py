"""Small explicit registry used by replaceable SEVC components."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable configuration keys to one canonical implementation."""

    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        self._items: dict[str, T] = {}

    def add(self, key: str, item: T) -> T:
        normalized = key.strip().lower()
        if not normalized or normalized != key:
            raise ValueError(f"invalid {self.component_name} key: {key!r}")
        if normalized in self._items:
            raise ValueError(
                f"duplicate {self.component_name} implementation key: {normalized}"
            )
        self._items[normalized] = item
        return item

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            return self.add(key, item)

        return decorator

    def get(self, key: str) -> T:
        normalized = key.strip().lower()
        try:
            return self._items[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"unknown {self.component_name} key {key!r}; available: {available}"
            ) from exc

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._items)

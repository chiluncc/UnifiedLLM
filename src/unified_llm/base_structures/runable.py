from __future__ import annotations

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import Self, Iterator, override
from copy import deepcopy


class Runable(ABC):
    def __init__(self) -> None:
        super().__init__()

    def __or__(self, value: Self) -> "RunableSequence" | NotImplementedType:
        if not isinstance(value, Runable):
            return NotImplemented
        return RunableSequence(self, value)
    
    @abstractmethod
    def invoke(self, params: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def stream(self, params: dict) -> Iterator[dict]:
        raise NotImplementedError


class RunableSequence(Runable):
    def __init__(self, *runables: Runable) -> None:
        super().__init__()
        self._runables = list(runables)

    def __or__(self, value: Self | Runable) -> Self | NotImplementedType:
        if isinstance(value, RunableSequence):
            return RunableSequence(*tuple(self._runables + value._runables))
        if isinstance(value, Runable):
            return RunableSequence(*tuple(self._runables + [value]))
        return NotImplemented

    @override
    def invoke(self, params: dict) -> dict:
        for runable in self._runables:
            params = runable.invoke(deepcopy(params))
        return deepcopy(params)

    @override
    def stream(self, params: dict) -> Iterator[dict]:
        for runable in self._runables:
            chunk: dict | None = None
            for chunk in runable.stream(deepcopy(params)):
                yield chunk
            if chunk is None:
                raise ValueError(f"{runable.__class__} return with None, input params: {params!r}")
            params = chunk
        yield deepcopy(params)
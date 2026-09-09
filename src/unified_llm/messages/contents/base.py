from abc import ABC, abstractmethod
from typing import Self, Any


class ContentException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class ContentBase(ABC):
    @abstractmethod
    def copy(self) -> Self: ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[str, Any]) -> Self: ...

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "values": None,
            }


class ContentSystemBase(ContentBase, ABC): ...


class ContentHumanBase(ContentBase, ABC): ...


class ContentAIBase(ContentBase, ABC): ...


class ContentToolBase(ContentBase, ABC): ...

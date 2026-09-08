from abc import ABC, abstractmethod
from typing import override


class ContentException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class ContentBase(ABC):
    @abstractmethod
    def copy(self) -> "ContentBase": ...


class ContentSystemBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentSystemBase": ...


class ContentHumanBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentHumanBase": ...


class ContentAIBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentAIBase": ...


class ContentToolBase(ContentBase, ABC):
    @override
    def copy(self) -> "ContentToolBase": ...

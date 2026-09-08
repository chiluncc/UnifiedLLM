from abc import ABC, abstractmethod
from typing import override, Any, Iterator
from copy import deepcopy

from .contents import (
    ContentAIBase,
    ContentAIToolCall,
    ContentBase,
    ContentHumanBase,
    ContentSystemBase,
    ContentToolBase,
)


class MessageException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class MessageBase(ABC):
    def __init__(self, contents: list[ContentBase] | ContentBase):
        super().__init__()
        self._contents: list[ContentBase] = [contents.copy()] if not isinstance(contents, list) else [c.copy() for c in contents]

    def __len__(self) -> int:
        return len(self._contents)
    
    def __getitem__(self, key):
        try:
            return self._contents[key].copy()
        except Exception as exc:
            raise MessageException(str(exc))
    
    def __iter__(self) -> Iterator[ContentBase]:
        return (c.copy() for c in self._contents)

    @abstractmethod
    def copy(self) -> "MessageBase": ...


################################


class MessageSystem(MessageBase):
    def __init__(self, contents: list[ContentSystemBase] | ContentSystemBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentSystemBase]:
        return super().__iter__()

    @override
    def copy(self) -> "MessageSystem":
        return MessageSystem(self._contents)


class MessageHuman(MessageBase):
    def __init__(self, contents: list[ContentHumanBase] | ContentHumanBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentHumanBase]:
        return super().__iter__()

    @override
    def copy(self) -> "MessageHuman":
        return MessageHuman(self._contents)


class MessageAI(MessageBase):
    def __init__(
        self,
        contents: list[ContentAIBase] | ContentAIBase,
        *,
        additions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(contents)
        self._additions: dict[str, Any] = dict() if additions is None else deepcopy(additions)
        if not self._contents:
            raise MessageException("MessageAI can't init: contents can't be empty")

    def __iter__(self) -> Iterator[ContentAIBase]:
        return super().__iter__()

    @override
    def copy(self) -> "MessageAI":
        return MessageAI(
            self._contents,
            additions=self._additions,
        )

    @property
    def additions(self) -> dict[str, Any]:
        return deepcopy(self._additions)


class MessageTool(MessageBase):
    def __init__(
        self,
        contents: list[ContentToolBase] | ContentToolBase,
        toolcall: ContentAIToolCall,
    ) -> None:
        super().__init__(contents)
        self._toolcall = toolcall

    def __iter__(self) -> Iterator[ContentToolBase]:
        return super().__iter__()

    @override
    def copy(self) -> "MessageTool":
        return MessageTool(self._contents, toolcall=self._toolcall.copy())

    @property
    def toolcall(self) -> ContentAIToolCall:
        return self._toolcall.copy()

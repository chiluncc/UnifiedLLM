from abc import ABC, abstractmethod
from typing import Any, Iterator, Self, override
from copy import deepcopy

from .contents import (
    ContentAIBase,
    ContentAIToolCall,
    ContentBase,
    ContentHumanBase,
    ContentSystemBase,
    ContentToolBase,
    contents_from_json,
    contents_to_json,
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
    def copy(self) -> Self: ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[str, Any]) -> Self: ...

    def to_json(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "contents": contents_to_json(self._contents),
            "values": None,
            }


################################


class MessageSystem(MessageBase):
    def __init__(self, contents: list[ContentSystemBase] | ContentSystemBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentSystemBase]:
        return super().__iter__()

    @override
    def copy(self) -> Self:
        return MessageSystem(self._contents)

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(contents_from_json(data["contents"]))


class MessageHuman(MessageBase):
    def __init__(self, contents: list[ContentHumanBase] | ContentHumanBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentHumanBase]:
        return super().__iter__()

    @override
    def copy(self) -> Self:
        return MessageHuman(self._contents)

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(contents_from_json(data["contents"]))


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
    def copy(self) -> Self:
        return MessageAI(
            self._contents,
            additions=self._additions,
        )

    @property
    def additions(self) -> dict[str, Any]:
        return deepcopy(self._additions)

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(
            contents_from_json(data["contents"]),
            additions=data["values"]["_additions"],
        )

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {
            "_additions": self._additions,
        }
        return json_block


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
    def copy(self) -> Self:
        return MessageTool(self._contents, toolcall=self._toolcall.copy())

    @property
    def toolcall(self) -> ContentAIToolCall:
        return self._toolcall.copy()

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(
            contents_from_json(data["contents"]),
            toolcall=ContentAIToolCall.from_json(data["values"]["_toolcall"]),
        )

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {
            "_toolcall": self._toolcall.to_json(),
        }
        return json_block


################################


_MESSAGE_JSON_CLASSES: tuple[type[MessageBase], ...] = (
    MessageSystem,
    MessageHuman,
    MessageAI,
    MessageTool,
)


def messages_to_json(messages: list[MessageBase]) -> list[dict]:
    if not isinstance(messages, list):
        raise MessageException(
            f"messages must be a list, got {type(messages).__name__}"
        )
    payload: list[dict] = []
    for message in messages:
        if not isinstance(message, MessageBase):
            raise MessageException(
                f"Expected MessageBase instance, got {type(message).__name__}"
            )
        payload.append(message.to_json())
    return payload


def messages_from_json(data: list[dict]) -> list[MessageBase]:
    if not isinstance(data, list):
        raise MessageException(
            f"messages json must be a list, got {type(data).__name__}"
        )
    restored: list[MessageBase] = []
    for item in data:
        if not isinstance(item, dict):
            raise MessageException(
                f"message json must be an object, got {type(item).__name__}"
            )
        cls = next(
            (cls for cls in _MESSAGE_JSON_CLASSES if cls.__name__ == item.get("class")),
            None,
        )
        if cls is None:
            raise MessageException(
                f"Unknown message class: {item.get('class')!r}"
            )
        restored.append(cls.from_json(item))
    return restored

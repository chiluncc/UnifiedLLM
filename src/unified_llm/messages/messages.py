from abc import ABC, abstractmethod
from typing import override, Any, Iterator
from copy import deepcopy

from .contents import ContentBase, ContentSystemBase, ContentHumanBase, ContentAIBase, ContentToolBase
from .contents import ContentReasoning, ContentToolCall


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


class SystemMessage(MessageBase):
    def __init__(self, contents: list[ContentSystemBase] | ContentSystemBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentSystemBase]:
        return super().__iter__()

    @override
    def copy(self) -> "SystemMessage":
        return SystemMessage(self._contents)


class HumanMessage(MessageBase):
    def __init__(self, contents: list[ContentHumanBase] | ContentHumanBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> Iterator[ContentHumanBase]:
        return super().__iter__()

    @override
    def copy(self) -> "HumanMessage":
        return HumanMessage(self._contents)


class AIMessage(MessageBase):
    def __init__(
        self,
        contents: list[ContentAIBase] | ContentAIBase,
        *,
        reasoning: ContentReasoning | None = None,
        toolcalls: list[ContentToolCall] | None = None,
        additions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(contents)
        self._reasoning: ContentReasoning | None = reasoning.copy() if reasoning is not None else None
        self._toolcalls: list[ContentToolCall] = list() if toolcalls is None else [t.copy() for t in toolcalls]
        self._additions: dict[str, Any] = dict() if additions is None else deepcopy(additions)
        if not self._contents and not self._toolcalls:
            raise MessageException(
                "AIMessage can't init: contents and toolcalls can't both be empty"
            )

    def __iter__(self) -> Iterator[ContentAIBase]:
        return super().__iter__()

    @override
    def copy(self) -> "AIMessage":
        return AIMessage(
            self._contents,
            reasoning=self._reasoning,
            toolcalls=self._toolcalls,
            additions=self._additions,
        )

    @property
    def reasoning(self) -> ContentReasoning | None:
        if self._reasoning is None:
            return None
        return self._reasoning.copy()

    @property
    def toolcalls(self) -> list[ContentToolCall]:
        return [t.copy() for t in self._toolcalls]

    @property
    def additions(self) -> dict[str, Any]:
        return deepcopy(self._additions)


class ToolMessage(MessageBase):
    def __init__(self, contents: list[ContentToolBase] | ContentToolBase, toolcall: ContentToolCall):
        super().__init__(contents)
        self._toolcall = toolcall

    def __iter__(self) -> Iterator[ContentToolBase]:
        return super().__iter__()

    @override
    def copy(self) -> "ToolMessage":
        return ToolMessage(self._contents, toolcall=self._toolcall.copy())

    @property
    def toolcall(self) -> ContentToolCall:
        return self._toolcall.copy()

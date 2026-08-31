from abc import ABC, abstractmethod
from typing import override, Any, Iterator

from .contents import ContentBase, ContentSystemBase, ContentHumanBase, ContentAIBase
from .contents import ContentReasoning


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


################################


class SystemMessage(MessageBase):
    def __init__(self, contents: list[ContentSystemBase] | ContentSystemBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> ContentSystemBase:
        return super().__iter__()


class HumanMessage(MessageBase):
    def __init__(self, contents: list[ContentHumanBase] | ContentHumanBase) -> None:
        super().__init__(contents)

    def __iter__(self) -> ContentHumanBase:
        return super().__iter__()


class AIMessage(MessageBase):
    def __init__(
        self,
        contents: list[ContentAIBase] | ContentAIBase,
        *,
        reasoning: ContentReasoning | None = None,
        additions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._contents: list[ContentAIBase] = [contents] if not isinstance(contents, list) else contents
        self._reasoning: ContentReasoning | None = reasoning
        self._additions: dict[str, Any] = additions if additions is not None else dict()

    def __iter__(self) -> ContentAIBase:
        return super().__iter__()

    @property
    def reasoning(self) -> ContentReasoning | None:
        return self._reasoning.copy()
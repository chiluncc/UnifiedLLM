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


################################


class ContentSystemText(ContentSystemBase):
    def __init__(self, content: str):
        super().__init__()
        self._text = content

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentSystemText":
        return ContentSystemText(self._text)


################################


class ContentHumanText(ContentHumanBase):
    def __init__(self, content: str):
        super().__init__()
        self._text = content

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentHumanText":
        return ContentHumanText(self._text)


################################


class ContentAIText(ContentAIBase):
    def __init__(self, content: str):
        super().__init__()
        self._text = content

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentAIText":
        return ContentAIText(self._text)


################################


class ContentReasoning(ContentBase):
    def __init__(self, *, reasoning_content: str | None = None, reasoning_summary: str | None = None):
        super().__init__()
        self._reasoning_content: str | None = reasoning_content
        self._reasoning_summary: str | None = reasoning_summary
        if self._reasoning_content is None and self._reasoning_summary is None:
            raise ContentException(
                "ContentAIReasoning can't init: at least one of reasoning_content or reasoning_summary must be provided"
            )

    @property
    def reasoning_content(self) -> str | None:
        return self._reasoning_content

    @property
    def reasoning_summary(self) -> str | None:
        return self._reasoning_summary

    @override
    def copy(self) -> "ContentReasoning":
        return ContentReasoning(
            reasoning_content=self._reasoning_content,
            reasoning_summary=self._reasoning_summary,
        )
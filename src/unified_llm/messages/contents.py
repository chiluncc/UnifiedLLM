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


################################


class ContentSystemText(ContentSystemBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentSystemText":
        return ContentSystemText(self._text)


################################


class ContentHumanText(ContentHumanBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentHumanText":
        return ContentHumanText(self._text)


################################


class ContentAIText(ContentAIBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

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


class ContentToolCall(ContentBase):
    def __init__(self, tool_name: str, tool_args: str, tool_id: str):
        super().__init__()
        self._tool_name: str = tool_name
        self._tool_args: str = tool_args
        self._tool_id: str = tool_id

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_args(self) -> str:
        return self._tool_args

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @override
    def copy(self) -> "ContentToolCall":
        return ContentToolCall(
            tool_name=self._tool_name,
            tool_args=self._tool_args,
            tool_id=self._tool_id,
        )


################################


class ContentToolText(ContentToolBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> "ContentToolText":
        return ContentToolText(self._text)

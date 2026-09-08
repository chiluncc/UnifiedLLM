from abc import ABC
from typing import override

from .base import ContentAIBase, ContentException


class ContentAIReasoningBase(ContentAIBase, ABC):
    @override
    def copy(self) -> "ContentAIReasoningBase": ...


class ContentAIReasoningText(ContentAIReasoningBase):
    def __init__(self, *, reasoning_content: str | None = None, reasoning_summary: str | None = None):
        super().__init__()
        self._reasoning_content: str | None = reasoning_content
        self._reasoning_summary: str | None = reasoning_summary
        if self._reasoning_content is None and self._reasoning_summary is None:
            raise ContentException(
                "ContentReasoningText can't init: at least one of reasoning_content or reasoning_summary must be provided"
            )

    @property
    def reasoning_content(self) -> str | None:
        return self._reasoning_content

    @property
    def reasoning_summary(self) -> str | None:
        return self._reasoning_summary

    @override
    def copy(self) -> "ContentAIReasoningText":
        return ContentAIReasoningText(
            reasoning_content=self._reasoning_content,
            reasoning_summary=self._reasoning_summary,
        )


class ContentAIText(ContentAIBase):
    def __init__(self, text: str, *, annotations: list[str] | None = None):
        super().__init__()
        self._text: str = text
        self._annotations = annotations if annotations is not None else []

    @property
    def text(self) -> str:
        return self._text

    @property
    def annotations(self) -> list[str]:
        return self._annotations.copy()

    @override
    def copy(self) -> "ContentAIText":
        return ContentAIText(self._text, annotations=self._annotations)


class ContentAIToolCall(ContentAIBase):
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
    def copy(self) -> "ContentAIToolCall":
        return ContentAIToolCall(
            tool_name=self._tool_name,
            tool_args=self._tool_args,
            tool_id=self._tool_id,
        )
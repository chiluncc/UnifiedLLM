from abc import ABC
from copy import deepcopy
from typing import Any, Literal, Self, override

from .base import ContentAIBase, ContentException


class ContentAIReasoningBase(ContentAIBase, ABC):
    @override
    def copy(self) -> Self: ...


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
    def copy(self) -> Self:
        return ContentAIReasoningText(
            reasoning_content=self._reasoning_content,
            reasoning_summary=self._reasoning_summary,
        )

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        values = data["values"]
        return cls(
            reasoning_content=values["_reasoning_content"],
            reasoning_summary=values["_reasoning_summary"],
        )

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {
            "_reasoning_content": self._reasoning_content,
            "_reasoning_summary": self._reasoning_summary,
        }
        return json_block


class ContentAIText(ContentAIBase):
    def __init__(self, text: str, *, annotations: list[dict[str, Any]] | None = None):
        super().__init__()
        self._text: str = text
        self._annotations: list[dict[str, Any]] = deepcopy(annotations) if annotations is not None else []

    @property
    def text(self) -> str:
        return self._text

    @property
    def annotations(self) -> list[dict[str, Any]]:
        return deepcopy(self._annotations)

    @override
    def copy(self) -> Self:
        return ContentAIText(self._text, annotations=self._annotations)

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        values = data["values"]
        return cls(text=values["_text"], annotations=values["_annotations"])

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {"_text": self._text, "_annotations": self._annotations}
        return json_block


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
    def copy(self) -> Self:
        return ContentAIToolCall(
            tool_name=self._tool_name,
            tool_args=self._tool_args,
            tool_id=self._tool_id,
        )

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        values = data["values"]
        return cls(
            tool_name=values["_tool_name"],
            tool_args=values["_tool_args"],
            tool_id=values["_tool_id"],
        )

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {
            "_tool_name": self._tool_name,
            "_tool_args": self._tool_args,
            "_tool_id": self._tool_id,
        }
        return json_block


class ContentAIServerBase(ContentAIBase, ABC): ...


class ContentAIServerWebSearchCall(ContentAIServerBase):
    def __init__(
        self,
        action_type: Literal["search", "open_page", "find_in_page"],
        *,
        queries: list[str] | None = None,
        url: str | None = None,
        pattern: str | None = None,
    ):
        super().__init__()
        if action_type == "search":
            if queries is None:
                raise ContentException(
                    "ContentAIServerWebSearchCall can't init: search requires queries"
                )
        elif action_type == "open_page":
            if url is None:
                raise ContentException(
                    "ContentAIServerWebSearchCall can't init: open_page requires url"
                )
        elif action_type == "find_in_page":
            if url is None or pattern is None:
                raise ContentException(
                    "ContentAIServerWebSearchCall can't init: find_in_page requires url and pattern"
                )
        else:
            raise ContentException(
                f"ContentAIServerWebSearchCall can't init: unknown action_type {action_type!r}"
            )

        self._action_type: Literal["search", "open_page", "find_in_page"] = action_type
        self._queries: list[str] | None = deepcopy(queries)
        self._url: str | None = url
        self._pattern: str | None = pattern

    @property
    def action_type(self) -> Literal["search", "open_page", "find_in_page"]:
        return self._action_type

    @property
    def queries(self) -> list[str] | None:
        return deepcopy(self._queries)

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def pattern(self) -> str | None:
        return self._pattern

    @override
    def copy(self) -> Self:
        return ContentAIServerWebSearchCall(
            self._action_type,
            queries=self._queries,
            url=self._url,
            pattern=self._pattern,
        )

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        values = data["values"]
        return cls(
            values["_action_type"],
            queries=values["_queries"],
            url=values["_url"],
            pattern=values["_pattern"],
        )

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {
            "_action_type": self._action_type,
            "_queries": self._queries,
            "_url": self._url,
            "_pattern": self._pattern,
        }
        return json_block

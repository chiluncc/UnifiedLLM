from .base import (
    ContentAIBase,
    ContentBase,
    ContentException,
    ContentHumanBase,
    ContentSystemBase,
    ContentToolBase,
)
from .system import ContentSystemText
from .human import ContentHumanText, ContentHumanImage
from .ai import ContentAIReasoningBase, ContentAIReasoningText, ContentAIText, ContentAIToolCall
from .tool import ContentToolText, ContentToolImage


_CONTENT_JSON_CLASSES: tuple[type[ContentBase], ...] = (
    ContentSystemText,
    ContentHumanText,
    ContentHumanImage,
    ContentAIReasoningText,
    ContentAIText,
    ContentAIToolCall,
    ContentToolText,
    ContentToolImage,
)


def contents_from_json(data: list[dict]) -> list[ContentBase]:
    if not isinstance(data, list):
        raise ContentException(
            f"contents json must be a list, got {type(data).__name__}"
        )
    restored: list[ContentBase] = []
    for item in data:
        if not isinstance(item, dict):
            raise ContentException(
                f"content json must be an object, got {type(item).__name__}"
            )
        cls = next(
            (cls for cls in _CONTENT_JSON_CLASSES if cls.__name__ == item.get("class")),
            None,
        )
        if cls is None:
            raise ContentException(
                f"Unknown content class: {item.get('class')!r}"
            )
        restored.append(cls.from_json(item))
    return restored


def contents_to_json(contents: list[ContentBase]) -> list[dict]:
    if not isinstance(contents, list):
        raise ContentException(
            f"contents must be a list, got {type(contents).__name__}"
        )
    payload: list[dict] = []
    for content in contents:
        if not isinstance(content, ContentBase):
            raise ContentException(
                f"Expected ContentBase instance, got {type(content).__name__}"
            )
        payload.append(content.to_json())
    return payload


__all__ = [
    "ContentAIBase",
    "ContentAIReasoningBase",
    "ContentAIReasoningText",
    "ContentAIText",
    "ContentAIToolCall",
    "ContentBase",
    "ContentException",
    "ContentHumanBase",
    "ContentHumanImage",
    "ContentHumanText",
    "ContentSystemBase",
    "ContentSystemText",
    "ContentToolBase",
    "ContentToolImage",
    "ContentToolText",

    "contents_from_json",
    "contents_to_json",
]

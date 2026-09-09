from typing import Self, override, Any

from .base import ContentSystemBase


class ContentSystemText(ContentSystemBase):
    def __init__(self, text: str):
        super().__init__()
        self._text: str = text

    @property
    def text(self) -> str:
        return self._text

    @override
    def copy(self) -> Self:
        return ContentSystemText(self._text)

    @classmethod
    @override
    def from_json(cls, data: dict[str, Any]) -> Self:
        return ContentSystemText(data["values"]["_text"])

    @override
    def to_json(self) -> dict[str, Any]:
        json_block = super().to_json()
        json_block["values"] = {"_text": self._text}
        return json_block
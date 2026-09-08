from typing import override

from .base import ContentSystemBase


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

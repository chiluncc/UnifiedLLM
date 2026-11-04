import filetype
import base64
from pathlib import Path
from typing import Literal, Self, Any
from pydantic import BaseModel, Field, model_validator


class BaseContent(BaseModel):
    ...


class ToolCallContent(BaseContent):
    tool_name: str
    tool_args: str
    tool_call_id: str


class ReasoningContent(BaseContent):
    reasoning: str
    summary: str
    reasoning_id: str | None = Field(default=None)


class TextContent(BaseContent):
    text: str


class ImageContent(BaseContent):
    source_type: Literal["data", "link"]
    mime_type: str
    base64: str | None = Field(default=None)
    url: str | None = Field(default=None)

    @model_validator(mode="after")
    def after_validate(self) -> Self:
        if not ((self.base64 is not None) ^ (self.url is not None)):
            raise ValueError(
                "Base64 and URL must be at least one of them "
                "and cannot be passed in simultaneously."
            )
        if self.source_type == "data" and self.base64 is None or self.source_type == "link" and self.url is None:
            raise ValueError("Kind and value not match.")
        return self

    @classmethod
    def from_local(cls, path: Path | str) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image path {path} not exists.")
        mime_typ = filetype.guess(path).mime
        base64_value = base64.b64encode(path.read_bytes()).decode("ascii")
        return cls(
            source_type="data",
            mime_type=mime_typ,
            base64=base64_value,
        )


class FileContent(BaseContent):
    file_id: str
    mime_type: str


class CustomContent(BaseContent):
    values: dict[str, Any] = Field(default_factory=dict)

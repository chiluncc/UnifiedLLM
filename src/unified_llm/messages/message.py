from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

from .content import BaseContent, ReasoningContent, TextContent, ImageContent, FileContent, CustomContent, ToolCallContent


class MetaData(BaseModel):
    token_input: int | None = Field(default=None)
    token_output: int | None = Field(default=None)
    token_all: int | None = Field(default=None)
    token_hited: int | None = Field(default=None)

    provider: str = Field(default="unknown")
    context_compatibility_group: str = Field(default="unknown")
    
    response_id: str = Field(default="unknown")
    stop_reason: Literal["stop", "interrupt", "max_token", "unknown_error"] = Field(default="stop")


class BaseMessage(BaseModel):
    contents: list[BaseContent] = Field(default_factory=list)
    addition: dict[str, Any] = Field(default_factory=dict)


class SystemMessage(BaseMessage):
    contents: list[TextContent] = Field(...)


class HumanMessage(BaseMessage):
    contents: list[TextContent | ImageContent | FileContent | CustomContent] = Field(...)


class AIMessage(BaseMessage):
    message_id: str
    contents: list[ReasoningContent | TextContent | ImageContent | FileContent | ToolCallContent | CustomContent] = Field(...)
    original: Any = Field(exclude=True)
    metadata: MetaData = Field(default_factory=MetaData)


class ToolMessage(BaseMessage):
    tool_call_content: ToolCallContent
    contents: list[TextContent | ImageContent | FileContent | CustomContent] = Field(...)
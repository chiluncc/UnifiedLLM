import uuid
from pydantic import BaseModel, Field


class StreamChunkBase(BaseModel, frozen=True): ...


class StreamChunkEmpty(StreamChunkBase):
    done: bool = Field(default=False)


class StreamChunkReasoning(StreamChunkBase):
    text: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class StreamChunkText(StreamChunkBase):
    text: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class StreamChunkToolCall(StreamChunkBase):
    index: int
    whole: bool = False
    tool_id: str | None = None
    tool_name: str | None = None
    tool_args: str | None = None


class StreamChunkWebSearch(StreamChunkBase):
    queries: list[str]
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class StreamChunkWebOpenPage(StreamChunkBase):
    url: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class StreamChunkWebFindInPage(StreamChunkBase):
    url: str
    pattern: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
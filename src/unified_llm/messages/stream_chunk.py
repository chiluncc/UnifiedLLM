from pydantic import BaseModel


class StreamChunkBase(BaseModel, frozen=True): ...


class StreamChunkEmpty(StreamChunkBase):
    pass


class StreamChunkReasoning(StreamChunkBase):
    text: str


class StreamChunkText(StreamChunkBase):
    text: str


class StreamChunkToolCall(StreamChunkBase):
    index: int
    finish: bool = False
    tool_id: str | None = None
    tool_name: str | None = None
    tool_args: str | None = None

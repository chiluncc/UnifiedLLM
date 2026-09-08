import queue
import threading
from dataclasses import dataclass
from typing import TypedDict, Iterator, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel

from unified_llm.messages.messages import MessageBase
from unified_llm.messages.messages import MessageAI
from unified_llm.messages.contents import ContentAIText, ContentAIReasoningText, ContentAIToolCall
from unified_llm.messages.stream_chunks import StreamChunkBase, StreamChunkEmpty
from unified_llm.tools import ToolExecutorBase, Tool


class ClientException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class TokenExpense(TypedDict):
    token_input: int | None
    token_output: int | None
    token_cached: int | None
    token_expense: float | None


@dataclass(frozen=True, slots=True)
class ClientResultInner:
    messages: list[MessageBase]
    expense: TokenExpense


@dataclass(frozen=True, slots=True)
class ClientResult:
    messages: list[MessageBase]
    expense: TokenExpense

    def _last_content(self, types: Any) -> list:
        if self.messages:
            last = self.messages[-1]
            if isinstance(last, MessageAI):
                type_filter = tuple(types) if isinstance(types, list) else (types,)
                return [content for content in last if isinstance(content, type_filter)]
        return []

    def get_reasonings(self) -> list[ContentAIReasoningText]:
        return self._last_content(ContentAIReasoningText)

    def get_contents(self) -> list[ContentAIText]:
        return self._last_content(ContentAIText)

    def get_toolcalls(self) -> list[ContentAIToolCall]:
        return self._last_content(ContentAIToolCall)


class ClientExecutor(ABC):
    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._result: ClientResult | None = None
        self._exception: Exception | None = None
        self._thread = threading.Thread(target=self._thread_container, daemon=True)

    def __iter__(self) -> Iterator[StreamChunkBase]:
        return self

    def __next__(self) -> StreamChunkBase:
        if self._done_event.is_set() and self._queue.empty():
            raise StopIteration
        try:
            return self._queue.get(timeout=0.1)
        except queue.Empty:
            return StreamChunkEmpty()

    def done(self) -> bool:
        return self._done_event.is_set()

    def cancel(self) -> bool:
        if self._done_event.is_set():
            return False
        self._cancel_event.set()
        return True

    def result(self) -> ClientResult | None:
        return self._result

    def exception(self) -> Exception | None:
        return self._exception

    def _thread_start(self) -> None:
        self._thread.start()

    def _thread_container(self) -> None:
        try:
            self._result = self._thread_content()
        except Exception as exc:
            self._exception = exc
        finally:
            self._done_event.set()

    def _push_chunk(self, chunk: StreamChunkBase) -> None:
        self._queue.put(chunk)

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @abstractmethod
    def _thread_content(self) -> ClientResult: ...


class ClientConfigBase(BaseModel, ABC, frozen=True):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class RequestConfigBase(BaseModel, ABC, frozen=True):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class ClientBase(ABC):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase) -> None:
        super().__init__()
        self._client_config = client_config
        self._request_config = request_config
        self._tools: list = []

    def set_tools(self, tools: ToolExecutorBase | list[Tool] | None = None) -> None:
        if isinstance(tools, ToolExecutorBase):
            self._tools = [self._serialize_tool(t) for t in tools.list_tools()]
        elif isinstance(tools, list) and all([isinstance(t, Tool) for t in tools]):
            self._tools = [self._serialize_tool(t) for t in tools]
        else:
            self._tools = []

    def get_tools(self) -> list:
        return self._tools

    def set_request_config(self, request_config: RequestConfigBase) -> None:
        self._request_config = request_config

    def get_request_config(self) -> RequestConfigBase:
        return self._request_config
    
    def get_client_config(self) -> ClientConfigBase:
        return self._client_config
        
    @abstractmethod
    def invoke(self, messages: list[MessageBase]) -> ClientResult: ...

    @abstractmethod
    async def ainvoke(self, messages: list[MessageBase]) -> ClientResult: ...

    @abstractmethod
    def execute(self, messages: list[MessageBase]) -> ClientExecutor: ...

    @abstractmethod
    def _serialize_tool(self, tool: Tool) -> Any: ...

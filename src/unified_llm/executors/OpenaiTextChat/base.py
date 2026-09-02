from __future__ import annotations

import inspect
import openai
from abc import ABC, abstractmethod
from docstring_parser import Style
from docstring_parser import parse as parse_docstring
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk
from typing import override, Callable

from unified_llm.messages.messages import MessageBase
from unified_llm.messages.stream_chunk import StreamChunkBase, StreamChunkReasoning, StreamChunkText, StreamChunkToolCall
from unified_llm.tools import Tool
from ..base import ClientBase, ClientConfigBase, RequestConfigBase
from ..base import ClientResult, ClientExecutor, ClientException


class OpenAITextChatClientExecutor(ClientExecutor):
    def __init__(self, client: OpenAITextChatClientBase, messages: list[MessageBase]):
        super().__init__()
        self._client: openai.Client = client._client
        self._request_config: dict = client.get_request_config().to_dict()
        self._request_config.update({"stream": True, "stream_options": {"include_usage": True}})
        self._messages: list = client._unserialize_messages(messages)
        self._conversation: list[MessageBase] = messages
        self._tools: list = client.get_tools()
        self._parse_stream_chunk: Callable[[ChatCompletionChunk], list[StreamChunkBase]] = client._parse_stream_chunk
        self._parse_stream_chunk_full: Callable[[list[ChatCompletionChunk]], ClientResult] = client._parse_stream_chunk_full
        self._thread_start()

    def _thread_content(self) -> ClientResult:
        if self._cancelled():
            raise ClientException("Stream execution cancelled")

        stream = self._client.chat.completions.create(
            messages=self._messages,
            tools=self._tools,
            **self._request_config,
        )

        raw_chunks: list[ChatCompletionChunk] = []
        reasoning: str = ""
        text: str = ""
        pending_toolcalls: dict[int, dict[str, str | None]] = {}

        def _flush_toolcalls() -> None:
            for index in sorted(pending_toolcalls):
                partial = pending_toolcalls[index]
                self._push_chunk(
                    StreamChunkToolCall(
                        index=index,
                        tool_id=partial["tool_id"],
                        tool_name=partial["tool_name"],
                        tool_args=partial["tool_args"],
                    )
                )
            pending_toolcalls.clear()

        for raw in stream:
            if self._cancelled():
                raise ClientException("Stream execution cancelled")
            raw_chunks.append(raw)
            for fragment in self._parse_stream_chunk(raw):
                match fragment:
                    case StreamChunkReasoning():
                        if fragment.text:
                            reasoning += fragment.text
                            self._push_chunk(StreamChunkReasoning(text=reasoning))
                    case StreamChunkText():
                        if fragment.text:
                            text += fragment.text
                            self._push_chunk(StreamChunkText(text=text))
                    case StreamChunkToolCall() as toolcall:
                        partial = pending_toolcalls.setdefault(
                            toolcall.index,
                            {"tool_id": None, "tool_name": None, "tool_args": ""},
                        )
                        if toolcall.tool_id is not None:
                            partial["tool_id"] = toolcall.tool_id
                        if toolcall.tool_name is not None:
                            partial["tool_name"] = toolcall.tool_name
                        if toolcall.tool_args:
                            partial["tool_args"] += toolcall.tool_args
                        if toolcall.finish:
                            _flush_toolcalls()
                    case _:
                        pass

        _flush_toolcalls()
        result = self._parse_stream_chunk_full(raw_chunks)
        return ClientResult(
            messages=self._conversation + result["messages"],
            expense=result["expense"],
        )


class OpenAITextChatClientBase(ClientBase, ABC):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)
        self._client = openai.Client(**client_config.to_dict())
        
    @override
    def invoke(self, messages: list[MessageBase]) -> ClientResult:
        response = self._client.chat.completions.create(
            messages=self._unserialize_messages(messages),
            tools=self.get_tools(),
            **self.get_request_config().to_dict(),
        )
        result = self._parse_response(response)
        return ClientResult(
            messages=messages + result["messages"],
            expense=result["expense"],
        )

    @override
    async def ainvoke(self, messages: list[MessageBase]) -> ClientResult:
        async with openai.AsyncClient(**self.get_client_config().to_dict()) as async_client:
            response = await async_client.chat.completions.create(
                messages=self._unserialize_messages(messages),
                tools=self.get_tools(),
                **self.get_request_config().to_dict(),
            )
        result = self._parse_response(response)
        return ClientResult(
            messages=messages + result["messages"],
            expense=result["expense"],
        )

    @override
    def execute(self, messages: list[MessageBase]) -> ClientExecutor:
        return OpenAITextChatClientExecutor(self, messages)

    @override
    def _serialize_tool(self, tool: Tool) -> dict:
        doc = parse_docstring(inspect.getdoc(tool.func) or "", style=Style.GOOGLE)
        description = "\n".join(part for part in (doc.short_description, doc.long_description) if part)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": description,
                "parameters": tool.args.model_json_schema(),
            },
        }
    
    @abstractmethod
    def _unserialize_messages(self, messages: list[MessageBase]) -> list[ChatCompletionMessageParam]: ...

    @abstractmethod
    def _parse_response(self, response: ChatCompletion) -> ClientResult: ...

    @abstractmethod
    def _parse_stream_chunk(self, response_chunk: ChatCompletionChunk) -> list[StreamChunkBase]: ...

    @abstractmethod
    def _parse_stream_chunk_full(self, response_chunks: list[ChatCompletionChunk]) -> ClientResult: ...

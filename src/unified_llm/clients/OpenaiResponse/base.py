from __future__ import annotations

import inspect
import uuid
import openai
from abc import ABC, abstractmethod
from docstring_parser import Style
from docstring_parser import parse as parse_docstring
from typing import Any, override, Callable

from openai.types.responses import (
    Response,
    ResponseInputParam,
    ResponseStreamEvent,
    ResponseFailedEvent,
    ResponseErrorEvent,
)

from unified_llm.messages import (
    MessageBase,
    StreamChunkBase,
    StreamChunkEmpty,
    StreamChunkReasoning,
    StreamChunkText,
    StreamChunkToolCall,
)
from unified_llm.tools import Tool
from ..base import (
    ClientBase,
    ClientConfigBase,
    RequestConfigBase,
    ClientResult,
    ClientResultInner,
    ClientExecutor,
    ClientException,
)


class OpenAIResponseClientExecutor(ClientExecutor):
    def __init__(self, client: OpenAIResponseClientBase, messages: list[MessageBase]):
        super().__init__()
        self._client: openai.Client = client._client
        self._request_config: dict[str, Any] = client._request_params()
        self._request_config.update({"stream": True})
        self._input: ResponseInputParam = client._serialize_input(messages)
        self._conversation: list[MessageBase] = messages
        self._tools: list[Any] = client.get_tools()
        self._parse_stream_event: Callable[
            [ResponseStreamEvent], list[StreamChunkBase]
        ] = client._parse_stream_event
        self._parse_stream_events_full: Callable[
            [list[ResponseStreamEvent]], ClientResultInner
        ] = client._parse_stream_events_full
        self._thread_start()

    def _thread_content(self) -> ClientResult:
        if self._cancelled():
            raise ClientException("Stream execution cancelled")

        stream = self._client.responses.create(
            input=self._input,
            tools=self._tools,
            **self._request_config,
        )

        raw_events: list[ResponseStreamEvent] = []
        pending_toolcalls: dict[int, dict[str, str | None]] = {}
        current_kind: type | None = None
        run_uuid: str | None = None

        def _flush_toolcalls() -> None:
            for index in sorted(pending_toolcalls):
                partial = pending_toolcalls[index]
                self._push_chunk(
                    StreamChunkToolCall(
                        index=index,
                        whole=True,
                        tool_id=partial["tool_id"],
                        tool_name=partial["tool_name"],
                        tool_args=partial["tool_args"],
                    )
                )
            pending_toolcalls.clear()

        for event in stream:
            if self._cancelled():
                raise ClientException("Stream execution cancelled")
            if isinstance(event, ResponseFailedEvent):
                raise ClientException(
                    f"Response stream failed: {event.response.error}"
                )
            if isinstance(event, ResponseErrorEvent):
                raise ClientException(f"Response stream error: {event.error}")
            raw_events.append(event)
            for fragment in self._parse_stream_event(event):
                match fragment:
                    case StreamChunkReasoning() as fragment:
                        if fragment.text:
                            if current_kind is not StreamChunkReasoning:
                                current_kind = StreamChunkReasoning
                                run_uuid = str(uuid.uuid4())
                            self._push_chunk(
                                fragment.model_copy(update={"uuid": run_uuid})
                            )
                    case StreamChunkText() as fragment:
                        if fragment.text:
                            if current_kind is not StreamChunkText:
                                current_kind = StreamChunkText
                                run_uuid = str(uuid.uuid4())
                            self._push_chunk(
                                fragment.model_copy(update={"uuid": run_uuid})
                            )
                    case StreamChunkToolCall() as toolcall:
                        current_kind = StreamChunkToolCall
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
                    case StreamChunkEmpty():
                        if fragment.done:
                            _flush_toolcalls()
                            current_kind = None
                            run_uuid = None
                    case _:
                        pass

        _flush_toolcalls()
        result = self._parse_stream_events_full(raw_events)
        return ClientResult(
            messages=self._conversation + result.messages,
            expense=result.expense,
        )


class OpenAIResponseClientBase(ClientBase, ABC):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)
        self._client = openai.Client(**client_config.to_dict())
        self._async_client = openai.AsyncClient(**client_config.to_dict())
        self._instructions: str | None = None

    def set_instructions(self, instructions: str | None) -> None:
        self._instructions = instructions

    def get_instructions(self) -> str | None:
        return self._instructions

    def _request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = self.get_request_config().to_dict()
        instructions = self.get_instructions()
        if instructions is not None:
            params["instructions"] = instructions
        return params

    @override
    def invoke(self, messages: list[MessageBase]) -> ClientResult:
        response = self._client.responses.create(
            input=self._serialize_input(messages),
            tools=self.get_tools(),
            **self._request_params(),
        )
        result = self._parse_response(response)
        return ClientResult(
            messages=messages + result.messages,
            expense=result.expense,
        )

    @override
    async def ainvoke(self, messages: list[MessageBase]) -> ClientResult:
        response = await self._async_client.responses.create(
            input=self._serialize_input(messages),
            tools=self.get_tools(),
            **self._request_params(),
        )
        result = self._parse_response(response)
        return ClientResult(
            messages=messages + result.messages,
            expense=result.expense,
        )

    @override
    def execute(self, messages: list[MessageBase]) -> OpenAIResponseClientExecutor:
        return OpenAIResponseClientExecutor(self, messages)

    @override
    def _serialize_tool(self, tool: Tool) -> dict:
        doc = parse_docstring(inspect.getdoc(tool.func) or "", style=Style.GOOGLE)
        description = "\n".join(
            part for part in (doc.short_description, doc.long_description) if part
        )
        return {
            "type": "function",
            "name": tool.name,
            "description": description,
            "parameters": tool.args.model_json_schema(),
        }
    
    @abstractmethod
    def _serialize_input(self, messages: list[MessageBase]) -> ResponseInputParam: ...

    @abstractmethod
    def _parse_response(self, response: Response) -> ClientResultInner: ...

    @abstractmethod
    def _parse_stream_event(self, response_event: ResponseStreamEvent) -> list[StreamChunkBase]: ...

    @abstractmethod
    def _parse_stream_events_full(self, response_events: list[ResponseStreamEvent]) -> ClientResultInner: ...


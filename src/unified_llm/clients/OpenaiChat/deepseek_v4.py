from datetime import datetime, time
from pydantic import Field
from typing import Any, override, Literal
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk

from .base import OpenAIChatClientBase
from ..base import ClientConfigBase, RequestConfigBase, TokenExpense, ClientException, ClientResultInner
from unified_llm.messages import (
    MessageBase,
    MessageAI,
    MessageHuman,
    MessageSystem,
    MessageTool,
    StreamChunkBase,
    StreamChunkEmpty,
    StreamChunkReasoning,
    StreamChunkText,
    StreamChunkToolCall,
    ContentAIBase,
    ContentAIText,
    ContentAIReasoningText,
    ContentAIToolCall,
    ContentHumanText,
    ContentHumanImage,
    ContentSystemText,
    ContentToolText,
    ContentToolImage,
)


class DeepSeekV4ClientConfig(ClientConfigBase):
    api_key: str
    base_url: str = Field(default="https://api.deepseek.com")

    @override
    def to_dict(self) -> dict[str, Any]:
        return {"api_key": self.api_key, "base_url": self.base_url}


class DeepSeekV4RequestConfig(RequestConfigBase):
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = Field(default="deepseek-flash")
    reasoning_effort: Literal["none", "low", "high", "max"] = Field(default="high")

    @override
    def to_dict(self) -> dict[str, Any]:
        if self.reasoning_effort == "none":
            return {
                "model": self.model,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        else:
            return {
                "model": self.model,
                "extra_body": {"thinking": {"type": "enabled"}},
                "reasoning_effort": self.reasoning_effort,
            }


class OpenAIChatClientDeepSeekV4(OpenAIChatClientBase):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)

    def _serialize_response(self, response: ChatCompletion) -> list[MessageBase]:
        if not response.choices:
            return []
        raw = response.choices[0].message

        contents: list[ContentAIBase] = []
        reasoning_content = getattr(raw, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            contents.append(ContentAIReasoningText(reasoning_content=reasoning_content))
        if raw.content is not None:
            contents.append(ContentAIText(raw.content))
        for tc in raw.tool_calls or []:
            contents.append(
                ContentAIToolCall(
                    tool_name=tc.function.name,
                    tool_args=tc.function.arguments,
                    tool_id=tc.id,
                )
            )

        return [MessageAI(contents=contents)] if contents else []

    def _compute_expense(self, model: str, usage) -> TokenExpense:
        _DEEPSEEK_V4_PRICES: dict[str, dict[str, tuple[float, float, float]]] = {
            "deepseek-v4-flash": {
                "peak": (0.04, 2.00, 8.00),
                "offpeak": (0.02, 1.00, 4.00),
            },
            "deepseek-v4-pro": {
                "peak": (0.30, 9.00, 27.00),
                "offpeak": (0.15, 4.50, 13.50),
            },
        }

        def _is_peak_period(now: datetime) -> bool:
            if now.weekday() >= 5:
                return False
            return time(9, 0) <= now.time() < time(12, 0) or time(14, 0) <= now.time() < time(18, 0)

        if usage is None:
            return TokenExpense(token_input=None, token_output=None, token_cached=None, token_expense=None)

        token_input = usage.prompt_tokens
        token_output = usage.completion_tokens
        token_cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if token_cached is None and usage.prompt_tokens_details is not None:
            token_cached = usage.prompt_tokens_details.cached_tokens

        token_expense: float | None = None
        prices = _DEEPSEEK_V4_PRICES.get(model)
        if prices is not None:
            cache_hit_price, cache_miss_price, output_price = prices[
                "peak" if _is_peak_period(datetime.now()) else "offpeak"
            ]
            cached = token_cached if token_cached is not None else 0
            token_expense = (
                cached * cache_hit_price
                + max(token_input - cached, 0) * cache_miss_price
                + token_output * output_price
            ) / 1_000_000

        return TokenExpense(
            token_input=token_input,
            token_output=token_output,
            token_cached=token_cached,
            token_expense=token_expense,
        )

    @override
    def _unserialize_messages(self, messages: list[MessageBase]) -> list[ChatCompletionMessageParam]:
        unserialized: list[ChatCompletionMessageParam] = []
        for message in messages:
            match message:
                case MessageSystem():
                    if len(message) != 1:
                        raise ClientException("SystemMessage only supports ContentSystemText")
                    content = message[0]
                    if not isinstance(content, ContentSystemText):
                        raise ClientException("SystemMessage only supports ContentSystemText")
                    unserialized.append({"role": "system", "content": content.text})
                case MessageHuman():
                    content_parts: list[dict[str, Any]] = []
                    for content in message:
                        match content:
                            case ContentHumanText():
                                content_parts.append({"type": "text", "text": content.text})
                            case ContentHumanImage():
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": content.img_url,
                                            "detail": content.detail,
                                        },
                                    }
                                )
                            case _:
                                raise ClientException(
                                    "HumanMessage only supports ContentHumanText/ContentHumanImage, "
                                    f"got {type(content).__name__}"
                                )
                    if not content_parts:
                        raise ClientException("HumanMessage can't be empty")
                    unserialized.append({"role": "user", "content": content_parts})
                case MessageAI():
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    toolcalls: list[ContentAIToolCall] = []
                    for content in message:
                        match content:
                            case ContentAIText():
                                text_parts.append(content.text)
                            case ContentAIReasoningText():
                                if content.reasoning_content is not None:
                                    reasoning_parts.append(content.reasoning_content)
                            case ContentAIToolCall():
                                toolcalls.append(content)
                            case _:
                                raise ClientException(
                                    "AIMessage only supports ContentAIText/ContentAIReasoningText/ContentAIToolCall, "
                                    f"got {type(content).__name__}"
                                )
                    if not text_parts and not toolcalls:
                        raise ClientException("AIMessage has empty content and no tool_calls")
                    assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
                    if text_parts:
                        assistant_msg["content"] = "".join(text_parts)
                    if reasoning_parts:
                        assistant_msg["reasoning_content"] = "".join(reasoning_parts)
                    if toolcalls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.tool_id,
                                "type": "function",
                                "function": {"name": tc.tool_name, "arguments": tc.tool_args},
                            }
                            for tc in toolcalls
                        ]
                    unserialized.append(assistant_msg)
                case MessageTool():
                    content_parts: list[dict[str, Any]] = []
                    for content in message:
                        match content:
                            case ContentToolText():
                                content_parts.append({"type": "text", "text": content.text})
                            case ContentToolImage():
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": content.img_url,
                                            "detail": content.detail,
                                        },
                                    }
                                )
                            case _:
                                raise ClientException(
                                    "ToolMessage only supports ContentToolText/ContentToolImage, "
                                    f"got {type(content).__name__}"
                                )
                    if not content_parts:
                        raise ClientException("ToolMessage can't be empty")
                    unserialized.append(
                        {
                            "role": "tool",
                            "content": content_parts,
                            "tool_call_id": message.toolcall.tool_id,
                        }
                    )
                case _:
                    raise ClientException(f"Unsupported message type: {type(message).__name__}")
        return unserialized

    @override
    def _parse_response(self, response: ChatCompletion) -> ClientResultInner:
        return ClientResultInner(
            messages=self._serialize_response(response),
            expense=self._compute_expense(response.model, response.usage),
        )

    @override
    def _parse_stream_chunk(self, response_chunk: ChatCompletionChunk) -> list[StreamChunkBase]:
        if not response_chunk.choices:
            return []
        choice = response_chunk.choices[0]
        fragments: list[StreamChunkBase] = []

        reasoning = getattr(choice.delta, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            fragments.append(StreamChunkReasoning(text=reasoning))
        if choice.delta.content:
            fragments.append(StreamChunkText(text=choice.delta.content))
        for tc in choice.delta.tool_calls or []:
            function = tc.function
            fragments.append(
                StreamChunkToolCall(
                    index=tc.index,
                    tool_id=tc.id,
                    tool_name=function.name if function else None,
                    tool_args=function.arguments if function else None,
                )
            )
        if choice.finish_reason is not None:
            fragments.append(StreamChunkEmpty(done=True))
        return fragments

    @override
    def _parse_stream_chunk_full(self, response_chunks: list[ChatCompletionChunk]) -> ClientResultInner:
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        toolcalls: dict[int, dict[str, str | None]] = {}
        model: str | None = None
        usage = None

        for chunk in response_chunks:
            if model is None and chunk.model:
                model = chunk.model
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
            if delta.content:
                text_parts.append(delta.content)
            for tc in delta.tool_calls or []:
                partial = toolcalls.setdefault(
                    tc.index,
                    {"tool_id": None, "tool_name": None, "tool_args": ""},
                )
                if tc.id:
                    partial["tool_id"] = tc.id
                function = tc.function
                if function:
                    if function.name:
                        partial["tool_name"] = function.name
                    if function.arguments:
                        partial["tool_args"] += function.arguments

        reasoning_content = "".join(reasoning_parts)
        contents: list[ContentAIBase] = []
        if reasoning_content:
            contents.append(ContentAIReasoningText(reasoning_content=reasoning_content))
        if text_parts:
            contents.append(ContentAIText("".join(text_parts)))
        for _, partial in sorted(toolcalls.items()):
            contents.append(
                ContentAIToolCall(
                    tool_name=partial["tool_name"] or "",
                    tool_args=partial["tool_args"],
                    tool_id=partial["tool_id"] or "",
                )
            )
        messages = [MessageAI(contents=contents)] if contents else []
        return ClientResultInner(
            messages=messages,
            expense=self._compute_expense(model or "", usage),
        )

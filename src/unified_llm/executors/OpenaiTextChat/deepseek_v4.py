from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, override, Literal
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .base import OpenAITextChatClientBase
from ..base import ClientConfigBase, RequestConfigBase, TokenExpense, ClientException
from unified_llm.messages.messages import MessageBase
from unified_llm.messages.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from unified_llm.messages.contents import ContentAIText
from unified_llm.messages.contents import ContentHumanText
from unified_llm.messages.contents import ContentSystemText
from unified_llm.messages.contents import ContentToolText
from unified_llm.messages.contents import ContentReasoning, ContentToolCall


@dataclass
class DeepSeekV4ClientConfig(ClientConfigBase):
    api_key: str
    base_url: str = field(default="https://api.deepseek.com")

    @override
    def to_dict(self) -> dict[str, Any]:
        return {"api_key": self.api_key, "base_url": self.base_url}


@dataclass
class DeepSeekV4RequestConfig(RequestConfigBase):
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"] = field(default="deepseek-v4-flash")
    reasoning_effort: Literal["none", "low", "high", "max"] = field(default="high")

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


class OpenAITextChatClientDeepSeekV4(OpenAITextChatClientBase):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)

    @override
    def _unserialize_messages(self, messages: list[MessageBase]) -> list[ChatCompletionMessageParam]:
        unserialized: list[ChatCompletionMessageParam] = []
        for message in messages:
            if len(message) > 1:
                raise ClientException("OpenAI text chat protocol only supports one content item per message")
            match message:
                case SystemMessage():
                    content = message[0]
                    if not isinstance(content, ContentSystemText):
                        raise ClientException("SystemMessage only supports ContentSystemText")
                    unserialized.append({"role": "system", "content": content.text})
                case HumanMessage():
                    content = message[0]
                    if not isinstance(content, ContentHumanText):
                        raise ClientException("HumanMessage only supports ContentHumanText")
                    unserialized.append({"role": "user", "content": content.text})
                case AIMessage():
                    assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
                    if len(message) == 1:
                        content = message[0]
                        if not isinstance(content, ContentAIText):
                            raise ClientException("AIMessage only supports ContentAIText")
                        assistant_msg["content"] = content.text
                    elif not message.toolcalls:
                        raise ClientException("AIMessage has empty content and no tool_calls")
                    reasoning = message.reasoning
                    if reasoning is not None and reasoning.reasoning_content is not None:
                        assistant_msg["reasoning_content"] = reasoning.reasoning_content
                    toolcalls = message.toolcalls
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
                case ToolMessage():
                    content = message[0]
                    if not isinstance(content, ContentToolText):
                        raise ClientException("ToolMessage only supports ContentToolText")
                    unserialized.append(
                        {
                            "role": "tool",
                            "content": content.text,
                            "tool_call_id": message.toolcall.tool_id,
                        }
                    )
                case _:
                    raise ClientException(f"Unsupported message type: {type(message).__name__}")
        return unserialized

    @override
    def _serialize_response(self, response: ChatCompletion) -> list[MessageBase]:
        if not response.choices:
            return []
        raw = response.choices[0].message

        contents: list[ContentAIText] = []
        if raw.content is not None:
            contents.append(ContentAIText(raw.content))

        reasoning: ContentReasoning | None = None
        reasoning_content = getattr(raw, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            reasoning = ContentReasoning(reasoning_content=reasoning_content)

        toolcalls: list[ContentToolCall] = []
        for tc in raw.tool_calls or []:
            toolcalls.append(
                ContentToolCall(
                    tool_name=tc.function.name,
                    tool_args=tc.function.arguments,
                    tool_id=tc.id,
                )
            )

        return [AIMessage(contents=contents, reasoning=reasoning, toolcalls=toolcalls)]

    @override
    def _extract_expense(self, response: ChatCompletion) -> TokenExpense:
        _DEEPSEEK_V4_PRICES: dict[str, dict[str, tuple[float, float, float]]] = {
            "deepseek-v4-flash": {
                "peak": (0.10, 3.00, 9.00),
                "offpeak": (0.05, 1.50, 4.50),
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

        usage = response.usage
        if usage is None:
            return TokenExpense(token_input=None, token_output=None, token_cached=None, token_expense=None)

        token_input = usage.prompt_tokens
        token_output = usage.completion_tokens
        token_cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if token_cached is None and usage.prompt_tokens_details is not None:
            token_cached = usage.prompt_tokens_details.cached_tokens

        token_expense: float | None = None
        prices = _DEEPSEEK_V4_PRICES.get(response.model)
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

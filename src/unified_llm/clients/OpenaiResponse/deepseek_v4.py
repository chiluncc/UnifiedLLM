from datetime import datetime, time
from typing import Any, Literal, override
from pydantic import Field
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseInputParam,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseIncompleteEvent,
    ResponseOutputText,
    ResponseOutputRefusal,
    ResponseReasoningItem,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItemDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)
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
    StreamChunkWebSearch,
    StreamChunkWebOpenPage,
    StreamChunkWebFindInPage,
    ContentAIBase,
    ContentAIText,
    ContentAIReasoningText,
    ContentAIToolCall,
    ContentAIServerWebSearchCall,
    ContentHumanText,
    ContentHumanImage,
    ContentSystemText,
    ContentToolText,
    ContentToolImage,
)
from ..base import (
    ClientConfigBase,
    RequestConfigBase,
    TokenExpense,
    ClientException,
    ClientResultInner,
)
from .base import OpenAIResponseClientBase


class DeepSeekV4ResponseRequestConfig(RequestConfigBase):
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = Field(default="deepseek-flash")
    reasoning_effort: Literal["none", "low", "high", "max"] = Field(default="high")
    web_search: bool = Field(default=False)

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
                "reasoning": {"effort": self.reasoning_effort},
            }


class OpenAIResponseClientDeepSeekV4(OpenAIResponseClientBase):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)

    def _serialize_reasoning_content(self, content: ContentAIReasoningText) -> dict[str, Any]:
        item: dict[str, Any] = {
            "type": "reasoning",
            "summary": (
                [{"type": "summary_text", "text": content.reasoning_summary}]
                if content.reasoning_summary is not None
                else []
            ),
        }
        if content.reasoning_content is not None:
            item["content"] = [{"type": "reasoning_text", "text": content.reasoning_content}]
        return item

    def _web_search_fields(
        self,
        action: Any,
    ) -> tuple[Literal["search", "open_page", "find_in_page"], dict[str, Any]] | None:
        action_type = getattr(action, "type", None)
        if action_type == "search":
            queries = list(getattr(action, "queries", None) or [])
            if not queries:
                query = getattr(action, "query", None)
                if query:
                    queries = [query]
            return "search", {"queries": queries}
        if action_type == "open_page":
            url = getattr(action, "url", None)
            if not url:
                return None
            return "open_page", {"url": url}
        if action_type == "find_in_page":
            url = getattr(action, "url", None)
            pattern = getattr(action, "pattern", None)
            if not url or not pattern:
                return None
            return "find_in_page", {"url": url, "pattern": pattern}
        return None

    def _web_search_action(self, content: ContentAIServerWebSearchCall) -> dict[str, Any]:
        if content.action_type == "search":
            return {"type": "search", "queries": content.queries or []}
        if content.action_type == "open_page":
            return {"type": "open_page", "url": content.url}
        return {"type": "find_in_page", "url": content.url, "pattern": content.pattern}

    def _serialize_ai_message(self, message: MessageAI) -> list[Any]:
        items: list[Any] = []
        for content in message:
            match content:
                case ContentAIReasoningText():
                    items.append(self._serialize_reasoning_content(content))
                case ContentAIText():
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content.text,
                                    "annotations": content.annotations,
                                }
                            ],
                        }
                    )
                case ContentAIToolCall():
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": content.tool_id,
                            "name": content.tool_name,
                            "arguments": content.tool_args,
                        }
                    )
                case ContentAIServerWebSearchCall():
                    items.append(
                        {
                            "type": "web_search_call",
                            "status": "completed",
                            "action": self._web_search_action(content),
                        }
                    )
                case _:
                    raise ClientException(f"Unsupported AI content type: {type(content).__name__}")
        if not items:
            raise ClientException("AIMessage has empty content and no tool_calls")
        response_ids = message.additions.get("response_ids")
        if isinstance(response_ids, list) and len(response_ids) == len(items):
            for item, item_id in zip(items, response_ids):
                if isinstance(item_id, str) and item_id:
                    item["id"] = item_id
        return items

    def _serialize_tool_message(self, message: MessageTool) -> dict[str, Any]:
        text_parts: list[str] = []
        parts: list[dict[str, Any]] = []
        for content in message:
            match content:
                case ContentToolText():
                    text_parts.append(content.text)
                    parts.append({"type": "input_text", "text": content.text})
                case ContentToolImage():
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": content.img_url,
                            "detail": content.detail,
                        }
                    )
                case _:
                    raise ClientException(f"Unsupported tool content type: {type(content).__name__}")
        if not parts:
            raise ClientException("ToolMessage can't be empty")
        output: str | list[dict[str, Any]]
        if len(text_parts) == len(parts):
            output = "".join(text_parts)
        else:
            output = parts
        return {
            "type": "function_call_output",
            "call_id": message.toolcall.tool_id,
            "output": output,
        }

    def _content_from_item(self, item: Any) -> list[ContentAIBase]:
        match item:
            case ResponseReasoningItem():
                reasoning_content = "".join(
                    part.text
                    for part in item.content or []
                    if isinstance(getattr(part, "text", None), str)
                ) or None
                reasoning_summary = "".join(
                    part.text
                    for part in item.summary or []
                    if isinstance(getattr(part, "text", None), str)
                ) or None
                if reasoning_content is None and reasoning_summary is None:
                    return []
                return [
                    ContentAIReasoningText(
                        reasoning_content=reasoning_content,
                        reasoning_summary=reasoning_summary,
                    )
                ]
            case ResponseOutputMessage():
                texts: list[ContentAIBase] = []
                for part in item.content or []:
                    if isinstance(part, ResponseOutputRefusal):
                        raise ClientException(f"Model refused to answer: {part.refusal}")
                    if not isinstance(part, ResponseOutputText):
                        continue
                    texts.append(
                        ContentAIText(
                            part.text,
                            annotations=[
                                annotation.model_dump(mode="json", exclude_none=True)
                                for annotation in part.annotations or []
                            ],
                        )
                    )
                return texts
            case ResponseFunctionToolCall():
                return [
                    ContentAIToolCall(
                        tool_name=item.name,
                        tool_args=item.arguments,
                        tool_id=item.call_id,
                    )
                ]
            case ResponseFunctionWebSearch():
                normalized = self._web_search_fields(item.action)
                if normalized is None:
                    return []
                action_type, fields = normalized
                return [ContentAIServerWebSearchCall(action_type, **fields)]
            case _:
                return []

    def _build_result(
        self,
        items: list[Any],
        expense: TokenExpense,
    ) -> ClientResultInner:
        contents: list[ContentAIBase] = []
        response_ids: list[str] = []
        for item in items:
            item_contents = self._content_from_item(item)
            if not item_contents:
                continue
            contents.extend(item_contents)
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str) and item_id:
                response_ids.append(item_id)
        messages: list[MessageBase] = []
        if contents:
            messages.append(MessageAI(contents=contents, additions={"response_ids": response_ids}))
        return ClientResultInner(messages=messages, expense=expense)

    def _compute_expense(self, model: str, usage: Any) -> TokenExpense:
        _DEEPSEEK_V4_PRICES: dict[str, dict[str, tuple[float, float, float]]] = {
            "deepseek-flash": {
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

        token_input = usage.input_tokens
        token_output = usage.output_tokens
        token_cached = getattr(usage.input_tokens_details, "cached_tokens", None)

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
    def _serialize_input(self, messages: list[MessageBase]) -> ResponseInputParam:
        items: list[Any] = []
        for message in messages:
            match message:
                case MessageSystem():
                    if len(message) != 1:
                        raise ClientException(
                            f"SystemMessage accepts exactly one content, got {len(message)}"
                        )
                    content = message[0]
                    if not isinstance(content, ContentSystemText):
                        raise ClientException(
                            f"Unsupported system content type: {type(content).__name__}"
                        )
                    items.append({"role": "system", "content": content.text})
                case MessageHuman():
                    content_parts: list[dict[str, Any]] = []
                    for content in message:
                        match content:
                            case ContentHumanText():
                                content_parts.append({"type": "input_text", "text": content.text})
                            case ContentHumanImage():
                                content_parts.append(
                                    {
                                        "type": "input_image",
                                        "image_url": content.img_url,
                                        "detail": content.detail,
                                    }
                                )
                            case _:
                                raise ClientException(
                                    f"Unsupported human content type: {type(content).__name__}"
                                )
                    if not content_parts:
                        raise ClientException("HumanMessage can't be empty")
                    items.append({"role": "user", "content": content_parts})
                case MessageAI():
                    items.extend(self._serialize_ai_message(message))
                case MessageTool():
                    items.append(self._serialize_tool_message(message))
                case _:
                    raise ClientException(f"Unsupported message type: {type(message).__name__}")
        return items

    @override
    def _parse_response(self, response: Response) -> ClientResultInner:
        return self._build_result(
            items=list(response.output or []),
            expense=self._compute_expense(response.model, response.usage),
        )

    @override
    def _parse_stream_event(self, response_event: ResponseStreamEvent) -> list[StreamChunkBase]:
        match response_event:
            case ResponseReasoningTextDeltaEvent() | ResponseReasoningSummaryTextDeltaEvent():
                if not response_event.delta:
                    return []
                return [StreamChunkReasoning(text=response_event.delta)]
            case ResponseTextDeltaEvent():
                if not response_event.delta:
                    return []
                return [StreamChunkText(text=response_event.delta)]
            case ResponseOutputItemAddedEvent():
                item = response_event.item
                if not isinstance(item, ResponseFunctionToolCall):
                    return []
                return [
                    StreamChunkToolCall(
                        index=response_event.output_index,
                        tool_id=item.call_id,
                        tool_name=item.name,
                        tool_args=None,
                    )
                ]
            case ResponseFunctionCallArgumentsDeltaEvent():
                if not response_event.delta:
                    return []
                return [
                    StreamChunkToolCall(
                        index=response_event.output_index,
                        tool_args=response_event.delta,
                    )
                ]
            case ResponseOutputItemDoneEvent():
                item = response_event.item
                if not isinstance(item, ResponseFunctionWebSearch):
                    return []
                normalized = self._web_search_fields(item.action)
                if normalized is None:
                    return []
                action_type, fields = normalized
                if action_type == "search":
                    return [StreamChunkWebSearch(queries=fields["queries"])]
                if action_type == "open_page":
                    return [StreamChunkWebOpenPage(url=fields["url"])]
                return [
                    StreamChunkWebFindInPage(url=fields["url"], pattern=fields["pattern"])
                ]
            case ResponseCompletedEvent() | ResponseIncompleteEvent():
                return [StreamChunkEmpty(done=True)]
            case _:
                return []

    @override
    def _parse_stream_events_full(
        self,
        response_events: list[ResponseStreamEvent],
    ) -> ClientResultInner:
        final_response: Response | None = None
        for event in response_events:
            if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                final_response = event.response
        if final_response is not None:
            return self._parse_response(final_response)
        
        items_by_index: dict[int, Any] = {}
        for event in response_events:
            if isinstance(event, ResponseOutputItemDoneEvent):
                items_by_index[event.output_index] = event.item
        return self._build_result(
            items=[items_by_index[index] for index in sorted(items_by_index)],
            expense=TokenExpense(
                token_input=None,
                token_output=None,
                token_cached=None,
                token_expense=None,
            ),
        )

    @override
    def get_tools(self) -> list[Any]:
        tools = list(super().get_tools())
        request_config = self.get_request_config()
        if request_config.web_search:
            tools.append({"type": "web_search"})
        return tools

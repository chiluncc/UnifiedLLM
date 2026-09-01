import inspect

import openai
from abc import ABC, abstractmethod
from docstring_parser import Style
from docstring_parser import parse as parse_docstring
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import override

from unified_llm.messages.messages import MessageBase
from unified_llm.tools import Tool
from ..base import ClientBase, ClientConfigBase, RequestConfigBase
from ..base import ClientResult, TokenExpense, ClientExecutor


class OpenAITextChatClientBase(ClientBase, ABC):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase):
        super().__init__(client_config, request_config)
        self._client = openai.Client(**client_config.to_dict())
        self._async_client = openai.AsyncClient(**client_config.to_dict())
        
    @override
    def invoke(self, messages: list[MessageBase]) -> ClientResult:
        response = self._client.chat.completions.create(
            messages=self._unserialize_messages(messages),
            tools=self.get_tools(),
            **self.get_request_conifg().to_dict(),
        )
        return ClientResult(
            messages=messages + self._serialize_response(response), 
            expense=self._extract_expense(response)
            )

    @override
    async def ainvoke(self, messages: list[MessageBase]) -> ClientResult:
        response = await self._async_client.chat.completions.create(
            messages=self._unserialize_messages(messages),
            tools=self.get_tools(),
            **self.get_request_conifg().to_dict(),
        )
        return ClientResult(
            messages=messages + self._serialize_response(response),
            expense=self._extract_expense(response),
        )

    @override
    def execute(self, messages: list[MessageBase]) -> ClientExecutor:
        return None

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
    def _serialize_response(self, response: ChatCompletion) -> list[MessageBase]: ...

    @abstractmethod
    def _extract_expense(self, response: ChatCompletion) -> TokenExpense: ...

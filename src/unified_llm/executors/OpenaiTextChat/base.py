from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from unified_llm.messages.messages import MessageBase, AIMessage
from ..base import Executer


class OpenAITextChatExecutorBase(ABC):
    def invoke(messages: list[MessageBase]) -> Executer:
        pass

    async def ainvoke(messages: list[MessageBase]) -> Executer:
        pass

    @abstractmethod
    def before_chat_parse_message(self, messages: list[MessageBase]) -> list[ChatCompletionMessageParam]: ...

    @abstractmethod
    def after_chat_parse_message(self, response: ChatCompletion) -> list[AIMessage]: ...

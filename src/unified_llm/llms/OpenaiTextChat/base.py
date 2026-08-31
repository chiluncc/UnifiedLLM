from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from unified_llm.messages.messages import MessageBase


class OpenAITextChatBase(ABC):
    @abstractmethod
    def before_chat(self, messages: list[MessageBase]) -> list[ChatCompletionMessageParam]:
        ...

    @abstractmethod
    def after_chat(self, response: ChatCompletion) -> MessageBase:
        ...

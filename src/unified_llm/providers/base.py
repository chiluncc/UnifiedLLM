import threading
from pydantic import BaseModel
from typing import Self
from abc import abstractmethod, ABC

from unified_llm.tools import Tool
from unified_llm.messages.message import BaseMessage


class BaseProviderConfig(BaseModel):
    ...


class BaseProvider(ABC):
    def __init__(self, config: BaseProviderConfig) -> None:
        super().__init__()

        self._raw_config: BaseProviderConfig = config
        self._raw_tools: list[Tool | str] = []
        self._stop_flag: threading.Event = threading.Event()

    @abstractmethod
    def get_config(self) -> BaseProviderConfig:
        return self._raw_config

    @abstractmethod
    def bind_tools(self, tools: list[Tool | str]) -> Self:
        self._raw_tools = tools

    @abstractmethod
    def get_tools(self) -> list[Tool | str]:
        return self._raw_tools
    
    @abstractmethod
    def invoke(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        raise NotImplementedError
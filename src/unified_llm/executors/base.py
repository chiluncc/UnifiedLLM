from typing import TypedDict, Iterator, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from unified_llm.messages.messages import MessageBase


class ClientException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class TokenExpense(TypedDict):
    token_input: int | None
    token_output: int | None
    token_cached: int | None
    token_expense: float | None


class ClientResult(TypedDict):
    messages: list[MessageBase]
    expense: TokenExpense


class Executor:
    def __init__(self) -> None: ...

    def __iter__(self) -> Iterator[Any]: ...

    def __next__(self) -> Any: ...

    def done(self) -> bool: ...

    def cancel(self) -> bool: ...

    def result(self) -> ClientResult: ...

    def exception(self) -> Exception: ...


@dataclass
class ClientConfigBase(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class RequestConfigBase(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class ClientBase(ABC):
    def __init__(self, client_config: ClientConfigBase, request_config: RequestConfigBase) -> None:
        super().__init__()
        self._client_config = client_config
        self._request_config = request_config
        self._tools: list = []

    def set_tools(self, tools_desp: list | None = None) -> None:
        self._tools = [] if tools_desp is None else tools_desp

    def get_tools(self) -> list:
        return self._tools

    def set_request_config(self, request_config: RequestConfigBase) -> None:
        self._request_config = request_config

    def get_request_conifg(self) -> RequestConfigBase:
        return self._request_config
    
    def get_client_config(self) -> ClientConfigBase:
        return self._client_config
        
    @abstractmethod
    def invoke(self, messages: list[MessageBase]) -> ClientResult: ...

    @abstractmethod
    async def ainvoke(self, messages: list[MessageBase]) -> ClientResult: ...

    @abstractmethod
    def execute(self, messages: list[MessageBase]) -> Executor: ...
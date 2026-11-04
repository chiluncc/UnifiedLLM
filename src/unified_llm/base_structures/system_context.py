from typing import Callable, TypedDict
from contextvars import ContextVar, Token
from logging import Logger

from unified_llm.utils.file_handles import get_default_logger


class SystemContext(TypedDict):
    logger: Logger
    stream: Callable[[tuple[str, str]], None]


_system_context: ContextVar[SystemContext] = ContextVar("system_context")


def set_system_context(system_context: SystemContext) -> Token[SystemContext]:
    return _system_context.set(system_context)


def get_system_context() -> SystemContext:
    try:
        return _system_context.get()
    except LookupError:
        system_context = SystemContext(
            logger=get_default_logger(),
            stream=lambda _: None,
        )
        _system_context.set(system_context)
        return system_context

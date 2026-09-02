import inspect
from typing import Any, Callable, get_type_hints
from abc import ABC, abstractmethod
from docstring_parser import Style
from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field, create_model

from unified_llm.messages.messages import ToolMessage
from unified_llm.messages.contents import ContentToolCall


class ToolException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Tool(BaseModel, frozen=True):
    name: str
    args: type[BaseModel]
    sync: bool
    func: Callable


def tool(func: Callable | None = None, *, name: str | None = None) -> Tool | Callable[..., Any]:
    def _parse_docstring(fn: Callable) -> dict[str, str]:
        doc = parse_docstring(inspect.getdoc(fn) or "", style=Style.GOOGLE)
        return {p.arg_name: p.description for p in doc.params if p.description}

    def _build_args_model(fn: Callable, param_descs: dict[str, str]) -> type[BaseModel]:
        type_hints = get_type_hints(fn)
        fields: dict[str, tuple[Any, Any]] = {}
        for param_name, param in inspect.signature(fn).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = type_hints.get(param_name, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default: Any = ... if param.default is inspect.Parameter.empty else param.default
            desc = param_descs.get(param_name)
            if desc:
                fields[param_name] = (annotation, Field(default, description=desc))
            else:
                fields[param_name] = (annotation, default)
        return create_model(f"{fn.__name__}Args", **fields)

    def _decorator(fn: Callable) -> Tool:
        tool_name = name if name is not None else fn.__name__
        param_descs = _parse_docstring(fn)
        args_model = _build_args_model(fn, param_descs)
        return Tool(
            name=tool_name,
            args=args_model,
            sync=not inspect.iscoroutinefunction(fn),
            func=fn,
        )

    if func is None:
        return _decorator
    return _decorator(func)


class ToolExecutorBase(ABC):
    @abstractmethod
    def list_tools(self) -> list[Tool]: ...

    @abstractmethod
    def execute(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]: ...

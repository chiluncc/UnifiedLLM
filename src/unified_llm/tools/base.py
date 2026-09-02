import inspect
import json
from typing import Any, Callable, get_type_hints

from docstring_parser import Style
from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field, ValidationError, create_model

from unified_llm.messages.messages import ToolMessage
from unified_llm.messages.contents import ContentToolCall
from unified_llm.messages.contents import ContentToolBase, ContentToolText


class ToolException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Tool(BaseModel, frozen=True):
    name: str
    args: type[BaseModel]
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
        return Tool(name=tool_name, args=args_model, func=fn)

    if func is None:
        return _decorator
    return _decorator(func)


class ToolExecutor:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, toolcall: ContentToolCall) -> ToolMessage:
        tool_def = self._tools.get(toolcall.tool_name)
        if tool_def is None:
            return ToolMessage(
                ContentToolText(f"Unknown tool: {toolcall.tool_name}"),
                toolcall=toolcall,
            )

        try:
            raw_args = json.loads(toolcall.tool_args or "{}")
        except json.JSONDecodeError as exc:
            raise ToolException(f"Invalid JSON args for tool {toolcall.tool_name}: {toolcall.tool_args!r}") from exc
        if not isinstance(raw_args, dict):
            raise ToolException(f"Tool args must be a JSON object, got: {toolcall.tool_args!r}")

        try:
            validated = tool_def.args.model_validate(raw_args)
        except ValidationError as exc:
            errors = exc.errors()
            missing = [str(err["loc"][0]) for err in errors if err["type"] == "missing"]
            if missing:
                return ToolMessage(
                    ContentToolText(f"Missing required arguments: {', '.join(missing)}"),
                    toolcall=toolcall,
                )
            details = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in errors
            )
            return ToolMessage(
                ContentToolText(f"Invalid arguments: {details}"),
                toolcall=toolcall,
            )

        result = tool_def.func(**validated.model_dump())
        contents: ContentToolBase | list[ContentToolBase]
        if isinstance(result, ContentToolBase):
            contents = [result]
        elif isinstance(result, list) and all(isinstance(item, ContentToolBase) for item in result):
            contents = result
        else:
            contents = ContentToolText(str(result))
        return ToolMessage(contents, toolcall=toolcall)

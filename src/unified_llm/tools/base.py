import inspect
from docstring_parser import parse
from typing import Callable, overload, Any
from pydantic import BaseModel, create_model, Field

from unified_llm.messages.content import ToolCallContent, TextContent, BaseContent
from unified_llm.messages.message import ToolMessage


class Tool:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if inspect.iscoroutinefunction(func):
            raise TypeError("Async function not supported.")

        tool_signature = inspect.signature(func)
        tool_docstring = parse(inspect.getdoc(func))

        self._name: str = func.__name__ if name is None else name
        self._description: str | None = tool_docstring.description if description is None else description

        par_name2desp = {i.arg_name: i.description for i in tool_docstring.params}
        fields = {}
        for name, param in tool_signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
                raise TypeError(
                    "Tool parameters of kind VAR_KEYWORD (**kwargs) or "
                    "VAR_POSITIONAL (*args) or POSITIONAL_ONLY (/) "
                    "are not supported."
                )
            
            annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (
                annotation,
                Field(default=default, description=par_name2desp.get(name)),
            )
        
        self._params: type[BaseModel] = create_model(
            f"{func.__name__}_config",
            __doc__=self._description,
            **fields,
        )
        self._func: Callable = func

    def _tool_call_parse(
        self, tool_call_content: ToolCallContent
    ) -> BaseModel | str:
        try:
            args_parse = self._params.model_validate_json(
                tool_call_content.tool_args
            )
        except Exception as exc:
            return str(exc)
        return args_parse

    def _standardized_result(
        self, tool_call_content: ToolCallContent, result: Any
    ) -> ToolMessage:
        if isinstance(result, list) and any([isinstance(item, BaseContent) for item in result]):
            result = [item if isinstance(item, BaseContent) else TextContent(text=str(item)) for item in result]
        elif isinstance(result, BaseContent):
            result = [result]
        else:
            result = [TextContent(text=str(result))]
        return ToolMessage(
            contents=result,
            tool_call_content=tool_call_content.model_copy(),
        )

    def invoke(
        self,
        tool_call_content: ToolCallContent,
        *,
        raise_tool_error: bool = True,
        tool_error_handler: Callable[[str, Exception], Any] = lambda name, exc: exc,
    ) -> ToolMessage:
        args = self._tool_call_parse(tool_call_content)
        if isinstance(args, str):
            return ToolMessage(
                contents=[TextContent(text=args)],
                tool_call_content=tool_call_content.model_copy(),
            )
        
        try:
            result = self._func(**dict(args))
            return self._standardized_result(tool_call_content, result)
        except Exception as exc:
            result = tool_error_handler(self._name, exc)
            if not isinstance(result, Exception):
                return self._standardized_result(tool_call_content, result)
            if raise_tool_error:
                raise result
            return ToolMessage(
                contents=[TextContent(text=str(result))],
                tool_call_content=tool_call_content.model_copy(),
            )
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def params(self) -> type[BaseModel]:
        return self._params


@overload
def tool(func: Callable) -> Tool:
    ...


@overload
def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Tool]:
    ...


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Tool] | Tool:
    def transfer_tool(func: Callable) -> Tool:
        return Tool(func, name=name, description=description)
    if func is not None:
        return transfer_tool(func)
    return transfer_tool

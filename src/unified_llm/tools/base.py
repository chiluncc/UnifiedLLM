import inspect
from docstring_parser import parse
from typing import Callable, override, overload, Iterator, Any
from pydantic import BaseModel, create_model, Field

from unified_llm.base_structures.runable import Runable


class Tool(Runable):
    def __init__(self, func: Callable, *, name: str | None = None, docstring: str | None = None) -> None:
        super().__init__()

        if inspect.iscoroutinefunction(func):
            raise TypeError("Async function not supported.")

        tool_signature = inspect.signature(func)
        tool_docstring = parse(inspect.getdoc(func))

        self._name: str = func.__name__ if name is None else name
        self._description: str | None = tool_docstring.description if docstring is None else docstring

        par_name2desp = {i.arg_name: i.description for i in tool_docstring.params}
        fields = {}
        for name, param in tool_signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                raise TypeError("Tool parameters of kind VAR_KEYWORD (**kwargs) or VAR_POSITIONAL (*args) are not supported.")
            
            annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, Field(default=default, description=par_name2desp.get(name)))
        
        self._params: type[BaseModel] = create_model(f"{func.__name__}_config", __doc__=self._description, **fields)
        self._func: Callable = func

    @override
    def invoke(self, params: dict) -> dict:
        # TODO 需要 ToolMessage 以及 AIMessage 的 AIMetaTool 定义后才能完成
        raise NotImplementedError

    @override
    def stream(self, params: dict) -> Iterator[dict]:
        # TODO 需要 ToolMessage 以及 AIMessage 的 AIMetaTool 定义后才能完成
        raise NotImplementedError

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
def tool(name: str | None = None, docstring: str | None = None) -> Callable[[Callable], Tool]:
    ...


def tool(func: Callable | None = None, *, name: str | None = None, docstring: str | None = None) -> Callable[[Callable], Tool] | Tool:
    def transfer_tool(func: Callable) -> Tool:
        return Tool(func, name=name, docstring=docstring)
    if func is not None:
        return transfer_tool(func)
    return transfer_tool
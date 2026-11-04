from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Callable, Any

from unified_llm.messages.content import ToolCallContent, TextContent
from unified_llm.messages.message import ToolMessage
from unified_llm.base_structures.system_context import get_system_context, set_system_context
from .base import Tool


class ToolManager:
    def __init__(
        self,
        *,
        raise_tool_error: bool = False,
        tool_error_handler: Callable[[str, Exception], Any] = lambda name, exc: exc,
        mode: Literal["parallel", "serial"] = "parallel",
        max_thread: int = 8,
    ):
        super().__init__()
        self._raise_tool_error: bool = raise_tool_error
        self._tool_error_handler: Callable[[str, Exception], Any] = tool_error_handler
        self._mode: Literal["parallel", "serial"] = mode
        self._max_thread: int = max_thread
        self._name2tool: dict[str, Tool] = {}

    @property
    def tools(self) -> list[Tool]:
        return list(self._name2tool.values())

    def add(self, tools: Tool | list[Tool]) -> None:
        tools = [tools] if isinstance(tools, Tool) else tools
        additions: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._name2tool or tool.name in additions:
                raise ValueError(f"Tool already registered: {tool.name!r}.")
            additions[tool.name] = tool
        self._name2tool.update(additions)

    def remove(self, tool_names: str | list[str]) -> None:
        tool_names = [tool_names] if isinstance(tool_names, str) else tool_names
        for name in tool_names:
            if name not in self._name2tool:
                raise KeyError(f"Unknown tool: {name!r}.")
        for name in dict.fromkeys(tool_names):
            del self._name2tool[name]

    def invoke(
        self, tool_call_contents: list[ToolCallContent]
    ) -> list[ToolMessage]:
        def execute_one(tool_call_content: ToolCallContent) -> ToolMessage:
            tool = self._name2tool.get(tool_call_content.tool_name)
            if tool is None:
                return ToolMessage(
                    contents=[
                        TextContent(
                            text=f"Unknown tool: {tool_call_content.tool_name!r}."
                        )
                    ],
                    tool_call_content=tool_call_content.model_copy(),
                )
            return tool.invoke(
                tool_call_content,
                raise_tool_error=self._raise_tool_error,
                tool_error_handler=self._tool_error_handler,
            )

        match self._mode:
            case "serial":
                return [execute_one(tool_call) for tool_call in tool_call_contents]
            case "parallel":
                if not tool_call_contents:
                    return []
                shared_context = get_system_context()
                with ThreadPoolExecutor(
                    max_workers=self._max_thread,
                    initializer=set_system_context,
                    initargs=(shared_context,),
                ) as executor:
                    return list(executor.map(execute_one, tool_call_contents))
            case _:
                raise ValueError(f"Unsupported execution mode: {self._mode!r}.")

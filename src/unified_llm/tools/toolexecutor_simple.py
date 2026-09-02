import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, override

from unified_llm.messages.contents import ContentToolCall
from unified_llm.messages.messages import ToolMessage
from .base import Tool, ToolExecutorBase


def _run_async_tool(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, awaitable)
        return future.result()


class ToolExecutorSimple(ToolExecutorBase):
    def __init__(self, tools: list[Tool]) -> None:
        super().__init__(tools)

    def _execute_serial(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]:
        results: list[ToolMessage] = []
        for toolcall in toolcalls:
            resolved = self._resolve_toolcall(toolcall)
            if isinstance(resolved, ToolMessage):
                results.append(resolved)
                continue
            if resolved.tool.sync:
                result = resolved.tool.func(**resolved.args.model_dump())
            else:
                result = _run_async_tool(resolved.tool.func(**resolved.args.model_dump()))
            results.append(self._wrap_result(toolcall, result))
        return results

    @override
    def sync_execute(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]:
        return self._execute_serial(toolcalls)

    @override
    def async_execute(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]:
        return self._execute_serial(toolcalls)

    @override
    def mixed_execute(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]:
        return self._execute_serial(toolcalls)

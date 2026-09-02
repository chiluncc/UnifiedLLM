import asyncio
import json
from typing import override

from pydantic import ValidationError

from unified_llm.messages.contents import ContentToolBase, ContentToolCall, ContentToolText
from unified_llm.messages.messages import ToolMessage
from .base import Tool, ToolExecutorBase, ToolException


class ToolExecutorSimple(ToolExecutorBase):
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    @override
    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    @override
    def execute(self, toolcalls: list[ContentToolCall]) -> list[ToolMessage]:
        results: list[ToolMessage] = []
        for toolcall in toolcalls:
            tool_def = self._tools.get(toolcall.tool_name)
            if tool_def is None:
                results.append(
                    ToolMessage(
                        ContentToolText(f"Unknown tool: {toolcall.tool_name}"),
                        toolcall=toolcall,
                    )
                )
                continue

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
                    results.append(
                        ToolMessage(
                            ContentToolText(f"Missing required arguments: {', '.join(missing)}"),
                            toolcall=toolcall,
                        )
                    )
                else:
                    details = "; ".join(
                        f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in errors
                    )
                    results.append(
                        ToolMessage(
                            ContentToolText(f"Invalid arguments: {details}"),
                            toolcall=toolcall,
                        )
                    )
                continue

            if tool_def.sync:
                result = tool_def.func(**validated.model_dump())
            else:
                result = asyncio.run(tool_def.func(**validated.model_dump()))

            contents: ContentToolBase | list[ContentToolBase]
            if isinstance(result, ContentToolBase):
                contents = [result]
            elif isinstance(result, list) and all(isinstance(item, ContentToolBase) for item in result):
                contents = result
            else:
                contents = ContentToolText(str(result))
            results.append(ToolMessage(contents, toolcall=toolcall))
        return results

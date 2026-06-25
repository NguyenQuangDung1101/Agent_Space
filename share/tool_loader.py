from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from jsonschema import ValidationError, validate

from share.registry import Registry
from share.schemas import ToolResult


class ToolLoader:
    def __init__(
        self,
        registry: Registry,
        assigned_tool_ids: list[str] | None = None,
    ) -> None:
        self.registry = registry
        self.assigned_tool_ids = list(
            dict.fromkeys(assigned_tool_ids or [])
        )

        self._tools: dict[
            str,
            dict[str, Any],
        ] = {}

        for tool_id in self.assigned_tool_ids:
            self._load_tool(tool_id)

    def _load_tool(
        self,
        tool_id: str,
    ) -> None:
        entry = self.registry.get_tool(tool_id)
        config = self.registry.get_tool_config(tool_id)

        tool_path = (
            self.registry.resolve_path(entry["path"])
            / "tool.py"
        )

        run_function = self._load_run_function(
            tool_id,
            tool_path,
        )

        tool_name = config["name"]

        if tool_name in self._tools:
            raise ValueError(
                f"Duplicate tool name: {tool_name}"
            )

        self._tools[tool_name] = {
            "id": tool_id,
            "config": config,
            "run": run_function,
        }

    @staticmethod
    def _load_run_function(
        tool_id: str,
        tool_path: Path,
    ) -> Callable[[dict[str, Any]], Any]:
        if not tool_path.is_file():
            raise FileNotFoundError(
                f"Tool file not found: {tool_path}"
            )

        module_name = f"agent_space_tool_{tool_id}"

        spec = importlib.util.spec_from_file_location(
            module_name,
            tool_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load tool: {tool_id}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_function = getattr(
            module,
            "run",
            None,
        )

        if not callable(run_function):
            raise AttributeError(
                f"Tool '{tool_id}' must define run(arguments)"
            )

        return run_function

    def get_tool_spec(self) -> list[dict[str, Any]]:
        return [
            dict(tool["config"])
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        tool = self._tools.get(name)

        if tool is None:
            return ToolResult(
                name=name,
                success=False,
                error="Tool is not assigned to this agent.",
            )

        if not isinstance(arguments, dict):
            return ToolResult(
                name=name,
                success=False,
                error="Tool arguments must be a JSON object.",
            )

        schema = tool["config"].get(
            "inputSchema",
            {
                "type": "object",
                "properties": {},
            },
        )

        try:
            validate(
                instance=arguments,
                schema=schema,
            )

            output = tool["run"](arguments)

            return ToolResult(
                name=name,
                success=True,
                output=output,
            )

        except ValidationError as error:
            return ToolResult(
                name=name,
                success=False,
                error=f"Invalid arguments: {error.message}",
            )

        except Exception as error:
            return ToolResult(
                name=name,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
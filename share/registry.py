from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TOOL_ASSIGNERS = {
    "manager",
    "analyze_agent",
    "planner_agent",
}


class Registry:
    def __init__(
        self,
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(
            project_root
            or Path(__file__).resolve().parents[1]
        ).resolve()

        self._agents = self._load_registry(
            self.project_root / "agent" / "agent.json",
            required_fields={
                "id",
                "type",
                "description",
                "capabilities",
                "selectable_as_worker",
                "path",
                "prompt_path",
                "allowed_callers",
            },
        )

        tool_entries = self._load_registry(
            self.project_root / "tool" / "tool.json",
            required_fields={
                "id",
                "path",
            },
        )

        self._tools = {
            tool_id: self._enrich_tool(entry)
            for tool_id, entry in tool_entries.items()
        }

        self._validate_agent_entries()

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(
                f"Registry file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _load_registry(
        self,
        path: Path,
        required_fields: set[str],
    ) -> dict[str, dict[str, Any]]:
        data = self._read_json(path)

        if not isinstance(data, list):
            raise ValueError(
                f"Registry must contain a JSON list: {path}"
            )

        registry: dict[str, dict[str, Any]] = {}

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid registry entry in {path}"
                )

            missing = required_fields - item.keys()

            if missing:
                raise ValueError(
                    f"Missing fields {sorted(missing)} in {path}"
                )

            item_id = item["id"]

            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    f"Invalid ID in {path}"
                )

            if item_id in registry:
                raise ValueError(
                    f"Duplicate ID '{item_id}' in {path}"
                )

            registry[item_id] = dict(item)

        return registry

    def _validate_agent_entries(self) -> None:
        for agent in self._agents.values():
            if not isinstance(
                agent["capabilities"],
                list,
            ):
                raise ValueError(
                    f"Agent '{agent['id']}' capabilities "
                    "must be a list."
                )

            if not isinstance(
                agent["selectable_as_worker"],
                bool,
            ):
                raise ValueError(
                    f"Agent '{agent['id']}' "
                    "selectable_as_worker must be a boolean."
                )

    def _enrich_tool(
        self,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        config_path = (
            self.resolve_path(entry["path"])
            / "config.json"
        )

        config = self._read_json(config_path)

        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid tool config: {config_path}"
            )

        for field in (
            "name",
            "description",
            "inputSchema",
        ):
            if field not in config:
                raise ValueError(
                    f"Tool config is missing '{field}': "
                    f"{config_path}"
                )

        return {
            **entry,
            "name": config["name"],
            "description": config["description"],
            "inputSchema": config["inputSchema"],
        }

    def resolve_path(
        self,
        relative_path: str,
    ) -> Path:
        path = (
            self.project_root / relative_path
        ).resolve()

        try:
            path.relative_to(self.project_root)
        except ValueError as error:
            raise ValueError(
                f"Path is outside project root: {relative_path}"
            ) from error

        return path

    def get_agent(
        self,
        agent_id: str,
    ) -> dict[str, Any]:
        try:
            return dict(self._agents[agent_id])
        except KeyError as error:
            raise KeyError(
                f"Unknown agent: {agent_id}"
            ) from error

    def get_tool(
        self,
        tool_id: str,
    ) -> dict[str, Any]:
        try:
            return dict(self._tools[tool_id])
        except KeyError as error:
            raise KeyError(
                f"Unknown tool: {tool_id}"
            ) from error

    def list_agents(self) -> list[dict[str, Any]]:
        return [
            dict(agent)
            for agent in self._agents.values()
        ]

    def list_selectable_agents(
        self,
    ) -> list[dict[str, Any]]:
        return [
            dict(agent)
            for agent in self._agents.values()
            if agent["selectable_as_worker"]
        ]

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            dict(tool)
            for tool in self._tools.values()
        ]

    def get_tool_config(
        self,
        tool_id: str,
    ) -> dict[str, Any]:
        tool = self.get_tool(tool_id)

        return {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
        }

    def validate_agent_ids(
        self,
        agent_ids: list[str],
        selectable_only: bool = False,
    ) -> list[dict[str, Any]]:
        agents = [
            self.get_agent(agent_id)
            for agent_id in agent_ids
        ]

        if selectable_only:
            invalid = [
                agent["id"]
                for agent in agents
                if not agent["selectable_as_worker"]
            ]

            if invalid:
                raise ValueError(
                    "Agents are not selectable workers: "
                    + ", ".join(invalid)
                )

        return agents

    def validate_tool_ids(
        self,
        tool_ids: list[str],
    ) -> list[dict[str, Any]]:
        return [
            self.get_tool(tool_id)
            for tool_id in tool_ids
        ]

    def can_call(
        self,
        caller_id: str,
        target_agent_id: str,
    ) -> bool:
        agent = self._agents.get(
            target_agent_id
        )

        if agent is None:
            return False

        return caller_id in agent.get(
            "allowed_callers",
            [],
        )

    def can_assign_tool(
        self,
        caller_id: str,
        tool_id: str,
    ) -> bool:
        return (
            caller_id in DEFAULT_TOOL_ASSIGNERS
            and tool_id in self._tools
        )

    def print_catalogs(self) -> None:
        print("\n[Agent Space] Registered agents:")

        for agent in self.list_agents():
            role = (
                "worker"
                if agent["selectable_as_worker"]
                else "control"
            )

            capabilities = ", ".join(
                agent["capabilities"]
            )

            print(
                f"  - {agent['id']} ({role})"
                f" | capabilities: {capabilities}"
            )

        print("[Agent Space] Registered tools:")

        for tool in self.list_tools():
            print(
                f"  - {tool['id']}"
                f" | {tool['description']}"
            )

        print()
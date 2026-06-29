from __future__ import annotations

import importlib
from typing import Any, Optional

from share.registry import Registry
from share.schemas import AgentRequest, AgentResult


class AgentRunner:
    def __init__(
        self,
        agent_id: str,
        service: Any,
        caller_id: str,
        assigned_tool_ids: list[str],
        runtime_system_prompt: Optional[str],
    ) -> None:
        self.agent_id = agent_id
        self.service = service
        self.caller_id = caller_id
        self.assigned_tool_ids = assigned_tool_ids
        self.runtime_system_prompt = runtime_system_prompt

    async def run(
        self,
        session_id: str,
        task: str,
        context: Optional[dict[str, Any]] = None,
        max_steps: int = 10,
    ) -> AgentResult:
        return await self.service.run(
            AgentRequest(
                session_id=session_id,
                caller_id=self.caller_id,
                task=task,
                context=context or {},
                assigned_tool_ids=self.assigned_tool_ids,
                runtime_system_prompt=(
                    self.runtime_system_prompt
                ),
                max_steps=max_steps,
            )
        )


class AgentFactory:
    def __init__(
        self,
        registry: Optional[Registry] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = registry or Registry()
        self.model = model
        self.base_url = base_url

    @staticmethod
    def _service_class_name(
        agent_id: str,
    ) -> str:
        return "".join(
            part.capitalize()
            for part in agent_id.split("_")
        ) + "Service"

    def _load_service_class(
        self,
        agent: dict[str, Any],
    ) -> type:
        module_path = ".".join(
            str(agent["path"])
            .replace("\\", "/")
            .strip("/")
            .split("/")
        )

        module = importlib.import_module(
            f"{module_path}.service"
        )

        class_name = self._service_class_name(
            agent["id"]
        )

        service_class = getattr(
            module,
            class_name,
            None,
        )

        if service_class is None:
            raise ImportError(
                f"Service class '{class_name}' was not found "
                f"for agent '{agent['id']}'."
            )

        return service_class

    def create(
        self,
        agent_id: str,
        caller_id: str = "manager",
        assigned_tool_ids: Optional[list[str]] = None,
        runtime_system_prompt: Optional[str] = None,
    ) -> AgentRunner:
        agent = self.registry.get_agent(agent_id)

        if not agent["selectable_as_worker"]:
            raise ValueError(
                f"Agent '{agent_id}' is not a selectable worker."
            )

        if not self.registry.can_call(
            caller_id,
            agent_id,
        ):
            raise PermissionError(
                f"'{caller_id}' cannot call '{agent_id}'."
            )

        tool_ids = list(
            dict.fromkeys(assigned_tool_ids or [])
        )

        self.registry.validate_tool_ids(
            tool_ids
        )

        for tool_id in tool_ids:
            if not self.registry.can_assign_tool(
                caller_id,
                tool_id,
            ):
                raise PermissionError(
                    f"'{caller_id}' cannot assign "
                    f"tool '{tool_id}'."
                )

        if (
            agent["type"] == "dynamic"
            and (
                not runtime_system_prompt
                or not runtime_system_prompt.strip()
            )
        ):
            raise ValueError(
                f"Agent '{agent_id}' requires a "
                "runtime system prompt."
            )

        service_class = self._load_service_class(
            agent
        )

        service = service_class(
            model=self.model,
            base_url=self.base_url,
        )

        if not callable(
            getattr(service, "run", None)
        ):
            raise TypeError(
                f"Agent service '{agent_id}' must define "
                "an async run(request) method."
            )

        return AgentRunner(
            agent_id=agent_id,
            service=service,
            caller_id=caller_id,
            assigned_tool_ids=tool_ids,
            runtime_system_prompt=(
                runtime_system_prompt.strip()
                if runtime_system_prompt
                else None
            ),
        )
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agent.planner_agent import builtin_tool
from share.agent import (
    ToolCallingAgent,
    build_strong_system_prompt,
)
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import AgentRequest, TeamPlan
from share.tool_loader import ToolLoader


AGENT_ID = "planner_agent"
SUPPORTED_ORCHESTRATION = {
    "sequential",
    "parallel",
}


class PlannerAgentService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = Registry()

        self.llm = Copilot(
            model=model,
            base_url=base_url,
        )

        self.system_prompt_text = Path(
            __file__
        ).with_name(
            "system_prompt.txt"
        ).read_text(
            encoding="utf-8"
        ).strip()

    def _validate_request(
        self,
        request: AgentRequest,
    ) -> None:
        if not self.registry.can_call(
            request.caller_id,
            AGENT_ID,
        ):
            raise PermissionError(
                f"'{request.caller_id}' cannot call "
                f"'{AGENT_ID}'."
            )

        for tool_id in request.assigned_tool_ids:
            if not self.registry.can_assign_tool(
                request.caller_id,
                tool_id,
            ):
                raise PermissionError(
                    f"Tool cannot be assigned: {tool_id}"
                )

    def _validate_plan(
        self,
        plan: TeamPlan,
    ) -> None:
        if plan.orchestration not in SUPPORTED_ORCHESTRATION:
            raise ValueError(
                "Only sequential and parallel "
                "orchestration are currently supported."
            )

        if not plan.members or not plan.tasks:
            raise ValueError(
                "Team plan must contain members and tasks."
            )

        member_ids = [
            member.instance_id
            for member in plan.members
        ]

        task_ids = [
            task.task_id
            for task in plan.tasks
        ]

        if len(member_ids) != len(set(member_ids)):
            raise ValueError(
                "Duplicate member instance IDs."
            )

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Duplicate task IDs."
            )

        member_id_set = set(member_ids)
        task_id_set = set(task_ids)

        for member in plan.members:
            if member.agent_id != "agent_of_requirement":
                raise ValueError(
                    "Team members must use "
                    "'agent_of_requirement'."
                )

            for tool_id in member.assigned_tool_ids:
                self.registry.get_tool(tool_id)

        dependencies = {}

        for task in plan.tasks:
            if task.agent_instance_id not in member_id_set:
                raise ValueError(
                    f"Unknown member in task: {task.task_id}"
                )

            if not set(task.dependencies) <= task_id_set:
                raise ValueError(
                    f"Unknown dependency in task: {task.task_id}"
                )

            if task.task_id in task.dependencies:
                raise ValueError(
                    f"Task cannot depend on itself: {task.task_id}"
                )

            dependencies[task.task_id] = set(
                task.dependencies
            )

        while dependencies:
            ready = [
                task_id
                for task_id, deps in dependencies.items()
                if not deps
            ]

            if not ready:
                raise ValueError(
                    "Task dependency cycle detected."
                )

            for task_id in ready:
                dependencies.pop(task_id)

            for deps in dependencies.values():
                deps.difference_update(ready)

    async def run(
        self,
        request: AgentRequest,
    ) -> TeamPlan:
        self._validate_request(request)

        tool_loader = ToolLoader(
            registry=self.registry,
            assigned_tool_ids=request.assigned_tool_ids,
        )

        tool_spec = (
            builtin_tool.get_tool_spec()
            + tool_loader.get_tool_spec()
        )

        system_prompt = build_strong_system_prompt(
            self.system_prompt_text,
            tool_spec,
            enable_communication=False,
        )

        def execute_tool(
            name: str,
            arguments: dict,
        ):
            if builtin_tool.has_tool(name):
                return builtin_tool.execute(
                    name,
                    arguments,
                )

            return tool_loader.execute(
                name,
                arguments,
            )

        agent = ToolCallingAgent(
            llm=self.llm,
            system_prompt=system_prompt,
            tool_executor=execute_tool,
            agent_id=AGENT_ID,
            instance_id=(
                f"{AGENT_ID}_{uuid4().hex[:10]}"
            ),
            max_steps=request.max_steps,
            history_mode="summary",
            enable_communication=False,
        )

        planning_context = {
            "original_request": request.task,
            "analysis_and_context": request.context,
            "available_agents": self.registry.list_agents(),
            "available_tools": self.registry.list_tools(),
            "supported_orchestration": sorted(
                SUPPORTED_ORCHESTRATION
            ),
        }

        final_answer = await agent.run(
            "Create a team plan using this context:\n\n"
            + json.dumps(
                planning_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        plan = TeamPlan.model_validate_json(
            final_answer.strip()
        )

        self._validate_plan(plan)
        return plan
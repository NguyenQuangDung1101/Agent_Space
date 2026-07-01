import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from agent.planner_agent import builtin_tool
from share.agent import (
    ToolCallingAgent,
    UserContactRaised,
    build_strong_system_prompt,
    build_waiting_result,
    run_resumable,
)
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import AgentRequest, AgentResult, ExecutionEvent, TeamPlan


AGENT_ID = "planner_agent"
SUPPORTED_ORCHESTRATION = {"sequential", "parallel", "supervisor"}
EventHandler = Callable[[ExecutionEvent], Any]


class PlannerAgentService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = Registry()
        self.llm = Copilot(model=model, base_url=base_url)
        self.system_prompt_text = Path(__file__).with_name(
            "system_prompt.txt"
        ).read_text(encoding="utf-8").strip()

    def _validate_request(self, request: AgentRequest) -> None:
        if not self.registry.can_call(request.caller_id, AGENT_ID):
            raise PermissionError(
                f"'{request.caller_id}' cannot call '{AGENT_ID}'."
            )

    def _validate_plan(
        self,
        plan: TeamPlan,
        worker_catalog: list[dict],
        tool_catalog: list[dict],
    ) -> None:
        if not plan.members or not plan.tasks:
            raise ValueError("Team plan must contain members and tasks.")

        allowed_agents = {item["id"]: item for item in worker_catalog}
        allowed_tools = {item["id"] for item in tool_catalog}
        member_ids = [member.instance_id for member in plan.members]
        task_ids = [task.task_id for task in plan.tasks]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Duplicate member instance IDs.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs.")

        for member in plan.members:
            agent = allowed_agents.get(member.agent_id)
            if agent is None:
                raise ValueError(f"Invalid worker agent: {member.agent_id}")
            if set(member.assigned_tool_ids) - allowed_tools:
                raise ValueError(f"Invalid tools for member: {member.instance_id}")
            if agent["type"] == "dynamic" and not member.system_prompt:
                raise ValueError(
                    f"Dynamic member requires system_prompt: {member.instance_id}"
                )

        member_id_set = set(member_ids)
        task_id_set = set(task_ids)
        dependencies = {}
        for task in plan.tasks:
            if task.agent_instance_id not in member_id_set:
                raise ValueError(f"Unknown member in task: {task.task_id}")
            if not set(task.dependencies) <= task_id_set:
                raise ValueError(f"Unknown dependency in task: {task.task_id}")
            if task.task_id in task.dependencies:
                raise ValueError(f"Task depends on itself: {task.task_id}")
            dependencies[task.task_id] = set(task.dependencies)

        while dependencies:
            ready = [task_id for task_id, deps in dependencies.items() if not deps]
            if not ready:
                raise ValueError("Task dependency cycle detected.")
            for task_id in ready:
                dependencies.pop(task_id)
            for deps in dependencies.values():
                deps.difference_update(ready)

        if plan.orchestration == "supervisor":
            if plan.supervisor is None:
                raise ValueError("Supervisor configuration is required.")
            supervisor_id = plan.supervisor.supervisor_instance_id
            if supervisor_id not in member_id_set:
                raise ValueError("Unknown supervisor instance ID.")
            if any(
                task.agent_instance_id == supervisor_id for task in plan.tasks
            ):
                raise ValueError("Supervisor must not own execution tasks.")
        elif plan.supervisor is not None:
            raise ValueError(
                "Supervisor configuration is only valid in supervisor mode."
            )

    async def run(
        self,
        request: AgentRequest,
        event_handler: Optional[EventHandler] = None,
    ) -> AgentResult:
        instance_id = request.instance_id or f"{AGENT_ID}_{uuid4().hex[:10]}"
        try:
            self._validate_request(request)
            worker_catalog = request.context.get(
                "worker_catalog", self.registry.list_selectable_agents()
            )
            tool_catalog = request.context.get(
                "tool_catalog", self.registry.list_tools()
            )
            agent = ToolCallingAgent(
                llm=self.llm,
                system_prompt=build_strong_system_prompt(
                    self.system_prompt_text,
                    builtin_tool.get_tool_spec(),
                    enable_communication=False,
                ),
                tool_executor=builtin_tool.execute,
                event_handler=event_handler,
                agent_id=AGENT_ID,
                instance_id=instance_id,
                max_steps=request.max_steps,
                history_mode="summary",
                enable_communication=False,
            )
            planning_input = {
                "original_request": request.task,
                "analysis": request.context.get("analysis"),
                "user_context": request.context.get("user_context", {}),
                "worker_catalog": worker_catalog,
                "tool_catalog": tool_catalog,
                "supported_orchestration": sorted(SUPPORTED_ORCHESTRATION),
            }
            try:
                answer = await run_resumable(
                    agent,
                    request,
                    "Create the team plan from this information:\n\n"
                    + json.dumps(
                        planning_input,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                )
            except UserContactRaised as signal:
                return build_waiting_result(request, agent, signal.contact)

            plan = TeamPlan.model_validate_json(answer.strip())
            self._validate_plan(plan, worker_catalog, tool_catalog)
            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=agent.instance_id,
                status="COMPLETED",
                final_answer=plan.model_dump_json(),
            )
        except Exception as error:
            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=instance_id,
                status="FAILED",
                error=f"{type(error).__name__}: {error}",
            )

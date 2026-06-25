import asyncio
from typing import Any, Optional

from agent.agent_of_requirement.service import (
    AgentOfRequirementService,
)
from share.schemas import (
    AgentRequest,
    AgentResult,
    ExecutionResult,
    TeamMember,
    TeamPlan,
    TeamTask,
)


class Orchestrator:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_steps: int = 10,
    ) -> None:
        self.worker = AgentOfRequirementService(
            model=model,
            base_url=base_url,
        )

        self.max_steps = max_steps

    @staticmethod
    def _validate_plan(
        plan: TeamPlan,
    ) -> None:
        if plan.orchestration not in {
            "sequential",
            "parallel",
        }:
            raise ValueError(
                "Unsupported orchestration mode."
            )

        member_ids = {
            member.instance_id
            for member in plan.members
        }

        task_ids = {
            task.task_id
            for task in plan.tasks
        }

        if len(member_ids) != len(plan.members):
            raise ValueError(
                "Duplicate member instance IDs."
            )

        if len(task_ids) != len(plan.tasks):
            raise ValueError(
                "Duplicate task IDs."
            )

        for task in plan.tasks:
            if task.agent_instance_id not in member_ids:
                raise ValueError(
                    f"Unknown member: {task.agent_instance_id}"
                )

            if not set(task.dependencies) <= task_ids:
                raise ValueError(
                    f"Unknown dependency in task: {task.task_id}"
                )

    async def _run_task(
        self,
        session_id: str,
        original_request: str,
        member: TeamMember,
        task: TeamTask,
        shared_context: dict[str, Any],
        task_outputs: dict[str, str],
    ) -> AgentResult:
        dependency_outputs = {
            dependency: task_outputs[dependency]
            for dependency in task.dependencies
            if dependency in task_outputs
        }

        runtime_prompt = (
            member.system_prompt
            or (
                f"You are the {member.role}.\n"
                f"Responsibility: {member.task}"
            )
        )

        result = await self.worker.run(
            AgentRequest(
                session_id=session_id,
                caller_id="manager",
                task=task.instruction,
                context={
                    **shared_context,
                    "original_request": original_request,
                    "role": member.role,
                    "dependency_outputs": dependency_outputs,
                },
                assigned_tool_ids=(
                    member.assigned_tool_ids
                ),
                runtime_system_prompt=runtime_prompt,
                max_steps=self.max_steps,
            )
        )

        # Keep the planned runtime identity in team results.
        result.instance_id = member.instance_id

        return result

    async def execute(
        self,
        session_id: str,
        original_request: str,
        plan: TeamPlan,
        context: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        try:
            self._validate_plan(plan)
        except Exception as error:
            return ExecutionResult(
                session_id=session_id,
                status="FAILED",
                execution_mode="team",
                errors=[
                    f"{type(error).__name__}: {error}"
                ],
            )

        context = context or {}

        members = {
            member.instance_id: member
            for member in plan.members
        }

        pending = {
            task.task_id: task
            for task in plan.tasks
        }

        completed: set[str] = set()
        task_outputs: dict[str, str] = {}
        agent_results: list[AgentResult] = []

        while pending:
            ready = [
                task
                for task in plan.tasks
                if (
                    task.task_id in pending
                    and set(task.dependencies) <= completed
                )
            ]

            if not ready:
                return ExecutionResult(
                    session_id=session_id,
                    status="FAILED",
                    execution_mode="team",
                    agent_results=agent_results,
                    errors=[
                        "No executable tasks remain. "
                        "The plan may contain a dependency cycle."
                    ],
                )

            if plan.orchestration == "sequential":
                ready = ready[:1]

            executions = [
                self._run_task(
                    session_id=session_id,
                    original_request=original_request,
                    member=members[
                        task.agent_instance_id
                    ],
                    task=task,
                    shared_context=context,
                    task_outputs=task_outputs,
                )
                for task in ready
            ]

            results = await asyncio.gather(
                *executions
            )

            for task, result in zip(
                ready,
                results,
            ):
                agent_results.append(result)

                if result.status != "COMPLETED":
                    return ExecutionResult(
                        session_id=session_id,
                        status="FAILED",
                        execution_mode="team",
                        agent_results=agent_results,
                        errors=[
                            result.error
                            or f"Task failed: {task.task_id}"
                        ],
                    )

                task_outputs[task.task_id] = (
                    result.final_answer or ""
                )

                completed.add(task.task_id)
                pending.pop(task.task_id)

        return ExecutionResult(
            session_id=session_id,
            status="COMPLETED",
            execution_mode="team",
            agent_results=agent_results,
        )
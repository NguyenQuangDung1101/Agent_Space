import asyncio
import json
from typing import Any, Callable, Optional

from share.agent_factory import AgentFactory
from share.registry import Registry
from share.schemas import (
    AgentResult,
    ExecutionEvent,
    ExecutionResult,
    Message,
    TeamMember,
    TeamPlan,
    TeamTask,
)


EventHandler = Callable[[ExecutionEvent], Any]


class Orchestrator:
    def __init__(
        self,
        registry: Optional[Registry] = None,
        factory: Optional[AgentFactory] = None,
        max_steps: int = 10,
    ) -> None:
        self.registry = registry or Registry()
        self.factory = factory or AgentFactory(
            registry=self.registry
        )
        self.max_steps = max_steps

    def validate_plan(self, plan: TeamPlan) -> None:
        if not plan.members or not plan.tasks:
            raise ValueError(
                "Team plan must contain members and tasks."
            )

        member_ids = [
            member.instance_id
            for member in plan.members
        ]
        task_ids = [task.task_id for task in plan.tasks]

        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Duplicate member instance IDs.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs.")

        self.registry.validate_agent_ids(
            [member.agent_id for member in plan.members],
            selectable_only=True,
        )

        for member in plan.members:
            self.registry.validate_tool_ids(
                member.assigned_tool_ids
            )
            if not self.registry.can_call(
                "manager",
                member.agent_id,
            ):
                raise PermissionError(
                    f"Manager cannot call: {member.agent_id}"
                )

            agent = self.registry.get_agent(member.agent_id)
            if agent["type"] == "dynamic" and not member.system_prompt:
                raise ValueError(
                    "Dynamic member requires system_prompt: "
                    f"{member.instance_id}"
                )

        member_id_set = set(member_ids)
        task_id_set = set(task_ids)
        dependencies: dict[str, set[str]] = {}

        for task in plan.tasks:
            if task.agent_instance_id not in member_id_set:
                raise ValueError(
                    f"Unknown member: {task.agent_instance_id}"
                )
            if not set(task.dependencies) <= task_id_set:
                raise ValueError(
                    f"Unknown dependency in task: {task.task_id}"
                )
            if task.task_id in task.dependencies:
                raise ValueError(
                    f"Task depends on itself: {task.task_id}"
                )
            dependencies[task.task_id] = set(task.dependencies)

        while dependencies:
            ready = [
                task_id
                for task_id, deps in dependencies.items()
                if not deps
            ]
            if not ready:
                raise ValueError("Task dependency cycle detected.")

            for task_id in ready:
                dependencies.pop(task_id)
            for deps in dependencies.values():
                deps.difference_update(ready)

        if plan.orchestration == "supervisor":
            if plan.supervisor is None:
                raise ValueError(
                    "Supervisor configuration is required."
                )
            supervisor_id = plan.supervisor.supervisor_instance_id
            if supervisor_id not in member_id_set:
                raise ValueError("Unknown supervisor instance ID.")
            if any(
                task.agent_instance_id == supervisor_id
                for task in plan.tasks
            ):
                raise ValueError(
                    "Supervisor must not own execution tasks."
                )
        elif plan.supervisor is not None:
            raise ValueError(
                "Supervisor configuration is only valid "
                "in supervisor mode."
            )

    @staticmethod
    def _record_event(
        events: list[ExecutionEvent],
        handler: Optional[EventHandler],
        event: ExecutionEvent,
    ) -> None:
        events.append(event)
        if handler:
            handler(event)

    @classmethod
    def _emit(
        cls,
        events: list[ExecutionEvent],
        handler: Optional[EventHandler],
        event_type: str,
        **kwargs: Any,
    ) -> None:
        cls._record_event(
            events,
            handler,
            ExecutionEvent(
                event_type=event_type,
                **kwargs,
            ),
        )

    @staticmethod
    def _parallel_batch(
        tasks: list[TeamTask],
    ) -> list[TeamTask]:
        selected: list[TeamTask] = []
        busy_members: set[str] = set()

        for task in tasks:
            if task.agent_instance_id not in busy_members:
                selected.append(task)
                busy_members.add(task.agent_instance_id)

        return selected

    @staticmethod
    def _route_messages(
        result: AgentResult,
        sender_id: str,
        member_ids: set[str],
        inboxes: dict[str, list[Message]],
        message_history: list[Message],
    ) -> None:
        normalized: list[Message] = []

        for message in result.messages:
            routed = message.model_copy(
                update={"sender": sender_id}
            )
            normalized.append(routed)
            message_history.append(routed)

            if routed.message_type == "direct":
                if (
                    routed.recipient in member_ids
                    and routed.recipient != sender_id
                ):
                    inboxes[routed.recipient].append(routed)

            elif routed.message_type == "broadcast":
                for recipient in member_ids - {sender_id}:
                    inboxes[recipient].append(
                        routed.model_copy(
                            update={"recipient": recipient}
                        )
                    )

        result.messages = normalized

    async def _run_member(
        self,
        session_id: str,
        original_request: str,
        member: TeamMember,
        task: str,
        task_id: Optional[str],
        context: dict[str, Any],
        dependency_outputs: dict[str, str],
        inbox: list[Message],
        members: list[TeamMember],
        max_steps: int,
        event_handler: Optional[EventHandler],
    ) -> AgentResult:
        runner = self.factory.create(
            agent_id=member.agent_id,
            caller_id="manager",
            assigned_tool_ids=member.assigned_tool_ids,
            runtime_system_prompt=member.system_prompt,
        )

        def forward_event(event: ExecutionEvent) -> None:
            if event_handler:
                event_handler(
                    event.model_copy(
                        update={
                            "agent_id": member.agent_id,
                            "instance_id": member.instance_id,
                            "task_id": task_id,
                            "task": task,
                            "assigned_tool_ids": (
                                member.assigned_tool_ids
                            ),
                        }
                    )
                )

        result = await runner.run(
            session_id=session_id,
            task=task,
            context={
                **context,
                "original_request": original_request,
                "team_role": member.role,
                "team_task": member.task,
                "current_task_id": task_id,
                "dependency_outputs": dependency_outputs,
                "inbox": [
                    item.model_dump(mode="json")
                    for item in inbox
                ],
                "team_members": [
                    {
                        "instance_id": item.instance_id,
                        "role": item.role,
                    }
                    for item in members
                ],
            },
            max_steps=max_steps,
            event_handler=forward_event,
        )
        result.instance_id = member.instance_id
        return result

    async def _choose_supervisor_task(
        self,
        session_id: str,
        original_request: str,
        supervisor: TeamMember,
        ready: list[TeamTask],
        completed: set[str],
        task_outputs: dict[str, str],
        context: dict[str, Any],
        inbox: list[Message],
        members: list[TeamMember],
        max_steps: int,
        event_handler: Optional[EventHandler],
    ) -> tuple[str, AgentResult]:
        selection_task = (
            "Choose exactly one next task from ready_tasks. "
            "Return only its task_id.\n\n"
            + json.dumps(
                {
                    "ready_tasks": [
                        task.model_dump()
                        for task in ready
                    ],
                    "completed_task_ids": sorted(completed),
                    "task_outputs": task_outputs,
                },
                ensure_ascii=False,
                default=str,
            )
        )

        result = await self._run_member(
            session_id=session_id,
            original_request=original_request,
            member=supervisor,
            task=selection_task,
            task_id=None,
            context=context,
            dependency_outputs={},
            inbox=inbox,
            members=members,
            max_steps=max_steps,
            event_handler=event_handler,
        )

        if result.status != "COMPLETED":
            raise RuntimeError(
                result.error or "Supervisor execution failed."
            )

        choice = (result.final_answer or "").strip().strip("`\"")
        if choice.startswith("{"):
            choice = str(
                json.loads(choice).get("task_id", "")
            ).strip()

        ready_ids = {task.task_id for task in ready}
        if choice not in ready_ids:
            raise ValueError(
                f"Supervisor selected invalid task: {choice}"
            )

        return choice, result

    async def execute(
        self,
        session_id: str,
        original_request: str,
        plan: TeamPlan,
        context: Optional[dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        event_handler: Optional[EventHandler] = None,
    ) -> ExecutionResult:
        events: list[ExecutionEvent] = []
        agent_results: list[AgentResult] = []
        messages: list[Message] = []

        try:
            self.validate_plan(plan)
            members = {
                member.instance_id: member
                for member in plan.members
            }
            member_ids = set(members)
            inboxes = {
                member_id: []
                for member_id in member_ids
            }
            pending = {
                task.task_id: task
                for task in plan.tasks
            }
            completed: set[str] = set()
            task_outputs: dict[str, str] = {}
            shared_context = dict(context or {})
            run_steps = max_steps or self.max_steps
            rounds = 0

            def worker_event(event: ExecutionEvent) -> None:
                self._record_event(
                    events,
                    event_handler,
                    event,
                )

            while pending:
                ready = [
                    task
                    for task in plan.tasks
                    if task.task_id in pending
                    and set(task.dependencies) <= completed
                ]
                if not ready:
                    raise ValueError(
                        "No executable tasks remain. "
                        "The plan may contain a cycle."
                    )

                if plan.orchestration == "sequential":
                    batch = ready[:1]

                elif plan.orchestration == "parallel":
                    batch = self._parallel_batch(ready)

                else:
                    rounds += 1
                    if rounds > plan.supervisor.max_rounds:
                        raise RuntimeError(
                            "Supervisor exceeded max_rounds."
                        )

                    supervisor_id = (
                        plan.supervisor.supervisor_instance_id
                    )
                    supervisor = members[supervisor_id]
                    supervisor_inbox = list(
                        inboxes[supervisor_id]
                    )
                    inboxes[supervisor_id].clear()

                    self._emit(
                        events,
                        event_handler,
                        "agent_running",
                        agent_id=supervisor.agent_id,
                        instance_id=supervisor_id,
                        task="Select the next ready team task.",
                        assigned_tool_ids=(
                            supervisor.assigned_tool_ids
                        ),
                    )

                    choice, supervisor_result = (
                        await self._choose_supervisor_task(
                            session_id=session_id,
                            original_request=original_request,
                            supervisor=supervisor,
                            ready=ready,
                            completed=completed,
                            task_outputs=task_outputs,
                            context={
                                **shared_context,
                                "message_history": [
                                    item.model_dump(mode="json")
                                    for item in messages
                                ],
                            },
                            inbox=supervisor_inbox,
                            members=plan.members,
                            max_steps=run_steps,
                            event_handler=worker_event,
                        )
                    )
                    agent_results.append(supervisor_result)
                    self._route_messages(
                        supervisor_result,
                        supervisor_id,
                        member_ids,
                        inboxes,
                        messages,
                    )

                    self._emit(
                        events,
                        event_handler,
                        "agent_completed",
                        agent_id=supervisor.agent_id,
                        instance_id=supervisor_id,
                        task="Select the next ready team task.",
                        assigned_tool_ids=(
                            supervisor.assigned_tool_ids
                        ),
                    )
                    self._emit(
                        events,
                        event_handler,
                        "supervisor_selected_task",
                        agent_id=supervisor.agent_id,
                        instance_id=supervisor_id,
                        task_id=choice,
                    )
                    batch = [pending[choice]]

                executions = []

                for task in batch:
                    member = members[task.agent_instance_id]
                    member_inbox = list(
                        inboxes[member.instance_id]
                    )
                    inboxes[member.instance_id].clear()
                    dependency_outputs = {
                        dependency: task_outputs[dependency]
                        for dependency in task.dependencies
                    }

                    self._emit(
                        events,
                        event_handler,
                        "agent_running",
                        agent_id=member.agent_id,
                        instance_id=member.instance_id,
                        task_id=task.task_id,
                        task=task.instruction,
                        assigned_tool_ids=(
                            member.assigned_tool_ids
                        ),
                    )

                    executions.append(
                        self._run_member(
                            session_id=session_id,
                            original_request=original_request,
                            member=member,
                            task=task.instruction,
                            task_id=task.task_id,
                            context={
                                **shared_context,
                                "message_history": [
                                    item.model_dump(mode="json")
                                    for item in messages
                                ],
                                "completed_task_outputs": dict(
                                    task_outputs
                                ),
                            },
                            dependency_outputs=dependency_outputs,
                            inbox=member_inbox,
                            members=plan.members,
                            max_steps=run_steps,
                            event_handler=worker_event,
                        )
                    )

                results = await asyncio.gather(*executions)

                for task, result in zip(batch, results):
                    member = members[task.agent_instance_id]
                    agent_results.append(result)
                    self._route_messages(
                        result,
                        member.instance_id,
                        member_ids,
                        inboxes,
                        messages,
                    )

                    self._emit(
                        events,
                        event_handler,
                        (
                            "agent_completed"
                            if result.status == "COMPLETED"
                            else "agent_failed"
                        ),
                        agent_id=member.agent_id,
                        instance_id=member.instance_id,
                        task_id=task.task_id,
                        task=task.instruction,
                        assigned_tool_ids=(
                            member.assigned_tool_ids
                        ),
                        details={
                            "error": result.error,
                        },
                    )

                    if result.status != "COMPLETED":
                        raise RuntimeError(
                            result.error
                            or f"Task failed: {task.task_id}"
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
                messages=messages,
                events=events,
            )

        except Exception as error:
            return ExecutionResult(
                session_id=session_id,
                status="FAILED",
                execution_mode="team",
                agent_results=agent_results,
                messages=messages,
                events=events,
                errors=[
                    f"{type(error).__name__}: {error}"
                ],
            )

import asyncio
import json
from typing import Any, Callable, Optional

from share.agent_factory import AgentFactory
from share.registry import Registry
from share.schemas import (
    AgentMessage,
    AgentResult,
    ExecutionEvent,
    ExecutionResult,
    TeamMember,
    TeamPlan,
    TeamTask,
    UserContactRequest,
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
        self.factory = factory or AgentFactory(registry=self.registry)
        self.max_steps = max_steps

    def validate_plan(self, plan: TeamPlan) -> None:
        if not plan.members or not plan.tasks:
            raise ValueError("Team plan must contain members and tasks.")

        member_ids = [item.instance_id for item in plan.members]
        task_ids = [item.task_id for item in plan.tasks]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Duplicate member instance IDs.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs.")

        self.registry.validate_agent_ids(
            [item.agent_id for item in plan.members],
            selectable_only=True,
        )
        for member in plan.members:
            self.registry.validate_tool_ids(member.assigned_tool_ids)
            if not self.registry.can_call("manager", member.agent_id):
                raise PermissionError(
                    f"Manager cannot call: {member.agent_id}"
                )
            agent = self.registry.get_agent(member.agent_id)
            if agent["type"] == "dynamic" and not member.system_prompt:
                raise ValueError(
                    "Dynamic member requires system_prompt: "
                    f"{member.instance_id}"
                )

        member_set = set(member_ids)
        task_set = set(task_ids)
        dependencies: dict[str, set[str]] = {}
        for task in plan.tasks:
            if task.agent_instance_id not in member_set:
                raise ValueError(
                    f"Unknown member: {task.agent_instance_id}"
                )
            if not set(task.dependencies) <= task_set:
                raise ValueError(
                    f"Unknown dependency in task: {task.task_id}"
                )
            if task.task_id in task.dependencies:
                raise ValueError(
                    f"Task depends on itself: {task.task_id}"
                )
            dependencies[task.task_id] = set(task.dependencies)

        while dependencies:
            ready = [key for key, value in dependencies.items() if not value]
            if not ready:
                raise ValueError("Task dependency cycle detected.")
            for task_id in ready:
                dependencies.pop(task_id)
            for value in dependencies.values():
                value.difference_update(ready)

        if plan.orchestration == "supervisor":
            if plan.supervisor is None:
                raise ValueError("Supervisor configuration is required.")
            supervisor_id = plan.supervisor.supervisor_instance_id
            if supervisor_id not in member_set:
                raise ValueError("Unknown supervisor instance ID.")
            if any(
                task.agent_instance_id == supervisor_id
                for task in plan.tasks
            ):
                raise ValueError("Supervisor must not own execution tasks.")
        elif plan.supervisor is not None:
            raise ValueError(
                "Supervisor configuration is only valid in supervisor mode."
            )

    @staticmethod
    def _emit(
        events: list[ExecutionEvent],
        handler: Optional[EventHandler],
        event_type: str,
        **kwargs: Any,
    ) -> None:
        event = ExecutionEvent(event_type=event_type, **kwargs)
        events.append(event)
        if handler:
            handler(event)

    @staticmethod
    def _parallel_batch(tasks: list[TeamTask]) -> list[TeamTask]:
        selected = []
        busy = set()
        for task in tasks:
            if task.agent_instance_id not in busy:
                selected.append(task)
                busy.add(task.agent_instance_id)
        return selected

    @staticmethod
    def _route_messages(
        result: AgentResult,
        sender_id: str,
        member_ids: set[str],
        inboxes: dict[str, list[AgentMessage]],
        history: list[AgentMessage],
    ) -> None:
        normalized = []
        for message in result.messages:
            routed = message.model_copy(update={"sender": sender_id})
            normalized.append(routed)
            history.append(routed)
            if routed.message_type == "direct":
                if routed.recipient in member_ids and routed.recipient != sender_id:
                    inboxes[routed.recipient].append(routed)
            elif routed.message_type == "broadcast":
                for recipient in member_ids - {sender_id}:
                    inboxes[recipient].append(
                        routed.model_copy(update={"recipient": recipient})
                    )
        result.messages = normalized

    async def _run_member(
        self,
        conversation_id: str,
        session_id: str,
        original_request: str,
        member: TeamMember,
        task: str,
        task_id: Optional[str],
        context: dict[str, Any],
        dependency_outputs: dict[str, str],
        inbox: list[AgentMessage],
        members: list[TeamMember],
        max_steps: int,
        event_handler: Optional[EventHandler],
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> AgentResult:
        runner = self.factory.create(
            agent_id=member.agent_id,
            caller_id="manager",
            assigned_tool_ids=member.assigned_tool_ids,
            runtime_system_prompt=member.system_prompt,
        )

        def forward(event: ExecutionEvent) -> None:
            if event_handler:
                event_handler(
                    event.model_copy(
                        update={
                            "agent_id": member.agent_id,
                            "instance_id": member.instance_id,
                            "task_id": task_id,
                            "task": task,
                            "assigned_tool_ids": member.assigned_tool_ids,
                        }
                    )
                )

        return await runner.run(
            conversation_id=conversation_id,
            session_id=session_id,
            instance_id=member.instance_id,
            task=task,
            context={
                **context,
                "original_request": original_request,
                "team_role": member.role,
                "team_task": member.task,
                "current_task_id": task_id,
                "dependency_outputs": dependency_outputs,
                "inbox": [item.model_dump(mode="json") for item in inbox],
                "team_members": [
                    {"instance_id": item.instance_id, "role": item.role}
                    for item in members
                ],
            },
            max_steps=max_steps,
            event_handler=forward,
            checkpoint=checkpoint,
            user_response=user_response,
        )

    @staticmethod
    def _supervisor_task(
        ready: list[TeamTask],
        completed: set[str],
        outputs: dict[str, str],
    ) -> str:
        return (
            "Choose exactly one next task from ready_tasks. "
            "Return only its task_id.\n\n"
            + json.dumps(
                {
                    "ready_tasks": [item.model_dump() for item in ready],
                    "completed_task_ids": sorted(completed),
                    "task_outputs": outputs,
                },
                ensure_ascii=False,
                default=str,
            )
        )

    @staticmethod
    def _parse_choice(result: AgentResult, ready_ids: set[str]) -> str:
        choice = (result.final_answer or "").strip().strip("`\"")
        if choice.startswith("{"):
            choice = str(json.loads(choice).get("task_id", "")).strip()
        if choice not in ready_ids:
            raise ValueError(f"Supervisor selected invalid task: {choice}")
        return choice

    @staticmethod
    def _serialize_state(
        pending: set[str],
        completed: set[str],
        outputs: dict[str, str],
        inboxes: dict[str, list[AgentMessage]],
        results: list[AgentResult],
        messages: list[AgentMessage],
        rounds: int,
        waiting: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "pending": sorted(pending),
            "completed": sorted(completed),
            "outputs": outputs,
            "inboxes": {
                key: [item.model_dump(mode="json") for item in value]
                for key, value in inboxes.items()
            },
            "agent_results": [
                item.model_dump(mode="json") for item in results
            ],
            "messages": [item.model_dump(mode="json") for item in messages],
            "rounds": rounds,
            "waiting": waiting,
        }

    def _waiting_result(
        self,
        session_id: str,
        events: list[ExecutionEvent],
        pending: set[str],
        completed: set[str],
        outputs: dict[str, str],
        inboxes: dict[str, list[AgentMessage]],
        results: list[AgentResult],
        messages: list[AgentMessage],
        rounds: int,
        waiting: list[dict[str, Any]],
    ) -> ExecutionResult:
        return ExecutionResult(
            session_id=session_id,
            status="WAITING_FOR_USER",
            execution_mode="team",
            agent_results=results,
            messages=messages,
            events=events,
            pending_contact=UserContactRequest.model_validate(
                waiting[0]["contact"]
            ),
            checkpoint=self._serialize_state(
                pending,
                completed,
                outputs,
                inboxes,
                results,
                messages,
                rounds,
                waiting,
            ),
        )

    async def execute(
        self,
        conversation_id: str,
        session_id: str,
        original_request: str,
        plan: TeamPlan,
        context: Optional[dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        event_handler: Optional[EventHandler] = None,
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> ExecutionResult:
        events: list[ExecutionEvent] = []
        try:
            self.validate_plan(plan)
            members = {item.instance_id: item for item in plan.members}
            tasks = {item.task_id: item for item in plan.tasks}
            member_ids = set(members)
            shared_context = dict(context or {})
            run_steps = max_steps or self.max_steps

            if checkpoint:
                pending = set(checkpoint.get("pending", []))
                completed = set(checkpoint.get("completed", []))
                outputs = dict(checkpoint.get("outputs", {}))
                inboxes = {
                    key: [AgentMessage.model_validate(item) for item in value]
                    for key, value in checkpoint.get("inboxes", {}).items()
                }
                results = [
                    AgentResult.model_validate(item)
                    for item in checkpoint.get("agent_results", [])
                ]
                messages = [
                    AgentMessage.model_validate(item)
                    for item in checkpoint.get("messages", [])
                ]
                rounds = int(checkpoint.get("rounds", 0))
                waiting = list(checkpoint.get("waiting", []))
            else:
                pending = set(tasks)
                completed = set()
                outputs: dict[str, str] = {}
                inboxes = {member_id: [] for member_id in member_ids}
                results: list[AgentResult] = []
                messages: list[AgentMessage] = []
                rounds = 0
                waiting: list[dict[str, Any]] = []

            def worker_event(event: ExecutionEvent) -> None:
                events.append(event)
                if event_handler:
                    event_handler(event)

            if waiting:
                item = waiting.pop(0)
                member = members[item["member_id"]]
                result = await self._run_member(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    original_request=original_request,
                    member=member,
                    task=item["task"],
                    task_id=item.get("task_id"),
                    context=item["context"],
                    dependency_outputs=item.get("dependency_outputs", {}),
                    inbox=[
                        AgentMessage.model_validate(message)
                        for message in item.get("inbox", [])
                    ],
                    members=plan.members,
                    max_steps=run_steps,
                    event_handler=worker_event,
                    checkpoint=item["checkpoint"],
                    user_response=user_response,
                )
                if result.status == "WAITING_FOR_USER":
                    item["checkpoint"] = result.checkpoint
                    item["contact"] = result.user_contact.model_dump(mode="json")
                    waiting.insert(0, item)
                    return self._waiting_result(
                        session_id,
                        events,
                        pending,
                        completed,
                        outputs,
                        inboxes,
                        results,
                        messages,
                        rounds,
                        waiting,
                    )
                if result.status != "COMPLETED":
                    raise RuntimeError(result.error or "Agent resume failed.")

                results.append(result)
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
                    "agent_completed",
                    agent_id=member.agent_id,
                    instance_id=member.instance_id,
                    task_id=item.get("task_id"),
                    task=item["task"],
                    assigned_tool_ids=member.assigned_tool_ids,
                )

                if item["kind"] == "task":
                    task_id = item["task_id"]
                    outputs[task_id] = result.final_answer or ""
                    completed.add(task_id)
                    pending.discard(task_id)
                else:
                    item["selected_task_id"] = self._parse_choice(
                        result, set(item["ready_task_ids"])
                    )
                    self._emit(
                        events,
                        event_handler,
                        "supervisor_selected_task",
                        agent_id=member.agent_id,
                        instance_id=member.instance_id,
                        task_id=item["selected_task_id"],
                    )

                if waiting:
                    return self._waiting_result(
                        session_id,
                        events,
                        pending,
                        completed,
                        outputs,
                        inboxes,
                        results,
                        messages,
                        rounds,
                        waiting,
                    )
                selected_task_id = item.get("selected_task_id")
            else:
                selected_task_id = None

            while pending:
                ready = [
                    task
                    for task in plan.tasks
                    if task.task_id in pending
                    and set(task.dependencies) <= completed
                ]
                if not ready:
                    raise ValueError(
                        "No executable tasks remain. The plan may contain a cycle."
                    )

                if selected_task_id:
                    batch = [tasks[selected_task_id]]
                    selected_task_id = None
                elif plan.orchestration == "sequential":
                    batch = ready[:1]
                elif plan.orchestration == "parallel":
                    batch = self._parallel_batch(ready)
                else:
                    rounds += 1
                    if rounds > plan.supervisor.max_rounds:
                        raise RuntimeError("Supervisor exceeded max_rounds.")
                    supervisor_id = plan.supervisor.supervisor_instance_id
                    supervisor = members[supervisor_id]
                    inbox = list(inboxes[supervisor_id])
                    inboxes[supervisor_id].clear()
                    selection_task = self._supervisor_task(
                        ready, completed, outputs
                    )
                    self._emit(
                        events,
                        event_handler,
                        "agent_running",
                        agent_id=supervisor.agent_id,
                        instance_id=supervisor_id,
                        task="Select the next ready team task.",
                        assigned_tool_ids=supervisor.assigned_tool_ids,
                    )
                    result = await self._run_member(
                        conversation_id=conversation_id,
                        session_id=session_id,
                        original_request=original_request,
                        member=supervisor,
                        task=selection_task,
                        task_id=None,
                        context={
                            **shared_context,
                            "message_history": [
                                item.model_dump(mode="json") for item in messages
                            ],
                        },
                        dependency_outputs={},
                        inbox=inbox,
                        members=plan.members,
                        max_steps=run_steps,
                        event_handler=worker_event,
                    )
                    if result.status == "WAITING_FOR_USER":
                        waiting.append(
                            {
                                "kind": "supervisor",
                                "member_id": supervisor_id,
                                "task_id": None,
                                "task": selection_task,
                                "context": shared_context,
                                "dependency_outputs": {},
                                "inbox": [
                                    item.model_dump(mode="json") for item in inbox
                                ],
                                "checkpoint": result.checkpoint,
                                "contact": result.user_contact.model_dump(
                                    mode="json"
                                ),
                                "ready_task_ids": [
                                    item.task_id for item in ready
                                ],
                            }
                        )
                        return self._waiting_result(
                            session_id,
                            events,
                            pending,
                            completed,
                            outputs,
                            inboxes,
                            results,
                            messages,
                            rounds,
                            waiting,
                        )
                    if result.status != "COMPLETED":
                        raise RuntimeError(
                            result.error or "Supervisor execution failed."
                        )
                    results.append(result)
                    self._route_messages(
                        result,
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
                        assigned_tool_ids=supervisor.assigned_tool_ids,
                    )
                    choice = self._parse_choice(
                        result, {item.task_id for item in ready}
                    )
                    self._emit(
                        events,
                        event_handler,
                        "supervisor_selected_task",
                        agent_id=supervisor.agent_id,
                        instance_id=supervisor_id,
                        task_id=choice,
                    )
                    batch = [tasks[choice]]

                executions = []
                metadata = []
                for task in batch:
                    member = members[task.agent_instance_id]
                    inbox = list(inboxes[member.instance_id])
                    inboxes[member.instance_id].clear()
                    dependency_outputs = {
                        key: outputs[key] for key in task.dependencies
                    }
                    member_context = {
                        **shared_context,
                        "message_history": [
                            item.model_dump(mode="json") for item in messages
                        ],
                        "completed_task_outputs": dict(outputs),
                    }
                    self._emit(
                        events,
                        event_handler,
                        "agent_running",
                        agent_id=member.agent_id,
                        instance_id=member.instance_id,
                        task_id=task.task_id,
                        task=task.instruction,
                        assigned_tool_ids=member.assigned_tool_ids,
                    )
                    executions.append(
                        self._run_member(
                            conversation_id=conversation_id,
                            session_id=session_id,
                            original_request=original_request,
                            member=member,
                            task=task.instruction,
                            task_id=task.task_id,
                            context=member_context,
                            dependency_outputs=dependency_outputs,
                            inbox=inbox,
                            members=plan.members,
                            max_steps=run_steps,
                            event_handler=worker_event,
                        )
                    )
                    metadata.append(
                        (task, member, inbox, dependency_outputs, member_context)
                    )

                batch_results = await asyncio.gather(*executions)
                for meta, result in zip(metadata, batch_results):
                    task, member, inbox, dependency_outputs, member_context = meta
                    if result.status == "WAITING_FOR_USER":
                        waiting.append(
                            {
                                "kind": "task",
                                "member_id": member.instance_id,
                                "task_id": task.task_id,
                                "task": task.instruction,
                                "context": member_context,
                                "dependency_outputs": dependency_outputs,
                                "inbox": [
                                    item.model_dump(mode="json") for item in inbox
                                ],
                                "checkpoint": result.checkpoint,
                                "contact": result.user_contact.model_dump(
                                    mode="json"
                                ),
                            }
                        )
                        continue

                    results.append(result)
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
                        "agent_completed"
                        if result.status == "COMPLETED"
                        else "agent_failed",
                        agent_id=member.agent_id,
                        instance_id=member.instance_id,
                        task_id=task.task_id,
                        task=task.instruction,
                        assigned_tool_ids=member.assigned_tool_ids,
                        details={"error": result.error},
                    )
                    if result.status != "COMPLETED":
                        raise RuntimeError(
                            result.error or f"Task failed: {task.task_id}"
                        )
                    outputs[task.task_id] = result.final_answer or ""
                    completed.add(task.task_id)
                    pending.discard(task.task_id)

                if waiting:
                    return self._waiting_result(
                        session_id,
                        events,
                        pending,
                        completed,
                        outputs,
                        inboxes,
                        results,
                        messages,
                        rounds,
                        waiting,
                    )

            return ExecutionResult(
                session_id=session_id,
                status="COMPLETED",
                execution_mode="team",
                agent_results=results,
                messages=messages,
                events=events,
            )

        except Exception as error:
            return ExecutionResult(
                session_id=session_id,
                status="FAILED",
                execution_mode="team",
                events=events,
                errors=[f"{type(error).__name__}: {error}"],
            )

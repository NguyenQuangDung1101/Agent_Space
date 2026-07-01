import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from agent.analyze_agent.service import AnalyzeAgentService
from agent.planner_agent.service import PlannerAgentService
from agent.synthesizer_agent.service import SynthesizerAgentService
from core.orchestrator import Orchestrator
from database.knowledge_repository import (
    close_knowledge_repository,
    get_knowledge_repository,
)
from database.neo4j_repository import Neo4jRepository
from share.agent_factory import AgentFactory
from share.conversation_store import ConversationStore
from share.event_broker import EventBroker
from share.registry import Registry
from share.schemas import (
    AgentRequest,
    AgentResult,
    AnalysisResult,
    Conversation,
    ExecutionEvent,
    ExecutionResult,
    Message,
    SessionRecord,
    SingleAgentPlan,
    TeamPlan,
    utc_now,
)


logger = logging.getLogger(__name__)


class AgentSpaceManager:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        conversations_dir: Optional[str | Path] = None,
    ) -> None:
        self.registry = Registry()
        self.factory = AgentFactory(
            registry=self.registry,
            model=model,
            base_url=base_url,
        )
        self.analyzer = AnalyzeAgentService(model=model, base_url=base_url)
        self.planner = PlannerAgentService(model=model, base_url=base_url)
        self.synthesizer = SynthesizerAgentService(
            model=model,
            base_url=base_url,
        )
        self.orchestrator = Orchestrator(
            registry=self.registry,
            factory=self.factory,
        )
        self.event_broker = EventBroker()
        self._background_tasks: set[asyncio.Task] = set()

        project_root = Path(__file__).resolve().parents[1]
        self.store = ConversationStore(
            conversations_dir or project_root / "data" / "conversations"
        )
        self.neo4j = Neo4jRepository()

    def initialize(self) -> None:
        self.neo4j.rebuild(self.store, self.registry)
        self.knowledge = (
            get_knowledge_repository() if self.neo4j.enabled else None
        )

    def close(self) -> None:
        if getattr(self, "knowledge", None):
            close_knowledge_repository()
            self.knowledge = None
        self.neo4j.close()

    def _mirror(self, method: str, *args: Any) -> None:
        if not self.neo4j.enabled:
            return
        try:
            getattr(self.neo4j, method)(*args)
        except Exception:
            logger.exception("Neo4j mirror failed: %s", method)

    def _save_conversation(self, conversation: Conversation) -> None:
        self.store.save_conversation(conversation)
        self._mirror("save_conversation", conversation)

    def _add_message(self, message: Message) -> Message:
        saved = self.store.add_message(message)
        self._mirror("save_message", saved)
        return saved

    def _save_session(self, session: SessionRecord) -> None:
        self.store.save_session(session)
        self._mirror("save_session", session)

    def _save_contact(self, contact, update: bool = False) -> None:
        if update:
            self.store.update_contact(contact)
        else:
            self.store.add_contact(contact)
        self._mirror("save_contact", contact)

    def create_conversation(
        self,
        title: Optional[str] = None,
    ) -> Conversation:
        conversation = self.store.create_conversation(title)
        self._mirror("save_conversation", conversation)
        memory = self.store.get_memory(conversation.conversation_id)
        self._mirror("save_memory", memory)
        return conversation

    def list_conversations(self) -> list[Conversation]:
        return self.store.list_conversations()

    def get_conversation(self, conversation_id: str) -> Conversation:
        return self.store.get_conversation(conversation_id)

    def get_messages(self, conversation_id: str) -> list[Message]:
        return self.store.list_messages(conversation_id)

    def get_session(self, session_id: str) -> SessionRecord:
        return self.store.get_session(session_id)

    def _write_json(
        self,
        session: SessionRecord,
        filename: str,
        value: Any,
    ) -> None:
        self.store.write_session_json(
            session.conversation_id,
            session.session_id,
            filename,
            value,
        )

    def _read_json(
        self,
        session: SessionRecord,
        filename: str,
        default: Any = None,
    ) -> Any:
        return self.store.read_session_json(
            session.conversation_id,
            session.session_id,
            filename,
            default,
        )

    def get_events(self, session_id: str) -> list[ExecutionEvent]:
        session = self.get_session(session_id)
        return [
            ExecutionEvent.model_validate(item)
            for item in self._read_json(session, "events.json", [])
        ]

    async def stream_events(
        self,
        session_id: str,
    ) -> AsyncIterator[Optional[ExecutionEvent]]:
        async for event in self.event_broker.stream(
            session_id,
            self.get_events(session_id),
        ):
            yield event

    def _record_event(
        self,
        session: SessionRecord,
        events: list[ExecutionEvent],
        event: ExecutionEvent,
        publish: bool = True,
    ) -> None:
        events.append(event)
        self._write_json(session, "events.json", events)
        if publish:
            self.event_broker.publish(session.session_id, event)

    def _event_recorder(
        self,
        session: SessionRecord,
        events: list[ExecutionEvent],
    ):
        def record(event: ExecutionEvent) -> None:
            self._record_event(session, events, event)

        return record

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _display_name(agent_id: str) -> str:
        return agent_id.replace("_", " ").title()

    def _tool_catalog(
        self,
        allowed_tool_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not allowed_tool_ids:
            return self.registry.list_tools()
        self.registry.validate_tool_ids(allowed_tool_ids)
        allowed = set(allowed_tool_ids)
        return [
            item
            for item in self.registry.list_tools()
            if item["id"] in allowed
        ]

    def _validate_single_plan(
        self,
        plan: SingleAgentPlan,
        tool_catalog: list[dict[str, Any]],
    ) -> None:
        agent = self.registry.get_agent(plan.agent_id)
        if not agent["selectable_as_worker"]:
            raise ValueError(
                f"Agent is not a selectable worker: {plan.agent_id}"
            )
        if not self.registry.can_call("manager", plan.agent_id):
            raise PermissionError(f"Manager cannot call: {plan.agent_id}")
        invalid = set(plan.assigned_tool_ids) - {
            item["id"] for item in tool_catalog
        }
        if invalid:
            raise ValueError(
                "Invalid assigned tools: " + ", ".join(sorted(invalid))
            )
        if agent["type"] == "dynamic" and not plan.system_prompt:
            raise ValueError("Dynamic worker requires a system_prompt.")

    def _create_session(
        self,
        conversation_id: str,
        message: Message,
        context: dict[str, Any],
        allowed_tool_ids: list[str],
        max_steps: int,
    ) -> SessionRecord:
        session = SessionRecord(
            session_id=f"session_{uuid4().hex[:12]}",
            conversation_id=conversation_id,
            trigger_message_id=message.message_id,
            original_request=message.content,
            status="RUNNING",
        )
        message.session_id = session.session_id
        self._add_message(message)
        self._write_json(
            session,
            "request.json",
            {
                "conversation_id": conversation_id,
                "session_id": session.session_id,
                "trigger_message_id": message.message_id,
                "task": message.content,
                "context": context,
                "allowed_tool_ids": allowed_tool_ids,
                "max_steps": max_steps,
            },
        )
        self._save_session(session)
        events: list[ExecutionEvent] = []
        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type="message_received",
                task=message.content,
                details={"message_id": message.message_id},
            ),
        )
        return session

    def _title_conversation(self, conversation_id: str, content: str) -> None:
        conversation = self.store.get_conversation(conversation_id)
        if not conversation.title:
            conversation.title = content[:80]
            self._save_conversation(conversation)

    def start_message(
        self,
        conversation_id: str,
        content: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
        attachments: Optional[list[str]] = None,
    ) -> tuple[SessionRecord, Message, bool]:
        self.store.get_conversation(conversation_id)
        clean_context = dict(context or {})
        if attachments:
            clean_context["attachments"] = list(attachments)
        allowed_tool_ids = list(dict.fromkeys(assigned_tool_ids or []))
        pending = self.store.pending_session(conversation_id)

        if pending and pending.pending_contact_id:
            contact = self.store.get_contact(
                conversation_id,
                pending.session_id,
                pending.pending_contact_id,
            )
            message = Message(
                message_id=f"message_{uuid4().hex[:12]}",
                conversation_id=conversation_id,
                role="user",
                content=content,
                session_id=pending.session_id,
                contact_request_id=contact.contact_id,
                attachments=attachments or [],
            )
            self._add_message(message)
            contact.status = "ANSWERED"
            contact.answer_message_id = message.message_id
            contact.response = content
            contact.answered_at = utc_now()
            self._save_contact(contact, update=True)

            pending.status = "RUNNING"
            pending.pending_contact_id = None
            self._save_session(pending)
            events = self.get_events(pending.session_id)
            self._record_event(
                pending,
                events,
                ExecutionEvent(
                    event_type="message_received",
                    task=content,
                    details={
                        "message_id": message.message_id,
                        "contact_id": contact.contact_id,
                    },
                ),
            )
            self._record_event(
                pending,
                events,
                ExecutionEvent(
                    event_type="user_contact_answered",
                    agent_id=contact.agent_id,
                    instance_id=contact.instance_id,
                    details={
                        "contact_id": contact.contact_id,
                        "agent_name": contact.agent_name,
                        "message_id": message.message_id,
                    },
                ),
            )
            self._record_event(
                pending,
                events,
                ExecutionEvent(
                    event_type="session_resumed",
                    agent_id=contact.agent_id,
                    instance_id=contact.instance_id,
                    task=pending.original_request,
                ),
            )
            self._spawn(self._resume_session(pending, content))
            return pending, message, True

        message = Message(
            message_id=f"message_{uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role="user",
            content=content,
            attachments=attachments or [],
        )
        self._title_conversation(conversation_id, content)
        session = self._create_session(
            conversation_id,
            message,
            clean_context,
            allowed_tool_ids,
            max_steps,
        )
        self._spawn(
            self._execute_session(
                session,
                clean_context,
                allowed_tool_ids,
                max_steps,
            )
        )
        return session, message, False

    async def handle_message(
        self,
        conversation_id: str,
        content: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
    ) -> ExecutionResult:
        clean_context = context or {}
        allowed_tool_ids = list(dict.fromkeys(assigned_tool_ids or []))
        if self.store.pending_session(conversation_id):
            raise RuntimeError(
                "Use start_message for user-contact replies so the existing "
                "event stream can remain active."
            )
        message = Message(
            message_id=f"message_{uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        self._title_conversation(conversation_id, content)
        session = self._create_session(
            conversation_id,
            message,
            clean_context,
            allowed_tool_ids,
            max_steps,
        )
        return await self._execute_session(
            session,
            clean_context,
            allowed_tool_ids,
            max_steps,
        )

    def _save_single_plan(
        self,
        session: SessionRecord,
        plan: SingleAgentPlan,
    ) -> None:
        self._write_json(session, "single_plan.json", plan)
        self._write_json(
            session,
            "assignments.json",
            {
                "execution_mode": "single",
                "agents": [
                    {
                        "agent_id": plan.agent_id,
                        "task": plan.task,
                        "assigned_tool_ids": plan.assigned_tool_ids,
                    }
                ],
            },
        )
        self._mirror("save_plan", session, plan)

    def _save_team_plan(self, session: SessionRecord, plan: TeamPlan) -> None:
        self._write_json(session, "team_plan.json", plan)
        self._write_json(
            session,
            "assignments.json",
            {
                "execution_mode": "team",
                "team_id": plan.team_id,
                "orchestration": plan.orchestration,
                "agents": [
                    {
                        "instance_id": item.instance_id,
                        "agent_id": item.agent_id,
                        "role": item.role,
                        "task": item.task,
                        "assigned_tool_ids": item.assigned_tool_ids,
                    }
                    for item in plan.members
                ],
            },
        )
        self._mirror("save_plan", session, plan)

    def _control_waiting(
        self,
        session: SessionRecord,
        result: AgentResult,
        stage: str,
        events: list[ExecutionEvent],
        *,
        execution_mode: Optional[str] = None,
        checkpoint_data: Optional[dict[str, Any]] = None,
        prior_results: Optional[list[AgentResult]] = None,
        prior_messages: Optional[list] = None,
    ) -> ExecutionResult:
        checkpoint = {
            "stage": stage,
            "instance_id": result.instance_id,
            "agent_checkpoint": result.checkpoint,
        }
        checkpoint.update(checkpoint_data or {})
        return ExecutionResult(
            session_id=session.session_id,
            status="WAITING_FOR_USER",
            execution_mode=execution_mode,
            agent_results=(prior_results or []) + [result],
            messages=(prior_messages or []) + result.messages,
            events=list(events),
            pending_contact=result.user_contact,
            checkpoint=checkpoint,
        )

    async def _run_analysis(
        self,
        session: SessionRecord,
        shared_context: dict[str, Any],
        worker_catalog: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
        max_steps: int,
        events: list[ExecutionEvent],
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> AnalysisResult | ExecutionResult:
        if checkpoint is None:
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="analysis_started",
                    agent_id="analyze_agent",
                    task=session.original_request,
                ),
            )

        result = await self.analyzer.run(
            AgentRequest(
                conversation_id=session.conversation_id,
                session_id=session.session_id,
                caller_id="manager",
                task=session.original_request,
                context={
                    "user_context": shared_context,
                    "conversation_memory": shared_context.get(
                        "conversation_memory", ""
                    ),
                    "worker_catalog": worker_catalog,
                    "tool_catalog": tool_catalog,
                },
                instance_id=checkpoint.get("instance_id") if checkpoint else None,
                checkpoint=(
                    checkpoint.get("agent_checkpoint") if checkpoint else None
                ),
                user_response=user_response,
                max_steps=max_steps,
            ),
            event_handler=self._event_recorder(session, events),
        )
        if result.status == "WAITING_FOR_USER":
            return self._control_waiting(
                session,
                result,
                "analysis",
                events,
            )
        if result.status != "COMPLETED" or not result.final_answer:
            raise RuntimeError(result.error or "Analyze Agent failed.")

        analysis = AnalysisResult.model_validate_json(result.final_answer)
        session.analysis = analysis
        session.execution_mode = analysis.execution_mode
        self._write_json(session, "analysis.json", analysis)
        self._save_session(session)
        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type="analysis_completed",
                agent_id="analyze_agent",
                instance_id=result.instance_id,
                task=session.original_request,
                details={
                    "execution_mode": analysis.execution_mode,
                    "reason": analysis.reason,
                },
            ),
        )
        return analysis

    async def _run_single(
        self,
        session: SessionRecord,
        analysis: AnalysisResult,
        context: dict[str, Any],
        tool_catalog: list[dict[str, Any]],
        max_steps: int,
        events: list[ExecutionEvent],
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> ExecutionResult:
        plan = analysis.single_plan
        if plan is None:
            raise ValueError("Analyze Agent did not provide single_plan.")
        self._validate_single_plan(plan, tool_catalog)

        if checkpoint is None:
            session.single_plan = plan
            self._save_single_plan(session, plan)
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="planning_completed",
                    agent_id="analyze_agent",
                    task=plan.task,
                    assigned_tool_ids=plan.assigned_tool_ids,
                    details={
                        "execution_mode": "single",
                        "selected_agent_id": plan.agent_id,
                    },
                ),
            )
            instance_id = f"{plan.agent_id}_{uuid4().hex[:10]}"
            agent_checkpoint = None
        else:
            instance_id = checkpoint["instance_id"]
            agent_checkpoint = checkpoint["agent_checkpoint"]

        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type="agent_running",
                agent_id=plan.agent_id,
                instance_id=instance_id,
                task=plan.task,
                assigned_tool_ids=plan.assigned_tool_ids,
            ),
        )

        def worker_event(event: ExecutionEvent) -> None:
            self._record_event(
                session,
                events,
                event.model_copy(
                    update={
                        "agent_id": plan.agent_id,
                        "instance_id": instance_id,
                        "task": plan.task,
                        "assigned_tool_ids": plan.assigned_tool_ids,
                    }
                ),
            )

        result = await self.factory.create(
            agent_id=plan.agent_id,
            caller_id="manager",
            assigned_tool_ids=plan.assigned_tool_ids,
            runtime_system_prompt=plan.system_prompt,
        ).run(
            conversation_id=session.conversation_id,
            session_id=session.session_id,
            instance_id=instance_id,
            task=plan.task,
            context={
                **context,
                "original_request": session.original_request,
                "analysis": analysis.model_dump(mode="json"),
            },
            max_steps=max_steps,
            event_handler=worker_event,
            checkpoint=agent_checkpoint,
            user_response=user_response,
        )

        if result.status == "WAITING_FOR_USER":
            return self._control_waiting(
                session,
                result,
                "single",
                events,
                execution_mode="single",
            )

        completed = result.status == "COMPLETED"
        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type="agent_completed" if completed else "agent_failed",
                agent_id=result.agent_id,
                instance_id=result.instance_id,
                task=plan.task,
                assigned_tool_ids=plan.assigned_tool_ids,
                details={"error": result.error},
            ),
        )
        return ExecutionResult(
            session_id=session.session_id,
            status="COMPLETED" if completed else "FAILED",
            execution_mode="single",
            final_answer=result.final_answer if completed else None,
            agent_results=[result],
            messages=result.messages,
            events=list(events),
            errors=[] if completed else [result.error or "Agent failed."],
        )

    async def _synthesize_team(
        self,
        session: SessionRecord,
        analysis: AnalysisResult,
        team_execution: ExecutionResult,
        max_steps: int,
        events: list[ExecutionEvent],
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> ExecutionResult:
        if checkpoint is None:
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="synthesis_started",
                    agent_id="synthesizer_agent",
                    task=session.original_request,
                ),
            )

        synthesis = await self.synthesizer.run(
            AgentRequest(
                conversation_id=session.conversation_id,
                session_id=session.session_id,
                caller_id="manager",
                task=session.original_request,
                context={
                    "analysis": analysis.model_dump(mode="json"),
                    "team_plan": session.team_plan.model_dump(mode="json"),
                    "team_execution": team_execution.model_dump(mode="json"),
                },
                instance_id=checkpoint.get("instance_id") if checkpoint else None,
                checkpoint=(
                    checkpoint.get("agent_checkpoint") if checkpoint else None
                ),
                user_response=user_response,
                max_steps=max_steps,
            ),
            event_handler=self._event_recorder(session, events),
        )
        if synthesis.status == "WAITING_FOR_USER":
            return self._control_waiting(
                session,
                synthesis,
                "synthesis",
                events,
                execution_mode="team",
                checkpoint_data={
                    "team_execution": team_execution.model_dump(mode="json")
                },
                prior_results=team_execution.agent_results,
                prior_messages=team_execution.messages,
            )

        synthesis_ok = synthesis.status == "COMPLETED"
        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type=(
                    "synthesis_completed" if synthesis_ok else "synthesis_failed"
                ),
                agent_id=synthesis.agent_id,
                instance_id=synthesis.instance_id,
                task=session.original_request,
                details={"error": synthesis.error},
            ),
        )
        errors = list(team_execution.errors)
        if not synthesis_ok:
            errors.append(synthesis.error or "Synthesis failed.")
        return ExecutionResult(
            session_id=session.session_id,
            status=(
                "COMPLETED"
                if team_execution.status == "COMPLETED" and synthesis_ok
                else "FAILED"
            ),
            execution_mode="team",
            final_answer=synthesis.final_answer if synthesis_ok else None,
            agent_results=team_execution.agent_results + [synthesis],
            messages=team_execution.messages,
            events=list(events),
            errors=errors,
        )

    async def _run_team(
        self,
        session: SessionRecord,
        analysis: AnalysisResult,
        context: dict[str, Any],
        worker_catalog: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
        max_steps: int,
        events: list[ExecutionEvent],
        checkpoint: Optional[dict[str, Any]] = None,
        user_response: Optional[str] = None,
    ) -> ExecutionResult:
        planning_resume = checkpoint and checkpoint.get("stage") == "planning"
        if checkpoint is None or planning_resume:
            if checkpoint is None:
                self._record_event(
                    session,
                    events,
                    ExecutionEvent(
                        event_type="planning_started",
                        agent_id="planner_agent",
                        task=session.original_request,
                    ),
                )
            planner_result = await self.planner.run(
                AgentRequest(
                    conversation_id=session.conversation_id,
                    session_id=session.session_id,
                    caller_id="manager",
                    task=session.original_request,
                    context={
                        "analysis": analysis.model_dump(mode="json"),
                        "user_context": context,
                        "worker_catalog": worker_catalog,
                        "tool_catalog": tool_catalog,
                    },
                    instance_id=(
                        checkpoint.get("instance_id") if planning_resume else None
                    ),
                    checkpoint=(
                        checkpoint.get("agent_checkpoint")
                        if planning_resume
                        else None
                    ),
                    user_response=user_response if planning_resume else None,
                    max_steps=max_steps,
                ),
                event_handler=self._event_recorder(session, events),
            )
            if planner_result.status == "WAITING_FOR_USER":
                return self._control_waiting(
                    session,
                    planner_result,
                    "planning",
                    events,
                    execution_mode="team",
                )
            if (
                planner_result.status != "COMPLETED"
                or not planner_result.final_answer
            ):
                raise RuntimeError(planner_result.error or "Planner Agent failed.")

            plan = TeamPlan.model_validate_json(planner_result.final_answer)
            self.orchestrator.validate_plan(plan)
            session.team_plan = plan
            self._save_team_plan(session, plan)
            self._save_session(session)
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="planning_completed",
                    agent_id="planner_agent",
                    instance_id=planner_result.instance_id,
                    task=session.original_request,
                    details={
                        "team_id": plan.team_id,
                        "orchestration": plan.orchestration,
                        "members": [
                            {
                                "instance_id": item.instance_id,
                                "agent_id": item.agent_id,
                                "assigned_tool_ids": item.assigned_tool_ids,
                            }
                            for item in plan.members
                        ],
                    },
                ),
            )
            orchestrator_checkpoint = None
            orchestration_response = None
        else:
            plan = session.team_plan
            if plan is None:
                raise ValueError("Missing team plan for resume.")
            orchestrator_checkpoint = checkpoint["orchestrator_checkpoint"]
            orchestration_response = user_response

        team_execution = await self.orchestrator.execute(
            conversation_id=session.conversation_id,
            session_id=session.session_id,
            original_request=session.original_request,
            plan=plan,
            context={
                **context,
                "analysis": analysis.model_dump(mode="json"),
            },
            max_steps=max_steps,
            event_handler=self._event_recorder(session, events),
            checkpoint=orchestrator_checkpoint,
            user_response=orchestration_response,
        )
        if team_execution.status == "WAITING_FOR_USER":
            team_execution.events = list(events)
            team_execution.checkpoint = {
                "stage": "team",
                "orchestrator_checkpoint": team_execution.checkpoint,
            }
            return team_execution
        return await self._synthesize_team(
            session,
            analysis,
            team_execution,
            max_steps,
            events,
        )

    async def _execute_session(
        self,
        session: SessionRecord,
        context: dict[str, Any],
        allowed_tool_ids: list[str],
        max_steps: int,
    ) -> ExecutionResult:
        events = self.get_events(session.session_id)
        self._record_event(
            session,
            events,
            ExecutionEvent(
                event_type="session_started",
                task=session.original_request,
            ),
        )
        try:
            memory = self.store.get_memory(session.conversation_id)
            shared_context = {
                **context,
                "conversation_id": session.conversation_id,
                "conversation_memory": memory.summary,
            }
            worker_catalog = self.registry.list_selectable_agents()
            tool_catalog = self._tool_catalog(allowed_tool_ids)
            analysis_result = await self._run_analysis(
                session,
                shared_context,
                worker_catalog,
                tool_catalog,
                max_steps,
                events,
            )
            if isinstance(analysis_result, ExecutionResult):
                execution = analysis_result
            elif analysis_result.execution_mode == "single":
                execution = await self._run_single(
                    session,
                    analysis_result,
                    shared_context,
                    tool_catalog,
                    max_steps,
                    events,
                )
            else:
                execution = await self._run_team(
                    session,
                    analysis_result,
                    shared_context,
                    worker_catalog,
                    tool_catalog,
                    max_steps,
                    events,
                )
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="execution_failed",
                    task=session.original_request,
                    details={"error": error_text},
                ),
            )
            execution = ExecutionResult(
                session_id=session.session_id,
                status="FAILED",
                execution_mode=session.execution_mode,
                events=list(events),
                errors=[error_text],
            )
        return self._finish_or_pause(session, execution, events)

    async def _resume_session(
        self,
        session: SessionRecord,
        user_response: str,
    ) -> ExecutionResult:
        events = self.get_events(session.session_id)
        request = self._read_json(session, "request.json", {})
        checkpoint = self._read_json(session, "checkpoint.json")
        if not checkpoint:
            execution = ExecutionResult(
                session_id=session.session_id,
                status="FAILED",
                execution_mode=session.execution_mode,
                errors=["Missing resumable checkpoint."],
            )
            return self._finish_or_pause(session, execution, events)

        try:
            memory = self.store.get_memory(session.conversation_id)
            context = {
                **request.get("context", {}),
                "conversation_id": session.conversation_id,
                "conversation_memory": memory.summary,
            }
            max_steps = int(request.get("max_steps", 10))
            tool_catalog = self._tool_catalog(
                request.get("allowed_tool_ids", [])
            )
            worker_catalog = self.registry.list_selectable_agents()
            stage = checkpoint.get("stage")

            if stage == "analysis":
                analysis_result = await self._run_analysis(
                    session,
                    context,
                    worker_catalog,
                    tool_catalog,
                    max_steps,
                    events,
                    checkpoint=checkpoint,
                    user_response=user_response,
                )
                if isinstance(analysis_result, ExecutionResult):
                    execution = analysis_result
                elif analysis_result.execution_mode == "single":
                    execution = await self._run_single(
                        session,
                        analysis_result,
                        context,
                        tool_catalog,
                        max_steps,
                        events,
                    )
                else:
                    execution = await self._run_team(
                        session,
                        analysis_result,
                        context,
                        worker_catalog,
                        tool_catalog,
                        max_steps,
                        events,
                    )
            else:
                analysis = session.analysis
                if analysis is None:
                    raise ValueError("Missing analysis for resume.")
                if stage == "single":
                    execution = await self._run_single(
                        session,
                        analysis,
                        context,
                        tool_catalog,
                        max_steps,
                        events,
                        checkpoint=checkpoint,
                        user_response=user_response,
                    )
                elif stage in {"planning", "team"}:
                    execution = await self._run_team(
                        session,
                        analysis,
                        context,
                        worker_catalog,
                        tool_catalog,
                        max_steps,
                        events,
                        checkpoint=checkpoint,
                        user_response=user_response,
                    )
                elif stage == "synthesis":
                    team_execution = ExecutionResult.model_validate(
                        checkpoint["team_execution"]
                    )
                    execution = await self._synthesize_team(
                        session,
                        analysis,
                        team_execution,
                        max_steps,
                        events,
                        checkpoint=checkpoint,
                        user_response=user_response,
                    )
                else:
                    raise ValueError("Unknown checkpoint stage.")
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            self._record_event(
                session,
                events,
                ExecutionEvent(
                    event_type="execution_failed",
                    task=session.original_request,
                    details={"error": error_text},
                ),
            )
            execution = ExecutionResult(
                session_id=session.session_id,
                status="FAILED",
                execution_mode=session.execution_mode,
                events=list(events),
                errors=[error_text],
            )
        return self._finish_or_pause(session, execution, events)

    def _persist_execution(
        self,
        session: SessionRecord,
        execution: ExecutionResult,
    ) -> None:
        self._write_json(session, "execution.json", execution)
        self._write_json(
            session,
            "outputs.json",
            [
                {
                    "agent_id": item.agent_id,
                    "instance_id": item.instance_id,
                    "status": item.status,
                    "output": item.final_answer,
                    "error": item.error,
                }
                for item in execution.agent_results
            ],
        )
        self._write_json(session, "messages.json", execution.messages)
        self._write_json(
            session,
            "tool_results.json",
            [
                {
                    "agent_id": item.agent_id,
                    "instance_id": item.instance_id,
                    "tool_calls": item.tool_calls,
                    "tool_results": item.tool_results,
                }
                for item in execution.agent_results
                if item.tool_calls or item.tool_results
            ],
        )
        failures = list(execution.errors)
        failures.extend(
            item.error for item in execution.agent_results if item.error
        )
        self._write_json(
            session,
            "failures.json",
            list(dict.fromkeys(failures)),
        )
        self._write_json(
            session,
            "final_response.json",
            {
                "status": execution.status,
                "final_answer": execution.final_answer,
                "errors": execution.errors,
            },
        )
        self._mirror("save_execution", session, execution)

    def _update_memory(
        self,
        session: SessionRecord,
        final_answer: str,
    ) -> None:
        memory = self.store.get_memory(session.conversation_id)
        entry = (
            f"User request: {session.original_request[:1500]}\n"
            f"Assistant response: {final_answer[:2500]}"
        )
        memory.summary = f"{memory.summary}\n\n{entry}".strip()[-8000:]
        memory.session_ids = (memory.session_ids + [session.session_id])[-20:]
        self.store.save_memory(memory)
        self._mirror("save_memory", memory)

    def _finish_or_pause(
        self,
        session: SessionRecord,
        execution: ExecutionResult,
        events: list[ExecutionEvent],
    ) -> ExecutionResult:
        if execution.status == "WAITING_FOR_USER":
            contact = execution.pending_contact
            if contact is None or execution.checkpoint is None:
                execution = execution.model_copy(
                    update={
                        "status": "FAILED",
                        "errors": ["Waiting execution has no contact/checkpoint."],
                    }
                )
            else:
                contact.agent_name = self._display_name(contact.agent_id)
                execution.pending_contact = contact
                session.status = "WAITING_FOR_USER"
                session.pending_contact_id = contact.contact_id
                session.final_result = execution
                self._write_json(
                    session,
                    "checkpoint.json",
                    execution.checkpoint,
                )
                self._save_contact(contact)
                self._record_event(
                    session,
                    events,
                    ExecutionEvent(
                        event_type="user_contact_requested",
                        agent_id=contact.agent_id,
                        instance_id=contact.instance_id,
                        task=session.original_request,
                        details={
                            "contact_id": contact.contact_id,
                            "agent_name": contact.agent_name,
                            "question": contact.question,
                            "reason": contact.reason,
                            "expected_response": contact.expected_response,
                        },
                    ),
                )
                execution.events = list(events)
                session.final_result = execution
                self._persist_execution(session, execution)
                self._mirror("save_contact", contact)
                self._save_session(session)
                return execution

        assistant_text = execution.final_answer
        if not assistant_text:
            assistant_text = (
                "Execution failed: "
                + ("; ".join(execution.errors) or "Unknown error")
            )
        assistant_message = Message(
            message_id=f"message_{uuid4().hex[:12]}",
            conversation_id=session.conversation_id,
            role="assistant",
            content=assistant_text,
            session_id=session.session_id,
        )
        self._add_message(assistant_message)
        if execution.status == "COMPLETED":
            self._update_memory(session, assistant_text)

        terminal = ExecutionEvent(
            event_type=(
                "session_completed"
                if execution.status == "COMPLETED"
                else "session_failed"
            ),
            task=session.original_request,
            details={
                "status": execution.status,
                "final_answer": execution.final_answer,
                "errors": execution.errors,
                "message_id": assistant_message.message_id,
            },
        )
        self._record_event(session, events, terminal, publish=False)
        execution.events = list(events)
        execution.checkpoint = None
        execution.pending_contact = None
        session.status = execution.status
        session.pending_contact_id = None
        session.final_result = execution
        self._persist_execution(session, execution)
        self._write_json(session, "checkpoint.json", {})
        self._save_session(session)
        self.event_broker.publish(session.session_id, terminal)
        return execution

    async def handle_task(
        self,
        user_request: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
    ) -> ExecutionResult:
        conversation = self.create_conversation()
        return await self.handle_message(
            conversation.conversation_id,
            user_request,
            context,
            assigned_tool_ids,
            max_steps,
        )

    def start_task(
        self,
        user_request: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
    ) -> SessionRecord:
        conversation = self.create_conversation()
        session, _, _ = self.start_message(
            conversation.conversation_id,
            user_request,
            context,
            assigned_tool_ids,
            max_steps,
        )
        return session

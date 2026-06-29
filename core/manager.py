import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from agent.analyze_agent.service import AnalyzeAgentService
from agent.planner_agent.service import PlannerAgentService
from agent.synthesizer_agent.service import SynthesizerAgentService
from core.orchestrator import Orchestrator
from share.agent_factory import AgentFactory
from share.registry import Registry
from share.schemas import (
    AgentRequest,
    AnalysisResult,
    ExecutionEvent,
    ExecutionResult,
    SessionRecord,
    SingleAgentPlan,
    utc_now,
)


class AgentSpaceManager:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        sessions_dir: Optional[str | Path] = None,
    ) -> None:
        self.registry = Registry()
        self.factory = AgentFactory(
            registry=self.registry,
            model=model,
            base_url=base_url,
        )
        self.analyzer = AnalyzeAgentService(model=model, base_url=base_url)
        self.planner = PlannerAgentService(model=model, base_url=base_url)
        self.synthesizer = SynthesizerAgentService(model=model, base_url=base_url)
        self.orchestrator = Orchestrator(
            registry=self.registry,
            factory=self.factory,
        )

        project_root = Path(__file__).resolve().parents[1]
        self.sessions_dir = Path(
            sessions_dir or project_root / "data" / "sessions"
        )
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_json_data(data: Any) -> Any:
        if hasattr(data, "model_dump"):
            return data.model_dump(mode="json")
        return data

    def _write_json(self, session_id: str, filename: str, data: Any) -> None:
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / filename).write_text(
            json.dumps(
                self._to_json_data(data),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _save_session(self, session: SessionRecord) -> None:
        session.updated_at = utc_now()
        self._write_json(session.session_id, "session.json", session)

    def _create_session(
        self,
        user_request: str,
        context: dict[str, Any],
        allowed_tool_ids: list[str],
        max_steps: int,
    ) -> SessionRecord:
        session_id = f"session_{uuid4().hex[:12]}"
        session = SessionRecord(
            session_id=session_id,
            original_request=user_request,
            status="RUNNING",
        )
        self._write_json(
            session_id,
            "request.json",
            {
                "session_id": session_id,
                "task": user_request,
                "context": context,
                "allowed_tool_ids": allowed_tool_ids,
                "max_steps": max_steps,
            },
        )
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> SessionRecord:
        path = self.sessions_dir / session_id / "session.json"
        if not path.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _tool_catalog(self, allowed_tool_ids: list[str]) -> list[dict[str, Any]]:
        if not allowed_tool_ids:
            return self.registry.list_tools()
        self.registry.validate_tool_ids(allowed_tool_ids)
        allowed = set(allowed_tool_ids)
        return [
            tool for tool in self.registry.list_tools()
            if tool["id"] in allowed
        ]

    def _validate_single_plan(
        self,
        plan: SingleAgentPlan,
        tool_catalog: list[dict[str, Any]],
    ) -> None:
        agent = self.registry.get_agent(plan.agent_id)
        if not agent["selectable_as_worker"]:
            raise ValueError(f"Agent is not a selectable worker: {plan.agent_id}")
        if not self.registry.can_call("manager", plan.agent_id):
            raise PermissionError(f"Manager cannot call: {plan.agent_id}")

        allowed_tools = {tool["id"] for tool in tool_catalog}
        invalid_tools = set(plan.assigned_tool_ids) - allowed_tools
        if invalid_tools:
            raise ValueError(
                "Invalid assigned tools: " + ", ".join(sorted(invalid_tools))
            )
        if agent["type"] == "dynamic" and not plan.system_prompt:
            raise ValueError("Dynamic worker requires a system_prompt.")

    def _event_recorder(
        self,
        session_id: str,
        events: list[ExecutionEvent],
    ):
        def record(event: ExecutionEvent) -> None:
            events.append(event)
            self._write_json(session_id, "events.json", events)
        return record

    async def _run_single(
        self,
        session: SessionRecord,
        analysis: AnalysisResult,
        context: dict[str, Any],
        tool_catalog: list[dict[str, Any]],
        max_steps: int,
        events: list[ExecutionEvent],
    ) -> ExecutionResult:
        plan = analysis.single_plan
        if plan is None:
            raise ValueError("Analyze Agent did not provide single_plan.")

        self._validate_single_plan(plan, tool_catalog)
        session.single_plan = plan
        self._write_json(session.session_id, "single_plan.json", plan)

        events.append(
            ExecutionEvent(
                event_type="agent_running",
                agent_id=plan.agent_id,
                details={"assigned_tool_ids": plan.assigned_tool_ids},
            )
        )
        self._write_json(session.session_id, "events.json", events)

        runner = self.factory.create(
            agent_id=plan.agent_id,
            caller_id="manager",
            assigned_tool_ids=plan.assigned_tool_ids,
            runtime_system_prompt=plan.system_prompt,
        )
        result = await runner.run(
            session_id=session.session_id,
            task=plan.task,
            context={
                **context,
                "original_request": session.original_request,
                "analysis": analysis.model_dump(mode="json"),
            },
            max_steps=max_steps,
        )

        events.append(
            ExecutionEvent(
                event_type=(
                    "agent_completed"
                    if result.status == "COMPLETED"
                    else "agent_failed"
                ),
                agent_id=result.agent_id,
                instance_id=result.instance_id,
            )
        )
        self._write_json(session.session_id, "events.json", events)

        completed = result.status == "COMPLETED"
        return ExecutionResult(
            session_id=session.session_id,
            status="COMPLETED" if completed else "FAILED",
            execution_mode="single",
            final_answer=result.final_answer if completed else None,
            agent_results=[result],
            messages=result.messages,
            events=events,
            errors=[] if completed else [result.error or "Agent execution failed."],
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
    ) -> ExecutionResult:
        events.append(ExecutionEvent(event_type="planning_started"))
        self._write_json(session.session_id, "events.json", events)

        plan = await self.planner.run(
            AgentRequest(
                session_id=session.session_id,
                caller_id="manager",
                task=session.original_request,
                context={
                    "analysis": analysis.model_dump(mode="json"),
                    "user_context": context,
                    "worker_catalog": worker_catalog,
                    "tool_catalog": tool_catalog,
                },
                max_steps=max_steps,
            )
        )
        self.orchestrator.validate_plan(plan)
        session.team_plan = plan
        self._write_json(session.session_id, "team_plan.json", plan)

        events.append(
            ExecutionEvent(
                event_type="team_planned",
                details={
                    "team_id": plan.team_id,
                    "orchestration": plan.orchestration,
                },
            )
        )
        self._write_json(session.session_id, "events.json", events)

        team_execution = await self.orchestrator.execute(
            session_id=session.session_id,
            original_request=session.original_request,
            plan=plan,
            context={
                **context,
                "analysis": analysis.model_dump(mode="json"),
            },
            max_steps=max_steps,
            event_handler=self._event_recorder(session.session_id, events),
        )

        events.append(ExecutionEvent(event_type="synthesis_started"))
        self._write_json(session.session_id, "events.json", events)

        synthesis = await self.synthesizer.run(
            AgentRequest(
                session_id=session.session_id,
                caller_id="manager",
                task=session.original_request,
                context={
                    "analysis": analysis.model_dump(mode="json"),
                    "team_plan": plan.model_dump(mode="json"),
                    "team_execution": team_execution.model_dump(mode="json"),
                },
                max_steps=max_steps,
            )
        )

        synthesis_ok = synthesis.status == "COMPLETED"
        events.append(
            ExecutionEvent(
                event_type=(
                    "synthesis_completed" if synthesis_ok else "synthesis_failed"
                ),
                agent_id=synthesis.agent_id,
                instance_id=synthesis.instance_id,
            )
        )
        self._write_json(session.session_id, "events.json", events)

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
            events=events,
            errors=errors,
        )

    async def handle_task(
        self,
        user_request: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
    ) -> ExecutionResult:
        context = context or {}
        allowed_tool_ids = list(dict.fromkeys(assigned_tool_ids or []))
        session = self._create_session(
            user_request,
            context,
            allowed_tool_ids,
            max_steps,
        )
        events = [ExecutionEvent(event_type="session_started")]
        self._write_json(session.session_id, "events.json", events)

        try:
            worker_catalog = self.registry.list_selectable_agents()
            tool_catalog = self._tool_catalog(allowed_tool_ids)

            analysis = await self.analyzer.run(
                AgentRequest(
                    session_id=session.session_id,
                    caller_id="manager",
                    task=user_request,
                    context={
                        "user_context": context,
                        "worker_catalog": worker_catalog,
                        "tool_catalog": tool_catalog,
                    },
                    max_steps=max_steps,
                )
            )
            session.analysis = analysis
            session.execution_mode = analysis.execution_mode
            self._write_json(session.session_id, "analysis.json", analysis)

            events.append(
                ExecutionEvent(
                    event_type="analysis_completed",
                    details={"execution_mode": analysis.execution_mode},
                )
            )
            self._write_json(session.session_id, "events.json", events)

            if analysis.execution_mode == "single":
                execution = await self._run_single(
                    session,
                    analysis,
                    context,
                    tool_catalog,
                    max_steps,
                    events,
                )
            else:
                execution = await self._run_team(
                    session,
                    analysis,
                    context,
                    worker_catalog,
                    tool_catalog,
                    max_steps,
                    events,
                )

        except Exception as error:
            events.append(
                ExecutionEvent(
                    event_type="execution_failed",
                    details={"error": f"{type(error).__name__}: {error}"},
                )
            )
            self._write_json(session.session_id, "events.json", events)
            execution = ExecutionResult(
                session_id=session.session_id,
                status="FAILED",
                execution_mode=session.execution_mode or "single",
                events=events,
                errors=[f"{type(error).__name__}: {error}"],
            )

        session.status = execution.status
        session.execution_mode = execution.execution_mode
        session.final_result = execution
        self._write_json(session.session_id, "execution.json", execution)
        self._save_session(session)
        return execution

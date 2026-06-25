import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from agent.agent_of_requirement.service import (
    AgentOfRequirementService,
)
from agent.analyze_agent.service import (
    AnalyzeAgentService,
)
from share.schemas import (
    AgentRequest,
    AnalysisResult,
    ExecutionResult,
    SessionRecord,
    utc_now,
)


class AgentSpaceManager:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        sessions_dir: Optional[str | Path] = None,
    ) -> None:
        self.analyzer = AnalyzeAgentService(
            model=model,
            base_url=base_url,
        )

        self.worker = AgentOfRequirementService(
            model=model,
            base_url=base_url,
        )

        project_root = Path(__file__).resolve().parents[1]

        self.sessions_dir = Path(
            sessions_dir
            or project_root / "data" / "sessions"
        )

        self.sessions_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Session storage
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_json_data(data: Any) -> Any:
        if hasattr(data, "model_dump"):
            return data.model_dump(mode="json")

        return data

    def _write_json(
        self,
        session_id: str,
        filename: str,
        data: Any,
    ) -> None:
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = session_dir / filename

        path.write_text(
            json.dumps(
                self._to_json_data(data),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _save_session(
        self,
        session: SessionRecord,
    ) -> None:
        session.updated_at = utc_now()

        self._write_json(
            session.session_id,
            "session.json",
            session,
        )

    def _create_session(
        self,
        user_request: str,
        context: dict[str, Any],
        assigned_tool_ids: list[str],
        max_steps: int,
    ) -> SessionRecord:
        session_id = (
            f"session_{uuid4().hex[:12]}"
        )

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
                "assigned_tool_ids": assigned_tool_ids,
                "max_steps": max_steps,
            },
        )

        self._save_session(session)

        return session

    def get_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        session_path = (
            self.sessions_dir
            / session_id
            / "session.json"
        )

        if not session_path.is_file():
            raise FileNotFoundError(
                f"Session not found: {session_id}"
            )

        return SessionRecord.model_validate_json(
            session_path.read_text(
                encoding="utf-8"
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Agent execution
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_runtime_prompt(
        analysis: AnalysisResult,
    ) -> str:
        constraints = (
            "\n".join(
                f"- {item}"
                for item in analysis.constraints
            )
            or "- None specified"
        )

        return (
            "You are a specialized execution agent.\n\n"
            f"Objective:\n{analysis.objective}\n\n"
            f"Expected output:\n"
            f"{analysis.expected_output}\n\n"
            f"Constraints:\n{constraints}\n\n"
            "Complete the task directly and accurately."
        )

    async def handle_task(
        self,
        user_request: str,
        context: Optional[dict[str, Any]] = None,
        assigned_tool_ids: Optional[list[str]] = None,
        max_steps: int = 10,
    ) -> ExecutionResult:
        context = context or {}
        assigned_tool_ids = assigned_tool_ids or []

        session = self._create_session(
            user_request=user_request,
            context=context,
            assigned_tool_ids=assigned_tool_ids,
            max_steps=max_steps,
        )

        try:
            analysis = await self.analyzer.run(
                AgentRequest(
                    session_id=session.session_id,
                    caller_id="manager",
                    task=user_request,
                    context=context,
                    assigned_tool_ids=[],
                    max_steps=max_steps,
                )
            )

            session.analysis = analysis

            self._write_json(
                session.session_id,
                "analysis.json",
                analysis,
            )

            worker_result = await self.worker.run(
                AgentRequest(
                    session_id=session.session_id,
                    caller_id="manager",
                    task=user_request,
                    context={
                        **context,
                        "analysis": analysis.model_dump(),
                    },
                    assigned_tool_ids=assigned_tool_ids,
                    runtime_system_prompt=(
                        self._build_runtime_prompt(
                            analysis
                        )
                    ),
                    max_steps=max_steps,
                )
            )

            completed = (
                worker_result.status == "COMPLETED"
            )

            execution_result = ExecutionResult(
                session_id=session.session_id,
                status=(
                    "COMPLETED"
                    if completed
                    else "FAILED"
                ),
                execution_mode="single",
                final_answer=(
                    worker_result.final_answer
                    if completed
                    else None
                ),
                agent_results=[
                    worker_result
                ],
                errors=(
                    []
                    if completed
                    else [
                        worker_result.error
                        or "Agent execution failed."
                    ]
                ),
            )

        except Exception as error:
            execution_result = ExecutionResult(
                session_id=session.session_id,
                status="FAILED",
                execution_mode="single",
                errors=[
                    f"{type(error).__name__}: {error}"
                ],
            )

        session.status = execution_result.status
        session.execution_mode = "single"
        session.final_result = execution_result

        self._write_json(
            session.session_id,
            "execution.json",
            execution_result,
        )

        self._save_session(session)

        return execution_result
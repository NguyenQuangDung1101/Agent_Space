from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from agent.knowledge_graph_modification_agent import builtin_tool
from share.agent import (
    ToolCallingAgent,
    UserContactRaised,
    build_strong_system_prompt,
    build_waiting_result,
    run_resumable,
)
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import (
    AgentMessage,
    AgentRequest,
    AgentResult,
    ExecutionEvent,
    ModificationPlan,
    ToolCall,
    ToolResult,
)


AGENT_ID = "knowledge_graph_modification_agent"
EventHandler = Callable[[ExecutionEvent], Any]


class KnowledgeGraphModificationAgentService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = Registry()
        self.llm = Copilot(model=model, base_url=base_url)
        self.system_prompt = Path(__file__).with_name(
            "system_prompt.txt"
        ).read_text(encoding="utf-8").strip()

    @staticmethod
    def _approved(response: str | None) -> bool:
        if not response:
            return False
        text = response.strip().lower()
        rejected = ("not approve", "do not", "don't", "change", "revise", "but")
        accepted = ("approve", "approved", "proceed", "accept", "yes", "đồng ý")
        return not any(word in text for word in rejected) and text.startswith(accepted)

    @staticmethod
    def _cancelled(response: str | None) -> bool:
        if not response:
            return False
        return response.strip().lower().startswith(
            ("cancel", "reject", "stop", "no, cancel", "hủy")
        )

    @staticmethod
    def _latest_plan(results: list[ToolResult]) -> ModificationPlan | None:
        for result in reversed(results):
            if result.name == "save_modification_plan" and result.success:
                return ModificationPlan.model_validate(result.output)
        return None

    async def run(
        self,
        request: AgentRequest,
        event_handler: Optional[EventHandler] = None,
    ) -> AgentResult:
        instance_id = request.instance_id or f"{AGENT_ID}_{uuid4().hex[:10]}"
        saved = request.checkpoint or {}
        tool_calls = [
            ToolCall.model_validate(item) for item in saved.get("tool_calls", [])
        ]
        tool_results = [
            ToolResult.model_validate(item) for item in saved.get("tool_results", [])
        ]
        try:
            if not self.registry.can_call(request.caller_id, AGENT_ID):
                raise PermissionError(
                    f"'{request.caller_id}' cannot call '{AGENT_ID}'."
                )
            phase = saved.get("phase", "analysis")
            saved_plan = saved.get("modification_plan")
            saved_messages = [
                AgentMessage.model_validate(item)
                for item in saved.get("agent", {}).get("outbox", [])
            ]
            if (
                request.checkpoint
                and phase == "review"
                and self._cancelled(request.user_response)
            ):
                return AgentResult(
                    agent_id=AGENT_ID,
                    instance_id=instance_id,
                    status="COMPLETED",
                    final_answer="Knowledge graph modification was cancelled. No changes were applied.",
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    messages=saved_messages,
                )
            if (
                request.checkpoint
                and phase == "review"
                and self._approved(request.user_response)
            ):
                if not saved_plan:
                    raise ValueError("Approved checkpoint has no modification plan.")
                plan = ModificationPlan.model_validate(saved_plan)
                call = ToolCall(
                    name="apply_modification_plan",
                    arguments={"plan": plan.model_dump(mode="json")},
                )
                tool_calls.append(call)
                if event_handler:
                    event = ExecutionEvent(
                        event_type="tool_running",
                        agent_id=AGENT_ID,
                        instance_id=instance_id,
                        details={"tool_name": call.name},
                    )
                    emitted = event_handler(event)
                    if inspect.isawaitable(emitted):
                        await emitted
                try:
                    output = builtin_tool.execute(
                        call.name, call.arguments
                    )
                    result = ToolResult(
                        name=call.name, success=True, output=output
                    )
                except Exception as error:
                    result = ToolResult(
                        name=call.name,
                        success=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                tool_results.append(result)
                if event_handler:
                    event = ExecutionEvent(
                        event_type=(
                            "tool_completed" if result.success else "tool_failed"
                        ),
                        agent_id=AGENT_ID,
                        instance_id=instance_id,
                        details={
                            "tool_name": call.name,
                            "success": result.success,
                            "error": result.error,
                        },
                    )
                    emitted = event_handler(event)
                    if inspect.isawaitable(emitted):
                        await emitted
                report = result.output if result.success else {
                    "success": False,
                    "failures": [result.error],
                }
                applied_ok = bool(result.success and report.get("success"))
                failure_text = "; ".join(report.get("failures", []))
                return AgentResult(
                    agent_id=AGENT_ID,
                    instance_id=instance_id,
                    status="COMPLETED" if applied_ok else "FAILED",
                    final_answer=(
                        "Knowledge graph modification result:\n"
                        + json.dumps(report, ensure_ascii=False, indent=2)
                    ),
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    messages=saved_messages,
                    error=None if applied_ok else (failure_text or result.error),
                )

            allowed_attachments = {
                str(Path(item).expanduser().resolve())
                for item in request.context.get("attachments", [])
            }

            def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
                tool_calls.append(ToolCall(name=name, arguments=arguments))
                if name == "read_document" and str(
                    Path(arguments.get("path", "")).expanduser().resolve()
                ) not in allowed_attachments:
                    result = ToolResult(
                        name=name,
                        success=False,
                        error="read_document accepts only supplied attachments.",
                    )
                elif name in builtin_tool.MUTATING_TOOLS:
                    result = ToolResult(
                        name=name,
                        success=False,
                        error="Explicit user approval is required before mutation.",
                    )
                else:
                    try:
                        result = ToolResult(
                            name=name,
                            success=True,
                            output=builtin_tool.execute(name, arguments),
                        )
                    except Exception as error:
                        result = ToolResult(
                            name=name,
                            success=False,
                            error=f"{type(error).__name__}: {error}",
                        )
                tool_results.append(result)
                return result

            prior_plan_count = sum(
                result.name == "save_modification_plan" and result.success
                for result in tool_results
            )
            agent = ToolCallingAgent(
                llm=self.llm,
                system_prompt=build_strong_system_prompt(
                    self.system_prompt,
                    builtin_tool.get_tool_spec(),
                    enable_communication=True,
                ),
                tool_executor=execute_tool,
                event_handler=event_handler,
                agent_id=AGENT_ID,
                instance_id=instance_id,
                max_steps=request.max_steps,
                history_mode="summary",
                enable_communication=True,
            )
            input_data = {
                "request": request.task,
                "context": request.context,
                "attachments": request.context.get("attachments", []),
            }
            try:
                await run_resumable(
                    agent,
                    request,
                    "Prepare a reviewed Knowledge graph modification plan:\n\n"
                    + json.dumps(input_data, ensure_ascii=False, indent=2, default=str),
                )
            except UserContactRaised as signal:
                return build_waiting_result(
                    request,
                    agent,
                    signal.contact,
                    checkpoint={
                        "phase": "analysis",
                        "tool_calls": [
                            item.model_dump(mode="json") for item in tool_calls
                        ],
                        "tool_results": [
                            item.model_dump(mode="json") for item in tool_results
                        ],
                    },
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )

            new_plan_count = sum(
                result.name == "save_modification_plan" and result.success
                for result in tool_results
            )
            if request.checkpoint and new_plan_count <= prior_plan_count:
                raise ValueError(
                    "Requested changes were not saved as a revised ModificationPlan."
                )
            plan = self._latest_plan(tool_results)
            if not plan:
                raise ValueError(
                    "The agent did not create a valid ModificationPlan."
                )
            contact = {
                "question": (
                    "Please review this Knowledge graph modification plan:\n\n"
                    + json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2)
                    + "\n\nReply APPROVE to apply it, or describe required changes."
                ),
                "reason": "Knowledge graph changes require user approval.",
                "expected_response": "APPROVE or free-text requested changes",
            }
            return build_waiting_result(
                request,
                agent,
                contact,
                checkpoint={
                    "phase": "review",
                    "modification_plan": plan.model_dump(mode="json"),
                    "tool_calls": [item.model_dump(mode="json") for item in tool_calls],
                    "tool_results": [item.model_dump(mode="json") for item in tool_results],
                },
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
        except Exception as error:
            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=instance_id,
                status="FAILED",
                tool_calls=tool_calls,
                tool_results=tool_results,
                error=f"{type(error).__name__}: {error}",
            )

import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from agent.agent_of_requirement import builtin_tool
from share.agent import ToolCallingAgent, build_strong_system_prompt
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import (
    AgentRequest,
    AgentResult,
    ExecutionEvent,
    Message,
    ToolCall,
    ToolResult,
)
from share.tool_loader import ToolLoader


AGENT_ID = "agent_of_requirement"
MAX_STEPS_MESSAGE = "Reached the maximum reasoning steps"
EventHandler = Callable[[ExecutionEvent], Any]


class AgentOfRequirementService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = Registry()
        self.llm = Copilot(model=model, base_url=base_url)
        self.base_prompt = Path(__file__).with_name(
            "base_prompt.txt"
        ).read_text(encoding="utf-8").strip()

    def _validate_request(self, request: AgentRequest) -> None:
        if not self.registry.can_call(request.caller_id, AGENT_ID):
            raise PermissionError(
                f"'{request.caller_id}' cannot call '{AGENT_ID}'."
            )
        if not request.runtime_system_prompt:
            raise ValueError("runtime_system_prompt is required.")
        self.registry.validate_tool_ids(request.assigned_tool_ids)

    async def run(
        self,
        request: AgentRequest,
        event_handler: Optional[EventHandler] = None,
    ) -> AgentResult:
        instance_id = f"{AGENT_ID}_{uuid4().hex[:10]}"
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []

        try:
            self._validate_request(request)
            tool_loader = ToolLoader(
                registry=self.registry,
                assigned_tool_ids=request.assigned_tool_ids,
            )
            tool_spec = (
                builtin_tool.get_tool_spec()
                + tool_loader.get_tool_spec()
            )

            runtime_prompt = (
                f"{self.base_prompt}\n\n"
                f"[ASSIGNED ROLE]\n"
                f"{request.runtime_system_prompt.strip()}"
            )
            system_prompt = build_strong_system_prompt(
                runtime_prompt,
                tool_spec,
                enable_communication=True,
            )

            def execute_tool(
                name: str,
                arguments: dict[str, Any],
            ) -> ToolResult:
                tool_calls.append(
                    ToolCall(name=name, arguments=arguments)
                )

                if builtin_tool.has_tool(name):
                    try:
                        result = ToolResult(
                            name=name,
                            success=True,
                            output=builtin_tool.execute(
                                name,
                                arguments,
                            ),
                        )
                    except Exception as error:
                        result = ToolResult(
                            name=name,
                            success=False,
                            error=(
                                f"{type(error).__name__}: {error}"
                            ),
                        )
                else:
                    result = tool_loader.execute(name, arguments)

                tool_results.append(result)
                return result

            agent = ToolCallingAgent(
                llm=self.llm,
                system_prompt=system_prompt,
                tool_executor=execute_tool,
                event_handler=event_handler,
                agent_id=AGENT_ID,
                instance_id=instance_id,
                max_steps=request.max_steps,
                history_mode="summary",
                enable_communication=True,
            )

            task_parts = [request.task]
            if request.context:
                task_parts.append(
                    "Context:\n"
                    + json.dumps(
                        request.context,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            final_answer = await agent.run("\n\n".join(task_parts))
            messages = [
                Message(**message)
                for message in agent.outbox
            ]

            if final_answer.startswith(MAX_STEPS_MESSAGE):
                return AgentResult(
                    agent_id=AGENT_ID,
                    instance_id=instance_id,
                    status="FAILED",
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    messages=messages,
                    error=final_answer,
                )

            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=instance_id,
                status="COMPLETED",
                final_answer=final_answer,
                tool_calls=tool_calls,
                tool_results=tool_results,
                messages=messages,
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

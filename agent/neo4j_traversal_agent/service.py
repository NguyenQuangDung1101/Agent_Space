from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from agent.neo4j_traversal_agent import builtin_tool
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
    ToolCall,
    ToolResult,
)


AGENT_ID = "neo4j_traversal_agent"
MAX_STEPS_MESSAGE = "Reached the maximum reasoning steps"
EventHandler = Callable[[ExecutionEvent], Any]


class Neo4jTraversalAgentService:
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
            ToolResult.model_validate(item)
            for item in saved.get("tool_results", [])
        ]

        try:
            if not self.registry.can_call(request.caller_id, AGENT_ID):
                raise PermissionError(
                    f"'{request.caller_id}' cannot call '{AGENT_ID}'."
                )

            def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
                tool_calls.append(ToolCall(name=name, arguments=arguments))
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
            task = request.task
            if request.context:
                task += "\n\nContext:\n" + json.dumps(
                    request.context,
                    ensure_ascii=False,
                    default=str,
                )

            try:
                final_answer = await run_resumable(agent, request, task)
            except UserContactRaised as signal:
                return build_waiting_result(
                    request,
                    agent,
                    signal.contact,
                    checkpoint={
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

            messages = [
                AgentMessage.model_validate(item) for item in agent.outbox
            ]
            if final_answer.startswith(MAX_STEPS_MESSAGE):
                return AgentResult(
                    agent_id=AGENT_ID,
                    instance_id=agent.instance_id,
                    status="FAILED",
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    messages=messages,
                    error=final_answer,
                )
            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=agent.instance_id,
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

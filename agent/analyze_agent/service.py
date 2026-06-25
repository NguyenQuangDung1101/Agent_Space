import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agent.analyze_agent import builtin_tool
from share.agent import (
    ToolCallingAgent,
    build_strong_system_prompt,
)
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import (
    AgentRequest,
    AnalysisResult,
)
from share.tool_loader import ToolLoader


AGENT_ID = "analyze_agent"


class AnalyzeAgentService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.registry = Registry()

        self.llm = Copilot(
            model=model,
            base_url=base_url,
        )

        self.system_prompt_text = Path(
            __file__
        ).with_name(
            "system_prompt.txt"
        ).read_text(
            encoding="utf-8"
        ).strip()

    def _validate_request(
        self,
        request: AgentRequest,
    ) -> None:
        if not self.registry.can_call(
            request.caller_id,
            AGENT_ID,
        ):
            raise PermissionError(
                f"'{request.caller_id}' cannot call "
                f"'{AGENT_ID}'."
            )

        for tool_id in request.assigned_tool_ids:
            if not self.registry.can_assign_tool(
                request.caller_id,
                tool_id,
            ):
                raise PermissionError(
                    f"'{request.caller_id}' cannot assign "
                    f"tool '{tool_id}'."
                )

    @staticmethod
    def _parse_result(
        final_answer: str,
    ) -> AnalysisResult:
        text = final_answer.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        return AnalysisResult.model_validate_json(
            text
        )

    async def run(
        self,
        request: AgentRequest,
    ) -> AnalysisResult:
        self._validate_request(request)

        tool_loader = ToolLoader(
            registry=self.registry,
            assigned_tool_ids=(
                request.assigned_tool_ids
            ),
        )

        tool_spec = (
            builtin_tool.get_tool_spec()
            + tool_loader.get_tool_spec()
        )

        system_prompt = build_strong_system_prompt(
            self.system_prompt_text,
            tool_spec,
            enable_communication=False,
        )

        def execute_tool(
            name: str,
            arguments: dict,
        ):
            if builtin_tool.has_tool(name):
                return builtin_tool.execute(
                    name,
                    arguments,
                )

            return tool_loader.execute(
                name,
                arguments,
            )

        agent = ToolCallingAgent(
            llm=self.llm,
            system_prompt=system_prompt,
            tool_executor=execute_tool,
            agent_id=AGENT_ID,
            instance_id=(
                f"{AGENT_ID}_{uuid4().hex[:10]}"
            ),
            max_steps=request.max_steps,
            history_mode="summary",
            enable_communication=False,
        )

        task_parts = [
            "Analyze this user request:",
            request.task,
        ]

        if request.context:
            task_parts.append(
                "Additional context:\n"
                + json.dumps(
                    request.context,
                    ensure_ascii=False,
                    default=str,
                )
            )

        final_answer = await agent.run(
            "\n\n".join(task_parts)
        )

        return self._parse_result(
            final_answer
        )
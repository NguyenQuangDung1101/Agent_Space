import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agent.synthesizer_agent import builtin_tool
from share.agent import ToolCallingAgent, build_strong_system_prompt
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import AgentRequest, AgentResult
from share.tool_loader import ToolLoader


AGENT_ID = "synthesizer_agent"


class SynthesizerAgentService:
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
                    f"Tool cannot be assigned: {tool_id}"
                )

    async def run(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        instance_id = (
            f"{AGENT_ID}_{uuid4().hex[:10]}"
        )

        try:
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
                instance_id=instance_id,
                max_steps=request.max_steps,
                history_mode="summary",
                enable_communication=False,
            )

            synthesis_input = {
                "original_request": request.task,
                "execution_context": request.context,
            }

            final_answer = await agent.run(
                "Create the final user response from "
                "the following information:\n\n"
                + json.dumps(
                    synthesis_input,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=instance_id,
                status="COMPLETED",
                final_answer=final_answer,
            )

        except Exception as error:
            return AgentResult(
                agent_id=AGENT_ID,
                instance_id=instance_id,
                status="FAILED",
                error=(
                    f"{type(error).__name__}: {error}"
                ),
            )
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agent.analyze_agent import builtin_tool
from share.agent import ToolCallingAgent, build_strong_system_prompt
from share.local_llm import Copilot
from share.registry import Registry
from share.schemas import AgentRequest, AnalysisResult


AGENT_ID = "analyze_agent"


class AnalyzeAgentService:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.registry = Registry()
        self.llm = Copilot(model=model, base_url=base_url)
        self.system_prompt_text = Path(__file__).with_name(
            "system_prompt.txt"
        ).read_text(encoding="utf-8").strip()

    def _validate_request(self, request: AgentRequest) -> None:
        if not self.registry.can_call(request.caller_id, AGENT_ID):
            raise PermissionError(
                f"'{request.caller_id}' cannot call '{AGENT_ID}'."
            )

    def _validate_result(
        self,
        result: AnalysisResult,
        worker_catalog: list[dict],
        tool_catalog: list[dict],
    ) -> None:
        if result.execution_mode == "team":
            if result.single_plan is not None:
                raise ValueError("single_plan must be null for team mode.")
            return

        plan = result.single_plan
        if plan is None:
            raise ValueError("single_plan is required for single mode.")

        agents = {item["id"]: item for item in worker_catalog}
        tools = {item["id"] for item in tool_catalog}

        if plan.agent_id not in agents:
            raise ValueError(f"Invalid worker agent: {plan.agent_id}")

        invalid_tools = set(plan.assigned_tool_ids) - tools
        if invalid_tools:
            raise ValueError(
                "Invalid assigned tools: " + ", ".join(sorted(invalid_tools))
            )

        if agents[plan.agent_id]["type"] == "dynamic" and not plan.system_prompt:
            raise ValueError("Dynamic workers require a system_prompt.")

    async def run(self, request: AgentRequest) -> AnalysisResult:
        self._validate_request(request)

        worker_catalog = request.context.get(
            "worker_catalog",
            self.registry.list_selectable_agents(),
        )
        tool_catalog = request.context.get(
            "tool_catalog",
            self.registry.list_tools(),
        )

        system_prompt = build_strong_system_prompt(
            self.system_prompt_text,
            builtin_tool.get_tool_spec(),
            enable_communication=False,
        )

        agent = ToolCallingAgent(
            llm=self.llm,
            system_prompt=system_prompt,
            tool_executor=builtin_tool.execute,
            agent_id=AGENT_ID,
            instance_id=f"{AGENT_ID}_{uuid4().hex[:10]}",
            max_steps=request.max_steps,
            history_mode="summary",
            enable_communication=False,
        )

        analysis_input = {
            "user_request": request.task,
            "user_context": request.context.get("user_context", {}),
            "worker_catalog": worker_catalog,
            "tool_catalog": tool_catalog,
        }

        final_answer = await agent.run(
            "Analyze and select the execution solution:\n\n"
            + json.dumps(
                analysis_input,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        result = AnalysisResult.model_validate_json(final_answer.strip())
        self._validate_result(result, worker_catalog, tool_catalog)
        return result

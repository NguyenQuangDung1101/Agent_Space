
from fastapi import FastAPI

from agent.agent_of_requirement.service import (
    AgentOfRequirementService,
)
from share.schemas import (
    AgentRequest,
    AgentResult,
)


app = FastAPI(
    title="Agent of Requirement",
)

service = AgentOfRequirementService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "agent_of_requirement",
    }


@app.post(
    "/run",
    response_model=AgentResult,
)
async def run_agent(
    request: AgentRequest,
) -> AgentResult:
    return await service.run(request)
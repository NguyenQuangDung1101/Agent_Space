from fastapi import FastAPI

from agent.planner_agent.service import PlannerAgentService
from share.schemas import AgentRequest, AgentResult


app = FastAPI(title="Planner Agent")
service = PlannerAgentService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "planner_agent"}


@app.post("/run", response_model=AgentResult)
async def run_agent(request: AgentRequest) -> AgentResult:
    return await service.run(request)

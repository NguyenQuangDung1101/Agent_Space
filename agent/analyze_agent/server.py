from fastapi import FastAPI

from agent.analyze_agent.service import AnalyzeAgentService
from share.schemas import AgentRequest, AgentResult


app = FastAPI(title="Analyze Agent")
service = AnalyzeAgentService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "analyze_agent"}


@app.post("/run", response_model=AgentResult)
async def run_agent(request: AgentRequest) -> AgentResult:
    return await service.run(request)

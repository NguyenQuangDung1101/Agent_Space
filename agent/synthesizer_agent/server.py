from fastapi import FastAPI

from agent.synthesizer_agent.service import (
    SynthesizerAgentService,
)
from share.schemas import AgentRequest, AgentResult


app = FastAPI(
    title="Synthesizer Agent",
)

service = SynthesizerAgentService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "synthesizer_agent",
    }


@app.post(
    "/run",
    response_model=AgentResult,
)
async def run_agent(
    request: AgentRequest,
) -> AgentResult:
    return await service.run(request)
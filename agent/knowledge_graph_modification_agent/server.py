from fastapi import FastAPI

from agent.knowledge_graph_modification_agent.service import (
    KnowledgeGraphModificationAgentService,
)
from share.schemas import AgentRequest, AgentResult


app = FastAPI(title="Knowledge Graph Modification Agent")
service = KnowledgeGraphModificationAgentService()


@app.post("/run", response_model=AgentResult)
async def run_agent(request: AgentRequest) -> AgentResult:
    return await service.run(request)

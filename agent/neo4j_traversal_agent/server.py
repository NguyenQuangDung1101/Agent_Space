from fastapi import FastAPI

from agent.neo4j_traversal_agent.service import Neo4jTraversalAgentService
from share.schemas import AgentRequest, AgentResult


app = FastAPI(title="Neo4j Traversal Agent")
service = Neo4jTraversalAgentService()


@app.post("/run", response_model=AgentResult)
async def run_agent(request: AgentRequest) -> AgentResult:
    return await service.run(request)

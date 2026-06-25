from fastapi import FastAPI, HTTPException

from agent.planner_agent.service import (
    PlannerAgentService,
)
from share.schemas import AgentRequest, TeamPlan


app = FastAPI(
    title="Planner Agent",
)

service = PlannerAgentService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "planner_agent",
    }


@app.post(
    "/run",
    response_model=TeamPlan,
)
async def run_agent(
    request: AgentRequest,
) -> TeamPlan:
    try:
        return await service.run(request)

    except PermissionError as error:
        raise HTTPException(
            status_code=403,
            detail=str(error),
        ) from error

    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error
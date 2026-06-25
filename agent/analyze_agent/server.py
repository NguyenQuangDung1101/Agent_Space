from fastapi import (
    FastAPI,
    HTTPException,
)

from agent.analyze_agent.service import (
    AnalyzeAgentService,
)
from share.schemas import (
    AgentRequest,
    AnalysisResult,
)


app = FastAPI(
    title="Analyze Agent",
)

service = AnalyzeAgentService()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "analyze_agent",
    }


@app.post(
    "/run",
    response_model=AnalysisResult,
)
async def run_agent(
    request: AgentRequest,
) -> AnalysisResult:
    try:
        return await service.run(request)

    except PermissionError as error:
        raise HTTPException(
            status_code=403,
            detail=str(error),
        ) from error

    except (
        KeyError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error
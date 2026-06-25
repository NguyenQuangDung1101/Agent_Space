from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from core.manager import AgentSpaceManager
from share.schemas import (
    ExecutionResult,
    SessionRecord,
)


class TaskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task: str = Field(
        min_length=1
    )

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    assigned_tool_ids: list[str] = Field(
        default_factory=list
    )

    max_steps: int = Field(
        default=10,
        ge=1,
        le=50,
    )


app = FastAPI(
    title="Agent Space",
    version="0.1.0",
)

manager = AgentSpaceManager()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "agent_space",
    }


@app.post(
    "/tasks",
    response_model=ExecutionResult,
)
async def create_task(
    request: TaskRequest,
) -> ExecutionResult:
    return await manager.handle_task(
        user_request=request.task,
        context=request.context,
        assigned_tool_ids=(
            request.assigned_tool_ids
        ),
        max_steps=request.max_steps,
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionRecord,
)
async def get_session(
    session_id: str,
) -> SessionRecord:
    try:
        return manager.get_session(
            session_id
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
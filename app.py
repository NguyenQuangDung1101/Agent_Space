import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.manager import AgentSpaceManager
from share.schemas import ExecutionResult, SessionRecord


class TaskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    assigned_tool_ids: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=50)


class TaskStartedResponse(BaseModel):
    session_id: str
    status: str
    events_url: str


app = FastAPI(
    title="Agent Space",
    version="0.3.0",
)

manager = AgentSpaceManager()
manager.registry.print_catalogs()

GUI_FILE = Path(__file__).resolve().parent / "gui" / "index.html"


@app.get("/", include_in_schema=False)
async def chat_gui() -> FileResponse:
    if not GUI_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="GUI file not found.",
        )
    return FileResponse(GUI_FILE)


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
        assigned_tool_ids=request.assigned_tool_ids,
        max_steps=request.max_steps,
    )


@app.post(
    "/tasks/start",
    response_model=TaskStartedResponse,
)
async def start_task(
    request: TaskRequest,
) -> TaskStartedResponse:
    session = manager.start_task(
        user_request=request.task,
        context=request.context,
        assigned_tool_ids=request.assigned_tool_ids,
        max_steps=request.max_steps,
    )
    return TaskStartedResponse(
        session_id=session.session_id,
        status=session.status,
        events_url=f"/sessions/{session.session_id}/events",
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionRecord,
)
async def get_session(
    session_id: str,
) -> SessionRecord:
    try:
        return manager.get_session(session_id)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.get("/sessions/{session_id}/events")
async def stream_session_events(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    try:
        manager.get_session(session_id)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    async def event_stream():
        async for event in manager.stream_events(session_id):
            if await request.is_disconnected():
                break

            if event is None:
                yield ": keep-alive\n\n"
                continue

            data = json.dumps(
                event.model_dump(mode="json"),
                ensure_ascii=False,
            )
            yield (
                f"event: {event.event_type}\n"
                f"data: {data}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

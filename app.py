import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.manager import AgentSpaceManager
from share.schemas import (
    Conversation,
    ExecutionEvent,
    ExecutionResult,
    Message,
    SessionRecord,
    UserContactRequest,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ConversationRequest(StrictModel):
    title: Optional[str] = None


class MessageRequest(StrictModel):
    content: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    assigned_tool_ids: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=50)


class MessageStartedResponse(StrictModel):
    conversation_id: str
    message_id: str
    session_id: str
    status: str
    resumed: bool
    events_url: str


class TaskRequest(StrictModel):
    task: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    assigned_tool_ids: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=50)


class TaskStartedResponse(StrictModel):
    conversation_id: str
    session_id: str
    status: str
    events_url: str


manager = AgentSpaceManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(manager.initialize)
    manager.registry.print_catalogs()
    yield
    manager.close()


app = FastAPI(title="Agent Space", version="0.7.0", lifespan=lifespan)
GUI_FILE = Path(__file__).resolve().parent / "gui" / "index.html"


@app.get("/", include_in_schema=False)
async def chat_gui() -> FileResponse:
    if not GUI_FILE.is_file():
        raise HTTPException(status_code=404, detail="GUI file not found.")
    return FileResponse(GUI_FILE)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent_space"}


@app.post("/conversations", response_model=Conversation)
async def create_conversation(
    request: ConversationRequest,
) -> Conversation:
    return manager.create_conversation(request.title)


@app.get("/conversations", response_model=list[Conversation])
async def list_conversations() -> list[Conversation]:
    return manager.list_conversations()


@app.get(
    "/conversations/{conversation_id}",
    response_model=Conversation,
)
async def get_conversation(conversation_id: str) -> Conversation:
    try:
        return manager.get_conversation(conversation_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[Message],
)
async def get_conversation_messages(
    conversation_id: str,
) -> list[Message]:
    try:
        return manager.get_messages(conversation_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/conversations/{conversation_id}/sessions",
    response_model=list[SessionRecord],
)
async def get_conversation_sessions(
    conversation_id: str,
) -> list[SessionRecord]:
    try:
        return manager.get_sessions(conversation_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/conversations/{conversation_id}/contacts",
    response_model=list[UserContactRequest],
)
async def get_conversation_contacts(
    conversation_id: str,
) -> list[UserContactRequest]:
    try:
        return manager.get_contacts(conversation_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageStartedResponse,
)
async def create_message(
    conversation_id: str,
    request: MessageRequest,
) -> MessageStartedResponse:
    try:
        session, message, resumed = manager.start_message(
            conversation_id=conversation_id,
            content=request.content,
            context=request.context,
            assigned_tool_ids=request.assigned_tool_ids,
            attachments=request.attachments,
            max_steps=request.max_steps,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return MessageStartedResponse(
        conversation_id=conversation_id,
        message_id=message.message_id,
        session_id=session.session_id,
        status=session.status,
        resumed=resumed,
        events_url=f"/sessions/{session.session_id}/events",
    )


@app.get("/sessions/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    try:
        return manager.get_session(session_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/sessions/{session_id}/events/history",
    response_model=list[ExecutionEvent],
)
async def get_session_events(session_id: str) -> list[ExecutionEvent]:
    try:
        return manager.get_events(session_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/sessions/{session_id}/events")
async def stream_session_events(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    try:
        manager.get_session(session_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    async def event_stream():
        async for event in manager.stream_events(session_id):
            if await request.is_disconnected():
                break
            if event is None:
                yield ": keep-alive\n\n"
                continue
            data = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False
            )
            yield f"event: {event.event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Backward-compatible task endpoints. Each task receives its own conversation.
@app.post("/tasks", response_model=ExecutionResult)
async def create_task(request: TaskRequest) -> ExecutionResult:
    return await manager.handle_task(
        user_request=request.task,
        context=request.context,
        assigned_tool_ids=request.assigned_tool_ids,
        max_steps=request.max_steps,
    )


@app.post("/tasks/start", response_model=TaskStartedResponse)
async def start_task(request: TaskRequest) -> TaskStartedResponse:
    session = manager.start_task(
        user_request=request.task,
        context=request.context,
        assigned_tool_ids=request.assigned_tool_ids,
        max_steps=request.max_steps,
    )
    return TaskStartedResponse(
        conversation_id=session.conversation_id,
        session_id=session.session_id,
        status=session.status,
        events_url=f"/sessions/{session.session_id}/events",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

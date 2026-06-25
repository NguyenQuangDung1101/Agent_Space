from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


AgentStatus = Literal[
    "PENDING",
    "RUNNING",
    "WAITING",
    "COMPLETED",
    "FAILED",
]

ExecutionMode = Literal["single", "team"]

OrchestrationMode = Literal[
    "sequential",
    "parallel",
    "supervisor",
]

MessageType = Literal[
    "direct",
    "broadcast",
    "manager",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas
# ──────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseSchema):
    name: str
    arguments: Dict[str, Any] = Field(
        default_factory=dict
    )


class ToolResult(BaseSchema):
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Communication schemas
# ──────────────────────────────────────────────────────────────────────────────

class Message(BaseSchema):
    sender: str
    recipient: str
    message_type: MessageType = "direct"
    content: str
    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ──────────────────────────────────────────────────────────────────────────────
# Agent schemas
# ──────────────────────────────────────────────────────────────────────────────

class AgentRequest(BaseSchema):
    session_id: str
    caller_id: str
    task: str

    context: Dict[str, Any] = Field(
        default_factory=dict
    )

    assigned_tool_ids: List[str] = Field(
        default_factory=list
    )

    runtime_system_prompt: Optional[str] = None
    max_steps: int = Field(default=10, ge=1)


class AgentResult(BaseSchema):
    agent_id: str
    instance_id: str
    status: AgentStatus

    final_answer: Optional[str] = None

    tool_calls: List[ToolCall] = Field(
        default_factory=list
    )

    tool_results: List[ToolResult] = Field(
        default_factory=list
    )

    messages: List[Message] = Field(
        default_factory=list
    )

    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Analyze Agent schemas
# ──────────────────────────────────────────────────────────────────────────────

class AnalysisResult(BaseSchema):
    objective: str

    constraints: List[str] = Field(
        default_factory=list
    )

    expected_output: str

    required_capabilities: List[str] = Field(
        default_factory=list
    )

    missing_information: List[str] = Field(
        default_factory=list
    )

    recommended_mode: ExecutionMode
    reason: str


# ──────────────────────────────────────────────────────────────────────────────
# Planner schemas
# ──────────────────────────────────────────────────────────────────────────────

class TeamMember(BaseSchema):
    instance_id: str
    agent_id: str
    role: str
    task: str

    system_prompt: Optional[str] = None

    assigned_tool_ids: List[str] = Field(
        default_factory=list
    )


class TeamTask(BaseSchema):
    task_id: str
    agent_instance_id: str
    instruction: str

    dependencies: List[str] = Field(
        default_factory=list
    )


class TeamPlan(BaseSchema):
    team_id: str
    orchestration: OrchestrationMode

    members: List[TeamMember] = Field(
        default_factory=list
    )

    tasks: List[TeamTask] = Field(
        default_factory=list
    )


# ──────────────────────────────────────────────────────────────────────────────
# Manager and session schemas
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionResult(BaseSchema):
    session_id: str
    status: AgentStatus
    execution_mode: ExecutionMode

    final_answer: Optional[str] = None
    agent_results: List[AgentResult] = Field(
        default_factory=list
    )

    errors: List[str] = Field(
        default_factory=list
    )


class SessionRecord(BaseSchema):
    session_id: str
    original_request: str
    status: AgentStatus = "PENDING"

    execution_mode: Optional[ExecutionMode] = None
    analysis: Optional[AnalysisResult] = None
    team_plan: Optional[TeamPlan] = None
    final_result: Optional[ExecutionResult] = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )
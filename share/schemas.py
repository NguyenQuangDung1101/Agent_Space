from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


AgentStatus = Literal[
    "PENDING",
    "RUNNING",
    "WAITING_FOR_USER",
    "COMPLETED",
    "FAILED",
]
ExecutionMode = Literal["single", "team"]
OrchestrationMode = Literal[
    "sequential",
    "parallel",
    "supervisor",
]
AgentMessageType = Literal[
    "direct",
    "broadcast",
    "manager",
]
ConversationRole = Literal["user", "assistant"]
ContactStatus = Literal["PENDING", "ANSWERED"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class Conversation(BaseSchema):
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Message(BaseSchema):
    message_id: str
    conversation_id: str
    role: ConversationRole
    content: str
    session_id: Optional[str] = None
    contact_request_id: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMemory(BaseSchema):
    conversation_id: str
    summary: str = ""
    session_ids: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class UserContactRequest(BaseSchema):
    contact_id: str
    conversation_id: str
    session_id: str
    agent_id: str
    agent_name: str
    instance_id: str
    question: str
    reason: str = ""
    expected_response: str = "Free-text response"
    status: ContactStatus = "PENDING"
    answer_message_id: Optional[str] = None
    response: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    answered_at: Optional[datetime] = None


KnowledgeNodeKind = Literal[
    "KnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeEntity",
]
KnowledgeRelationshipType = Literal[
    "HAS_DOCUMENT",
    "HAS_CHUNK",
    "HAS_ENTITY",
    "MENTIONS",
    "RELATED_TO",
    "SUPPORTED_BY",
]


class ImportNode(BaseSchema):
    id: str
    kind: KnowledgeNodeKind
    name: str
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None


class ImportRelationship(BaseSchema):
    source_id: str
    target_id: str
    type: KnowledgeRelationshipType
    properties: Dict[str, Any] = Field(default_factory=dict)


class ImportPlan(BaseSchema):
    nodes: List[ImportNode] = Field(default_factory=list)
    relationships: List[ImportRelationship] = Field(default_factory=list)


class DeleteRelationship(BaseSchema):
    source_id: str
    target_id: str
    type: KnowledgeRelationshipType


class DeletePlan(BaseSchema):
    node_ids: List[str] = Field(default_factory=list)
    relationships: List[DeleteRelationship] = Field(default_factory=list)


class ModificationPlan(BaseSchema):
    operation: Literal["import", "delete"]
    summary: str
    import_plan: Optional[ImportPlan] = None
    delete_plan: Optional[DeletePlan] = None

    @model_validator(mode="after")
    def validate_operation_plan(self):
        if self.operation == "import":
            if self.import_plan is None or self.delete_plan is not None:
                raise ValueError("Import operation requires only import_plan.")
        elif self.delete_plan is None or self.import_plan is not None:
            raise ValueError("Delete operation requires only delete_plan.")
        return self


class ModificationResult(BaseSchema):
    success: bool
    applied_nodes: List[str] = Field(default_factory=list)
    applied_relationships: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseSchema):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseSchema):
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None


class AgentMessage(BaseSchema):
    sender: str
    recipient: str
    message_type: AgentMessageType = "direct"
    content: str
    timestamp: datetime = Field(default_factory=utc_now)


class ExecutionEvent(BaseSchema):
    event_type: str
    agent_id: Optional[str] = None
    instance_id: Optional[str] = None
    task_id: Optional[str] = None
    task: Optional[str] = None
    assigned_tool_ids: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class AgentRequest(BaseSchema):
    session_id: str
    conversation_id: Optional[str] = None
    caller_id: str
    task: str
    context: Dict[str, Any] = Field(default_factory=dict)
    assigned_tool_ids: List[str] = Field(default_factory=list)
    runtime_system_prompt: Optional[str] = None
    instance_id: Optional[str] = None
    checkpoint: Optional[Dict[str, Any]] = None
    user_response: Optional[str] = None
    max_steps: int = Field(default=10, ge=1)


class AgentResult(BaseSchema):
    agent_id: str
    instance_id: str
    status: AgentStatus
    final_answer: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)
    user_contact: Optional[UserContactRequest] = None
    checkpoint: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SingleAgentPlan(BaseSchema):
    agent_id: str
    task: str
    system_prompt: Optional[str] = None
    assigned_tool_ids: List[str] = Field(default_factory=list)


class AnalysisResult(BaseSchema):
    objective: str
    constraints: List[str] = Field(default_factory=list)
    expected_output: str
    required_capabilities: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    execution_mode: ExecutionMode = Field(
        validation_alias=AliasChoices(
            "execution_mode",
            "recommended_mode",
        )
    )
    single_plan: Optional[SingleAgentPlan] = None
    reason: str


class TeamMember(BaseSchema):
    instance_id: str
    agent_id: str
    role: str
    task: str
    system_prompt: Optional[str] = None
    assigned_tool_ids: List[str] = Field(default_factory=list)


class TeamTask(BaseSchema):
    task_id: str
    agent_instance_id: str
    instruction: str
    dependencies: List[str] = Field(default_factory=list)


class SupervisorConfig(BaseSchema):
    supervisor_instance_id: str
    max_rounds: int = Field(default=20, ge=1)


class TeamPlan(BaseSchema):
    team_id: str
    orchestration: OrchestrationMode
    members: List[TeamMember] = Field(default_factory=list)
    tasks: List[TeamTask] = Field(default_factory=list)
    supervisor: Optional[SupervisorConfig] = None


class ExecutionResult(BaseSchema):
    session_id: str
    status: AgentStatus
    execution_mode: Optional[ExecutionMode] = None
    final_answer: Optional[str] = None
    agent_results: List[AgentResult] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)
    events: List[ExecutionEvent] = Field(default_factory=list)
    pending_contact: Optional[UserContactRequest] = None
    checkpoint: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class SessionRecord(BaseSchema):
    session_id: str
    conversation_id: str
    trigger_message_id: str
    original_request: str
    status: AgentStatus = "PENDING"
    execution_mode: Optional[ExecutionMode] = None
    analysis: Optional[AnalysisResult] = None
    single_plan: Optional[SingleAgentPlan] = None
    team_plan: Optional[TeamPlan] = None
    pending_contact_id: Optional[str] = None
    final_result: Optional[ExecutionResult] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

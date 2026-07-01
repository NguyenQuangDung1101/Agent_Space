from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase

from share.schemas import (
    Conversation,
    ConversationMemory,
    ExecutionResult,
    Message,
    SessionRecord,
    UserContactRequest,
)


logger = logging.getLogger(__name__)
AGENT_SPACE_DOMAIN = "agent_space"


@dataclass(frozen=True)
class Neo4jSettings:
    enabled: bool
    uri: str
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        load_dotenv()
        enabled = os.getenv("NEO4J_ENABLED", "true").lower() not in {
            "0", "false", "no",
        }
        return cls(
            enabled=enabled,
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "agent-space-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


def json_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


class Neo4jRepository:
    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        self.settings = settings or Neo4jSettings.from_env()
        self.driver = None
        if self.settings.enabled:
            self.driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
            )

    @property
    def enabled(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        if not self.driver:
            return []
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.settings.database,
        )
        return [record.data() for record in records]

    def verify(self) -> None:
        if self.driver:
            self.driver.verify_connectivity()

    def apply_schema(self) -> None:
        path = Path(__file__).with_name("schema.cypher")
        for statement in path.read_text(encoding="utf-8").split(";"):
            if statement.strip():
                self._run(statement.strip())

    def reset_agent_space(self) -> None:
        self._run(
            "MATCH (n {domain: $domain}) DETACH DELETE n",
            domain=AGENT_SPACE_DOMAIN,
        )

    def sync_registry(
        self,
        agents: Iterable[dict[str, Any]],
        tools: Iterable[dict[str, Any]],
    ) -> None:
        self._run(
            "MERGE (s:System {id: 'agent_space', domain: $domain}) "
            "SET s.name = 'Agent Space'",
            domain=AGENT_SPACE_DOMAIN,
        )
        for agent in agents:
            self._run(
                "MATCH (s:System {id: 'agent_space', domain: $domain}) "
                "MERGE (a:Agent {id: $id, domain: $domain}) "
                "SET a.type = $type, a.description = $description, "
                "a.capabilities = $capabilities, "
                "a.selectable_as_worker = $selectable "
                "MERGE (s)-[:REGISTERS]->(a)",
                domain=AGENT_SPACE_DOMAIN,
                id=agent["id"],
                type=agent["type"],
                description=agent["description"],
                capabilities=agent.get("capabilities", []),
                selectable=agent.get("selectable_as_worker", False),
            )
        for tool in tools:
            self._run(
                "MATCH (s:System {id: 'agent_space', domain: $domain}) "
                "MERGE (t:Tool {id: $id, domain: $domain}) "
                "SET t.name = $name, t.description = $description, "
                "t.input_schema_json = $schema "
                "MERGE (s)-[:REGISTERS]->(t)",
                domain=AGENT_SPACE_DOMAIN,
                id=tool["id"],
                name=tool.get("name", tool["id"]),
                description=tool.get("description", ""),
                schema=json_text(tool.get("inputSchema", {})),
            )

    def save_conversation(self, conversation: Conversation) -> None:
        self._run(
            "MERGE (c:Conversation {id: $id, domain: $domain}) "
            "SET c.title = $title, c.created_at = datetime($created_at), "
            "c.updated_at = datetime($updated_at)",
            id=conversation.conversation_id,
            domain=AGENT_SPACE_DOMAIN,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
        )

    def save_message(self, message: Message) -> None:
        self._run(
            "MATCH (c:Conversation {id: $conversation_id, domain: $domain}) "
            "MERGE (m:Message {id: $id, domain: $domain}) "
            "SET m.role = $role, m.content = $content, "
            "m.attachments = $attachments, m.created_at = datetime($created_at), "
            "m.session_id = $session_id, m.contact_request_id = $contact_id "
            "MERGE (c)-[:HAS_MESSAGE]->(m)",
            domain=AGENT_SPACE_DOMAIN,
            conversation_id=message.conversation_id,
            id=message.message_id,
            role=message.role,
            content=message.content,
            attachments=message.attachments,
            created_at=message.created_at.isoformat(),
            session_id=message.session_id,
            contact_id=message.contact_request_id,
        )

    def save_memory(self, memory: ConversationMemory) -> None:
        memory_id = f"memory_{memory.conversation_id}"
        self._run(
            "MATCH (c:Conversation {id: $conversation_id, domain: $domain}) "
            "MERGE (m:MemorySnapshot {id: $id, domain: $domain}) "
            "SET m.summary = $summary, m.session_ids = $session_ids, "
            "m.updated_at = datetime($updated_at) "
            "MERGE (c)-[:HAS_MEMORY]->(m)",
            domain=AGENT_SPACE_DOMAIN,
            conversation_id=memory.conversation_id,
            id=memory_id,
            summary=memory.summary,
            session_ids=memory.session_ids,
            updated_at=memory.updated_at.isoformat(),
        )

    def save_session(self, session: SessionRecord) -> None:
        self._run(
            "MATCH (c:Conversation {id: $conversation_id, domain: $domain}) "
            "MERGE (s:Session {id: $id, domain: $domain}) "
            "SET s.status = $status, s.original_request = $request, "
            "s.execution_mode = $mode, s.pending_contact_id = $contact_id, "
            "s.created_at = datetime($created_at), "
            "s.updated_at = datetime($updated_at) "
            "MERGE (c)-[:HAS_SESSION]->(s) "
            "WITH s OPTIONAL MATCH (m:Message {id: $message_id, domain: $domain}) "
            "FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END | "
            "MERGE (s)-[:TRIGGERED_BY]->(m))",
            domain=AGENT_SPACE_DOMAIN,
            conversation_id=session.conversation_id,
            id=session.session_id,
            status=session.status,
            request=session.original_request,
            mode=session.execution_mode,
            contact_id=session.pending_contact_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            message_id=session.trigger_message_id,
        )
        task_id = f"task_{session.session_id}"
        self._run(
            "MATCH (s:Session {id: $session_id, domain: $domain}) "
            "MERGE (t:Task {id: $task_id, domain: $domain}) "
            "SET t.instruction = $instruction "
            "MERGE (s)-[:HAS_TASK]->(t)",
            domain=AGENT_SPACE_DOMAIN,
            session_id=session.session_id,
            task_id=task_id,
            instruction=session.original_request,
        )
        if session.analysis:
            analysis_id = f"analysis_{session.session_id}"
            self._run(
                "MATCH (s:Session {id: $session_id, domain: $domain}) "
                "MERGE (a:Analysis {id: $id, domain: $domain}) "
                "SET a.data_json = $data "
                "MERGE (s)-[:HAS_ANALYSIS]->(a)",
                domain=AGENT_SPACE_DOMAIN,
                session_id=session.session_id,
                id=analysis_id,
                data=json_text(session.analysis),
            )

    def save_plan(self, session: SessionRecord, plan: Any) -> None:
        solution_id = f"solution_{session.session_id}"
        task_id = f"task_{session.session_id}"
        self._run(
            "MATCH (t:Task {id: $task_id, domain: $domain}) "
            "MERGE (s:Solution {id: $id, domain: $domain}) "
            "SET s.execution_mode = $mode, s.plan_json = $plan "
            "MERGE (t)-[:SOLVED_BY]->(s)",
            domain=AGENT_SPACE_DOMAIN,
            task_id=task_id,
            id=solution_id,
            mode=session.execution_mode,
            plan=json_text(plan),
        )
        for relationship_type in ("USES_AGENT", "USES_TOOL"):
            self._run(
                f"MATCH (s:Solution {{id: $id, domain: $domain}})"
                f"-[r:{relationship_type}]->() DELETE r",
                id=solution_id,
                domain=AGENT_SPACE_DOMAIN,
            )
        if session.single_plan:
            agents = [session.single_plan.agent_id]
            tools = session.single_plan.assigned_tool_ids
        elif session.team_plan:
            agents = [member.agent_id for member in session.team_plan.members]
            tools = [
                tool_id
                for member in session.team_plan.members
                for tool_id in member.assigned_tool_ids
            ]
        else:
            agents, tools = [], []
        for agent_id in dict.fromkeys(agents):
            self._run(
                "MATCH (s:Solution {id: $solution_id, domain: $domain}), "
                "(a:Agent {id: $agent_id, domain: $domain}) "
                "MERGE (s)-[:USES_AGENT]->(a)",
                domain=AGENT_SPACE_DOMAIN,
                solution_id=solution_id,
                agent_id=agent_id,
            )
        for tool_id in dict.fromkeys(tools):
            self._run(
                "MATCH (s:Solution {id: $solution_id, domain: $domain}), "
                "(t:Tool {id: $tool_id, domain: $domain}) "
                "MERGE (s)-[:USES_TOOL]->(t)",
                domain=AGENT_SPACE_DOMAIN,
                solution_id=solution_id,
                tool_id=tool_id,
            )

    def save_contact(self, contact: UserContactRequest) -> None:
        self._run(
            "MATCH (s:Session {id: $session_id, domain: $domain}) "
            "MERGE (c:UserContact {id: $id, domain: $domain}) "
            "SET c.status = $status, c.question = $question, c.reason = $reason, "
            "c.expected_response = $expected, c.response = $response, "
            "c.agent_id = $agent_id, c.instance_id = $instance_id, "
            "c.created_at = datetime($created_at), "
            "c.answered_at = CASE WHEN $answered_at IS NULL THEN null "
            "ELSE datetime($answered_at) END "
            "MERGE (s)-[:HAS_CONTACT]->(c) "
            "WITH c OPTIONAL MATCH (r:AgentRun {instance_id: $instance_id, "
            "session_id: $session_id, domain: $domain}) "
            "FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END | "
            "MERGE (c)-[:REQUESTED_BY]->(r)) "
            "WITH c OPTIONAL MATCH (m:Message {id: $message_id, domain: $domain}) "
            "FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END | "
            "MERGE (c)-[:ANSWERED_BY]->(m))",
            domain=AGENT_SPACE_DOMAIN,
            session_id=contact.session_id,
            id=contact.contact_id,
            status=contact.status,
            question=contact.question,
            reason=contact.reason,
            expected=contact.expected_response,
            response=contact.response,
            agent_id=contact.agent_id,
            instance_id=contact.instance_id,
            created_at=contact.created_at.isoformat(),
            answered_at=(contact.answered_at.isoformat() if contact.answered_at else None),
            message_id=contact.answer_message_id,
        )

    def save_execution(
        self,
        session: SessionRecord,
        execution: ExecutionResult,
    ) -> None:
        execution_id = f"execution_{session.session_id}"
        self._run(
            "MATCH (s:Session {id: $session_id, domain: $domain}) "
            "MERGE (e:Execution {id: $id, domain: $domain}) "
            "SET e.status = $status, e.final_answer = $answer, "
            "e.errors_json = $errors "
            "MERGE (s)-[:HAS_EXECUTION]->(e)",
            domain=AGENT_SPACE_DOMAIN,
            session_id=session.session_id,
            id=execution_id,
            status=execution.status,
            answer=execution.final_answer,
            errors=json_text(execution.errors),
        )
        for index, result in enumerate(execution.agent_results):
            run_id = f"{execution_id}:agent:{index}:{result.instance_id}"
            self._run(
                "MATCH (e:Execution {id: $execution_id, domain: $domain}) "
                "MERGE (r:AgentRun {id: $id, domain: $domain}) "
                "SET r.instance_id = $instance_id, r.session_id = $session_id, "
                "r.status = $status, r.final_answer = $answer, r.error = $error "
                "MERGE (e)-[:HAS_AGENT_RUN]->(r) "
                "WITH r OPTIONAL MATCH (a:Agent {id: $agent_id, domain: $domain}) "
                "FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END | "
                "MERGE (r)-[:INSTANCE_OF]->(a))",
                domain=AGENT_SPACE_DOMAIN,
                execution_id=execution_id,
                id=run_id,
                session_id=session.session_id,
                instance_id=result.instance_id,
                agent_id=result.agent_id,
                status=result.status,
                answer=result.final_answer,
                error=result.error,
            )
            for tool_index, tool_result in enumerate(result.tool_results):
                tool_run_id = f"{run_id}:tool:{tool_index}"
                self._run(
                    "MATCH (r:AgentRun {id: $run_id, domain: $domain}) "
                    "MERGE (t:ToolRun {id: $id, domain: $domain}) "
                    "SET t.tool_id = $tool_id, t.success = $success, "
                    "t.output_json = $output, t.error = $error "
                    "MERGE (r)-[:HAS_TOOL_RUN]->(t)",
                    domain=AGENT_SPACE_DOMAIN,
                    run_id=run_id,
                    id=tool_run_id,
                    tool_id=tool_result.name,
                    success=tool_result.success,
                    output=json_text(tool_result.output),
                    error=tool_result.error,
                )

    def rebuild(self, store: Any, registry: Any) -> None:
        self.verify()
        self.apply_schema()
        self.reset_agent_space()
        self.sync_registry(registry.list_agents(), registry.list_tools())
        for conversation in store.list_conversations():
            self.save_conversation(conversation)
            for message in store.list_messages(conversation.conversation_id):
                self.save_message(message)
            self.save_memory(store.get_memory(conversation.conversation_id))
            for session in store.list_sessions(conversation.conversation_id):
                self.save_session(session)
                plan = session.single_plan or session.team_plan
                if plan:
                    self.save_plan(session, plan)
                raw = store.read_session_json(
                    conversation.conversation_id,
                    session.session_id,
                    "execution.json",
                )
                if raw:
                    try:
                        self.save_execution(
                            session, ExecutionResult.model_validate(raw)
                        )
                    except Exception:
                        logger.exception(
                            "Could not rebuild execution %s", session.session_id
                        )
                for contact in store.list_contacts(
                    conversation.conversation_id, session.session_id
                ):
                    self.save_contact(contact)

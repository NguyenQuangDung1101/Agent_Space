# AGENT SPACE

From the project folder:

```bash
cp .env.example .env
docker compose up -d neo4j
```

Make sure Ollama is running and pull the models:

```bash
ollama signin
ollama pull qwen3-embedding:0.6b
ollama pull gemma4:31b-cloud
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

Open:

* Agent Space GUI: `http://127.0.0.1:8000`
* API docs: `http://127.0.0.1:8000/docs`
* Health check: `http://127.0.0.1:8000/health`
* Neo4j Browser: `http://localhost:7474`

Neo4j login:

```text
Username: neo4j
Password: agent-space-password
```

In Neo4j Browser, inspect all nodes with:

```cypher
MATCH (n)-[r]-(m)
RETURN n, r, m
```


## 1. Core entities

```text
Conversation
    ├── Contains user and assistant Messages
    ├── Contains multiple execution Sessions
    ├── Stores shared ConversationMemory
    └── Must exist before a user task is sent

Session
    ├── Represents one user task execution
    ├── Belongs to exactly one Conversation
    ├── Contains analysis, plans, executions, events, contacts, and result
    └── May pause and resume when waiting for user input

Message
    ├── Belongs to a Conversation
    ├── May start a new Session
    ├── May answer a pending UserContactRequest
    └── May contain attachments

ConversationMemory
    ├── Stores a compact summary of previous completed Sessions
    ├── Contains useful requests, decisions, plans, and results
    └── Is loaded as shared context for new Sessions

UserContactRequest
    ├── Belongs to a Session
    ├── Is raised by any agent
    ├── Contains a free-text question, reason, and expected response
    └── Pauses the Session until the user replies
```

A normal user message creates a new Session.

A message answering a pending UserContactRequest resumes the existing Session instead of creating another Session.

---

## 2. Project structure

```text
AGENT_SPACE/
│
├── app.py
│   # FastAPI entry point for conversations, messages, sessions,
│   # attachments, events, health check, and GUI.
│
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── gui/
│   └── index.html
│       # Conversation-based chat GUI.
│       # Starts with no selected conversation.
│       # Displays history, session state, events, and user-contact messages.
│
├── agent/
│   ├── agent.json
│   │   # Registry for control agents and selectable worker agents.
│   │
│   ├── analyze_agent/
│   │   # Determines single-agent or team execution.
│   │   ├── service.py
│   │   ├── server.py
│   │   ├── system_prompt.txt
│   │   └── builtin_tool.py
│   │
│   ├── planner_agent/
│   │   # Creates team execution plans.
│   │   ├── service.py
│   │   ├── server.py
│   │   ├── system_prompt.txt
│   │   └── builtin_tool.py
│   │
│   ├── agent_of_requirement/
│   │   # General selectable worker agent (system prompt defined and tools given by caller).
│   │   ├── service.py
│   │   ├── server.py
│   │   ├── base_prompt.txt
│   │   └── builtin_tool.py
│   │
│   ├── synthesizer_agent/
│   │   # Gathers and combines team results into the final response.
│   │   ├── service.py
│   │   ├── server.py
│   │   ├── system_prompt.txt
│   │   └── builtin_tool.py
│   │
│   ├── knowledge_graph_modification_agent/
│   │   # Selectable worker for inspecting and modifying external Knowledge data.
│   │   ├── service.py
│   │   ├── server.py
│   │   ├── system_prompt.txt
│   │   └── builtin_tool.py
│   │       # Private document, node, relationship, and validation tools.
│   │
│   └── neo4j_traversal_agent/
│       # Selectable worker for retrieving external Knowledge data.
│       ├── service.py
│       ├── server.py
│       ├── system_prompt.txt
│       └── builtin_tool.py
│           # Private anchor search, expansion, and path-finding tools.
│
├── tool/
│   ├── tool.json
│   │   # Registry of shared tools available to normal workers.
│   └── get_current_datetime/
│       ├── config.json
│       └── tool.py
│
├── share/
│   ├── local_llm.py
│   ├── embedder.py
│   │   # Ollama embedder using qwen3-embedding:0.6b.
│   ├── agent.py
│   │   # Generic agent runtime supporting tools, communication,
│   │   # final answers, user contact, checkpointing, and resume.
│   ├── schemas.py
│   │   # Conversation, Message, Memory, Session, Contact,
│   │   # analysis, plans, executions, and events.
│   ├── registry.py
│   ├── tool_loader.py
│   ├── agent_factory.py
│   ├── event_broker.py
│   └── conversation_store.py
│       # File-based conversation, session, attachment,
│       # event, artifact, and checkpoint persistence.
│
├── core/
│   ├── manager.py
│   │   # Controls conversations, Sessions, execution branches,
│   │   # pause/resume, memory updates, and persistence.
│   └── orchestrator.py
│       # Executes teams, routes internal messages,
│       # manages shared context, and pauses the whole team when required.
│
├── database/
│   ├── neo4j_repository.py
│   │   # Agent Space graph persistence and startup synchronization.
│   ├── knowledge_repository.py
│   │   # Access to the separate external Knowledge domain.
│   ├── schema.cypher
│   │   # Agent Space constraints and indexes.
│   └── knowledge_schema.cypher
│       # External Knowledge constraints and indexes.
│
└── data/
    ├── neo4j/
    │   ├── data/
    │   ├── logs/
    │   └── import/
    │
    └── conversations/
        └── {conversation_id}/
            ├── conversation.json
            ├── messages.json
            ├── memory.json
            ├── attachments/
            └── sessions/
                └── {session_id}/
                    ├── request.json
                    ├── analysis.json
                    ├── single_plan.json or team_plan.json
                    ├── assignments.json
                    ├── events.json
                    ├── outputs.json
                    ├── messages.json
                    ├── tool_results.json
                    ├── failures.json
                    ├── checkpoint.json
                    ├── final_response.json
                    └── session.json
```

Built-in tools are private. They are not registered in `tool/tool.json` and are not assigned separately by Analyzer or Planner.

Worker agents are free to choose their own tool flow. The system must not force a specific tool order unless required for data integrity.

---

## 3. Conversation and Session flow

```text
Create or select Conversation
    ↓
User sends Message
    ↓
Manager saves Message
    ↓
Check for pending UserContactRequest
    ├── Pending contact exists
    │       ↓
    │   Link Message as the response
    │       ↓
    │   Resume the existing Session
    │
    └── No pending contact
            ↓
        Create new Session
            ↓
        Load ConversationMemory
            ↓
        Analyze Agent
            ├── Single-agent execution
            └── Team execution
                    ↓
                Planner Agent
                    ↓
                Orchestrator
            ↓
        Execute selected worker or team
            ↓
        Synthesize when required
            ↓
        Save final assistant Message
            ↓
        Mark Session COMPLETED
            ↓
        Update ConversationMemory
```

Session artifacts contain detailed execution data.

ConversationMemory contains only compact, useful context required by later Sessions.

A Session is completed only after its final response is saved and added to the Conversation.

---

## 4. User-contact flow

Every agent may contact the user during execution.

User contact is separate from direct or broadcast agent communication because it pauses execution.

An agent may emit:

```text
<user_contact>
{
  "question": "Please review the proposed graph changes.",
  "reason": "User feedback is needed before continuing.",
  "expected_response": "Approval, rejection, or requested changes"
}
</user_contact>
```

Flow:

```text
Agent raises UserContactRequest
    ↓
Manager receives the request
    ↓
Session becomes WAITING_FOR_USER
    ↓
Save agent and Session checkpoint
    ↓
Emit user_contact_requested
    ↓
GUI displays:
    Agent name
    Contact message
    ↓
User sends a free-text response
    ↓
Response Message is linked to the contact
    ↓
Emit user_contact_answered
    ↓
Session becomes RUNNING
    ↓
Emit session_resumed
    ↓
Restore and resume the same logical agent instance
```

For team execution, the entire Session pauses while waiting for the user.

User contact may be used for clarification, review, approval, rejection, or general feedback.

A modification plan is only the free-text proposal sent through user contact. It is not required to be a separate stored plan schema or special tool result.

---

## 5. Graph data model

Two graph domains exist in the same Neo4j database.

### Agent Space domain

```text
(System)-[:REGISTERS]->(Agent)
(System)-[:REGISTERS]->(Tool)

(Conversation)-[:HAS_MESSAGE]->(Message)
(Conversation)-[:HAS_SESSION]->(Session)
(Conversation)-[:HAS_MEMORY]->(MemorySnapshot)

(Session)-[:TRIGGERED_BY]->(Message)
(Session)-[:HAS_TASK]->(Task)
(Session)-[:HAS_ANALYSIS]->(Analysis)

(Task)-[:SOLVED_BY]->(Solution)
(Solution)-[:USES_AGENT]->(Agent)
(Solution)-[:USES_TOOL]->(Tool)

(Session)-[:HAS_EXECUTION]->(Execution)
(Execution)-[:HAS_AGENT_RUN]->(AgentRun)
(AgentRun)-[:INSTANCE_OF]->(Agent)
(AgentRun)-[:HAS_TOOL_RUN]->(ToolRun)

(Session)-[:HAS_CONTACT]->(UserContact)
(UserContact)-[:REQUESTED_BY]->(AgentRun)
(UserContact)-[:ANSWERED_BY]->(Message)
```

### External Knowledge domain

```text
(Knowledge {id: "external_knowledge"})
    -[:HAS_DOCUMENT]->(KnowledgeDocument)

(KnowledgeDocument)-[:HAS_CHUNK]->(KnowledgeChunk)
(Knowledge)-[:HAS_ENTITY]->(KnowledgeEntity)

(KnowledgeChunk)-[:MENTIONS]->(KnowledgeEntity)
(KnowledgeEntity)-[:RELATED_TO]->(KnowledgeEntity)
(KnowledgeEntity)-[:SUPPORTED_BY]->(KnowledgeChunk)
```

Every Knowledge-domain node contains:

```text
id
name
description
properties_json
embedding
```

`embedding` is a direct Neo4j node attribute.

It must not be:

```text
Stored inside properties_json
Stored in conversation or Session JSON files
Included in public tool responses
Included when concatenating visible node attributes
```

When a Knowledge node is created or updated:

```text
Visible node attributes
    ↓
Remove embedding fields
    ↓
Concatenate id, name, description, and properties
    ↓
Generate embedding using Ollama
    ↓
Store vector as node.embedding
```

The Knowledge root node cannot be deleted.

The Agent Space repository must ignore Knowledge-domain nodes during normal control-graph queries.

Only explicit Knowledge agents may access the Knowledge domain.

Optional provenance relationships may connect the two domains:

```text
(AgentRun)-[:IMPORTED]->(KnowledgeDocument)
(AgentRun)-[:RETRIEVED_FROM]->(KnowledgeEntity)
```

---

## 6. Persistence behavior

File storage remains the primary audit log.

Neo4j stores synchronized graph representations of Agent Space execution data.

At application startup:

```text
Connect to Neo4j
    ↓
Apply constraints and indexes
    ↓
Delete only Agent Space domain data
    ↓
Synchronize registered Agents and shared Tools
    ↓
Rebuild Conversations, Messages, Sessions,
plans, executions, contacts, and memory
from file storage
    ↓
Keep the external Knowledge domain unchanged
```

Knowledge graph data and embeddings must persist through the Docker Neo4j volume under:

```text
data/neo4j/
```

---

## 7. Session statuses and events

```text
Session statuses:
    PENDING
    RUNNING
    WAITING_FOR_USER
    COMPLETED
    FAILED
```

Required events:

```text
conversation_created
message_received
session_started

analysis_started
analysis_completed

planning_started
planning_completed

agent_running
agent_completed

tool_running
tool_completed

user_contact_requested
session_paused
user_contact_answered
session_resumed

synthesis_started
synthesis_completed

session_completed
session_failed
```

The GUI display:

```text
No conversation selected
Running
Paused and waiting for user
Resumed
Completed
Failed
```

A user-contact message must display the requesting agent’s name above the message.

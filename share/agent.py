from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
from uuid import uuid4

from share.local_llm import Copilot
from share.schemas import (
    AgentMessage,
    AgentRequest,
    AgentResult,
    ExecutionEvent,
    ToolCall,
    ToolResult,
    UserContactRequest,
)


logger = logging.getLogger(__name__)
Handler = Callable[..., Any]
HistoryMode = Literal["trim", "summary"]

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
COMMUNICATION_RE = re.compile(
    r"<communication>\s*(.*?)\s*</communication>", re.DOTALL
)
USER_CONTACT_RE = re.compile(
    r"<user_contact>\s*(.*?)\s*</user_contact>", re.DOTALL
)
FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL
)


class UserContactRaised(Exception):
    def __init__(self, contact: Dict[str, Any]) -> None:
        super().__init__(contact["question"])
        self.contact = contact


def render_tools_contract(tool_spec: List[Dict[str, Any]]) -> str:
    if not tool_spec:
        return "- No tools are assigned."

    blocks = []
    for tool in tool_spec:
        schema = tool.get("inputSchema", {})
        blocks.append(
            f"- name: {tool.get('name', '')}\n"
            f"  description: {tool.get('description', '')}\n"
            f"  required: {schema.get('required', [])}\n"
            "  properties: "
            + json.dumps(schema.get("properties", {}), ensure_ascii=False)
        )
    return "\n".join(blocks)


def build_strong_system_prompt(
    user_system_prompt: str,
    tool_spec: List[Dict[str, Any]],
    enable_communication: bool = True,
    enable_user_contact: bool = True,
) -> str:
    tags = ["<tool_call>"]
    communication_doc = ""
    contact_doc = ""

    if enable_communication:
        tags.append("<communication>")
        communication_doc = """
For inter-agent communication, use:
<communication>{"recipient":"<recipient_id>","message_type":"direct","content":"<message>"}</communication>
message_type must be direct, broadcast, or manager.
"""

    if enable_user_contact:
        tags.append("<user_contact>")
        contact_doc = """
When required user input is missing and execution cannot safely continue, use:
<user_contact>{"question":"<question>","reason":"<why it is needed>","expected_response":"<free-text response expected>"}</user_contact>
Ask one clear free-text question and pause. Do not use user contact for optional details.
"""

    tags.append("<final_answer>")
    control = f"""
Only output these tags: {', '.join(tags)}.

Tool call format:
<tool_call>{{"name":"<tool_name>","arguments":{{...}}}}</tool_call>
{communication_doc}{contact_doc}
Final answer format:
<final_answer>...final answer...</final_answer>

Available tools:
{render_tools_contract(tool_spec)}

Rules:
- Do not output text outside tags.
- Multiple tool or communication tags are allowed in one response.
- <user_contact> and <final_answer> must each be standalone.
- Read tool results before continuing.
"""
    return f"{user_system_prompt}\n\n{control}".strip()


def _extract_json_tags(
    text: str,
    pattern: re.Pattern[str],
) -> List[Dict[str, Any]]:
    values = []
    for match in pattern.finditer(text):
        try:
            value = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    return [
        item
        for item in _extract_json_tags(text, TOOL_CALL_RE)
        if isinstance(item.get("name"), str)
    ]


def extract_communications(text: str) -> List[Dict[str, Any]]:
    return [
        item
        for item in _extract_json_tags(text, COMMUNICATION_RE)
        if isinstance(item.get("recipient"), str)
        and isinstance(item.get("content"), str)
    ]


def extract_user_contacts(text: str) -> List[Dict[str, Any]]:
    return [
        item
        for item in _extract_json_tags(text, USER_CONTACT_RE)
        if isinstance(item.get("question"), str)
        and item["question"].strip()
    ]


def extract_final_answer(text: str) -> Optional[str]:
    match = FINAL_ANSWER_RE.search(text)
    return match.group(1).strip() if match else None


class ToolCallingAgent:
    def __init__(
        self,
        llm: Copilot,
        system_prompt: str,
        tool_executor: Optional[Handler] = None,
        communication_handler: Optional[Handler] = None,
        event_handler: Optional[Handler] = None,
        agent_id: str = "agent",
        instance_id: Optional[str] = None,
        max_steps: int = 6,
        max_history: int = 10,
        history_mode: HistoryMode = "trim",
        enable_communication: bool = True,
        enable_user_contact: bool = True,
    ) -> None:
        if max_steps < 1 or max_history < 1:
            raise ValueError("max_steps and max_history must be at least 1.")
        if history_mode not in ("trim", "summary"):
            raise ValueError("history_mode must be 'trim' or 'summary'.")

        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_executor = tool_executor
        self.communication_handler = communication_handler
        self.event_handler = event_handler
        self.agent_id = agent_id
        self.instance_id = instance_id or agent_id
        self.max_steps = max_steps
        self.max_history = max_history
        self.history_mode = history_mode
        self.enable_communication = enable_communication
        self.enable_user_contact = enable_user_contact

        self.user_turn = 0
        self.turn_step = 0
        self.summary_count = 0
        self.conversation: List[str] = []
        self.outbox: List[Dict[str, Any]] = []

    def _compose_prompt(self) -> str:
        return "\n".join(self.conversation)

    def _append_user(self, text: str) -> None:
        self.user_turn += 1
        self.turn_step = 0
        self.conversation.append(f"\n[USER {self.user_turn}]\n{text}")

    @staticmethod
    def _is_history_block(item: str) -> bool:
        item = item.lstrip()
        return item.startswith("[INFERENCE ") or item.startswith("[SUMMARY ")

    def _history_count(self) -> int:
        return sum(self._is_history_block(item) for item in self.conversation)

    def _trim_history(self, target: int) -> None:
        while self._history_count() > target:
            start = next(
                (
                    index
                    for index, item in enumerate(self.conversation)
                    if self._is_history_block(item)
                ),
                None,
            )
            if start is None:
                return
            end = len(self.conversation)
            for index in range(start + 1, len(self.conversation)):
                item = self.conversation[index].lstrip()
                if item.startswith(("[INFERENCE ", "[SUMMARY ", "[USER ")):
                    end = index
                    break
            del self.conversation[start:end]

    async def _prepare_history(self) -> None:
        if self._history_count() + 1 <= self.max_history:
            return

        if self.history_mode == "summary":
            try:
                summary = await asyncio.to_thread(
                    self.llm.infer,
                    user_prompt=self._compose_prompt(),
                    system_prompt=(
                        "Summarize concisely. Preserve objectives, constraints, "
                        "decisions, tool results, unresolved questions, and all "
                        "information needed to continue. Return only the summary."
                    ),
                    think=False,
                )
                if summary and summary.strip():
                    self.summary_count += 1
                    self.conversation = [
                        f"[SUMMARY {self.summary_count}]\n{summary.strip()}"
                    ]
                    return
            except Exception:
                logger.exception("Conversation summarization failed.")

        self._trim_history(self.max_history - 1)

    def export_checkpoint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "instance_id": self.instance_id,
            "conversation": list(self.conversation),
            "outbox": list(self.outbox),
            "user_turn": self.user_turn,
            "turn_step": self.turn_step,
            "summary_count": self.summary_count,
        }

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.instance_id = str(
            checkpoint.get("instance_id") or self.instance_id
        )
        self.conversation = list(checkpoint.get("conversation", []))
        self.outbox = list(checkpoint.get("outbox", []))
        self.user_turn = int(checkpoint.get("user_turn", 0))
        self.turn_step = int(checkpoint.get("turn_step", 0))
        self.summary_count = int(checkpoint.get("summary_count", 0))

    @staticmethod
    async def _invoke(handler: Handler, *args: Any) -> Any:
        result = handler(*args)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _format_result(result: Any) -> str:
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    async def _emit_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_handler:
            await self._invoke(
                self.event_handler,
                ExecutionEvent(
                    event_type=event_type,
                    agent_id=self.agent_id,
                    instance_id=self.instance_id,
                    details=details or {},
                ),
            )

    async def _execute_tool(self, call: Dict[str, Any]) -> None:
        name = call["name"]
        arguments = call.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            self.conversation.append(
                f"[TOOL:{name}:ERROR]\nTool arguments must be a JSON object."
            )
            return
        if self.tool_executor is None:
            self.conversation.append(
                f"[TOOL:{name}:ERROR]\nNo tool executor is configured."
            )
            return

        await self._emit_event(
            "tool_running", {"tool_name": name, "arguments": arguments}
        )
        try:
            result = await self._invoke(self.tool_executor, name, arguments)
            self.conversation.append(
                f"[TOOL:{name}:RESULT]\n{self._format_result(result)}"
            )
            success = bool(getattr(result, "success", True))
            error = getattr(result, "error", None)
            if isinstance(result, dict) and "success" in result:
                success = bool(result.get("success"))
                error = result.get("error")
            await self._emit_event(
                "tool_completed" if success else "tool_failed",
                {"tool_name": name, "success": success, "error": error},
            )
        except Exception as error:
            logger.exception("Tool execution failed: %s", name)
            error_text = f"{type(error).__name__}: {error}"
            self.conversation.append(f"[TOOL:{name}:ERROR]\n{error_text}")
            await self._emit_event(
                "tool_failed", {"tool_name": name, "error": error_text}
            )

    async def _handle_communication(self, item: Dict[str, Any]) -> None:
        message = {
            "sender": self.instance_id,
            "recipient": item["recipient"],
            "message_type": item.get("message_type", "direct"),
            "content": item["content"],
        }
        if message["message_type"] not in ("direct", "broadcast", "manager"):
            self.conversation.append(
                "[COMMUNICATION:ERROR]\nInvalid message_type."
            )
            return

        self.outbox.append(message)
        self.conversation.append(
            "[COMMUNICATION:OUT]\n"
            + json.dumps(message, ensure_ascii=False)
        )
        if self.communication_handler:
            try:
                result = await self._invoke(self.communication_handler, message)
                self.conversation.append(
                    "[COMMUNICATION:RESULT]\n" + self._format_result(result)
                )
            except Exception as error:
                logger.exception("Communication routing failed.")
                self.conversation.append(
                    f"[COMMUNICATION:ERROR]\n{type(error).__name__}: {error}"
                )

    async def step_once(self) -> Optional[str]:
        self.turn_step += 1
        await self._prepare_history()
        self.conversation.append(f"\n[INFERENCE {self.turn_step}]\n")

        try:
            output = await asyncio.to_thread(
                self.llm.infer,
                user_prompt=self._compose_prompt(),
                system_prompt=self.system_prompt,
            )
        except Exception as error:
            logger.exception("LLM inference failed.")
            self.conversation.append(
                f"[AGENT]\nLLM error: {type(error).__name__}: {error}"
            )
            return None

        if not output:
            self.conversation.append("[AGENT]\nThe LLM returned no response.")
            return None

        self.conversation.append(f"[AGENT]\nLLM raw output:\n{output}")
        tool_calls = extract_tool_calls(output)
        communications = extract_communications(output)
        contacts = extract_user_contacts(output)
        final_answer = extract_final_answer(output)

        exclusive = int(final_answer is not None) + len(contacts)
        if exclusive > 1 or (exclusive and (tool_calls or communications)):
            self.conversation.append(
                "[AGENT]\n<final_answer> and <user_contact> must be standalone."
            )
            return None
        if len(contacts) > 1:
            self.conversation.append(
                "[AGENT]\nOnly one <user_contact> is allowed."
            )
            return None

        for call in tool_calls:
            await self._execute_tool(call)
        if communications:
            if self.enable_communication:
                for message in communications:
                    await self._handle_communication(message)
            else:
                self.conversation.append(
                    "[COMMUNICATION:ERROR]\nCommunication is disabled."
                )
        if tool_calls or communications:
            return None

        if contacts:
            if not self.enable_user_contact:
                self.conversation.append(
                    "[USER_CONTACT:ERROR]\nUser contact is disabled."
                )
                return None
            contact = {
                "question": contacts[0]["question"].strip(),
                "reason": str(contacts[0].get("reason", "")).strip(),
                "expected_response": str(
                    contacts[0].get(
                        "expected_response", "Free-text response"
                    )
                ).strip(),
            }
            self.conversation.append(
                "[USER_CONTACT:REQUESTED]\n"
                + json.dumps(contact, ensure_ascii=False)
            )
            raise UserContactRaised(contact)

        if final_answer is not None:
            self.conversation.append(
                f"[FINAL FOR USER {self.user_turn}]\n{final_answer}"
            )
            return final_answer

        valid = ["<tool_call>"]
        if self.enable_communication:
            valid.append("<communication>")
        if self.enable_user_contact:
            valid.append("<user_contact>")
        valid.append("<final_answer>")
        self.conversation.append(
            "[AGENT]\nNo valid tag found. Use: " + ", ".join(valid)
        )
        return None

    async def _continue(self) -> str:
        for _ in range(self.max_steps):
            final_answer = await self.step_once()
            if final_answer is not None:
                return final_answer
        return (
            "Reached the maximum reasoning steps without a "
            "<final_answer>. Please refine the request."
        )

    async def run(self, user_prompt: str) -> str:
        self._append_user(user_prompt)
        return await self._continue()

    async def resume(self, user_response: str) -> str:
        self._append_user(
            "Response to the pending user-contact request:\n" + user_response
        )
        return await self._continue()


async def run_resumable(
    agent: ToolCallingAgent,
    request: AgentRequest,
    initial_prompt: str,
) -> str:
    if request.checkpoint:
        if request.user_response is None:
            raise ValueError("user_response is required when resuming.")
        agent.restore_checkpoint(
            request.checkpoint.get("agent", request.checkpoint)
        )
        return await agent.resume(request.user_response)
    return await agent.run(initial_prompt)


def build_waiting_result(
    request: AgentRequest,
    agent: ToolCallingAgent,
    contact: Dict[str, Any],
    *,
    checkpoint: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[ToolCall]] = None,
    tool_results: Optional[List[ToolResult]] = None,
    messages: Optional[List[AgentMessage]] = None,
) -> AgentResult:
    payload = {"agent": agent.export_checkpoint()}
    payload.update(checkpoint or {})
    user_contact = UserContactRequest(
        contact_id=f"contact_{uuid4().hex[:12]}",
        conversation_id=request.conversation_id or "",
        session_id=request.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_id.replace("_", " ").title(),
        instance_id=agent.instance_id,
        question=contact["question"],
        reason=contact.get("reason", ""),
        expected_response=contact.get(
            "expected_response", "Free-text response"
        ),
    )
    return AgentResult(
        agent_id=agent.agent_id,
        instance_id=agent.instance_id,
        status="WAITING_FOR_USER",
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        messages=messages or [
            AgentMessage.model_validate(item) for item in agent.outbox
        ],
        user_contact=user_contact,
        checkpoint=payload,
    )


def build_agent(
    system_prompt_path: str,
    model: str,
    base_url: str,
    tool_spec: Optional[List[Dict[str, Any]]] = None,
    tool_executor: Optional[Handler] = None,
    communication_handler: Optional[Handler] = None,
    event_handler: Optional[Handler] = None,
    agent_id: str = "agent",
    instance_id: Optional[str] = None,
    max_steps: int = 24,
    max_history: int = 16,
    history_mode: HistoryMode = "trim",
    enable_communication: bool = True,
    enable_user_contact: bool = True,
) -> ToolCallingAgent:
    prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()
    return ToolCallingAgent(
        llm=Copilot(model=model, base_url=base_url),
        system_prompt=build_strong_system_prompt(
            prompt,
            tool_spec or [],
            enable_communication=enable_communication,
            enable_user_contact=enable_user_contact,
        ),
        tool_executor=tool_executor,
        communication_handler=communication_handler,
        event_handler=event_handler,
        agent_id=agent_id,
        instance_id=instance_id,
        max_steps=max_steps,
        max_history=max_history,
        history_mode=history_mode,
        enable_communication=enable_communication,
        enable_user_contact=enable_user_contact,
    )

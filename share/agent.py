import asyncio
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from share.local_llm import Copilot


logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

COMMUNICATION_RE = re.compile(
    r"<communication>\s*(.*?)\s*</communication>",
    re.DOTALL,
)

FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.DOTALL,
)

HistoryMode = Literal["trim", "summary"]
Handler = Callable[..., Any]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt and output parsing
# ──────────────────────────────────────────────────────────────────────────────

def render_tools_contract(
    tool_spec: List[Dict[str, Any]],
) -> str:
    if not tool_spec:
        return "- No tools are assigned."

    blocks = []

    for tool in tool_spec:
        schema = tool.get("inputSchema", {})

        blocks.append(
            f"- name: {tool.get('name', '')}\n"
            f"  description: {tool.get('description', '')}\n"
            f"  required: {schema.get('required', [])}\n"
            f"  properties: "
            f"{json.dumps(schema.get('properties', {}), ensure_ascii=False)}"
        )

    return "\n".join(blocks)


def build_strong_system_prompt(
    user_system_prompt: str,
    tool_spec: List[Dict[str, Any]],
    enable_communication: bool = True,
) -> str:
    communication_doc = ""
    allowed_tags = "<tool_call> or <final_answer>"

    if enable_communication:
        allowed_tags = (
            "<tool_call>, <communication>, or <final_answer>"
        )

        communication_doc = """
When you need to send information to another runtime agent or the manager,
use this EXACT format:

<communication>{"recipient":"<recipient_id>", "message_type":"direct", "content":"<message>"}</communication>

- message_type must be "direct", "broadcast", or "manager".
- Use communication only for inter-agent coordination.
- Do not use communication to answer the user.
"""

    control = f"""
Follow the instructions precisely. Only output {allowed_tags} tags.

You can call tools using this EXACT format:

<tool_call>{{"name":"<tool_name>", "arguments":{{...}}}}</tool_call>

- Only output a tool call when you actually want it to be executed.
{communication_doc}
When you are ready to answer the user, output:

<final_answer>...your final answer for the user...</final_answer>

Available tools (schema):
{render_tools_contract(tool_spec)}

Rules:
- Output only valid tags.
- Do not include commentary outside the tags.
- You may output multiple non-final action tags in one response.
- <final_answer> must be standalone and must not appear with other tags.
- If a tool returns data, read the result and continue reasoning.
- If required information is missing, use <final_answer> to ask the user
  for the missing information.
"""

    return f"{user_system_prompt}\n\n{control}".strip()


def _extract_json_tags(
    text: str,
    pattern: re.Pattern,
) -> List[Dict[str, Any]]:
    result = []

    for match in pattern.finditer(text):
        try:
            value = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            result.append(value)

    return result


def extract_tool_calls(
    text: str,
) -> List[Dict[str, Any]]:
    return [
        item
        for item in _extract_json_tags(text, TOOL_CALL_RE)
        if isinstance(item.get("name"), str)
    ]


def extract_communications(
    text: str,
) -> List[Dict[str, Any]]:
    return [
        item
        for item in _extract_json_tags(text, COMMUNICATION_RE)
        if isinstance(item.get("recipient"), str)
        and isinstance(item.get("content"), str)
    ]


def extract_final_answer(
    text: str,
) -> Optional[str]:
    match = FINAL_ANSWER_RE.search(text)

    if match is None:
        return None

    return match.group(1).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Generic tool-calling agent
# ──────────────────────────────────────────────────────────────────────────────

class ToolCallingAgent:
    def __init__(
        self,
        llm: Copilot,
        system_prompt: str,
        tool_executor: Optional[Handler] = None,
        communication_handler: Optional[Handler] = None,
        agent_id: str = "agent",
        instance_id: Optional[str] = None,
        max_steps: int = 6,
        max_history: int = 10,
        history_mode: HistoryMode = "trim",
        enable_communication: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError(
                "max_steps must be at least 1."
            )

        if max_history < 1:
            raise ValueError(
                "max_history must be at least 1."
            )

        if history_mode not in ("trim", "summary"):
            raise ValueError(
                "history_mode must be 'trim' or 'summary'."
            )

        self.llm = llm
        self.system_prompt = system_prompt

        self.tool_executor = tool_executor
        self.communication_handler = communication_handler

        self.agent_id = agent_id
        self.instance_id = instance_id or agent_id

        self.max_steps = max_steps
        self.max_history = max_history
        self.history_mode = history_mode
        self.enable_communication = enable_communication

        self.user_turn = 0
        self.turn_step = 0
        self.summary_count = 0

        self.conversation: List[str] = []

        # Messages waiting for the future Orchestrator.
        self.outbox: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Conversation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compose_prompt(self) -> str:
        return "\n".join(self.conversation)

    def _append_user(
        self,
        text: str,
    ) -> None:
        self.user_turn += 1
        self.turn_step = 0

        self.conversation.append(
            f"\n[USER {self.user_turn}]\n{text}"
        )

    @staticmethod
    def _is_history_block(
        item: str,
    ) -> bool:
        item = item.lstrip()

        return (
            item.startswith("[INFERENCE ")
            or item.startswith("[SUMMARY ")
        )

    def _count_history_blocks(self) -> int:
        return sum(
            self._is_history_block(item)
            for item in self.conversation
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Trim history mode
    # ──────────────────────────────────────────────────────────────────────────

    def _trim_history_to(
        self,
        target: int,
    ) -> None:
        while self._count_history_blocks() > target:
            start_index = next(
                (
                    index
                    for index, item in enumerate(self.conversation)
                    if self._is_history_block(item)
                ),
                None,
            )

            if start_index is None:
                return

            end_index = len(self.conversation)

            for index in range(
                start_index + 1,
                len(self.conversation),
            ):
                item = self.conversation[index].lstrip()

                if (
                    item.startswith("[INFERENCE ")
                    or item.startswith("[SUMMARY ")
                    or item.startswith("[USER ")
                ):
                    end_index = index
                    break

            del self.conversation[start_index:end_index]

    # ──────────────────────────────────────────────────────────────────────────
    # Summary history mode
    # ──────────────────────────────────────────────────────────────────────────

    async def _replace_history_with_summary(
        self,
    ) -> bool:
        current_conversation = self._compose_prompt().strip()

        if not current_conversation:
            return True

        summary_system_prompt = (
            "Summarize the conversation concisely. "
            "Preserve objectives, constraints, decisions, important tool "
            "results, unresolved questions, and information needed to "
            "continue. Do not invent information. Return only the summary."
        )

        try:
            summary = await asyncio.to_thread(
                self.llm.infer,
                user_prompt=current_conversation,
                system_prompt=summary_system_prompt,
                think=False,
            )

        except Exception:
            logger.exception(
                "Conversation summarization failed."
            )
            return False

        if not summary or not summary.strip():
            return False

        self.summary_count += 1

        # Replace the whole current conversation with one summary block.
        self.conversation = [
            f"[SUMMARY {self.summary_count}]\n"
            f"{summary.strip()}"
        ]

        return True

    async def _prepare_history(
        self,
    ) -> None:
        # Include the next inference block in the calculation.
        projected_count = (
            self._count_history_blocks() + 1
        )

        if projected_count <= self.max_history:
            return

        if self.history_mode == "summary":
            summarized = (
                await self._replace_history_with_summary()
            )

            if summarized:
                return

        # Trim is also used as fallback when summarization fails.
        self._trim_history_to(
            self.max_history - 1
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Result and handler helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_result(
        result: Any,
    ) -> str:
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        # Supports MCP-style results.
        if (
            isinstance(result, dict)
            and isinstance(result.get("content"), list)
        ):
            text_parts = [
                str(item.get("text", ""))
                for item in result["content"]
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                )
            ]

            text = "\n".join(text_parts).strip()

            if text:
                return text

        if isinstance(result, str):
            return result

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    async def _invoke(
        handler: Handler,
        *args: Any,
    ) -> Any:
        result = handler(*args)

        if inspect.isawaitable(result):
            return await result

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Tool execution
    # ──────────────────────────────────────────────────────────────────────────

    async def _execute_tool(
        self,
        call: Dict[str, Any],
    ) -> None:
        name = call["name"]
        arguments = call.get("arguments", {}) or {}

        if not isinstance(arguments, dict):
            self.conversation.append(
                f"[TOOL:{name}:ERROR]\n"
                "Tool arguments must be a JSON object."
            )
            return

        if self.tool_executor is None:
            self.conversation.append(
                f"[TOOL:{name}:ERROR]\n"
                "No tool executor is configured."
            )
            return

        try:
            result = await self._invoke(
                self.tool_executor,
                name,
                arguments,
            )

            self.conversation.append(
                f"[TOOL:{name}:RESULT]\n"
                f"{self._format_result(result)}"
            )

        except Exception as error:
            logger.exception(
                "Tool execution failed: %s",
                name,
            )

            self.conversation.append(
                f"[TOOL:{name}:ERROR]\n"
                f"{type(error).__name__}: {error}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Agent communication
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_communication(
        self,
        message: Dict[str, Any],
    ) -> None:
        message = {
            "sender": self.instance_id,
            "recipient": message["recipient"],
            "message_type": message.get(
                "message_type",
                "direct",
            ),
            "content": message["content"],
        }

        if message["message_type"] not in (
            "direct",
            "broadcast",
            "manager",
        ):
            self.conversation.append(
                "[COMMUNICATION:ERROR]\n"
                "Invalid message_type."
            )
            return

        self.outbox.append(message)

        self.conversation.append(
            "[COMMUNICATION:OUT]\n"
            + json.dumps(
                message,
                ensure_ascii=False,
            )
        )

        # The future Orchestrator may collect messages from outbox.
        if self.communication_handler is None:
            self.conversation.append(
                "[COMMUNICATION:STATUS]\n"
                "Queued for the orchestrator."
            )
            return

        try:
            result = await self._invoke(
                self.communication_handler,
                message,
            )

            self.conversation.append(
                "[COMMUNICATION:RESULT]\n"
                f"{self._format_result(result)}"
            )

        except Exception as error:
            logger.exception(
                "Communication routing failed."
            )

            self.conversation.append(
                "[COMMUNICATION:ERROR]\n"
                f"{type(error).__name__}: {error}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    async def step_once(
        self,
    ) -> Optional[str]:
        self.turn_step += 1

        await self._prepare_history()

        self.conversation.append(
            f"\n[INFERENCE {self.turn_step}]\n"
        )

        try:
            llm_output = await asyncio.to_thread(
                self.llm.infer,
                user_prompt=self._compose_prompt(),
                system_prompt=self.system_prompt,
            )

        except Exception as error:
            logger.exception(
                "LLM inference failed."
            )

            self.conversation.append(
                f"[AGENT]\nLLM error: {type(error).__name__}: {error}"
            )

            return None

        if not llm_output:
            self.conversation.append(
                "[AGENT]\nThe LLM did not return a response."
            )
            return None

        self.conversation.append(
            f"[AGENT]\nLLM raw output:\n{llm_output}"
        )

        tool_calls = extract_tool_calls(
            llm_output
        )

        communications = extract_communications(
            llm_output
        )

        final_answer = extract_final_answer(
            llm_output
        )

        # A final answer must be exclusive.
        if (
            final_answer is not None
            and (tool_calls or communications)
        ):
            self.conversation.append(
                "[AGENT]\n"
                "Invalid output. <final_answer> cannot appear "
                "with <tool_call> or <communication>. Try again."
            )

            return None

        for call in tool_calls:
            await self._execute_tool(call)

        if communications:
            if self.enable_communication:
                for message in communications:
                    await self._handle_communication(
                        message
                    )
            else:
                self.conversation.append(
                    "[COMMUNICATION:ERROR]\n"
                    "Communication is disabled for this agent."
                )

        # Continue inference after performing actions.
        if tool_calls or communications:
            return None

        if final_answer is not None:
            self.conversation.append(
                f"[FINAL FOR USER {self.user_turn}]\n"
                f"{final_answer}"
            )

            return final_answer

        if self.enable_communication:
            valid_tags = (
                "<tool_call>, <communication>, "
                "or <final_answer>"
            )
        else:
            valid_tags = (
                "<tool_call> or <final_answer>"
            )

        self.conversation.append(
            "[AGENT]\n"
            "Your previous output did not include a valid "
            f"{valid_tags} tag. Try again."
        )

        return None

    async def run(
        self,
        user_prompt: str,
    ) -> str:
        self._append_user(user_prompt)

        for _ in range(self.max_steps):
            final_answer = await self.step_once()

            if final_answer is not None:
                return final_answer

        return (
            "Reached the maximum reasoning steps without a "
            "<final_answer>. Please refine the request."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CLI testing
    # ──────────────────────────────────────────────────────────────────────────

    async def chat_cli(
        self,
        first_user_prompt: Optional[str] = None,
    ) -> None:
        pending_prompt = first_user_prompt

        print(
            "Interactive mode. "
            "Enter=continue, new text=new turn, exit()=stop.\n"
        )

        while True:
            try:
                if pending_prompt is not None:
                    user_text = pending_prompt
                    pending_prompt = None
                else:
                    user_text = input(
                        "You: "
                    ).strip()

            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return

            if not user_text:
                continue

            if user_text.lower() == "exit()":
                print("Bye.")
                return

            self._append_user(user_text)

            for _ in range(self.max_steps):
                final_answer = await self.step_once()

                if final_answer is not None:
                    print(
                        "\n=== Final Answer ===\n"
                    )
                    print(final_answer)
                    break

                try:
                    follow_up = input(
                        "(Enter=continue, "
                        "new text=new turn): "
                    ).strip()

                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")
                    return

                if not follow_up:
                    continue

                if follow_up.lower() == "exit()":
                    print("Bye.")
                    return

                pending_prompt = follow_up
                break

            else:
                print(
                    "Reached the maximum reasoning steps "
                    "without a <final_answer>."
                )


# ──────────────────────────────────────────────────────────────────────────────
# Agent builder
# ──────────────────────────────────────────────────────────────────────────────

def build_agent(
    system_prompt_path: str,
    model: str,
    base_url: str,
    tool_spec: Optional[List[Dict[str, Any]]] = None,
    tool_executor: Optional[Handler] = None,
    communication_handler: Optional[Handler] = None,
    agent_id: str = "agent",
    instance_id: Optional[str] = None,
    max_steps: int = 24,
    max_history: int = 16,
    history_mode: HistoryMode = "trim",
    enable_communication: bool = True,
) -> ToolCallingAgent:
    system_prompt_text = Path(
        system_prompt_path
    ).read_text(
        encoding="utf-8"
    ).strip()

    llm = Copilot(
        model=model,
        base_url=base_url,
    )

    system_prompt = build_strong_system_prompt(
        system_prompt_text,
        tool_spec or [],
        enable_communication=enable_communication,
    )

    return ToolCallingAgent(
        llm=llm,
        system_prompt=system_prompt,
        tool_executor=tool_executor,
        communication_handler=communication_handler,
        agent_id=agent_id,
        instance_id=instance_id,
        max_steps=max_steps,
        max_history=max_history,
        history_mode=history_mode,
        enable_communication=enable_communication,
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        first_prompt = "hello"

        agent = build_agent(
            system_prompt_path="./agent/analyze_agent/system_prompt.txt",
            model="gemma4:31b-cloud",
            base_url="http://localhost:11434",
        )

        await agent.chat_cli(
            first_user_prompt=first_prompt,
        )

    asyncio.run(main())
import os
import re
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
logger = logging.getLogger("OLLama")
logging.basicConfig(level=logging.INFO)
# Reuse your existing LLM + tools
from local_llm import Copilot, load_system_prompt

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)

def render_tools_contract(tool_spec: List[Dict[str, Any]]) -> str:
    lines = []
    for t in tool_spec:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("inputSchema", {})
        required = schema.get("required", [])
        props = schema.get("properties", {})
        lines.append(f"- name: {name}\n  description: {desc}\n  required: {required}\n  properties: {json.dumps(props, ensure_ascii=False)}")
    return "\n".join(lines)

def build_strong_system_prompt(user_system_prompt: str, tool_spec: List[Dict[str, Any]]) -> str:
    tools_doc = render_tools_contract(tool_spec)
    control = f"""

Follow the instructions precisely. Only output "
<tool_call>...</tool_call> or <final_answer>...</final_answer>."

You can call tools using this EXACT format (one per line):

<tool_call>{{"name":"<tool_name>", "arguments":{{...}}}}</tool_call>

- Only output a tool call when you actually want me to execute it.

- After one or multiple tool calls, when you are ready to answer the user, output:

<final_answer>...your final answer for the user...</final_answer>

Available tools (schema):
{tools_doc}

Rules:
- Output ONLY either <tool_call>...</tool_call>, <final_answer>...</final_answer> at each step.
- <final_answer> must be a standalone response, mutually exclusive with <tool_call> and should not be called at the same time (in an inference response) with <tool_call>.
- Do NOT include extra commentary outside those tags (except thinking string, which is not be returned to user).
- You can call multiple tools in one response by outputting multiple <tool_call>...</tool_call>.
- If a tool returns tabular data (CSV/text), read it and continue reasoning.
- If the prompt lacks necessary arguments or clarity, you can request the missing information explicitly by using <final_answer> to ask the user to clarify the missing details or provide the required arguments.
"""
    return f"{user_system_prompt}\n\n{control}".strip()


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except Exception:
            pass
    return calls

def extract_final_answer(text: str) -> Optional[str]:
    m = FINAL_ANSWER_RE.search(text)
    return m.group(1).strip() if m else None

# ──────────────────────────────────────────────────────────────────────────────
# Agent (multi-turn capable)
# ──────────────────────────────────────────────────────────────────────────────

class ToolCallingAgent:
    def __init__(self, llm: Copilot, system_prompt: str, max_steps: int = 6, max_history: int = 10):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_history = max_history + 1  # total [INFERENCE] blocks to retain across all turns
        self.user_turn = 0
        self.turn_step = 0  # resets per user turn
        self.conversation: List[str] = []
        self.conversation_sunmmary = None
        self.conversation_sunmmary_updated = False

    # ──────────────────────────────────────────────────────────────────────────────
    # Helper functions
    # ──────────────────────────────────────────────────────────────────────────────

    def _compose_prompt(self) -> str:
        return "\n".join(self.conversation)

    def _append_user(self, text: str):
        self.user_turn += 1
        self.turn_step = 0
        self.conversation.append(f"\n[USER {self.user_turn}]\n{text}")
        self._trim_history_multi()

    def _append_agent(self, text: str):
        self.conversation.append(f"[AGENT]\n{text}")
        self._trim_history_multi()

    def _append_final(self, text: str):
        self.conversation.append(f"[Final for user {self.user_turn}]\n{text}")
        self._trim_history_multi()

    def _append_tool_result(self, name: str, result: Dict[str, Any]):
        pieces = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                pieces.append(item.get("text", ""))
        payload = "\n".join(pieces).strip()
        safe = payload if payload else "<empty result>"
        # print(f"[TOOL:{name}:RESULT]\n{safe}\n")
        self.conversation.append(f"[TOOL:{name}:RESULT]\n{safe}")
        self._trim_history_multi()

    def _count_inference_blocks(self) -> int:
        count = 0
        for item in self.conversation:
            s = item.lstrip()
            if s.startswith("[INFERENCE "):
                count += 1
        return count

    def _trim_history_multi(self):
        while self._count_inference_blocks() > self.max_history:
            N = len(self.conversation)
            start_idx = None
            for idx, item in enumerate(self.conversation):
                s = item.lstrip()
                if s.startswith("[INFERENCE "):
                    start_idx = idx
                    break
            if start_idx is None:
                return
            end_idx = N
            for j in range(start_idx + 1, N):
                sj = self.conversation[j].lstrip()
                if sj.startswith("[INFERENCE ") or sj.startswith("[USER "):
                    end_idx = j
                    break
            del self.conversation[start_idx:end_idx]

    def update_conversation_summary(self):
        curr_conversation = self._compose_prompt()
        self.conversation_sunmmary = self.llm.infer(
            user_prompt=curr_conversation,
            system_prompt="[SYSTEM]\nSummarize the conversation concisely, focusing on key points and briefly note the decisions made. Do not use icons"
        )
        self.conversation_sunmmary_updated = True

    # ──────────────────────────────────────────────────────────────────────────────
    # Inference function
    # ──────────────────────────────────────────────────────────────────────────────

    async def step_once(self) -> Optional[str]:
        self.conversation_sunmmary_updated = False
        self.turn_step += 1
        self.conversation.append(f"\n[INFERENCE {self.turn_step}]:\n")
        self._trim_history_multi()

        prompt_now = self._compose_prompt()
        # print(prompt_now)
        llm_out = self.llm.infer(
            user_prompt=prompt_now,
            system_prompt=self.system_prompt
        )
        # print(f"[LLM OUTPUT]:\n{llm_out}\n")
        if not llm_out:
            self._append_agent("The LLM did not return a response.")
            return None

        self._append_agent(f"LLM output: {llm_out}")

        calls = extract_tool_calls(llm_out)
        if calls:
            for call in calls:
                name = call.get("name")
                args = call.get("arguments", {}) or {}


        final = extract_final_answer(llm_out)
        if final is not None:
            self._append_final(final)
            return final

        if not any([calls, final]):
            self._append_agent("Your previous output didn't include a valid <tool_call>...</tool_call> or <final_answer>...</final_answer>. This will be considered as thinking string. Please try again.")
            return None

        return None

    async def run(self, user_prompt: str) -> str:
        self._append_user(user_prompt)
        for _ in range(1, self.max_steps + 1):
            final = await self.step_once()
            self._trim_history_multi()
            if final is not None:
                return final
        return "Reached max reasoning steps without a <final_answer>. Please refine your request."

    async def chat_cli(self, first_user_prompt=None):
        check_first_prompt = True if first_user_prompt else False
        print("Interactive mode. After each step, press Enter to continue reasoning or type a new prompt to start a new turn.\n")
        while True:
            try:
                if first_user_prompt and check_first_prompt:
                    user_text = first_user_prompt
                    check_first_prompt = False
                else:
                    user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return
            if not user_text:
                continue
            if user_text.lower() == "exit()":
                print("Bye.")
                return
            self._append_user(user_text)
            i = 0
            while True:
                i += 1
                final = await self.step_once()
                self._trim_history_multi()


                if final is not None:
                    print("\n=== Final Answer ===\n")
                    print(final)
                    break
                try:
                    if i >= self.max_steps:
                        print("Reached max reasoning steps without a <final_answer>. Please refine your request.")
                        i = 0
                    follow = input("(Enter=continue to answer the current prompt, or type a new prompt): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")
                    return
                if follow == "":
                    continue
                elif follow == "exit()":
                    print("Bye.")
                    return
                else:
                    self._append_user(follow)
                    continue

# ──────────────────────────────────────────────────────────────────────────────
# Example CLI usage
# ──────────────────────────────────────────────────────────────────────────────

def build_agent(system_prompt_text: str, model: str, host: str) -> ToolCallingAgent:
    llm = Copilot(host=host, model=model)
    sp = build_strong_system_prompt(system_prompt_text, TOOL_SPEC)
    return ToolCallingAgent(llm=llm, system_prompt=sp, max_steps=20, max_history=13)


# if __name__ == "__main__":
    # role_sys_prompt = load_system_prompt('./system_prompt_doc/role.txt')
    # instruction_sys_prompt = load_system_prompt('./system_prompt_doc/instruction.txt')

    # parts = [role_sys_prompt, instruction_sys_prompt]
    # sys_prompt = "\n\n".join([p for p in parts if p])
    # agent = build_agent(sys_prompt, model="gpt-oss:20b-cloud")

    # # Interactive multi-turn terminal chat:
    # asyncio.run(agent.chat_cli())


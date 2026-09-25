"""ReAct-style agent loop around the fine-tuned model.

Each step the model replies with exactly one JSON object:
    {"action": "<tool_name>", "args": {...}}   -> run the tool, feed back the observation
    {"action": "final", "answer": "..."}       -> stop and return the answer

The loop runs at most MAX_STEPS tool steps, then forces a final answer.
"""

from __future__ import annotations

import json
import re

from .model import generate
from .tools import TOOLS, call_tool

MAX_STEPS = 6

AGENT_SYSTEM = """You are Don't Agent, an expert AI programming agent. You solve tasks step by step and you can use tools.

Available tools:
{tools}

On each step, reply with EXACTLY one JSON object and no other text:
- To use a tool: {{"action": "run_python", "args": {{"code": "print(2 + 2)"}}}}
- When you are done: {{"action": "final", "answer": "your final answer here"}}

Rules:
- Think through the problem, then use tools to verify code by running it.
- Never invent tool results; always wait for the Observation.
- Keep the final answer concise and in the user's language.
"""


def _tool_docs() -> str:
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in TOOLS.items())


def _parse_action(text: str) -> dict | None:
    """Extract the JSON action object from model output (fences tolerated)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else text.strip()
    if not m:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None
    return obj


def run_agent(message: str, history: list[dict] | None = None) -> dict:
    """Run the ReAct loop. Returns {"reply": str, "steps": int}."""
    messages = [{"role": "system", "content": AGENT_SYSTEM.format(tools=_tool_docs())}]
    for h in history or []:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:4000]})
    messages.append({"role": "user", "content": message})

    steps = 0
    for _ in range(MAX_STEPS):
        steps += 1
        raw = generate(messages)
        action = _parse_action(raw)
        if action is None:
            # Model didn't follow the JSON format: treat its text as the answer.
            messages.append({"role": "assistant", "content": raw})
            return {"reply": raw.strip(), "steps": steps}
        name = action["action"]
        if name == "final":
            answer = str(action.get("answer", "")).strip() or raw.strip()
            messages.append({"role": "assistant", "content": answer})
            return {"reply": answer, "steps": steps}
        args = action.get("args")
        if not isinstance(args, dict):
            args = {}
        observation = call_tool(name, args)
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {"role": "user", "content": f"Observation from tool '{name}':\n{observation}"}
        )

    # Out of steps: force a final plain-text answer.
    messages.append(
        {
            "role": "user",
            "content": "You are out of tool steps. Give your final answer now as plain text.",
        }
    )
    final = generate(messages)
    action = _parse_action(final)
    if action and action.get("action") == "final":
        reply = str(action.get("answer", "")).strip() or final.strip()
    else:
        reply = final.strip()
    return {"reply": reply, "steps": steps}

"""Agent tools, sandboxed to WORKSPACE_DIR.

Tools:
    run_python(code)        execute Python in a subprocess (15s timeout)
    read_file(path)         read a text file
    write_file(path, ...)   write text to a file (creates parent dirs)
    list_dir(path=".")      list a directory

Every path is resolved inside WORKSPACE_DIR; escapes raise ToolError.
Only the standard library is used here so the tools stay dependency-free.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/tmp/agent_workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

MAX_OUTPUT_CHARS = 6000


class ToolError(Exception):
    """Raised when a tool call is invalid or unsafe."""


def _safe_path(rel: str) -> Path:
    p = (WORKSPACE_DIR / rel).resolve()
    if p != WORKSPACE_DIR and WORKSPACE_DIR not in p.parents:
        raise ToolError(f"path escapes workspace: {rel!r}")
    return p


def run_python(code: str) -> str:
    """Execute Python code in a subprocess with a 15s timeout.

    Returns captured stdout + stderr (truncated).
    """
    script = WORKSPACE_DIR / "_tool_run.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(WORKSPACE_DIR),
        )
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 15 seconds"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no output)"
    return out[:MAX_OUTPUT_CHARS]


def read_file(path: str, max_chars: int = 8000) -> str:
    """Read a text file inside the workspace."""
    p = _safe_path(path)
    if not p.is_file():
        raise ToolError(f"file not found: {path!r}")
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def write_file(path: str, content: str) -> str:
    """Write text to a file inside the workspace (creates parent dirs)."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    """List a directory inside the workspace."""
    p = _safe_path(path)
    if not p.is_dir():
        raise ToolError(f"not a directory: {path!r}")
    names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(names) if names else "(empty)"


TOOLS = {
    "run_python": {
        "func": run_python,
        "description": (
            "Run Python code and return stdout/stderr (15s timeout). "
            'args: {"code": string}'
        ),
    },
    "read_file": {
        "func": read_file,
        "description": 'Read a text file. args: {"path": string}',
    },
    "write_file": {
        "func": write_file,
        "description": (
            "Write text to a file, creating parent dirs. "
            'args: {"path": string, "content": string}'
        ),
    },
    "list_dir": {
        "func": list_dir,
        "description": 'List a directory. args: {"path": string (optional)}',
    },
}


def call_tool(name: str, args: dict) -> str:
    """Dispatch a tool call by name; never raises, returns an error string."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool {name!r}. Available: {', '.join(TOOLS)}"
    try:
        return str(tool["func"](**args))
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except ToolError as e:
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001 - tools must not crash the loop
        return f"ERROR: {type(e).__name__}: {e}"

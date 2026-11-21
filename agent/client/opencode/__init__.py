"""Detection of OpenCode IDE agent sessions and extraction of user context."""
from typing import Callable
from pathlib import Path

from agent.client import is_agent, check_agent
from agent.client.user_context import UserContext
from agent.multimodal.content_parts import content_text
from agent.openai.chat_completions_api import ChatCompletionMessageParam

OPENCODE_MARKER = "You are an AI agent running in OpenCode"


def is_opencode_agent(messages: list[ChatCompletionMessageParam]) -> bool:
    """Check that the conversation is an OpenCode IDE agent session."""
    return is_agent(messages, OPENCODE_MARKER)


def get_opencode_context(messages: list[ChatCompletionMessageParam]) -> UserContext:
    """Extract user context from the first message (mirrors Veai pattern).

    Returns a UserContext with whatever can be parsed from the system prompt.
    Falls back to empty/default values when no project_information block is found.
    """
    first_message = messages[0] if messages else None
    if not first_message:
        return UserContext()

    content = content_text(first_message.content)
    return _get_context(content)


def _get_context(system_prompt: str | list | None) -> UserContext:
    """Parse OS info and workdir from a <project_information> block."""
    context = UserContext()
    if isinstance(system_prompt, str):
        has_start = system_prompt.find("<project_information>")
        has_end = system_prompt.find("</project_information>")
        if 0 <= has_start < has_end < len(system_prompt):
            project_info = system_prompt[has_start:has_end].splitlines()
            for line in project_info:
                stripped = line.strip()
                if stripped.startswith("OS info:") or stripped.startswith("OS:"):
                    os_full = stripped.split(":", 1)[1].strip()
                    context.os_full = os_full
                    context.os_type = _detect_os(os_full)
                elif "Project absolute path:" in stripped or "Project path:" in stripped:
                    path_marker = "Project absolute path:" if "Project absolute path:" in stripped else "Project path:"
                    context.workdir = Path((path_marker + stripped[len(path_marker):]).strip())
    return context


def _detect_os(os_full: str):
    """Heuristic: guess OS from the OS info string."""
    from agent.client.user_context import OS
    lower = os_full.lower()
    if "windows" in lower:
        return OS.Windows
    if "linux" in lower:
        return OS.Linux
    if "mac" in lower or "darwin" in lower:
        return OS.MacOS
    return None
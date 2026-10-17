"""Client-profile detection: GitHub Copilot agent marker."""
from agent.client import is_agent
from agent.openai.chat_completions_api import ChatCompletionMessageParam


def is_github_copilot(messages: list[ChatCompletionMessageParam]) -> bool:
    """Check that the conversation is a GitHub Copilot agent session."""
    return is_agent(messages, "When asked for your name, you must respond with \"GitHub Copilot\"")

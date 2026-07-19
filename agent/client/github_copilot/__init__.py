from agent.client import is_agent
from agent.openai.chat_completions_api import ChatCompletionMessageParam


def is_github_copilot(messages: list[ChatCompletionMessageParam]) -> bool:
    return is_agent(messages, "When asked for your name, you must respond with \"GitHub Copilot\"")

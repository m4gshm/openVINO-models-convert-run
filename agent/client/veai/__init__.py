from agent.openai.chat_completions_api import ChatCompletionMessageParam


def is_veai_agent(messages: list[ChatCompletionMessageParam]) -> bool:
    first_message = messages[0] if messages else None
    if first_message:
        content = first_message.content
        if isinstance(content, str):
            str_content = content
            return "You are Veai Agent" in str_content
    return False

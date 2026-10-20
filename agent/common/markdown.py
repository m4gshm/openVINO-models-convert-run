"""Markdown formatting helpers shared by the inference, client and REST layers.

These helpers used to live in :mod:`agent.inference.token_handler`, but they are
also needed by the client tools and by the REST layer. Keeping them in a leaf
module of :mod:`agent.common` removes the inverted dependency of the lower
layers on the token handler.
"""


def markdown_bold(value) -> str:
    """Wrap the value into markdown bold."""
    return f"**{value}**"


def markdown_back_tick(value) -> str:
    """Wrap the value into markdown backticks."""
    return f"`{value}`"


def markdown_file_content(value, mime: str = "") -> str:
    """Wrap the value into a markdown fenced code block."""
    return f"```{mime}\n{value}\n```"


def markdown_json(value: str) -> str:
    """Wrap the value into a markdown json code block."""
    return markdown_file_content(value, "json")


def markdown_tool_call_loop_error(loop_cause: str | None, loop_error: str) -> str:
    """Format a tool call loop error as a markdown warning block."""
    return ((markdown_bold("WARNING: " + loop_cause) + "\n") if loop_cause else "") + markdown_file_content(
        loop_error)

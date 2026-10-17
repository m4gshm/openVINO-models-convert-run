"""User context extracted from the conversation: OS, workdir and tool-produced files."""
from enum import Enum
from pathlib import Path

from agent.openai.chat_completions_api import ChatCompletionMessageParam

DEFAULT_DEPTH = 1
MAX_DEPTH = 4

ROOT = "."


class UserContextFiles:
    """Files read or written through tools during the conversation."""

    def __init__(self, file_content: dict[Path, bytes] | None = None):
        """Store file contents keyed by path."""
        self.file_content: dict[Path, bytes] = file_content if file_content else {}

    def get_files(self) -> list[Path]:
        """List all tracked file paths."""
        return list(self.file_content.keys())

    def get_file_content(self, file_name: str) -> bytes | None:
        """Return the tracked content of the file, if any."""
        path = Path(file_name)
        return self.file_content.get(path)


class OS(Enum):
    """Operating system families seen in the environment block."""

    Windows = 'Windows'  # pylint: disable=invalid-name  # mirrors OS naming used by out-of-zone code
    MacOS = 'MacOS'  # pylint: disable=invalid-name  # mirrors OS naming used by out-of-zone code
    Linux = 'Linux'  # pylint: disable=invalid-name  # mirrors OS naming used by out-of-zone code
    TOOL_RESPONSE = 4
    FIM_MIDDLE = 5


class UserContext:  # pylint: disable=too-few-public-methods  # mutable data holder
    """Conversation-scoped context: OS, project dir, architectures and files."""

    def __init__(self, model_architectures: set[str] | None = None):
        """Initialize empty context with optional model architectures."""
        super().__init__()
        self.os_full: str | None = None
        self.os_type: OS | None = None
        self.workdir: Path | None = None
        self.model_architectures: set[str] = model_architectures if model_architectures else {}
        self.messages: list[ChatCompletionMessageParam] = []
        self.files: UserContextFiles = UserContextFiles()

    def max_list_dir_depth(self, directory: str) -> int | None:
        """Depth allowed for list_dir: shallow for the project root, else unlimited."""
        if directory == ROOT or directory == str(self.workdir):
            return DEFAULT_DEPTH
        return None

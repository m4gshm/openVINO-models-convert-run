from pathlib import Path

from agent.openai.chat_completions_api import ChatCompletionMessageParam


class UserContextFiles:
    def __init__(self, file_content: dict[Path, bytes] | None = None):
        self.file_content: dict[Path, bytes] = file_content if file_content else {}

    def get_files(self) -> list[Path]:
        return list(self.file_content.keys())

    def get_file_content(self, file_name: str) -> bytes | None:
        path = Path(file_name)
        return self.file_content.get(path)


class UserContext:

    def __init__(self, model_architectures: set[str] | None = None):
        super().__init__()
        self.os: str | None = None
        self.workdir: Path | None = None
        self.model_architectures: set[str] = model_architectures if model_architectures else {}
        self.messages: list[ChatCompletionMessageParam] = []
        self.files: UserContextFiles = UserContextFiles()

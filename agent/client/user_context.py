from agent.openai.chat_completions_api import ChatCompletionMessageParam


class UserContextFiles:
    def __init__(self, file_content: dict[str, bytes] | None = None):
        self.file_content: dict[str, bytes] = file_content if file_content else {}

    def get_files(self) -> list[str]:
        return list(self.file_content.keys())

    def get_file_content(self, file_name: str) -> bytes | None:
        return self.file_content.get(file_name)


class UserContext:

    def __init__(self, model_architectures: set[str] | None = None):
        super().__init__()
        self.os: str | None = None
        self.workdir: str | None = None
        self.model_architectures: set[str] = model_architectures if model_architectures else {}
        self.messages: list[ChatCompletionMessageParam] = []
        self.files: UserContextFiles = UserContextFiles()

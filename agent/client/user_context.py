from agent.openai.chat_completions_api import ChatCompletionMessageParam


class UserContextFiles:
    def get_files(self) -> list[str]:
        return []


class UserContext():

    def __init__(self, model_architectures: set[str] | None = None):
        super().__init__()
        self.os: str | None = None
        self.workdir: str | None = None
        self.model_architectures: set[str] = model_architectures if model_architectures else {}
        self.messages: list[ChatCompletionMessageParam] = []
        self.files: UserContextFiles = UserContextFiles()

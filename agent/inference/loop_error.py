class LoopError(Exception):
    def __init__(self, message, payload: str | list[str]):
        super().__init__(message)
        self.payload = payload

    def __str__(self):
        return f"LoopError: {self.args[0]}={self.payload}"

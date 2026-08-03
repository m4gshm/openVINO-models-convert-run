ERROR_APPEARS_TO_BE_A_LOOP = "Generated content appears to be a loop"


class LoopError(Exception):
    def __init__(self, payload: str, message=ERROR_APPEARS_TO_BE_A_LOOP):
        super().__init__(message)
        self.message = message
        self.payload = payload

    def __str__(self):
        return f"LoopError: {self.message}={self.payload}"

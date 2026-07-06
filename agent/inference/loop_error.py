class LoopError(Exception):
    def __init__(self, message, payload: str):
        super().__init__(message)
        self.message = message
        self.payload = payload

    def __str__(self):
        return f"LoopError: {self.message}={self.payload}"

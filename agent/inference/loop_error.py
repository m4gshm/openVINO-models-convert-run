class LoopError(Exception):
    def __init__(self, payload: str, message="Generated content appears to be a loop"):
        super().__init__(message)
        self.message = message
        self.payload = payload

    def __str__(self):
        return f"LoopError: {self.message}={self.payload}"

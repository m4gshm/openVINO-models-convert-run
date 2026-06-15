from pydantic import BaseModel


class GenerateConfig(BaseModel):
    reasoning_supported: bool = True
    default_max_new_tokens: int = 4096
    default_max_tokens: int = 65536
    default_temperature: float = 0.8
    default_top_p: float = 0.95
    default_top_k: int = 40
    default_min_p: float = 0.05
    default_repetition_penalty: float = 1.1

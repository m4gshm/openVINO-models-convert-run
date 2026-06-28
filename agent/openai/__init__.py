from pydantic import BaseModel


class GenerateConfig(BaseModel):
    max_new_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    preprocess_prompt_by_parser: bool = True


def default_generate_config():
    return GenerateConfig(
        max_new_tokens=4096
    )

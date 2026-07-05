from openai import BaseModel


class UserContext(BaseModel):
    os: str | None = None
    workdir: str | None = None

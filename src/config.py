from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    database_url: str = "sqlite:///screening.db"

    # Anthropic routing: cheap model for understanding and simple replies,
    # stronger model for emotionally loaded turns and the closing summary.
    model_understand: str = "claude-haiku-4-5"
    model_reply: str = "claude-haiku-4-5"
    model_reply_escalated: str = "claude-sonnet-4-6"

    # Groq fallback (used only if Anthropic fails).
    groq_model_understand: str = "openai/gpt-oss-120b"
    groq_model_reply: str = "openai/gpt-oss-20b"

    history_turns_in_context: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()

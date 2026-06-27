from pydantic import BaseModel, Field

from ..fsm.enums import Availability, Intent, Language, Schedule, Sentiment


class TurnUnderstanding(BaseModel):
    """Structured reading of a single candidate message.

    The model fills only the fields actually present in the latest message;
    everything else stays null so the orchestrator can keep what it already knew.
    """

    language: Language
    intent: Intent
    sentiment: Sentiment
    confirmation: bool | None = None

    consent: bool | None = None
    full_name: str | None = None
    has_license: bool | None = None
    city: str | None = None
    availability: Availability | None = None
    preferred_schedule: Schedule | None = None
    experience_years: float | None = None
    experience_platforms: list[str] = Field(default_factory=list)
    start_date_text: str | None = None
    vehicle_type: str | None = None

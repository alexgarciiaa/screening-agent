from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .enums import (
    Availability,
    Intent,
    Language,
    Modality,
    Outcome,
    Schedule,
    Sentiment,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Experience(BaseModel):
    years: float | None = None
    platforms: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    consent: bool | None = None
    full_name: str | None = None
    has_license: bool | None = None
    city: str | None = None
    matched_location: str | None = None
    city_in_service_area: bool | None = None
    availability: Availability | None = None
    preferred_schedule: Schedule | None = None
    experience: Experience = Field(default_factory=Experience)
    start_date_text: str | None = None
    vehicle_type: str | None = None


class Message(BaseModel):
    role: str
    text: str
    modality: Modality = Modality.TEXT
    transcription_confidence: float | None = None
    timestamp: datetime = Field(default_factory=_now)


class ConversationState(BaseModel):
    conversation_id: str
    candidate_id: str
    language: Language = Language.ES
    outcome: Outcome = Outcome.IN_PROGRESS
    profile: CandidateProfile = Field(default_factory=CandidateProfile)
    messages: list[Message] = Field(default_factory=list)
    last_intent: Intent | None = None
    last_sentiment: Sentiment = Sentiment.NEUTRAL
    awaiting_confirmation: bool = False
    last_confirmation: bool | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def add_message(
        self,
        role: str,
        text: str,
        modality: Modality = Modality.TEXT,
        transcription_confidence: float | None = None,
    ) -> None:
        self.messages.append(
            Message(
                role=role,
                text=text,
                modality=modality,
                transcription_confidence=transcription_confidence,
            )
        )
        self.updated_at = _now()

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..fsm.enums import Availability, Outcome, Schedule
from ..fsm.models import CandidateProfile, ConversationState

_RECOMMENDED_ACTION = {
    Outcome.QUALIFIED: "Contactar al candidato para agendar la entrevista.",
    Outcome.OUT_OF_AREA: "Guardar en lista de espera para cuando haya cobertura en su zona.",
    Outcome.DISQUALIFIED_NO_LICENSE: "Archivar: no cumple el requisito de carnet de conducir.",
    Outcome.CONSENT_DECLINED: "Archivar: el candidato no dio su consentimiento.",
    Outcome.OPTED_OUT: "Archivar: el candidato pidio detener el proceso.",
    Outcome.ABANDONED: "Reintentar el contacto mas adelante.",
}

_AVAILABILITY_LABEL = {
    Availability.FULL_TIME: "jornada completa",
    Availability.PART_TIME: "media jornada",
    Availability.WEEKENDS: "fines de semana",
}

_SCHEDULE_LABEL = {
    Schedule.MORNING: "mañana",
    Schedule.AFTERNOON: "tarde",
    Schedule.EVENING: "noche",
    Schedule.FLEXIBLE: "flexible",
}


class RecruiterHandoff(BaseModel):
    """Structured result passed to a human recruiter once screening ends."""

    conversation_id: str
    candidate_id: str
    outcome: Outcome
    qualified: bool
    summary: str
    recommended_action: str
    profile: CandidateProfile
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def build_handoff(state: ConversationState) -> RecruiterHandoff:
    return RecruiterHandoff(
        conversation_id=state.conversation_id,
        candidate_id=state.candidate_id,
        outcome=state.outcome,
        qualified=state.outcome is Outcome.QUALIFIED,
        summary=_summary(state.profile),
        recommended_action=_RECOMMENDED_ACTION.get(state.outcome, "Revisar manualmente."),
        profile=state.profile,
    )


def _summary(profile: CandidateProfile) -> str:
    parts: list[str] = []
    if profile.full_name:
        parts.append(profile.full_name)
    if profile.city:
        parts.append(f"zona: {profile.matched_location or profile.city}")
    if profile.has_license is not None:
        parts.append(f"carnet: {'si' if profile.has_license else 'no'}")
    if profile.availability:
        parts.append(f"disponibilidad: {_AVAILABILITY_LABEL[profile.availability]}")
    if profile.preferred_schedule:
        parts.append(f"horario: {_SCHEDULE_LABEL[profile.preferred_schedule]}")
    if profile.experience.years is not None:
        platforms = ", ".join(profile.experience.platforms) or "sin plataformas"
        parts.append(f"experiencia: {profile.experience.years:g} anos ({platforms})")
    if profile.start_date_text:
        parts.append(f"inicio: {profile.start_date_text}")
    return " | ".join(parts) if parts else "Sin datos recogidos."

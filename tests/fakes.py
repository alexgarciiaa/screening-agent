"""Deterministic provider double for tests.

Mirrors just enough of the understanding step to drive full conversations
without calling an API: each message is mapped to the field currently being
asked. Replies are not asserted on, so they are a placeholder.
"""

import re

from src.agents.schemas import TurnUnderstanding
from src.fsm.enums import Availability, Intent, Language, Schedule, Sentiment, Stage
from src.fsm.flow import next_missing_stage
from src.fsm.models import ConversationState

_YES = {"si", "sí", "claro", "vale", "ok", "correcto", "yes", "yep", "sure"}
_NO = {"no", "nop", "nope", "negativo"}
_STOP_WORDS = {"para", "parar", "stop", "cancelar", "dejalo", "déjalo", "basta"}
_STOP_PHRASES = ("no quiero", "no sigo", "lo dejo")

_AVAILABILITY_HINTS = {
    Availability.FULL_TIME: ("full", "completa", "completo"),
    Availability.PART_TIME: ("part", "parcial", "medio"),
    Availability.WEEKENDS: ("finde", "fin de semana", "weekend", "fines"),
}
_SCHEDULE_HINTS = {
    Schedule.MORNING: ("mañana", "manana", "morning"),
    Schedule.AFTERNOON: ("tarde", "afternoon"),
    Schedule.EVENING: ("noche", "evening", "night"),
    Schedule.FLEXIBLE: ("flex", "da igual", "cualquier", "indiferente"),
}
_PLATFORMS = ("glovo", "uber eats", "uber", "rappi", "just eat", "deliveroo", "didi")


class FakeProvider:
    def __init__(self, settings=None) -> None:
        self._settings = settings

    def understand(self, state: ConversationState) -> TurnUnderstanding:
        text = state.messages[-1].text if state.messages else ""
        lowered = text.lower().strip()
        intent = self._intent(lowered)
        result = TurnUnderstanding(
            language=Language.ES, intent=intent, sentiment=Sentiment.NEUTRAL
        )
        if intent is not Intent.ANSWER:
            return result

        if state.awaiting_confirmation:
            result.confirmation = self._yes_no(lowered)
            return result

        self._fill_stage(next_missing_stage(state.profile), text, lowered, result)
        return result

    def reply(
        self, state, decision, escalate: bool = False, context: str | None = None
    ) -> str:
        return "(fake reply)"

    @staticmethod
    def _intent(lowered: str) -> Intent:
        tokens = set(re.findall(r"\w+", lowered))
        if tokens & _STOP_WORDS or any(p in lowered for p in _STOP_PHRASES):
            return Intent.STOP
        if "?" in lowered:
            return Intent.QUESTION
        return Intent.ANSWER

    def _fill_stage(
        self, stage: Stage | None, text: str, lowered: str, result: TurnUnderstanding
    ) -> None:
        match stage:
            case Stage.CONSENT:
                result.consent = self._yes_no(lowered)
            case Stage.NAME:
                result.full_name = text.strip()
            case Stage.LICENSE:
                result.has_license = self._yes_no(lowered)
            case Stage.CITY:
                result.city = text.strip()
            case Stage.AVAILABILITY:
                result.availability = self._match(lowered, _AVAILABILITY_HINTS)
            case Stage.SCHEDULE:
                result.preferred_schedule = self._match(lowered, _SCHEDULE_HINTS)
            case Stage.EXPERIENCE:
                years = re.search(r"\d+(?:[.,]\d+)?", lowered)
                result.experience_years = (
                    float(years.group().replace(",", ".")) if years else 0.0
                )
                result.experience_platforms = [p for p in _PLATFORMS if p in lowered]
            case Stage.START_DATE:
                result.start_date_text = text.strip()
            case _:
                pass

    @staticmethod
    def _yes_no(lowered: str) -> bool | None:
        tokens = set(re.findall(r"\w+", lowered)) | {lowered}
        if tokens & _YES:
            return True
        if tokens & _NO:
            return False
        return None

    @staticmethod
    def _match(lowered: str, hints: dict):
        for value, needles in hints.items():
            if any(needle in lowered for needle in needles):
                return value
        return None

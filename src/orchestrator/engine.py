from dataclasses import dataclass

from ..config import Settings
from ..fsm.enums import (
    Action,
    Modality,
    Outcome,
    Sentiment,
    TERMINAL_OUTCOMES,
)
from ..fsm.flow import Decision
from ..fsm.models import ConversationState
from ..agents.provider import LLMProvider
from .decision import decide
from .validation import apply_understanding, resolve_service_area

_ESCALATING_SENTIMENTS = {Sentiment.FRUSTRATED, Sentiment.CONFUSED}
_ESCALATING_ACTIONS = {Action.CONFIRM_SUMMARY, Action.CLOSE_QUALIFIED}


@dataclass
class AgentTurn:
    reply: str
    decision: Decision
    outcome: Outcome
    finished: bool


class ScreeningEngine:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def start(self, state: ConversationState) -> str:
        decision = decide(state)
        reply = self._provider.reply(state, decision)
        state.add_message("agent", reply)
        return reply

    def handle(
        self,
        state: ConversationState,
        text: str,
        modality: Modality = Modality.TEXT,
        transcription_confidence: float | None = None,
    ) -> AgentTurn:
        state.add_message("candidate", text, modality, transcription_confidence)

        understanding = self._provider.understand(state)
        state.language = understanding.language
        state.last_intent = understanding.intent
        state.last_sentiment = understanding.sentiment
        state.last_confirmation = understanding.confirmation

        apply_understanding(state.profile, understanding)
        resolve_service_area(state.profile)

        decision = decide(state)
        state.awaiting_confirmation = decision.action is Action.CONFIRM_SUMMARY
        if decision.outcome is not None:
            state.outcome = decision.outcome

        escalate = (
            decision.action in _ESCALATING_ACTIONS
            or state.last_sentiment in _ESCALATING_SENTIMENTS
        )
        reply = self._provider.reply(state, decision, escalate=escalate)
        state.add_message("agent", reply)

        return AgentTurn(
            reply=reply,
            decision=decision,
            outcome=state.outcome,
            finished=state.outcome in TERMINAL_OUTCOMES,
        )

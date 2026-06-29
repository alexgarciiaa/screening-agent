import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..fsm.enums import (
    Action,
    Intent,
    Language,
    Modality,
    Outcome,
    Sentiment,
    TERMINAL_OUTCOMES,
)
from ..fsm.flow import Decision, next_missing_stage
from ..fsm.models import CandidateProfile, ConversationState
from ..agents.provider import LLMProvider
from ..agents.retrieval import Retriever
from .decision import decide
from .validation import apply_understanding, resolve_service_area

_ESCALATING_SENTIMENTS = {Sentiment.FRUSTRATED, Sentiment.CONFUSED}
_ESCALATING_ACTIONS = {Action.CONFIRM_SUMMARY, Action.CLOSE_QUALIFIED}

_NPS_RE = re.compile(r"\b(10|[0-9])\b")

_NPS_QUESTION = {
    Language.ES: (
        "¡Gracias a ti! 🙏 Una última cosa: del 0 al 10, "
        "¿qué te ha parecido el proceso de selección?"
    ),
    Language.EN: (
        "Thank you! 🙏 One last thing: from 0 to 10, "
        "how would you rate this screening experience?"
    ),
}
_NPS_THANKS = {
    Language.ES: "¡Gracias por tu valoración{name}! Nos ayuda a mejorar. 🙌",
    Language.EN: "Thanks for your rating{name}! It helps us improve. 🙌",
}
_NPS_RETRY = {
    Language.ES: "Solo necesito un número del 0 al 10 🙂 ¿Cómo valorarías el proceso?",
    Language.EN: "I just need a number from 0 to 10 🙂 How would you rate the process?",
}

# Nudges for a candidate who goes quiet mid-screening: after 2h and 24h of
# silence. Same fixed-template approach as the NPS messages.
_REMINDER_DELAYS = (timedelta(hours=2), timedelta(hours=24))
_REMINDER_MESSAGES = (
    {
        Language.ES: (
            "¡Hola{name}! 👋 ¿Seguimos con tu solicitud en Grupo Sazón? "
            "Podemos retomarla donde la dejaste cuando quieras."
        ),
        Language.EN: (
            "Hi{name}! 👋 Shall we continue your application with Grupo Sazón? "
            "We can pick up right where you left off whenever you're ready."
        ),
    },
    {
        Language.ES: (
            "¡Hola de nuevo{name}! Tu solicitud sigue abierta. Si quieres "
            "continuar, solo responde a este mensaje 🙂"
        ),
        Language.EN: (
            "Hi again{name}! Your application is still open. If you'd like to "
            "continue, just reply to this message 🙂"
        ),
    },
)


def _name_suffix(profile: CandidateProfile) -> str:
    """', <first name>' when the name is known, else '' — for greeting templates."""
    first = (profile.full_name or "").split(" ")[0]
    return f", {first}" if first else ""


def _parse_nps(text: str) -> int | None:
    match = _NPS_RE.search(text)
    return int(match.group()) if match else None


@dataclass
class AgentTurn:
    reply: str
    outcome: Outcome
    finished: bool
    decision: Decision | None = None


class ScreeningEngine:
    def __init__(
        self, provider: LLMProvider, retriever: Retriever | None = None
    ) -> None:
        self._provider = provider
        self._retriever = retriever

    def start(self, state: ConversationState) -> str:
        decision = decide(state)
        state.last_asked_stage = decision.stage
        reply = self._provider.reply(state, decision)
        state.add_message("agent", reply)
        return reply

    def handle(
        self,
        state: ConversationState,
        text: str,
        modality: Modality = Modality.TEXT,
    ) -> AgentTurn:
        if state.outcome in TERMINAL_OUTCOMES:
            return AgentTurn(reply="", outcome=state.outcome, finished=True)

        state.add_message("candidate", text, modality)
        state.last_candidate_at = datetime.now(timezone.utc)
        state.reminders_sent = 0

        understanding = self._provider.understand(state)
        state.language = understanding.language
        state.last_intent = understanding.intent
        state.last_sentiment = understanding.sentiment
        state.last_confirmation = understanding.confirmation

        # Only capture profile fields from an actual answer. A field mentioned
        # inside a question ("do you operate in Barcelona?") or chit-chat must
        # not be stored as the candidate's own data.
        if understanding.intent is Intent.ANSWER:
            apply_understanding(
                state.profile,
                understanding,
                pending=next_missing_stage(state.profile),
            )
            resolve_service_area(state.profile)

        decision = decide(state)
        state.last_asked_stage = decision.stage
        state.awaiting_confirmation = decision.action is Action.CONFIRM_SUMMARY
        if decision.outcome is not None:
            state.outcome = decision.outcome

        escalate = (
            decision.action in _ESCALATING_ACTIONS
            or state.last_sentiment in _ESCALATING_SENTIMENTS
        )
        context = self._retrieve_context(decision, text)
        reply = self._provider.reply(
            state, decision, escalate=escalate, context=context
        )
        state.add_message("agent", reply)

        return AgentTurn(
            reply=reply,
            decision=decision,
            outcome=state.outcome,
            finished=state.outcome in TERMINAL_OUTCOMES,
        )

    def _retrieve_context(self, decision: Decision, question: str) -> str | None:
        """Knowledge-base passages for an answerable question, or None."""
        if self._retriever is None or decision.action is not Action.ANSWER_QUESTION:
            return None
        chunks = self._retriever.search(question)
        if not chunks:
            return None
        return "\n\n".join(f"[{c.source}] {c.content}" for c in chunks)

    def follow_up(self, state: ConversationState, text: str) -> str | None:
        """Post-screening NPS, run after the screening has finished.

        The candidate's first message triggers the rating question; the next is
        read as the score (0-10). An out-of-range or missing number is re-asked.
        Returns the message to send, or None once a score has been recorded.
        """
        if state.nps_done:
            return None
        state.add_message("candidate", text)
        if not state.nps_asked:
            state.nps_asked = True
            reply = _NPS_QUESTION[state.language]
        else:
            score = _parse_nps(text)
            if score is None:
                # Out of range or no number: stay here and ask once more.
                reply = _NPS_RETRY[state.language]
            else:
                state.nps_score = score
                state.nps_done = True
                reply = _NPS_THANKS[state.language].format(
                    name=_name_suffix(state.profile)
                )
        state.add_message("agent", reply)
        return reply

    def reminder(
        self, state: ConversationState, now: datetime | None = None
    ) -> str | None:
        """Next nudge for a silent in-progress candidate, or None.

        Reminders fire 2h and 24h after the candidate's last message. Calling
        this advances the counter so each nudge is sent once; the counter resets
        when the candidate replies (see handle).
        """
        now = now or datetime.now(timezone.utc)
        sent = state.reminders_sent
        if state.outcome is not Outcome.IN_PROGRESS or sent >= len(_REMINDER_DELAYS):
            return None
        baseline = state.last_candidate_at or state.created_at
        if now - baseline < _REMINDER_DELAYS[sent]:
            return None
        state.reminders_sent = sent + 1
        return _REMINDER_MESSAGES[sent][state.language].format(
            name=_name_suffix(state.profile)
        )

import json
import logging
from typing import Protocol

from ..config import Settings
from ..fsm.flow import Decision
from ..fsm.models import ConversationState
from . import prompts
from .schemas import TurnUnderstanding

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    def understand(self, state: ConversationState) -> TurnUnderstanding: ...

    def reply(
        self, state: ConversationState, decision: Decision, escalate: bool = False
    ) -> str: ...


class AnthropicProvider:
    """Primary provider backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=30.0
        )

    def understand(self, state: ConversationState) -> TurnUnderstanding:
        response = self._client.messages.parse(
            model=self._settings.model_understand,
            max_tokens=1000,
            temperature=0.1,
            system=prompts.SYSTEM_UNDERSTAND,
            messages=[
                {"role": "user", "content": prompts.understand_user_message(state)}
            ],
            output_format=TurnUnderstanding,
        )
        return response.parsed_output

    def reply(
        self, state: ConversationState, decision: Decision, escalate: bool = False
    ) -> str:
        model = (
            self._settings.model_reply_escalated
            if escalate
            else self._settings.model_reply
        )
        response = self._client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.3,
            system=prompts.SYSTEM_REPLY,
            messages=[
                {"role": "user", "content": prompts.reply_user_message(state, decision)}
            ],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()


class GroqProvider:
    """Fallback provider backed by Groq's OpenAI-compatible API."""

    def __init__(self, settings: Settings) -> None:
        import groq

        self._settings = settings
        self._client = groq.Groq(api_key=settings.groq_api_key, timeout=30.0)

    def understand(self, state: ConversationState) -> TurnUnderstanding:
        schema = json.dumps(TurnUnderstanding.model_json_schema())
        system = (
            prompts.SYSTEM_UNDERSTAND
            + "\n\nReturn only a JSON object matching this schema:\n"
            + schema
        )
        response = self._client.chat.completions.create(
            model=self._settings.groq_model_understand,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompts.understand_user_message(state)},
            ],
        )
        return TurnUnderstanding.model_validate_json(
            response.choices[0].message.content
        )

    def reply(
        self, state: ConversationState, decision: Decision, escalate: bool = False
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.groq_model_reply,
            temperature=0.3,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_REPLY},
                {"role": "user", "content": prompts.reply_user_message(state, decision)},
            ],
        )
        return response.choices[0].message.content.strip()


class FallbackProvider:
    """Try the primary provider and fall back to the secondary on any failure."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    def understand(self, state: ConversationState) -> TurnUnderstanding:
        try:
            return self._primary.understand(state)
        except Exception:
            logger.warning(
                "Primary provider failed on understand, using fallback",
                exc_info=True,
            )
            return self._fallback.understand(state)

    def reply(
        self, state: ConversationState, decision: Decision, escalate: bool = False
    ) -> str:
        try:
            return self._primary.reply(state, decision, escalate=escalate)
        except Exception:
            logger.warning(
                "Primary provider failed on reply, using fallback",
                exc_info=True,
            )
            return self._fallback.reply(state, decision, escalate=escalate)


def build_provider(settings: Settings) -> LLMProvider:
    primary = AnthropicProvider(settings) if settings.anthropic_api_key else None
    fallback = GroqProvider(settings) if settings.groq_api_key else None

    if primary and fallback:
        return FallbackProvider(primary, fallback)
    provider = primary or fallback
    if provider is None:
        raise RuntimeError(
            "No LLM provider configured: set ANTHROPIC_API_KEY or GROQ_API_KEY"
        )
    return provider

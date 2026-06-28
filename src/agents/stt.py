"""Speech-to-text for voice notes, via Groq Whisper"""

import io
import logging

from ..config import Settings

logger = logging.getLogger(__name__)


def transcribe(settings: Settings, audio: bytes, filename: str = "audio.ogg") -> str | None:
    """Transcribe a voice note. Returns the text, or None if it can't be done."""
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set; cannot transcribe voice messages")
        return None

    import groq

    client = groq.Groq(api_key=settings.groq_api_key)
    buffer = io.BytesIO(audio)
    buffer.name = filename
    response = client.audio.transcriptions.create(
        file=buffer, model=settings.groq_stt_model
    )
    text = (response.text or "").strip()
    return text or None

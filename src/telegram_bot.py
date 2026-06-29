"""Telegram entrypoint: runs the screening agent as an always-on bot via long polling.

This is the main entry point in production. The CLI (`main.py`) is kept for testing.
Conversations are persisted via the repository (SQLite locally, Postgres/Supabase in
production), keyed by chat id, so the bot remembers each candidate across restarts.
"""

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agents import stt
from .agents.provider import build_provider
from .agents.retrieval import build_retriever
from .config import get_settings
from .data import service_areas
from .fsm.enums import Modality, Outcome
from .orchestrator.engine import ScreeningEngine
from .orchestrator.handoff import build_handoff
from .storage.repository import ConversationRepository

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_locks: dict[int, asyncio.Lock] = {}


def _lock(chat_id: int) -> asyncio.Lock:
    return _locks.setdefault(chat_id, asyncio.Lock())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    engine: ScreeningEngine = context.application.bot_data["engine"]
    repo: ConversationRepository = context.application.bot_data["repository"]
    candidate_id = str(chat_id)
    async with _lock(chat_id):
        await asyncio.to_thread(repo.reset, candidate_id)
        state = await asyncio.to_thread(repo.get_or_create, candidate_id)
        greeting = await asyncio.to_thread(engine.start, state)
        await asyncio.to_thread(repo.save, state)
    await update.message.reply_text(greeting)


async def _process_turn(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    modality: Modality = Modality.TEXT,
) -> str:
    engine: ScreeningEngine = context.application.bot_data["engine"]
    repo: ConversationRepository = context.application.bot_data["repository"]
    candidate_id = str(chat_id)
    try:
        async with _lock(chat_id):
            state = await asyncio.to_thread(repo.get_active, candidate_id)
            if state is not None:
                turn = await asyncio.to_thread(engine.handle, state, text, modality)
                reply = turn.reply
                if turn.finished:
                    logger.info(
                        "Handoff for chat %s:\n%s",
                        chat_id,
                        build_handoff(state).model_dump_json(indent=2),
                    )
                await asyncio.to_thread(repo.save, state)
            else:
                finished = await asyncio.to_thread(repo.latest, candidate_id)
                if finished is not None:
                    # Screening is over; collect a one-off NPS rating.
                    nps_reply = await asyncio.to_thread(
                        engine.follow_up, finished, text
                    )
                    if nps_reply is not None:
                        await asyncio.to_thread(repo.save, finished)
                        reply = nps_reply
                    else:
                        reply = (
                            "Ya hemos terminado tu proceso de selección. "
                            "Si quieres empezar de nuevo, escribe /start."
                        )
                else:
                    # Brand-new candidate who messaged without /start.
                    state = await asyncio.to_thread(repo.get_or_create, candidate_id)
                    reply = await asyncio.to_thread(engine.start, state)
                    await asyncio.to_thread(repo.save, state)
    except Exception:
        logger.exception("Error handling update from chat %s", chat_id)
        reply = "Ha ocurrido un error. Escribe /start para empezar de nuevo."
    return reply


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reply = await _process_turn(context, update.effective_chat.id, update.message.text)
    await update.message.reply_text(reply)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    settings = context.application.bot_data["settings"]
    try:
        tg_file = await context.bot.get_file(update.message.voice.file_id)
        audio = bytes(await tg_file.download_as_bytearray())
        text = await asyncio.to_thread(stt.transcribe, settings, audio)
    except Exception:
        logger.exception("Voice transcription failed for chat %s", chat_id)
        text = None

    if not text:
        await update.message.reply_text(
            "No pude entender el audio. ¿Puedes repetirlo o escribirlo, por favor?"
        )
        return

    reply = await _process_turn(context, chat_id, text, Modality.VOICE)
    await update.message.reply_text(reply)


_REMINDER_SCAN_SECONDS = 600


async def _reminder_loop(app) -> None:
    """Nudge candidates who went quiet mid-screening (after 2h and 24h)."""
    engine: ScreeningEngine = app.bot_data["engine"]
    repo: ConversationRepository = app.bot_data["repository"]
    while True:
        await asyncio.sleep(_REMINDER_SCAN_SECONDS)
        try:
            active = await asyncio.to_thread(repo.list_by_outcome, Outcome.IN_PROGRESS)
        except Exception:
            logger.exception("Reminder scan failed to load conversations")
            continue
        for snapshot in active:
            try:
                chat_id = int(snapshot.candidate_id)
                async with _lock(chat_id):
                    state = await asyncio.to_thread(repo.get_active, snapshot.candidate_id)
                    if state is None:
                        continue
                    message = engine.reminder(state)
                    if message is None:
                        continue
                    await app.bot.send_message(chat_id=chat_id, text=message)
                    await asyncio.to_thread(repo.save, state)
            except Exception:
                logger.exception("Reminder failed for %s", snapshot.candidate_id)


async def _announce(app) -> None:
    me = await app.bot.get_me()
    logger.info("Bot @%s is live: https://t.me/%s", me.username, me.username)
    app.create_task(_reminder_loop(app))


def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in the environment (from @BotFather).")
    try:
        provider = build_provider(settings)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    service_areas.configure(settings.database_url)

    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .post_init(_announce)
        .build()
    )
    app.bot_data["engine"] = ScreeningEngine(provider, build_retriever(settings))
    app.bot_data["repository"] = ConversationRepository(settings.database_url)
    app.bot_data["settings"] = settings
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    if settings.telegram_webhook_url:
        logger.info("Starting in webhook mode on port %s.", settings.port)
        app.run_webhook(
            listen="0.0.0.0",
            port=settings.port,
            url_path=settings.telegram_bot_token,
            webhook_url=f"{settings.telegram_webhook_url.rstrip('/')}/{settings.telegram_bot_token}",
        )
    else:
        logger.info("Starting in polling mode (local). Ctrl+C to stop.")
        app.run_polling()


if __name__ == "__main__":
    main()

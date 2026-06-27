"""Telegram entrypoint: runs the screening agent as an always-on bot via long polling.

This is the main entry point in production. The CLI (`main.py`) is kept for testing.
Sessions are held in memory, keyed by Telegram chat id, so a restart starts fresh.
"""

import asyncio
import logging
import uuid

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agents.provider import build_provider
from .config import get_settings
from .orchestrator.engine import ScreeningEngine
from .orchestrator.handoff import build_handoff
from .fsm.models import ConversationState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_sessions: dict[int, ConversationState] = {}
_locks: dict[int, asyncio.Lock] = {}


def _new_session(chat_id: int) -> ConversationState:
    return ConversationState(conversation_id=uuid.uuid4().hex, candidate_id=str(chat_id))


def _lock(chat_id: int) -> asyncio.Lock:
    return _locks.setdefault(chat_id, asyncio.Lock())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    engine: ScreeningEngine = context.application.bot_data["engine"]
    async with _lock(chat_id):
        state = _new_session(chat_id)
        _sessions[chat_id] = state
        greeting = await asyncio.to_thread(engine.start, state)
    await update.message.reply_text(greeting)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    engine: ScreeningEngine = context.application.bot_data["engine"]
    try:
        async with _lock(chat_id):
            state = _sessions.get(chat_id)
            if state is None:
                state = _new_session(chat_id)
                _sessions[chat_id] = state
                reply = await asyncio.to_thread(engine.start, state)
            else:
                turn = await asyncio.to_thread(engine.handle, state, update.message.text)
                reply = turn.reply
                if turn.finished:
                    logger.info(
                        "Handoff for chat %s:\n%s",
                        chat_id,
                        build_handoff(state).model_dump_json(indent=2),
                    )
                    _sessions.pop(chat_id, None)
    except Exception:
        logger.exception("Error handling update from chat %s", chat_id)
        reply = "Ha ocurrido un error. Escribe /start para empezar de nuevo."
    await update.message.reply_text(reply)


def _render_qr(link: str) -> None:
    try:
        import qrcode
        from qrcode.image.svg import SvgImage
    except ImportError:
        logger.info("Install 'qrcode' to render the access QR for %s", link)
        return
    qr = qrcode.QRCode(border=2)
    qr.add_data(link)
    qr.make(fit=True)
    try:
        qr.print_ascii(invert=True)  # best effort; fails on non-UTF-8 consoles
    except Exception:
        logger.debug("Could not print the ASCII QR", exc_info=True)
    try:
        qrcode.make(link, image_factory=SvgImage).save("bot_qr.svg")
        logger.info("Access QR saved to bot_qr.svg")
    except Exception:
        logger.debug("Could not save the SVG QR", exc_info=True)


async def _announce(app) -> None:
    me = await app.bot.get_me()
    link = f"https://t.me/{me.username}"
    logger.info("Bot @%s is live. Open: %s", me.username, link)
    _render_qr(link)


def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in the environment (from @BotFather).")
    try:
        provider = build_provider(settings)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .post_init(_announce)
        .build()
    )
    app.bot_data["engine"] = ScreeningEngine(provider, settings)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("Starting Telegram bot (long polling). Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

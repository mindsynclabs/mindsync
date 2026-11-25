"""
Telegram integration that runs INSIDE the FastAPI app: one process, no separate bot.
It calls the RAG code directly (the same functions the HTTP endpoints use).

  pip install "python-telegram-bot>=20"

Two modes (TELEGRAM_MODE):

  polling  (default) The app asks Telegram for new messages in the background.
           No public URL needed, so it works on a machine behind a home router.
           Run with a single uvicorn worker.

  webhook  Telegram calls POST /telegram/webhook on your FastAPI app.
           Needs a public HTTPS URL (a server, or a tunnel such as Cloudflare
           Tunnel / ngrok). Set TELEGRAM_WEBHOOK_URL and TELEGRAM_WEBHOOK_SECRET.

Environment variables:
  TELEGRAM_BOT_TOKEN       token from @BotFather (Telegram is disabled if unset)
  TELEGRAM_ALLOWED_IDS     comma-separated Telegram user ids that may use the bot
  TELEGRAM_MODE            polling | webhook
  TELEGRAM_WEBHOOK_URL     public base URL, e.g. https://abc.example.com   (webhook only)
  TELEGRAM_WEBHOOK_SECRET  long random string checked on every webhook call (webhook only)
  SHOW_SOURCES             1 (default) to list source documents under each answer
"""
import asyncio
import functools
import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

log = logging.getLogger("telegram_service")
logging.getLogger("httpx").setLevel(logging.WARNING)  # keep the bot token out of logs

# ---- Config -------------------------------------------------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
MODE = os.getenv("TELEGRAM_MODE", "polling").lower()
WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED = {
    int(x) for x in os.getenv("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x
}
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "1") == "1"
MAX_FILE_BYTES = 20 * 1024 * 1024  # Telegram Bot API download limit
ALLOWED_EXT = (".pdf", ".txt", ".md")
TG_MESSAGE_LIMIT = 4000  # Telegram's hard limit is 4096 characters

HELP = (
    "I answer questions about your documents.\n\n"
    "• Send a text message to ask a question\n"
    "• Send a PDF, .txt or .md file to add it (a caption sets its doc id)\n"
    "• /reset  clear this conversation's memory\n"
    "• /delete <doc_id>  remove a document\n"
    "• /help  show this message"
)

# The RAG and OCR modules are injected by main.py (see configure) so this file never
# imports them a second time under a different name.
_rag = None
_ocr = None
_empty_index_error: tuple = ()
_ptb: Optional[Application] = None

router = APIRouter(prefix="/telegram", tags=["telegram"])


def configure(rag_module, ocr_module) -> None:
    global _rag, _ocr, _empty_index_error
    _rag, _ocr = rag_module, ocr_module
    err = getattr(rag_module, "EmptyIndexError", None)
    _empty_index_error = (err,) if err else ()


# ---- Endpoints ---------------------------------------------------------------
@router.post("/webhook")
async def webhook(request: Request):
    """Telegram calls this in webhook mode. Returns immediately; work happens in the queue."""
    if _ptb is None or MODE != "webhook":
        raise HTTPException(404, "Telegram webhook is not active.")
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(secret, WEBHOOK_SECRET):
        raise HTTPException(403, "Bad secret token.")
    update = Update.de_json(await request.json(), _ptb.bot)
    await _ptb.update_queue.put(update)
    return {"ok": True}


@router.get("/status")
async def status():
    return {
        "enabled": _ptb is not None,
        "mode": MODE if _ptb is not None else None,
        "allowed_users": len(ALLOWED),
    }


# ---- Helpers -----------------------------------------------------------------
def session_id(update: Update) -> str:
    return f"tg-{update.effective_chat.id}"


async def send_long(update: Update, text: str) -> None:
    text = text.strip() or "(empty answer)"
    for i in range(0, len(text), TG_MESSAGE_LIMIT):
        await update.effective_message.reply_text(text[i : i + TG_MESSAGE_LIMIT])


async def _typing_loop(update: Update, context, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def with_typing(update: Update, context, awaitable):
    """Show 'typing...' in the chat while slow work (LLM, OCR) runs."""
    stop = asyncio.Event()
    task = asyncio.create_task(_typing_loop(update, context, stop))
    try:
        return await awaitable
    finally:
        stop.set()
        await task


def extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return content.decode("utf-8")
    if name.endswith(".pdf"):
        return _ocr.extract_pdf_text(content)
    raise ValueError("Unsupported file type")


def guarded(handler):
    """Only allow listed users. Everyone else is ignored."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None or update.effective_message is None:
            return
        if not ALLOWED:
            # Setup mode: nobody is allowed yet, so just reveal the user's own id.
            await update.effective_message.reply_text(
                f"This bot is locked. Your Telegram user id is {user.id}.\n"
                "Put it in TELEGRAM_ALLOWED_IDS and restart the app."
            )
            return
        if user.id not in ALLOWED:
            log.warning("Ignored message from unauthorized user id %s", user.id)
            return
        await handler(update, context)

    return wrapper


# ---- Handlers (they call the RAG functions directly) ---------------------------
@guarded
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP)


@guarded
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _rag.reset_session(session_id(update))
    await update.effective_message.reply_text("Conversation memory cleared.")


@guarded
async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text("Usage: /delete <doc_id>")
        return
    doc_id = " ".join(context.args)
    await run_in_threadpool(_rag.delete_document, doc_id)
    await update.effective_message.reply_text(f"Deleted document '{doc_id}'.")


@guarded
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.effective_message.text or "").strip()
    if not question:
        return
    try:
        result = await with_typing(
            update, context, run_in_threadpool(_rag.chat, session_id(update), question)
        )
    except _empty_index_error:
        await update.effective_message.reply_text(
            "No documents yet. Send me a PDF or text file first."
        )
        return
    except Exception as e:
        log.exception("chat failed")
        await update.effective_message.reply_text(
            f"Chat failed (is Ollama running and the model pulled?): {e}"
        )
        return

    answer = result.get("answer", "")
    if result.get("grounded") is False:
        answer = "Not found in your documents. Answering from general knowledge:\n\n" + answer
    if SHOW_SOURCES and result.get("sources"):
        names = []
        for s in result["sources"]:
            name = s.get("doc_id")
            if name and name not in names:
                names.append(name)
        if names:
            answer += "\n\nSources: " + ", ".join(names)
    await send_long(update, answer)


@guarded
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    doc = msg.document
    filename = doc.file_name or "upload"
    if not filename.lower().endswith(ALLOWED_EXT):
        await msg.reply_text("Unsupported file type. Send .pdf, .txt or .md")
        return
    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await msg.reply_text("File is too large (Telegram bots can download up to 20 MB).")
        return

    await msg.reply_text(f"Received {filename}. Processing, scanned PDFs can take several minutes...")
    doc_id = (msg.caption or "").strip() or filename

    async def process():
        tg_file = await doc.get_file()
        content = bytes(await tg_file.download_as_bytearray())
        text = await run_in_threadpool(extract_text, filename, content)
        text = text.replace("\r", "").strip()
        if not text:
            return None
        meta = {
            "source": "telegram",
            "filename": filename,
            "file_hash": hashlib.sha256(content).hexdigest(),
        }
        return await run_in_threadpool(_rag.ingest_text, doc_id, text, meta)

    try:
        chunks = await with_typing(update, context, process())
    except Exception as e:
        log.exception("upload failed")
        await msg.reply_text(f"Upload failed: {e}")
        return

    if chunks is None:
        await msg.reply_text(
            "No text found. For scanned PDFs, make sure the OCR model is pulled and PDF_OCR is not 'off'."
        )
    else:
        await msg.reply_text(f"Added '{doc_id}' ({chunks} chunks). You can ask questions now.")


# ---- Lifecycle (called from main.py's lifespan) --------------------------------
async def startup() -> None:
    global _ptb
    if not TOKEN:
        log.info("Telegram is disabled (TELEGRAM_BOT_TOKEN is not set).")
        return
    if _rag is None:
        log.error("telegram_service.configure(rag, ocr) was not called; Telegram is disabled.")
        return
    if MODE not in ("polling", "webhook"):
        log.error("TELEGRAM_MODE must be 'polling' or 'webhook'; Telegram is disabled.")
        return
    if MODE == "webhook" and not (WEBHOOK_URL and WEBHOOK_SECRET):
        log.error("Webhook mode needs TELEGRAM_WEBHOOK_URL and TELEGRAM_WEBHOOK_SECRET; Telegram is disabled.")
        return
    if not ALLOWED:
        log.warning("TELEGRAM_ALLOWED_IDS is empty: the bot is locked and only tells users their id.")

    try:
        builder = Application.builder().token(TOKEN).concurrent_updates(True)
        if MODE == "webhook":
            builder = builder.updater(None)  # we feed updates from the FastAPI endpoint
        app = builder.build()

        private = filters.ChatType.PRIVATE  # ignore groups and channels
        app.add_handler(CommandHandler(["start", "help"], cmd_start, filters=private))
        app.add_handler(CommandHandler("reset", cmd_reset, filters=private))
        app.add_handler(CommandHandler("delete", cmd_delete, filters=private))
        app.add_handler(MessageHandler(private & filters.Document.ALL, on_document))
        app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, on_text))

        await app.initialize()
        await app.start()
        if MODE == "webhook":
            await app.bot.set_webhook(
                url=f"{WEBHOOK_URL}/telegram/webhook",
                secret_token=WEBHOOK_SECRET,
                allowed_updates=Update.ALL_TYPES,
            )
            log.info("Telegram webhook set: %s/telegram/webhook", WEBHOOK_URL)
        else:
            await app.bot.delete_webhook()  # polling and webhook cannot be active together
            await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            log.info("Telegram polling started.")
        _ptb = app
    except Exception:
        # Never take the whole API down because Telegram is unreachable or the token is wrong.
        log.exception("Telegram failed to start; the API keeps running without it.")
        _ptb = None


async def shutdown() -> None:
    global _ptb
    if _ptb is None:
        return
    try:
        if MODE == "polling" and _ptb.updater and _ptb.updater.running:
            await _ptb.updater.stop()
        await _ptb.stop()
        await _ptb.shutdown()
    except Exception:
        log.exception("Error while stopping Telegram.")
    _ptb = None
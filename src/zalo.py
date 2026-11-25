#!/usr/bin/env python3
"""
Zalo bot for the RAG app: a user messages the bot on Zalo, the question goes to your
RAG backend (POST /chat), and the answer is sent back. Same idea as a Telegram bot.

This uses the Zalo BOT PLATFORM (bot.zaloplatforms.com), NOT the Zalo Official Account API.

Flow
  user ──Zalo──► Zalo Bot Platform ──► this script ──POST /chat──► your RAG app (rag.py)
  user ◄─Zalo─── sendMessage ◄─────── this script ◄──answer─────── your RAG app
Two ways to receive messages (they exclude each other):
  polling   this script asks Zalo for new messages (getUpdates). No public URL needed.
            Zalo's docs recommend it for local development and testing.
  webhook   Zalo calls YOUR public HTTPS URL. Recommended for production.

Setup
  1. In the Zalo app, open "Zalo Bot Manager" -> create a bot (its name must start with "Bot")
     -> the bot token (looks like 123456789:abc...) arrives as a Zalo message.
  2. export ZALO_BOT_TOKEN="123456789:abc..."
     export ZALO_ALLOWED_CHAT_IDS="..."      # optional but strongly advised, see below
  3. Start your RAG app (uvicorn ...), then:
        python zalo_bot.py me          # check the token
        python zalo_bot.py polling     # run the bot

Webhook mode (production)
  - in your FastAPI main.py:   import zalo_bot;  app.include_router(zalo_bot.build_webhook_router())
  - export ZALO_WEBHOOK_SECRET="8-256 random characters"
  - python zalo_bot.py set-webhook https://your-domain/zalo/webhook
    (must be public HTTPS; localhost is rejected. For local tests use a tunnel such as
     Cloudflare Tunnel or ngrok.) Then python zalo_bot.py test-webhook.
  - to go back to polling:  python zalo_bot.py delete-webhook

Security: anyone who finds your bot can ask it questions, so it can reveal your documents.
Set ZALO_ALLOWED_CHAT_IDS (comma separated). Send the bot a message with ZALO_DEBUG=1 to
see your chat id in the log.

Install: pip install httpx     (fastapi only for webhook mode; your app already has it)
"""
import argparse
import hmac
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

API_BASE = "https://bot-api.zaloplatforms.com"
BOT_TOKEN = os.getenv("ZALO_BOT_TOKEN", "")
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/chat")
ALLOWED = {x.strip() for x in os.getenv("ZALO_ALLOWED_CHAT_IDS", "").split(",") if x.strip()}
WEBHOOK_SECRET = os.getenv("ZALO_WEBHOOK_SECRET", "")
POLL_TIMEOUT = int(os.getenv("ZALO_POLL_TIMEOUT", "30"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
WORKERS = int(os.getenv("ZALO_WORKERS", "2"))
DEBUG = os.getenv("ZALO_DEBUG", "0") == "1"
ZALO_TEXT_LIMIT = 2000  # sendMessage accepts 1-2000 characters

WELCOME = ("Xin chào! Hãy gửi câu hỏi, mình sẽ trả lời dựa trên tài liệu của công ty.\n"
           "Gõ /reset để bắt đầu cuộc trò chuyện mới.")
UNSUPPORTED = "Hiện mình chỉ đọc được tin nhắn văn bản."
ERROR_TEXT = "Xin lỗi, hệ thống đang gặp lỗi. Bạn thử lại sau nhé."
NO_DOCS = "Hệ thống chưa có tài liệu nào."
NOT_ALLOWED = "Bạn chưa được cấp quyền sử dụng bot này."
TOO_LONG = f"Câu hỏi quá dài (tối đa {MAX_QUESTION_CHARS} ký tự)."

log = logging.getLogger("zalo")


# =============================================================================
# Zalo Bot API client
# =============================================================================
class ZaloError(Exception):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class ZaloClient:
    """Thin client. The token is part of every URL, so it is masked in all error text."""

    GET_METHODS = {"getMe", "getWebhookInfo"}  # parameterless reads (falls back to POST)

    def __init__(self, token: str, poll_timeout: int = POLL_TIMEOUT):
        if not token:
            raise SystemExit("Set ZALO_BOT_TOKEN first (see the instructions at the top of this file).")
        import httpx

        self.token = token
        self._http = httpx.Client(timeout=httpx.Timeout(poll_timeout + 15.0))

    def _mask(self, text: Any) -> str:
        return str(text).replace(self.token, "***")

    def call(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{API_BASE}/bot{self.token}/{method}"
        try:
            if method in self.GET_METHODS:
                resp = self._http.get(url)
                if resp.status_code >= 400:
                    resp = self._http.post(url, json=payload or {})
            else:
                resp = self._http.post(url, json=payload or {})
            data = resp.json()
        except Exception as e:  # network error, bad JSON ...
            raise ZaloError(f"{method}: {self._mask(e)}") from None
        if not data.get("ok"):
            code = data.get("error_code")
            raise ZaloError(f"{method}: error {code} {data.get('description', '')}", code)
        return data.get("result")

    def get_me(self):
        return self.call("getMe")

    def get_updates(self, timeout: int = POLL_TIMEOUT):
        # docs list `timeout` as a String parameter
        return self.call("getUpdates", {"timeout": str(timeout)})

    def send_chat_action(self, chat_id: str, action: str = "typing") -> None:
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": action})
        except ZaloError:
            pass  # the typing indicator is optional

    def send_message(self, chat_id: str, text: str) -> None:
        for part in split_message(text):
            self.call("sendMessage", {"chat_id": chat_id, "text": part})

    def set_webhook(self, url: str, secret_token: str):
        return self.call("setWebhook", {"url": url, "secret_token": secret_token})

    def delete_webhook(self):
        return self.call("deleteWebhook")

    def get_webhook_info(self):
        return self.call("getWebhookInfo")

    def test_webhook(self):
        return self.call("testWebhook")


# =============================================================================
# Pure helpers
# =============================================================================
def split_message(text: str, limit: int = ZALO_TEXT_LIMIT) -> List[str]:
    """Cut a long answer into parts of at most `limit` characters, at line breaks when possible."""
    text = (text or "").strip() or "..."
    parts: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    parts.append(text)
    return [p for p in parts if p]


def extract_updates(payload: Any) -> List[Dict[str, Any]]:
    """getUpdates and the webhook body share one shape: {"ok":true,"result":{event_name, message}}.
    Accepts a wrapped body, a list, or a bare update."""
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if isinstance(payload, list):
        return [u for u in payload if isinstance(u, dict)]
    if isinstance(payload, dict) and payload:
        return [payload]
    return []


# =============================================================================
# Handling one update
# =============================================================================
_seen: "OrderedDict[Any, bool]" = OrderedDict()
_seen_lock = threading.Lock()
_chat_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)  # one question at a time per chat
_epoch: Dict[str, int] = defaultdict(int)  # /reset bumps this -> a fresh RAG session


def already_seen(key: Any) -> bool:
    """Webhooks can be delivered twice; ignore repeats."""
    with _seen_lock:
        if key in _seen:
            return True
        _seen[key] = True
        while len(_seen) > 1000:
            _seen.popitem(last=False)
        return False


def ask_backend(session_id: str, text: str) -> str:
    import httpx

    resp = httpx.post(CHAT_API_URL, json={"message": text, "session_id": session_id}, timeout=300)
    if resp.status_code == 409:  # your app: no documents ingested yet
        return NO_DOCS
    resp.raise_for_status()
    return (resp.json().get("answer") or "").strip() or ERROR_TEXT


def handle_update(client: ZaloClient, update: Dict[str, Any]) -> None:
    if DEBUG:
        print("[zalo] raw update:", json.dumps(update, ensure_ascii=False), flush=True)
    event = update.get("event_name", "")
    msg = update.get("message") or {}
    chat_id = str((msg.get("chat") or {}).get("id") or "")
    if not chat_id:
        return
    if already_seen((event, msg.get("message_id"))):
        return
    if ALLOWED and chat_id not in ALLOWED:
        log.warning("blocked chat %s (not in ZALO_ALLOWED_CHAT_IDS)", chat_id)
        client.send_message(chat_id, NOT_ALLOWED)
        return
    if event == "message.unsupported.received":
        return  # protected accounts: no content is delivered
    if event != "message.text.received":
        client.send_message(chat_id, UNSUPPORTED)  # image / sticker / voice
        return

    text = (msg.get("text") or "").strip()
    if not text:
        return
    command = text.split()[0].lower()
    if command in ("/start", "/help"):
        client.send_message(chat_id, WELCOME)
        return
    if command == "/reset":
        _epoch[chat_id] += 1
        client.send_message(chat_id, "Đã bắt đầu cuộc trò chuyện mới.")
        return
    if len(text) > MAX_QUESTION_CHARS:
        client.send_message(chat_id, TOO_LONG)
        return

    with _chat_locks[chat_id]:
        client.send_chat_action(chat_id)
        session_id = f"zalo-{chat_id}-{_epoch[chat_id]}"
        try:
            answer = ask_backend(session_id, text)
        except Exception as e:
            log.error("backend failed for chat %s: %s", chat_id, e)
            answer = ERROR_TEXT
        client.send_message(chat_id, answer)


def safe_handle(client: ZaloClient, update: Dict[str, Any]) -> None:
    try:
        handle_update(client, update)
    except Exception:
        log.exception("failed to handle update")


# =============================================================================
# Receiving: polling and webhook
# =============================================================================
def run_polling(client: ZaloClient) -> None:
    client.delete_webhook()  # polling does not work while a webhook is registered
    me = client.get_me()
    log.info("Polling as %s. Ctrl+C to stop.", me)
    if not ALLOWED:
        log.warning("ZALO_ALLOWED_CHAT_IDS is empty: ANYONE who finds the bot can query your documents.")
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    failures = 0
    while True:
        try:
            result = client.get_updates(POLL_TIMEOUT)
            failures = 0
        except ZaloError as e:
            if e.code == 408:  # nothing new during the long poll
                continue
            failures += 1
            log.error("%s (retry in %ds)", e, min(30, 2 * failures))
            time.sleep(min(30, 2 * failures))
            continue
        for update in extract_updates(result):
            pool.submit(safe_handle, client, update)


def build_webhook_router(client: Optional[ZaloClient] = None):
    """FastAPI router for webhook mode: app.include_router(zalo_bot.build_webhook_router())"""
    from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

    router = APIRouter()
    zalo = client or ZaloClient(BOT_TOKEN)

    @router.post("/zalo/webhook")
    async def zalo_webhook(
        request: Request,
        background: BackgroundTasks,
        x_bot_api_secret_token: str = Header(default=""),
    ):
        # Zalo sends the secret you registered in this header (plain comparison, no HMAC)
        if not WEBHOOK_SECRET or not hmac.compare_digest(x_bot_api_secret_token, WEBHOOK_SECRET):
            raise HTTPException(status_code=403, detail="Unauthorized")
        body = await request.json()
        for update in extract_updates(body):
            background.add_task(safe_handle, zalo, update)  # answer after replying 200 quickly
        return {"message": "Success"}

    return router


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Zalo bot for the RAG app")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("me", help="check the bot token")
    sub.add_parser("polling", help="run the bot with long polling (no public URL needed)")
    p = sub.add_parser("set-webhook", help="register your public HTTPS URL (switches off polling)")
    p.add_argument("url")
    sub.add_parser("delete-webhook", help="remove the webhook (back to polling)")
    sub.add_parser("webhook-info", help="show the registered webhook")
    sub.add_parser("test-webhook", help="ask Zalo to test your webhook endpoint")
    a = ap.parse_args()

    client = ZaloClient(BOT_TOKEN)
    try:
        if a.cmd == "me":
            print(client.get_me())
        elif a.cmd == "polling":
            run_polling(client)
        elif a.cmd == "set-webhook":
            if not 8 <= len(WEBHOOK_SECRET) <= 256:
                sys.exit("Set ZALO_WEBHOOK_SECRET to 8-256 characters first.")
            print(json.dumps(client.set_webhook(a.url, WEBHOOK_SECRET), indent=2))
        elif a.cmd == "delete-webhook":
            print(client.delete_webhook())
        elif a.cmd == "webhook-info":
            print(client.get_webhook_info())
        elif a.cmd == "test-webhook":
            print(json.dumps(client.test_webhook(), indent=2))
    except ZaloError as e:
        sys.exit(f"Zalo error: {e}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
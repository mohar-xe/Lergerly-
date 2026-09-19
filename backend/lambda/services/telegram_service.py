"""
Ledgerly - Telegram Bot API Service (Demo: Top 5 + English)
Handles Telegram webhook parsing, file download (voice), and message sending.
Mirrors WhatsAppService interface for handler reuse.
"""

import os
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple


class TelegramError(Exception):
    pass


class TelegramUnavailableError(TelegramError):
    pass


class TelegramValidationError(TelegramError):
    pass


TELEGRAM_API_BASE = "https://api.telegram.org"


def is_transient_telegram_error(exc: Exception) -> bool:
    if isinstance(exc, TelegramValidationError):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code <= 599
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    return False


class TelegramService:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        bot_username: Optional[str] = None,
        verify_token: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.bot_username = bot_username or os.environ.get("TELEGRAM_BOT_USERNAME", "")
        self.verify_token = verify_token or os.environ.get("TELEGRAM_VERIFY_TOKEN", "")

    def _api_url(self, method: str) -> str:
        if not self.bot_token:
            raise TelegramUnavailableError("TELEGRAM_BOT_TOKEN is not configured.")
        return f"{TELEGRAM_API_BASE}/bot{self.bot_token}/{method}"

    # ---------------------------------------------------------------------
    # 1. Verify webhook (secret token header or query)
    # ---------------------------------------------------------------------
    def verify_webhook(self, headers: Dict[str, Any], query_params: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Validates Telegram webhook via X-Telegram-Bot-Api-Secret-Token header.
        If TELEGRAM_VERIFY_TOKEN not set, skip verification (allow all).
        """
        if not self.verify_token:
            return True, "ok"
        # Check header (case-insensitive)
        secret = ""
        if headers:
            for k, v in headers.items():
                if k.lower() == "x-telegram-bot-api-secret-token" and v:
                    secret = str(v).strip()
                    break
        # Also allow query param ?secret_token= for testing
        if not secret and query_params:
            secret = str(query_params.get("secret_token") or query_params.get("verify_token") or "").strip()
        if secret != self.verify_token:
            return False, "Invalid Telegram secret token"
        return True, "ok"

    def verify_signature(self, raw_body: str, signature_header: str) -> bool:
        # Telegram uses secret token header, not HMAC; keep for parity
        if not self.verify_token:
            return True
        if not signature_header:
            return False
        return signature_header.strip() == self.verify_token

    # ---------------------------------------------------------------------
    # 2. Parse inbound webhook (Telegram Update)
    # ---------------------------------------------------------------------
    def parse_webhook(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts normalized messages from Telegram Update.
        Returns list of dicts: {from, chat_id, message_id, type, text, audio_id, mime_type, timestamp}
        Supports text and voice/audio.
        """
        messages: List[Dict[str, Any]] = []
        if not isinstance(body, dict):
            return messages
        # Telegram update contains "message" or "edited_message" or "channel_post"
        for key in ("message", "edited_message", "channel_post"):
            msg_obj = body.get(key)
            if not isinstance(msg_obj, dict):
                continue
            chat = msg_obj.get("chat", {}) if isinstance(msg_obj.get("chat"), dict) else {}
            chat_id = str(chat.get("id", "")) if chat else ""
            from_user = msg_obj.get("from", {}) if isinstance(msg_obj.get("from"), dict) else {}
            from_id = str(from_user.get("id", "") or chat_id)
            msg_id = str(msg_obj.get("message_id", ""))
            date_ts = str(msg_obj.get("date", ""))
            # Determine type
            if "text" in msg_obj and isinstance(msg_obj.get("text"), str):
                messages.append({
                    "from": chat_id,  # Telegram chat_id is the recipient for reply
                    "chat_id": chat_id,
                    "from_user_id": from_id,
                    "message_id": msg_id,
                    "type": "text",
                    "text": msg_obj.get("text", ""),
                    "timestamp": date_ts,
                    "raw": msg_obj,
                })
            elif "voice" in msg_obj and isinstance(msg_obj.get("voice"), dict):
                voice = msg_obj.get("voice", {})
                file_id = voice.get("file_id", "")
                mime = voice.get("mime_type", "audio/ogg")
                # Telegram voice is ogg/opus
                messages.append({
                    "from": chat_id,
                    "chat_id": chat_id,
                    "from_user_id": from_id,
                    "message_id": msg_id,
                    "type": "voice",
                    "audio_id": file_id,
                    "mime_type": mime or "audio/ogg",
                    "timestamp": date_ts,
                    "raw": msg_obj,
                })
            elif "audio" in msg_obj and isinstance(msg_obj.get("audio"), dict):
                audio = msg_obj.get("audio", {})
                file_id = audio.get("file_id", "")
                mime = audio.get("mime_type", "audio/mpeg")
                messages.append({
                    "from": chat_id,
                    "chat_id": chat_id,
                    "from_user_id": from_id,
                    "message_id": msg_id,
                    "type": "audio",
                    "audio_id": file_id,
                    "mime_type": mime or "audio/mpeg",
                    "timestamp": date_ts,
                    "raw": msg_obj,
                })
            elif "document" in msg_obj and isinstance(msg_obj.get("document"), dict):
                doc = msg_obj.get("document", {})
                mime = doc.get("mime_type", "")
                # Only handle audio documents (e.g. ogg, mp3)
                if mime and mime.startswith("audio/"):
                    file_id = doc.get("file_id", "")
                    messages.append({
                        "from": chat_id,
                        "chat_id": chat_id,
                        "from_user_id": from_id,
                        "message_id": msg_id,
                        "type": "audio",
                        "audio_id": file_id,
                        "mime_type": mime,
                        "timestamp": date_ts,
                        "raw": msg_obj,
                    })
                else:
                    messages.append({
                        "from": chat_id,
                        "chat_id": chat_id,
                        "message_id": msg_id,
                        "type": "document",
                        "raw": msg_obj,
                        "timestamp": date_ts,
                    })
            else:
                # Unsupported (photo, sticker, etc)
                messages.append({
                    "from": chat_id,
                    "chat_id": chat_id,
                    "message_id": msg_id,
                    "type": "unsupported",
                    "raw": msg_obj,
                    "timestamp": date_ts,
                })
        return messages

    # ---------------------------------------------------------------------
    # 3. Download file (voice/audio) via file_id
    # ---------------------------------------------------------------------
    def download_media(self, file_id: str) -> bytes:
        """
        Downloads file bytes via Telegram getFile + download.
        """
        if not file_id or not file_id.strip():
            raise TelegramValidationError("file_id cannot be empty")
        if not self.bot_token:
            raise TelegramUnavailableError("TELEGRAM_BOT_TOKEN is not configured.")

        # Step 1: getFile
        getfile_url = self._api_url("getFile")
        params = urllib.parse.urlencode({"file_id": file_id.strip()}).encode()
        req = urllib.request.Request(getfile_url, data=params, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise TelegramUnavailableError(f"Telegram getFile failed [{e.code}]: {body}")
        except urllib.error.URLError as e:
            raise TelegramUnavailableError(f"Telegram connection failed: {str(e)}")
        except Exception as e:
            raise TelegramUnavailableError(f"Telegram getFile error: {str(e)}")

        if not data.get("ok"):
            raise TelegramUnavailableError(f"Telegram getFile not ok: {data}")
        file_path = data.get("result", {}).get("file_path", "")
        if not file_path:
            raise TelegramUnavailableError(f"Telegram getFile no file_path: {data}")

        # Step 2: download via file path
        file_url = f"{TELEGRAM_API_BASE}/file/bot{self.bot_token}/{file_path}"
        req2 = urllib.request.Request(file_url, method="GET")
        try:
            with urllib.request.urlopen(req2, timeout=20) as resp2:
                file_bytes = resp2.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise TelegramUnavailableError(f"Telegram file download failed [{e.code}]: {body}")
        except urllib.error.URLError as e:
            raise TelegramUnavailableError(f"Telegram file download connection failed: {str(e)}")
        except Exception as e:
            raise TelegramUnavailableError(f"Telegram file download error: {str(e)}")

        if not file_bytes:
            raise TelegramUnavailableError("Downloaded Telegram file is empty.")

        max_bytes = int(os.environ.get("MAX_VOICE_BYTES", str(10 * 1024 * 1024)))
        if len(file_bytes) > max_bytes:
            raise TelegramValidationError(f"Telegram file too large ({len(file_bytes)} bytes > {max_bytes} bytes).")
        return file_bytes

    # ---------------------------------------------------------------------
    # 4. Send text message
    # ---------------------------------------------------------------------
    def send_text(
        self,
        to: str,
        text: str,
        chat_id: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sends Telegram text via sendMessage.
        """
        if not to or not to.strip():
            raise TelegramValidationError("Recipient 'to' cannot be empty")
        if text is None or not str(text).strip():
            raise TelegramValidationError("Message text cannot be empty")
        if not self.bot_token:
            raise TelegramUnavailableError("TELEGRAM_BOT_TOKEN is not configured.")

        target = (chat_id or to).strip()
        url = self._api_url("sendMessage")
        payload = {
            "chat_id": target,
            "text": str(text).strip()[:4096],
            "parse_mode": "Markdown",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        retries = max_retries if max_retries is not None else int(os.environ.get("TELEGRAM_SEND_MAX_RETRIES", "2"))
        base_delay = backoff_base if backoff_base is not None else float(os.environ.get("TELEGRAM_SEND_RETRY_BACKOFF_BASE", "0.2"))

        def _do_send():
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        from services.retry_helper import retry_with_backoff

        try:
            return retry_with_backoff(
                _do_send,
                max_retries=retries,
                base_delay=base_delay,
                max_delay=2.0,
                is_retryable_fn=is_transient_telegram_error,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise TelegramUnavailableError(f"Telegram send failed [{e.code}]: {body}")
        except urllib.error.URLError as e:
            raise TelegramUnavailableError(f"Telegram send connection failed: {str(e)}")
        except (TelegramValidationError, TelegramUnavailableError):
            raise
        except Exception as e:
            raise TelegramUnavailableError(f"Telegram send error: {str(e)}")

    def resolve_shop_id(self, chat_id: str) -> str:
        """
        Resolves shopId from chat_id via TELEGRAM_SHOP_MAP or DEFAULT_SHOP_ID.
        """
        default_shop = os.environ.get("DEFAULT_SHOP_ID", "shop001")
        map_str = os.environ.get("TELEGRAM_SHOP_MAP", os.environ.get("SHOP_PHONE_MAP", ""))
        if map_str:
            try:
                mapping = json.loads(map_str)
                if isinstance(mapping, dict) and chat_id in mapping:
                    return str(mapping[chat_id]).strip() or default_shop
            except Exception:
                pass
        # Also support TELEGRAM mapping via phone map style
        return default_shop

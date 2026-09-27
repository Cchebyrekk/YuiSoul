# -*- coding: utf-8 -*-
"""
Telegram-бот Юи на голом Bot API (requests, long polling) — без сторонних библиотек.

Входящие сообщения владельца кладутся в общую input_queue как ("telegram", текст, metadata):
  metadata = {"timestamp", "sent_at", "chat_id", "kind": "text"|"voice"|"photo", "images": [data_url]}
Голосовые распознаёт переданная функция transcribe(bytes) -> (text, language) (Whisper из STTManager),
фото уходят картинкой в ход (vision). Всё от других пользователей игнорируется.
"""
import base64
import contextlib
import io
import json
import os
import threading
import time
import uuid
from typing import Callable, Optional

import requests
from PIL import Image

from scripts.config import (TELEGRAM_BATCH_WINDOW, TELEGRAM_GUEST_MAX_PER_HOUR, TELEGRAM_GUESTS_ENABLED,
                            TELEGRAM_MAX_MESSAGE, TELEGRAM_NOTIFY_OWNER_ABOUT_GUESTS, TELEGRAM_POLL_TIMEOUT)

API_URL = "https://api.telegram.org/bot{token}/{method}"
FILE_URL = "https://api.telegram.org/file/bot{token}/{path}"
PHOTO_MAX_SIDE = 1600  # из размеров, которые отдаёт Telegram, берём самый крупный не больше этого

# Реакции, которые бот может поставить (ReactionTypeEmoji в Bot API); другие Telegram отклоняет
REACTION_EMOJI = ("👍 👎 ❤ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳 ❤‍🔥 🌚 🌭 💯 🤣 ⚡ 🍌 🏆 💔 "
                  "🤨 😐 🍓 🍾 💋 🖕 😈 😴 😭 🤓 👻 👨‍💻 👀 🎃 🙈 😇 😨 🤝 ✍ 🤗 🫡 🎅 🎄 ☃ 💅 🤪 🗿 🆒 💘 🙉 🦄 😘 💊 🙊 "
                  "😎 👾 🤷‍♂ 🤷 🤷‍♀ 😡").split()


def split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> list:
    """Режет длинный текст на части не длиннее limit — по абзацам, строкам, пробелам."""
    parts = []
    text = text.strip()
    while len(text) > limit:
        cut = max(text.rfind(sep, 0, limit) for sep in ("\n\n", "\n", " "))
        cut = cut if cut > limit // 2 else limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


class TelegramBot:
    def __init__(self, token: str, owner_id: int, input_queue=None,
                 transcribe: Optional[Callable[[bytes], tuple]] = None,
                 session: Optional[requests.Session] = None):
        self.token = token
        self.owner_id = owner_id
        self.input_queue = input_queue
        self.transcribe = transcribe
        self.http = session or requests.Session()  # своя сессия: long polling не должен занимать пул llama-server
        self._offset = 0
        self._pending = {}  # токен кнопок подтверждения -> {event, granted}
        self.last_message_id = 0  # последнее сообщение владельца — для реакций по умолчанию
        self.recent_stickers = {}  # sticker_id (file_unique_id) -> стикер, присланный владельцем (для сохранения)
        self.batch_window = TELEGRAM_BATCH_WINDOW
        self._batches = {}        # chat_id -> копящаяся пачка сообщений
        self._batch_timers = {}
        self._guest_times = {}    # user_id гостя -> время его сообщений за последний час (лимит)
        self._known_guests = set()
        self._batch_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---------- Bot API ----------
    def _hide_token(self, text) -> str:
        """В сообщениях об ошибках requests бывает URL — а в нём токен бота."""
        return str(text).replace(self.token, "<token>") if self.token else str(text)

    def api(self, method: str, timeout: float = 30, files=None, **params):
        """Вызов метода Bot API; возвращает result или None (ошибка уже напечатана)."""
        try:
            response = self.http.post(API_URL.format(token=self.token, method=method),
                                      data=params if files else None, json=None if files else params,
                                      files=files, timeout=timeout)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[TELEGRAM] {method}: {type(e).__name__}: {self._hide_token(e)}")
            return None
        if not data.get("ok"):
            print(f"[TELEGRAM] {method}: {data.get('description', 'ошибка')}")
            return None
        return data.get("result")

    def download(self, file_id: str) -> Optional[bytes]:
        info = self.api("getFile", file_id=file_id)
        if not info or not info.get("file_path"):
            return None
        try:
            response = self.http.get(FILE_URL.format(token=self.token, path=info["file_path"]), timeout=60)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"[TELEGRAM] Не удалось скачать файл: {type(e).__name__}: {self._hide_token(e)}")
            return None

    def send_text(self, text: str, chat_id: Optional[int] = None, reply_to: int = 0) -> bool:
        chat_id = chat_id or self.owner_id
        if not chat_id or not text.strip():
            return False
        ok = True
        for i, part in enumerate(split_message(text)):
            params = {"chat_id": chat_id, "text": part}
            if reply_to and i == 0:
                params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
            ok = self.api("sendMessage", **params) is not None and ok
        return ok

    def react(self, emoji: str, message_id: int = 0, chat_id: Optional[int] = None) -> str:
        """Реакция на сообщение (по умолчанию — на последнее от владельца)."""
        emoji = (emoji or "").replace("\ufe0f", "").strip()   # ❤️ -> ❤: с вариантным селектором Telegram отклоняет
        if emoji not in REACTION_EMOJI:
            return f"Такую реакцию поставить нельзя. Можно только: {' '.join(REACTION_EMOJI)}"
        message_id = message_id or self.last_message_id
        if not message_id:
            return "Не на что реагировать: сообщений от пользователя ещё не было."
        ok = self.api("setMessageReaction", chat_id=chat_id or self.owner_id, message_id=message_id,
                      reaction=[{"type": "emoji", "emoji": emoji}]) is not None
        return f"Реакция {emoji} поставлена." if ok else "Не удалось поставить реакцию."

    def send_sticker(self, file_id: str, chat_id: Optional[int] = None) -> bool:
        return self.api("sendSticker", chat_id=chat_id or self.owner_id, sticker=file_id) is not None

    def send_voice(self, ogg: bytes, caption: str = "", chat_id: Optional[int] = None, reply_to: int = 0) -> bool:
        chat_id = chat_id or self.owner_id
        if not chat_id:
            return False
        params = {"chat_id": chat_id}
        if caption:
            params["caption"] = caption[:1024]  # лимит подписи в Telegram
        if reply_to:
            params["reply_parameters"] = json.dumps({"message_id": reply_to, "allow_sending_without_reply": True})
        return self.api("sendVoice", timeout=60, files={"voice": ("yui.ogg", ogg, "audio/ogg")}, **params) is not None

    def send_photo(self, jpeg: bytes, caption: str = "", chat_id: Optional[int] = None) -> bool:
        params = {"chat_id": chat_id or self.owner_id}
        if caption:
            params["caption"] = caption[:1024]
        return self.api("sendPhoto", timeout=120, files={"photo": ("screen.jpg", jpeg, "image/jpeg")}, **params) is not None

    def send_document(self, path: str, caption: str = "", chat_id: Optional[int] = None) -> bool:
        params = {"chat_id": chat_id or self.owner_id}
        if caption:
            params["caption"] = caption[:1024]
        with open(path, "rb") as f:
            return self.api("sendDocument", timeout=300, files={"document": (os.path.basename(path), f)}, **params) is not None

    def ask_buttons(self, text: str, buttons: list, on_answer, chat_id: Optional[int] = None) -> Optional[str]:
        """
        Сообщение с кнопками [(подпись, значение)]; нажатие владельца -> on_answer(значение) (в потоке опроса).
        Не ждёт: возвращает токен (для cancel_buttons) или None, если отправить не удалось.
        Кнопки нажимает только человек — поэтому через них подтверждаются опасные действия и факты.
        """
        chat_id = chat_id or self.owner_id
        token = uuid.uuid4().hex[:12]
        labels = dict(buttons)
        markup = {"inline_keyboard": [[{"text": label, "callback_data": f"yui:{token}:{value}"}
                                       for label, value in buttons]]}
        self._pending[token] = {"on_answer": on_answer, "labels": {v: k for k, v in labels.items()}}
        sent = self.api("sendMessage", chat_id=chat_id, text=text, reply_markup=markup)
        if sent is None:
            self._pending.pop(token, None)
            return None
        self._pending[token]["message"] = (chat_id, sent["message_id"], text)
        return token

    def cancel_buttons(self, token: str, note: str):
        """Кнопки больше не нужны (время вышло): убрать их и дописать note."""
        waiter = self._pending.pop(token, None)
        if waiter and waiter.get("message"):
            chat_id, message_id, text = waiter["message"]
            self.api("editMessageText", chat_id=chat_id, message_id=message_id, text=f"{text}\n\n{note}")

    def ask_confirmation(self, text: str, chat_id: Optional[int] = None, timeout: float = 120) -> bool:
        """«Выполнить / Отмена» и ждать нажатия владельца. Нет ответа за timeout — отказ."""
        answered, result = threading.Event(), {}
        token = self.ask_buttons(text, [("✅ Выполнить", "yes"), ("❌ Отмена", "no")],
                                 lambda value: (result.update(value=value), answered.set()), chat_id=chat_id)
        if token is None:
            return False
        if not answered.wait(timeout):
            self.cancel_buttons(token, "⌛ Нет ответа — не выполнено.")
            return False
        return result.get("value") == "yes"

    def _handle_callback(self, query: dict):
        if (query.get("from") or {}).get("id") != self.owner_id:
            return
        parts = (query.get("data") or "").split(":")
        waiter = self._pending.pop(parts[1], None) if len(parts) == 3 and parts[0] == "yui" else None
        label = waiter["labels"].get(parts[2], parts[2]) if waiter else ""
        self.api("answerCallbackQuery", callback_query_id=query["id"], text=label or "Уже неактуально")
        if waiter is None:
            return
        message = query.get("message") or {}
        if message:
            self.api("editMessageText", chat_id=message["chat"]["id"], message_id=message["message_id"],
                     text=f"{message.get('text', '')}\n\n{label}")
        try:
            waiter["on_answer"](parts[2])
        except Exception as e:
            print(f"[TELEGRAM] Ошибка обработки ответа на кнопку: {e}")

    @contextlib.contextmanager
    def chat_action(self, action: str = "typing", chat_id: Optional[int] = None):
        """«Юи печатает…» / «записывает голосовое…», пока идёт ход (Telegram гасит статус через ~5 с)."""
        chat_id = chat_id or self.owner_id
        stop = threading.Event()

        def keep_alive():
            while not stop.is_set():
                self.api("sendChatAction", timeout=10, chat_id=chat_id, action=action)
                stop.wait(4.5)

        threading.Thread(target=keep_alive, daemon=True).start()
        try:
            yield
        finally:
            stop.set()

    # ---------- Приём сообщений ----------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        me = self.api("getMe", timeout=10)
        name = f"@{me['username']}" if me else "(пока нет связи с Telegram)"
        print(f"[TELEGRAM] Бот {name} слушает сообщения."
              + ("" if self.owner_id else " Напиши ему /start — он назовёт твой ID для .env."))

    def stop(self):
        self._running = False

    def _poll_loop(self):
        failures = 0
        while self._running:
            updates = self._get_updates()
            if updates is None:
                failures += 1
                time.sleep(min(60, 5 * failures))  # нет сети — не долбим сервер
                continue
            failures = 0
            for update in updates:
                self._offset = max(self._offset, update.get("update_id", 0) + 1)
                try:
                    self.handle_update(update)
                except Exception as e:
                    print(f"[TELEGRAM] Ошибка обработки сообщения: {type(e).__name__}: {self._hide_token(e)}")

    def _get_updates(self):
        try:
            response = self.http.post(API_URL.format(token=self.token, method="getUpdates"),
                                      json={"offset": self._offset, "timeout": TELEGRAM_POLL_TIMEOUT,
                                            "allowed_updates": ["message", "callback_query"]},
                                      timeout=TELEGRAM_POLL_TIMEOUT + 10)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[TELEGRAM] Нет связи: {type(e).__name__}")
            return None
        if not data.get("ok"):
            print(f"[TELEGRAM] getUpdates: {data.get('description', 'ошибка')}")
            return None
        return data.get("result", [])

    def handle_update(self, update: dict):
        if update.get("callback_query"):
            self._handle_callback(update["callback_query"])
            return
        message = update.get("message")
        if not message:
            return
        user_id = (message.get("from") or {}).get("id")
        chat_id = message["chat"]["id"]
        if not self.owner_id:
            print(f"[TELEGRAM] Сообщение от ID {user_id}. Если это ты — впиши в .env: YUI_TELEGRAM_OWNER_ID={user_id}")
            self.send_text(f"Твой Telegram ID: {user_id}. Впиши его в .env как YUI_TELEGRAM_OWNER_ID "
                           f"и перезапусти Юи.", chat_id=chat_id)
            return
        guest = user_id != self.owner_id
        if guest:
            # Другие люди: только личные чаты, если гости разрешены, и с лимитом сообщений
            if not TELEGRAM_GUESTS_ENABLED or message["chat"].get("type") != "private":
                print(f"[TELEGRAM] Игнорирую сообщение от чужого ID {user_id}.")
                return
            if not self._guest_allowed(user_id):
                print(f"[TELEGRAM] Гость {user_id} превысил лимит сообщений в час — пропускаю.")
                return
            self._notify_owner_about_guest(message.get("from") or {})

        text = (message.get("text") or message.get("caption") or "").strip()
        if text.startswith("/"):
            if text.split()[0] != "/start":
                return
            if not guest:
                self.send_text("Я на связи.", chat_id=chat_id)
                return
            text = "(открыл чат с тобой и нажал «Старт»)"  # гость пришёл впервые — пусть Юи сама поздоровается

        if not guest:
            self.last_message_id = message.get("message_id", self.last_message_id)
        kind, images, extra = "text", [], {}
        if message.get("sticker"):
            kind = "sticker"
            sticker = message["sticker"]
            extra = {"sticker_id": sticker.get("file_unique_id", ""), "emoji": sticker.get("emoji", ""),
                     "set_name": sticker.get("set_name", "")}
            self.recent_stickers[extra["sticker_id"]] = {"file_id": sticker["file_id"], **extra}
            image = self._sticker_data_url(sticker)
            if image:
                images.append(image)
        elif message.get("voice") or message.get("audio"):
            kind = "voice"
            spoken = self._transcribe((message.get("voice") or message.get("audio"))["file_id"])
            if not spoken:
                self.send_text("Не разобрала голосовое, напиши текстом?", chat_id=chat_id)
                return
            text = f"{text}\n{spoken}".strip()
        elif message.get("photo"):
            kind = "photo"
            image = self._photo_data_url(message["photo"])
            if not image:
                self.send_text("Не получилось скачать фото.", chat_id=chat_id)
                return
            images.append(image)
        elif not text:
            self.send_text("Пока понимаю только текст, голосовые и фото.", chat_id=chat_id)
            return

        print(f"\n[TELEGRAM INPUT] ({kind}): {text or extra.get('emoji') or '[фото]'}")
        sender = message.get("from") or {}
        guest_info = {"guest": {"id": user_id, "name": sender.get("first_name", ""),
                                "username": sender.get("username", "")}} if guest else {}
        self._add_to_batch(text, {
            "timestamp": time.time(), "sent_at": message.get("date", time.time()),
            "chat_id": chat_id, "kind": kind, "images": images,
            "message_id": message.get("message_id", 0), **extra, **guest_info,
        }, source="telegram_guest" if guest else "telegram")

    # ---------- Пачки сообщений ----------
    # В мессенджере пишут несколькими сообщениями подряд (и альбомом фото): ждём TELEGRAM_BATCH_WINDOW секунд
    # тишины и отдаём Юи одной репликой — иначе первое сообщение запускало ход, а остальные вклеивались в него.
    # Пачки — отдельно по каждому чату (владелец и гости не смешиваются).
    def _add_to_batch(self, text: str, meta: dict, source: str = "telegram"):
        chat_id = meta["chat_id"]
        with self._batch_lock:
            batch = self._batches.get(chat_id)
            if batch is None:
                self._batches[chat_id] = {"texts": [text] if text else [], **meta, "source": source,
                                          "message_ids": [meta["message_id"]]}
            else:
                if text:
                    batch["texts"].append(text)
                batch["images"] = batch["images"] + meta["images"]
                batch["message_ids"].append(meta["message_id"])
                batch["message_id"] = meta["message_id"]
                # Голосом — если хоть одно голосовое; стикер помним последний
                kind = "voice" if "voice" in (batch["kind"], meta["kind"]) else meta["kind"]
                batch.update({k: v for k, v in meta.items() if k not in ("images", "sent_at", "kind", "timestamp")})
                batch["kind"] = kind
            timer = self._batch_timers.pop(chat_id, None)
            if timer is not None:
                timer.cancel()
            flush_now = self.batch_window <= 0  # без ожидания (тесты)
            if not flush_now:
                timer = threading.Timer(self.batch_window, self._flush_batch, args=(chat_id,))
                timer.daemon = True
                self._batch_timers[chat_id] = timer
                timer.start()
        if flush_now:
            self._flush_batch(chat_id)

    def _flush_batch(self, chat_id=None):
        with self._batch_lock:
            if chat_id is None:  # старый вызов без чата — первый попавшийся
                chat_id = next(iter(self._batches), None)
            batch = self._batches.pop(chat_id, None)
            self._batch_timers.pop(chat_id, None)
        if batch is None or self.input_queue is None:
            return
        text = "\n".join(batch.pop("texts"))
        source = batch.pop("source")
        if len(batch["message_ids"]) > 1:
            print(f"[TELEGRAM] Пачка из {len(batch['message_ids'])} сообщений -> одна реплика.")
        self.input_queue.put((source, text, batch))

    # ---------- Гости: другие люди пишут Юи ----------
    def _guest_allowed(self, user_id: int) -> bool:
        """Не больше TELEGRAM_GUEST_MAX_PER_HOUR сообщений в час от одного гостя."""
        now = time.time()
        recent = [t for t in self._guest_times.get(user_id, []) if now - t < 3600]
        allowed = len(recent) < TELEGRAM_GUEST_MAX_PER_HOUR
        if allowed:
            recent.append(now)
        self._guest_times[user_id] = recent
        return allowed

    def _notify_owner_about_guest(self, user: dict):
        if not TELEGRAM_NOTIFY_OWNER_ABOUT_GUESTS or user.get("id") in self._known_guests:
            return
        self._known_guests.add(user.get("id"))
        who = user.get("first_name", "") + (f" (@{user['username']})" if user.get("username") else "")
        self.send_text(f"📨 Юи пишет новый человек: {who.strip() or user.get('id')}. Про тебя она ему не рассказывает.")

    def _transcribe(self, file_id: str) -> str:
        if self.transcribe is None:
            return ""
        audio = self.download(file_id)
        if not audio:
            return ""
        try:
            text, _ = self.transcribe(audio)
            return text
        except Exception as e:
            print(f"[TELEGRAM] Не удалось распознать голосовое: {type(e).__name__}: {e}")
            return ""

    def _sticker_data_url(self, sticker: dict) -> Optional[str]:
        """Картинка стикера для зрения: у анимированных и видео — превью, у обычных — сам webp (-> JPEG)."""
        animated = sticker.get("is_animated") or sticker.get("is_video")
        file_id = (sticker.get("thumbnail") or {}).get("file_id") if animated else sticker.get("file_id")
        data = self.download(file_id) if file_id else None
        if not data:
            return None
        try:
            with Image.open(io.BytesIO(data)) as img:
                rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))   # прозрачный фон -> белый
            flat.paste(rgba, mask=rgba.split()[3])
            buf = io.BytesIO()
            flat.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"[TELEGRAM] Не удалось открыть стикер: {e}")
            return None

    def _photo_data_url(self, sizes: list) -> Optional[str]:
        fitting = [s for s in sizes if max(s.get("width", 0), s.get("height", 0)) <= PHOTO_MAX_SIDE]
        best = max(fitting or sizes, key=lambda s: s.get("width", 0) * s.get("height", 0))
        data = self.download(best["file_id"])
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii") if data else None

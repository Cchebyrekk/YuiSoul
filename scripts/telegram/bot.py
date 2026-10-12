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
import re
import threading
import time
import uuid
from typing import Callable, Optional

import requests
from PIL import Image

from scripts.agent.rooms import calls_yui
from scripts.tools.frames import SHEET_FRAMES, VIDEO_SHEET_FRAMES, animation_sheet, lottie_sheet
from scripts.config import (TELEGRAM_BATCH_WINDOW, TELEGRAM_GROUP_BATCH_WINDOW, TELEGRAM_GROUP_REQUIRE_OWNER,
                            TELEGRAM_GROUPS_ENABLED, TELEGRAM_GUEST_MAX_PER_HOUR, TELEGRAM_GUESTS_ENABLED,
                            TELEGRAM_MAX_MESSAGE, TELEGRAM_NOTIFY_OWNER_ABOUT_GUESTS, TELEGRAM_POLL_TIMEOUT)

API_URL = "https://api.telegram.org/bot{token}/{method}"
FILE_URL = "https://api.telegram.org/file/bot{token}/{path}"
PHOTO_MAX_SIDE = 1600  # из размеров, которые отдаёт Telegram, берём самый крупный не больше этого
RECENT_MESSAGES_KEPT = 100  # сколько входящих сообщений на чат помнить для цитаты по тексту
STICKER_MIN_SIDE = 512      # превью анимированного стикера (128 px) увеличиваем до этого
ANIMATION_MAX_BYTES = 20 * 1024 * 1024  # больше бот скачать не может (лимит Bot API) — тогда только превью

# Реакции, которые бот может поставить (ReactionTypeEmoji в Bot API); другие Telegram отклоняет
REACTION_EMOJI = ("👍 👎 ❤ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳 ❤‍🔥 🌚 🌭 💯 🤣 ⚡ 🍌 🏆 💔 "
                  "🤨 😐 🍓 🍾 💋 🖕 😈 😴 😭 🤓 👻 👨‍💻 👀 🎃 🙈 😇 😨 🤝 ✍ 🤗 🫡 🎅 🎄 ☃ 💅 🤪 🗿 🆒 💘 🙉 🦄 😘 💊 🙊 "
                  "😎 👾 🤷‍♂ 🤷 🤷‍♀ 😡").split()


def _jpeg_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
        self.recent_messages = {}  # chat_id -> последние входящие [(message_id, текст)] — для цитаты по тексту
        self.batch_window = TELEGRAM_BATCH_WINDOW
        self._batches = {}        # chat_id -> копящаяся пачка сообщений
        self._batch_timers = {}
        self._guest_times = {}    # user_id гостя -> время его сообщений за последний час (лимит)
        self._known_guests = set()
        self._groups_checked = {}  # chat_id группы -> можно ли там говорить (есть ли в ней владелец)
        self.me = None             # сам бот (getMe): id и username — чтобы понять, что в группе обращаются к Юи
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

    def remember_message(self, chat_id: int, message_id: int, text: str):
        if message_id and text:
            recent = self.recent_messages.setdefault(chat_id, [])
            recent.append((message_id, text))
            del recent[:-RECENT_MESSAGES_KEPT]

    def find_message(self, chat_id: Optional[int], quote: str) -> int:
        """message_id недавнего сообщения в чате, где был этот текст (сначала точное совпадение, потом вхождение); 0 — нет."""
        def norm(s):
            return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())
        quote = norm(quote or "")
        recent = list(reversed(self.recent_messages.get(chat_id or self.owner_id, [])))
        if not quote:
            return 0
        for message_id, text in recent:
            if norm(text) == quote:
                return message_id
        for message_id, text in recent:
            text = norm(text)
            if len(quote) >= 3 and (quote in text or (len(text) >= 3 and text in quote)):
                return message_id
        return 0

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
        me = self.me = self.api("getMe", timeout=10)  # до опроса: в группе по нему узнаём обращения к Юи
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
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
        except requests.ReadTimeout:
            return []  # долгий опрос не дождался ответа — обычное дело при плохой сети, просто спрашиваем снова
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
        group = message["chat"].get("type") in ("group", "supergroup")
        guest = user_id != self.owner_id and not group
        if group:
            # Группа: только если включено и ты там есть (иначе любой мог бы добавить Юи к себе)
            if not TELEGRAM_GROUPS_ENABLED or (message.get("from") or {}).get("is_bot") or \
                    not self._group_allowed(chat_id, user_id):
                return
        elif guest:
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
            command = text.split()[0]
            if command in ("/good", "/bad") and not guest and not group:
                # Оценка последнего обмена для набора диалогов (scripts/agent/dataset.py) — ставит основной цикл
                if self.input_queue is not None:
                    self.input_queue.put(("rate", command[1:], {"chat_id": chat_id}))
                return
            if group or command != "/start":
                return
            if not guest:
                self.send_text("Я на связи.", chat_id=chat_id)
                return
            text = "(открыл чат с тобой и нажал «Старт»)"  # гость пришёл впервые — пусть Юи сама поздоровается

        if not guest and not group:
            self.last_message_id = message.get("message_id", self.last_message_id)
        self.remember_message(chat_id, message.get("message_id", 0), text)
        kind, images, extra = "text", [], {}
        # GIF в Telegram — это animation (mp4), видео и кружки — video/video_note: Юи видит их превью-кадр
        moving = message.get("animation") or message.get("video") or message.get("video_note")
        if message.get("sticker"):
            kind = "sticker"
            sticker = message["sticker"]
            extra = {"sticker_id": sticker.get("file_unique_id", ""), "emoji": sticker.get("emoji", ""),
                     "set_name": sticker.get("set_name", ""),
                     "animated": bool(sticker.get("is_animated") or sticker.get("is_video"))}
            self.recent_stickers[extra["sticker_id"]] = {"file_id": sticker["file_id"], **extra}
            image, extra["frames"] = self._sticker_data_url(sticker)
            print(f"[TELEGRAM] Стикер {extra['emoji']}: " + (
                f"раскадровка из {extra['frames']} кадров." if image and extra["frames"] else
                "картинка для зрения есть." if image else "картинку получить не удалось — Юи знает только эмодзи."))
            if image:
                images.append(image)
        elif moving:
            kind = "gif" if message.get("animation") else "video"
            thumb = (moving.get("thumbnail") or {}).get("file_id")
            image, extra["frames"], speech = self._animation_data_url(
                moving.get("file_id"), thumb, moving.get("file_size", 0),
                frames=SHEET_FRAMES if kind == "gif" else VIDEO_SHEET_FRAMES, with_speech=kind == "video")
            if speech:
                text = f"{text}\n[в видео говорят: {speech}]".strip()
            if image:
                images.append(image)
        elif message.get("voice") or message.get("audio"):
            kind = "voice"
            spoken = self._transcribe((message.get("voice") or message.get("audio"))["file_id"])
            if not spoken and not group:
                self.send_text("Не разобрала голосовое, напиши текстом?", chat_id=chat_id)
                return
            text = f"{text}\n{spoken or '[голосовое, не разобрала]'}".strip()
        elif message.get("photo"):
            kind = "photo"
            image = self._photo_data_url(message["photo"])
            if not image and not group:
                self.send_text("Не получилось скачать фото.", chat_id=chat_id)
                return
            images += [image] if image else []
        elif (message.get("document") or {}).get("mime_type", "").startswith("image/"):
            kind = "photo"  # картинка, присланная файлом
            image = self._image_data_url(message["document"]["file_id"])
            if not image and not group:
                self.send_text("Не получилось открыть картинку.", chat_id=chat_id)
                return
            images += [image] if image else []
        elif not text:
            if not group:  # в группе на всё подряд не отвечаем
                self.send_text("Пока понимаю только текст, голосовые, фото, стикеры и гифки.", chat_id=chat_id)
            return

        if group:
            self._add_group_message(message, text, kind, images, extra)
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

    # ---------- Групповые чаты ----------
    def _group_allowed(self, chat_id: int, sender_id: int) -> bool:
        """Группа, где есть владелец (проверяется один раз через getChatMember и запоминается)."""
        if not TELEGRAM_GROUP_REQUIRE_OWNER:
            return True
        if chat_id not in self._groups_checked:
            if sender_id == self.owner_id:
                allowed = True
            else:
                member = self.api("getChatMember", timeout=10, chat_id=chat_id, user_id=self.owner_id)
                allowed = bool(member) and member.get("status") in ("creator", "administrator", "member", "restricted")
            if not allowed:
                print(f"[TELEGRAM] Группа {chat_id}: тебя там нет — Юи в ней молчит.")
            self._groups_checked[chat_id] = allowed
        return self._groups_checked[chat_id]

    def addresses_me(self, message: dict, text: str) -> bool:
        """В группе обращаются к Юи: @упоминание бота, ответ на её сообщение или её имя."""
        username = (self.me or {}).get("username", "")
        if username and f"@{username.lower()}" in text.lower():
            return True
        replied = ((message.get("reply_to_message") or {}).get("from") or {}).get("id")
        if replied and replied == (self.me or {}).get("id"):
            return True
        return calls_yui(text)

    def _add_group_message(self, message: dict, text: str, kind: str, images: list, extra: dict):
        sender = message.get("from") or {}
        user_id, chat_id = sender.get("id"), message["chat"]["id"]
        name = sender.get("first_name") or sender.get("username") or "кто-то"
        if user_id == self.owner_id:
            name += " (твой человек)"
        what = text or {"sticker": f"[стикер {extra.get('emoji', '')}]", "photo": "[картинка]", "gif": "[гифка]",
                        "video": "[видео]"}.get(kind, "[сообщение]")
        print(f"\n[TELEGRAM ГРУППА {message['chat'].get('title', chat_id)}] {name}: {what}")
        addressed = self.addresses_me(message, text)
        self._add_to_batch(f"{name}: {what}", {
            "timestamp": time.time(), "sent_at": message.get("date", time.time()), "chat_id": chat_id,
            "kind": kind, "images": images, "message_id": message.get("message_id", 0),
            "group": {"id": chat_id, "title": message["chat"].get("title", "")},
            "speakers": [user_id], "addressed_by": [user_id] if addressed else [], **extra,
        }, source="telegram_group")

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
                for key in ("speakers", "addressed_by"):  # группа: кто писал и кто звал Юи — за всю пачку
                    if key in meta:
                        meta = {**meta, key: batch.get(key, []) + meta[key]}
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
            window = max(self.batch_window, TELEGRAM_GROUP_BATCH_WINDOW) if source == "telegram_group" else self.batch_window
            if not flush_now:
                timer = threading.Timer(window, self._flush_batch, args=(chat_id,))
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
        return self._transcribe_data(self.download(file_id))

    def _transcribe_data(self, audio: Optional[bytes]) -> str:
        """Речь из аудио или видео (Whisper сам достаёт звуковую дорожку); нет звука или речи — пусто."""
        if self.transcribe is None or not audio:
            return ""
        try:
            text, _ = self.transcribe(audio)
            return text
        except Exception as e:
            print(f"[TELEGRAM] Не удалось распознать речь: {type(e).__name__}: {e}")
            return ""

    def _sticker_data_url(self, sticker: dict) -> tuple:
        """
        Картинка стикера для зрения -> (data URL или None, кадров в раскадровке). Обычный — сам webp; видео-стикер
        (webm) — раскадровка из 4 кадров; анимированный .tgs (Lottie, отрисовать нечем) — превью, увеличенное:
        превью всего 128 px, и надпись на нём модель не читала.
        """
        thumb = (sticker.get("thumbnail") or {}).get("file_id")
        if sticker.get("is_video"):
            return self._animation_data_url(sticker["file_id"], thumb, sticker.get("file_size", 0))[:2]
        if sticker.get("is_animated"):
            data = self.download(sticker["file_id"])
            sheet, count = lottie_sheet(data) if data else (None, 0)
            if sheet is not None:
                return _jpeg_data_url(sheet), count
            return (self._image_data_url(thumb, min_side=STICKER_MIN_SIDE) if thumb else None), 0
        return self._image_data_url(sticker["file_id"]), 0

    def _animation_data_url(self, file_id: str, thumb_id: Optional[str], size: int = 0,
                            frames: int = SHEET_FRAMES, with_speech: bool = False) -> tuple:
        """
        Гифка/видео/видео-стикер -> (раскадровка, число кадров, речь из звуковой дорожки); не вышло — (превью, 0, "").
        Речь — только если with_speech (видео и кружки: там главное часто то, что говорят).
        """
        data = self.download(file_id) if file_id and (size or 0) <= ANIMATION_MAX_BYTES else None
        speech = self._transcribe_data(data) if data and with_speech else ""
        sheet, count = animation_sheet(data, frames) if data else (None, 0)
        if sheet is not None:
            return _jpeg_data_url(sheet), count, speech
        return (self._image_data_url(thumb_id, min_side=STICKER_MIN_SIDE) if thumb_id else None), 0, speech

    def _image_data_url(self, file_id: str, min_side: int = 0) -> Optional[str]:
        """Файл-картинка из Telegram (webp, png, jpeg, первый кадр gif) -> JPEG data URL; прозрачность -> белый фон."""
        data = self.download(file_id)
        if not data:
            return None
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.seek(0)
                rgba = img.convert("RGBA")
            rgba.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
            if min_side and max(rgba.size) < min_side:  # крошечное превью — увеличить, чтобы разобрать надпись
                scale = min_side / max(rgba.size)
                rgba = rgba.resize((round(rgba.width * scale), round(rgba.height * scale)), Image.LANCZOS)
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[3])
            return _jpeg_data_url(flat)
        except Exception as e:
            print(f"[TELEGRAM] Не удалось открыть картинку: {e}")
            return None

    def _photo_data_url(self, sizes: list) -> Optional[str]:
        fitting = [s for s in sizes if max(s.get("width", 0), s.get("height", 0)) <= PHOTO_MAX_SIDE]
        best = max(fitting or sizes, key=lambda s: s.get("width", 0) * s.get("height", 0))
        data = self.download(best["file_id"])
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii") if data else None

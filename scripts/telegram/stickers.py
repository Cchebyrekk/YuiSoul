# -*- coding: utf-8 -*-
"""
Стикеры, которые Юи сохранила себе из присланных пользователем (как у kuni: сохраняет понравившиеся и шлёт сама).
memory/telegram_stickers.json — [{"sticker_id", "file_id", "emoji", "set_name", "note", "saved_at"}].
file_id привязан к боту: отправить стикер можно, только если бот его уже видел.
"""
import datetime
import json
import os
import threading

from scripts.config import MEMORY_DIR

STICKERS_PATH = os.path.join(MEMORY_DIR, "telegram_stickers.json")


class StickerGallery:
    def __init__(self, path: str = STICKERS_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> list:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _write(self, stickers: list):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(stickers, f, ensure_ascii=False, indent=1)

    def save(self, sticker: dict, note: str = "") -> str:
        with self._lock:
            stickers = [s for s in self._load() if s["sticker_id"] != sticker["sticker_id"]]
            stickers.append({**sticker, "note": note.strip(),
                             "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
            self._write(stickers)
        return f"Стикер {sticker.get('emoji', '')} сохранён ({len(stickers)} в коллекции)."

    def get(self, sticker_id: str):
        return next((s for s in self._load() if s["sticker_id"] == sticker_id), None)

    def listing(self) -> str:
        stickers = self._load()
        if not stickers:
            return "Коллекция пуста: сохраняй понравившиеся стикеры, которые присылает пользователь (action=save)."
        return "Твои стикеры:\n" + "\n".join(
            f"- sticker_id={s['sticker_id']} {s.get('emoji', '')}"
            f"{' — ' + s['note'] if s.get('note') else ''}{' (набор ' + s['set_name'] + ')' if s.get('set_name') else ''}"
            for s in stickers)


def sticker_handler(gallery: StickerGallery, telegram, action: str, sticker_id: str = "", note: str = "",
                    chat_id=None) -> str:
    if telegram is None:
        return "Telegram не подключён."
    action = (action or "").strip().lower()
    sticker_id = (sticker_id or "").strip()
    if action == "list":
        return gallery.listing()
    if action == "save":
        sticker = telegram.recent_stickers.get(sticker_id)
        if sticker is None:
            return "Такого стикера нет среди присланных с последнего запуска — сохранить можно только их."
        return gallery.save(sticker, note)
    if action == "send":
        sticker = gallery.get(sticker_id) or telegram.recent_stickers.get(sticker_id)
        if sticker is None:
            return "Нет такого стикера. Посмотри свою коллекцию: action=list."
        return ("Стикер отправлен." if telegram.send_sticker(sticker["file_id"], chat_id=chat_id)
                else "Не удалось отправить стикер.")
    return "Неизвестное действие: нужно list, save или send."

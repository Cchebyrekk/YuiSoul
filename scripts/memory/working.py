# -*- coding: utf-8 -*-
"""
Рабочая память (идея из kuni): обещания, договорённости, дела и напоминания на ближайшие дни — то, что
важно 1-3 дня, но не для долговременной памяти. Всегда видна Юи в хвосте контекста (<things_to_remember>),
устаревшее исчезает само. Напоминание с remind_at будит Юи в нужное время внутренним ходом.

memory/working_memory.json — [{"id", "text", "created", "expires": "YYYY-MM-DD", "remind_at": "YYYY-MM-DD HH:MM"|null,
"reminded": bool}]
"""
import datetime
import json
import os
import queue
import threading
import time
from typing import Callable, Optional

from scripts.config import MEMORY_DIR

WORKING_MEMORY_PATH = os.path.join(MEMORY_DIR, "working_memory.json")
DEFAULT_DAYS = 3
MAX_DAYS = 14
REMINDER_POLL_INTERVAL = 30

REMINDER_PROMPT = (
    "<system_event>Пора напомнить пользователю: «{text}» (напоминание на {remind_at}). {where}\n"
    "Напомни коротко и по-своему: у компьютера — speak_aloud, если он, похоже, отошёл — send_telegram. "
    "Когда напомнила — убери запись (working_memory action=done, id={id}), если дело на этом закончено, "
    "и вызови task_complete.</system_event>"
)

_DATE_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d")


def parse_time(value: str) -> Optional[datetime.datetime]:
    value = (value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class WorkingMemory:
    def __init__(self, path: str = WORKING_MEMORY_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _load(self, now: datetime.datetime) -> list:
        try:
            with open(self.path, encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, ValueError):
            return []
        today = now.strftime("%Y-%m-%d")
        # Устаревшее уходит само; несработавшее напоминание живёт, пока не сработает
        return [i for i in items if i.get("expires", today) >= today or (i.get("remind_at") and not i.get("reminded"))]

    def _write(self, items: list):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)

    def add(self, text: str, days: int = DEFAULT_DAYS, remind_at: str = "", now: datetime.datetime = None) -> str:
        now = now or datetime.datetime.now()
        text = (text or "").strip()
        if not text:
            return "Ошибка: пустая запись."
        when = None
        if remind_at:
            when = parse_time(remind_at)
            if when is None:
                return "Ошибка: не поняла время напоминания — нужно ГГГГ-ММ-ДД ЧЧ:ММ (например 2026-09-28 10:00)."
            if when <= now:
                return f"Ошибка: {when:%Y-%m-%d %H:%M} уже прошло (сейчас {now:%Y-%m-%d %H:%M})."
        days = max(1, min(MAX_DAYS, int(days or DEFAULT_DAYS)))
        expires = max(now + datetime.timedelta(days=days), when or now)
        with self._lock:
            items = self._load(now)
            item = {"id": max((i["id"] for i in items), default=0) + 1, "text": text,
                    "created": now.strftime("%Y-%m-%d %H:%M"), "expires": expires.strftime("%Y-%m-%d"),
                    "remind_at": when.strftime("%Y-%m-%d %H:%M") if when else None, "reminded": False}
            items.append(item)
            self._write(items)
        return f"Записала (id={item['id']})" + (f", напомню {item['remind_at']}." if when else f", помню до {item['expires']}.")

    def done(self, item_id: int, now: datetime.datetime = None) -> str:
        with self._lock:
            items = self._load(now or datetime.datetime.now())
            left = [i for i in items if i["id"] != int(item_id)]
            if len(left) == len(items):
                return f"Нет записи с id={item_id}."
            self._write(left)
        return f"Запись {item_id} убрана."

    def context(self, now: datetime.datetime = None) -> str:
        """Блок для контекста каждого хода (пусто, если помнить нечего)."""
        items = self._load(now or datetime.datetime.now())
        if not items:
            return ""
        lines = [f"- [id={i['id']}, с {i['created']}] {i['text']}"
                 + (f" (напомнить {i['remind_at']})" if i.get("remind_at") and not i.get("reminded") else "")
                 for i in items]
        return "<things_to_remember>\n" + "\n".join(lines) + "\n</things_to_remember>"

    def due(self, now: datetime.datetime = None) -> list:
        """Напоминания, время которых пришло; помечаются сработавшими, чтобы не повторяться."""
        now = now or datetime.datetime.now()
        with self._lock:
            items = self._load(now)
            fired = [i for i in items if i.get("remind_at") and not i.get("reminded")
                     and parse_time(i["remind_at"]) <= now]
            for item in fired:
                item["reminded"] = True
            if fired:
                self._write(items)
        return fired


def working_memory_handler(wm: WorkingMemory, action: str, text: str = "", item_id=None, days=None,
                           remind_at: str = "") -> str:
    action = (action or "").strip().lower()
    if action == "add":
        return wm.add(text, days=days or DEFAULT_DAYS, remind_at=remind_at or "")
    if action == "done":
        try:
            return wm.done(int(item_id))
        except (TypeError, ValueError):
            return "Ошибка: укажи id записи (он есть в <things_to_remember>)."
    if action == "list":
        return wm.context() or "Рабочая память пуста."
    return "Неизвестное действие: нужно add, done или list."


class ReminderWatcher:
    """Фоновый поток: пришло время напоминания — внутренний ход «reminder» в очередь основного цикла."""

    def __init__(self, input_queue: queue.Queue, wm: WorkingMemory, where: Optional[Callable[[], str]] = None,
                 poll_interval: int = REMINDER_POLL_INTERVAL):
        self.input_queue = input_queue
        self.wm = wm
        self.where = where or (lambda: "")
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def is_due(self) -> bool:
        return True  # напоминание не ждёт тишины: время пришло — значит пора

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        print(f"[REMINDER] Запущено: проверка напоминаний каждые {self.poll_interval} с.")

    def stop(self):
        self._stop.set()

    def check(self):
        for item in self.wm.due():
            print(f"\n[REMINDER] Пора напомнить: {item['text']}")
            self.input_queue.put(("reminder", REMINDER_PROMPT.format(where=self.where(), **item),
                                  {"timestamp": time.time(), "item_id": item["id"]}))

    def _run(self):
        while not self._stop.wait(self.poll_interval):
            try:
                self.check()
            except Exception as e:
                print(f"[REMINDER] Ошибка проверки: {e}")

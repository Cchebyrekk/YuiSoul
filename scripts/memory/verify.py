# -*- coding: utf-8 -*-
"""
Подтверждение фактов пользователем (идея из kuni: доверие 1 — «якорь» — ставит только человек, не модель).
Юи спрашивает (verify_fact), пользователь отвечает кнопками — в Telegram или окном на ПК:
  «Верно» -> факт становится якорем (доверие 1: сон его не переписывает и сверяет с ним остальное);
  «Нет»   -> факт удаляется;  «Не важно» -> просто снимается с вопроса.
Сомнительные факты подсказывает сон (reflection.on_doubtful) — список лежит в memory/to_verify.json
[{"path", "text", "added"}], Юи видит его в рефлексии.
Вопрос не блокирует ход: ответ приходит, когда удобно (одновременно — один вопрос).
"""
import ctypes
import datetime
import json
import os
import threading
from typing import Callable, Optional

from scripts.config import MEMORY_DIR
from scripts.memory.manager import MemoryManager, content_stems

TO_VERIFY_PATH = os.path.join(MEMORY_DIR, "to_verify.json")
MAX_DOUBTFUL = 20
PC_ANSWER_TIMEOUT = 600  # окно на ПК закрывается само через 10 минут — это «не важно»

MB_YESNOCANCEL, MB_ICONQUESTION, MB_TOPMOST = 0x3, 0x20, 0x40000
IDYES, IDNO = 6, 7


def find_record(records: list, text: str) -> Optional[int]:
    """Индекс факта, о котором спрашивают: совпадение текста, иначе — больше всего общих слов (не меньше половины)."""
    wanted = (text or "").strip().rstrip(".").lower()
    for i, r in enumerate(records):
        if r["text"].strip().rstrip(".").lower() == wanted:
            return i
    stems = content_stems(text)
    best, best_share = None, 0.5
    for i, r in enumerate(records):
        share = len(stems & content_stems(r["text"])) / max(1, len(stems))
        if share >= best_share:
            best, best_share = i, share
    return best


class FactVerifier:
    def __init__(self, mm: MemoryManager, telegram=None, channel: Callable[[], Optional[dict]] = lambda: None,
                 prefer_telegram: Callable[[], bool] = lambda: False, path: str = TO_VERIFY_PATH):
        self.mm = mm
        self.telegram = telegram
        self.channel = channel                  # reply_channel текущего хода (Telegram — если пишет оттуда)
        self.prefer_telegram = prefer_telegram  # вне хода: последний раз он писал из Telegram?
        self.path = path
        self._lock = threading.Lock()
        self.asking = None                      # (path, text) вопроса, который ждёт ответа

    # ---------- список «под вопросом» (его пополняет сон) ----------
    def _load(self) -> list:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _write(self, items: list):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)

    def add_doubtful(self, path: str, text: str):
        with self._lock:
            items = [i for i in self._load() if not (i["path"] == path and i["text"] == text)]
            items.append({"path": path, "text": text, "added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
            self._write(items[-MAX_DOUBTFUL:])

    def doubtful(self) -> list:
        return self._load()

    def _forget_doubtful(self, path: str, text: str):
        with self._lock:
            self._write([i for i in self._load() if not (i["path"] == path and i["text"] == text)])

    # ---------- вопрос и ответ ----------
    def ask(self, path: str, text: str) -> str:
        records = self.mm.read_fact_records(path)
        index = find_record(records, text)
        if index is None:
            return f"В {path} нет такого факта — проверь формулировку (search_memory FILE:{path})."
        fact = records[index]["text"]
        if records[index]["confidence"] >= 1.0:
            return "Этот факт уже подтверждён пользователем."
        with self._lock:
            if self.asking is not None:
                return "Ты уже задала вопрос и ждёшь ответа — спроси следующий, когда он ответит."
            self.asking = (path, fact)
        question = f"Юи хочет уточнить, правильно ли она помнит:\n«{fact}»"
        use_telegram = self.telegram is not None and (self.channel() is not None or self.prefer_telegram())
        if use_telegram:
            token = self.telegram.ask_buttons(question, [("✅ Верно", "yes"), ("❌ Нет", "no"), ("🤷 Не важно", "skip")],
                                              lambda answer: self.resolve(path, fact, answer))
            if token is None:
                self.asking = None
                return "Не удалось отправить вопрос в Telegram (нет связи?)."
            return f"Спросила в Telegram, верно ли «{fact}». Ответ придёт кнопкой — ждать не нужно."
        threading.Thread(target=self._ask_on_pc, args=(question, path, fact), daemon=True).start()
        return f"Спросила окном на компьютере, верно ли «{fact}». Ответ придёт, когда он нажмёт — ждать не нужно."

    def _ask_on_pc(self, question: str, path: str, fact: str):
        try:
            answer = ctypes.windll.user32.MessageBoxTimeoutW(
                None, question + "\n\nДа — верно, Нет — неверно, Отмена — не важно.", "Юи уточняет",
                MB_YESNOCANCEL | MB_ICONQUESTION | MB_TOPMOST, 0, PC_ANSWER_TIMEOUT * 1000)
        except Exception as e:
            print(f"[VERIFY] Не удалось показать окно: {e}")
            answer = 0
        self.resolve(path, fact, {IDYES: "yes", IDNO: "no"}.get(answer, "skip"))

    def resolve(self, path: str, fact: str, answer: str) -> str:
        """Ответ пользователя: yes — якорь, no — удалить, иначе — снять с вопроса."""
        self.asking = None
        self._forget_doubtful(path, fact)
        records = self.mm.read_fact_records(path)
        index = find_record(records, fact)
        if index is None:
            return "Факта уже нет."
        if answer == "yes":
            records[index]["confidence"] = 1.0
            result = f"подтверждён: «{fact}» — теперь это якорь"
        elif answer == "no":
            records.pop(index)
            result = f"опровергнут и удалён: «{fact}»"
        else:
            print(f"[VERIFY] Пользователю не важно: «{fact}»")
            return "skip"
        self.mm.write_fact_records(path, records)
        print(f"[VERIFY] Факт {result}.")
        return result

# -*- coding: utf-8 -*-
"""
Сбор диалогов для будущего дообучения: каждый обмен владельца с Юи — строкой в datasets/dialogues.jsonl.

{"time": "ГГГГ-ММ-ДД ЧЧ:ММ", "source": "telegram"|"voice"|"text", "rating": null|"good"|"bad",
 "messages": [{"role": "user", "content": ...}, {"role": "assistant", ..., "tool_calls": [...]}, {"role": "tool", ...}]}

В сообщениях — только сам разговор: без «души» и найденной памяти (<injected_context>), без служебных пометок и
картинок. Оценку ставит владелец командой /good или /bad — она относится к последнему обмену. Учить модель стоит
на отмеченных «good»: в сырых логах есть и зацикливания, и неудачные ответы. Гости и группы сюда не пишутся.
"""
import datetime
import json
import os
import re
import threading
from typing import Optional

from scripts.config import BASE_DIR
from scripts.tools.vision import content_text

DATASET_PATH = os.path.join(BASE_DIR, "datasets", "dialogues.jsonl")
_SERVICE_RE = re.compile(r"<(injected_context|system_note|system_warning|system_event|things_to_remember)>.*?</\1>",
                         re.DOTALL)
_lock = threading.Lock()


def clean_turn(messages: list) -> list:
    """Сообщения одного обмена без служебного: пустые после чистки (пометки, заглушки картинок) выбрасываются."""
    cleaned = []
    for message in messages:
        text = _SERVICE_RE.sub("", content_text(message.get("content"))).strip()
        entry = {"role": message["role"], "content": text}
        if message.get("tool_calls"):
            entry["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                   for c in message["tool_calls"]]
        if text or entry.get("tool_calls"):
            cleaned.append(entry)
    return cleaned


def turn_slice(messages: list) -> list:
    """Последний обмен в истории: от последней реплики пользователя с <injected_context> (её ставит каждый ход)."""
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "user" and "<injected_context>" in content_text(messages[i].get("content")):
            return messages[i:]
    return []


def log_turn(messages: list, source: str, path: Optional[str] = None, now: Optional[datetime.datetime] = None) -> bool:
    """Дописать последний обмен в набор. False — записывать нечего (Юи промолчала или обмен пустой)."""
    turn = clean_turn(turn_slice(messages))
    if not any(m["role"] == "assistant" for m in turn) or not any(m["role"] == "user" for m in turn):
        return False
    record = {"time": (now or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"), "source": source,
              "rating": None, "messages": turn}
    path = path or DATASET_PATH
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def rate_last(rating: str, path: Optional[str] = None) -> str:
    """Оценка последнего записанного обмена: good / bad. Возвращает, что получилось (для ответа пользователю)."""
    path = path or DATASET_PATH
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            lines = []
        if not lines:
            return "Оценивать пока нечего: ни одного обмена не записано."
        record = json.loads(lines[-1])
        record["rating"] = rating
        lines[-1] = json.dumps(record, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        good = sum('"rating": "good"' in line for line in lines)
    mark = "👍 хороший" if rating == "good" else "👎 плохой"
    return f"Отметила последний обмен как {mark}. Хороших в наборе: {good} из {len(lines)}."

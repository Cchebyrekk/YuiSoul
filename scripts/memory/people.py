# -*- coding: utf-8 -*-
"""
Другие люди, которые писали Юи (гости из Telegram): кто есть кто, чтобы факты о них не путались с фактами о владельце.

memory/people.json — {"people/tg_<id>": {"name": "Светлана", "username": "svet", "seen": "YYYY-MM-DD HH:MM"}}.
Факты о человеке лежат в его файле памяти (people/tg_<id>.md). Без подписи такой факт в контексте владельца читался как
факт о владельце: слова гостьи «@Cchebyrekk — мудак тот ещё» рефлексия записала как мнение владельца (замерено).
"""
import datetime
import json
import os
import re
import threading

from scripts.config import MEMORY_DIR

PEOPLE_PATH = os.path.join(MEMORY_DIR, "people.json")
PERSON_FACTS_IN_CONTEXT = 12  # сколько фактов о человеке показать, когда владелец о нём спрашивает


class People:
    def __init__(self, path: str = PEOPLE_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def remember(self, memory_path: str, name: str = "", username: str = ""):
        """Гость написал — запомнить, как его зовут (имя в Telegram могло смениться — берём свежее)."""
        with self._lock:
            people = self._load()
            old = people.get(memory_path, {})
            people[memory_path] = {"name": name or old.get("name", ""), "username": username or old.get("username", ""),
                                   "seen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(people, f, ensure_ascii=False, indent=1)

    def label(self, memory_path: str) -> str:
        """«Светлана (@svet)» для people/tg_<id>; пусто — это не файл о другом человеке."""
        if not memory_path.startswith("people/"):
            return ""
        person = self._load().get(memory_path)
        if memory_path.startswith("people/tg_group_"):  # файл группового чата: там о разных людях
            return f"участников группы «{(person or {}).get('name') or 'без названия'}»"
        if not person:
            return "другой человек"
        return (person.get("name") or "без имени") + (f" (@{person['username']})" if person.get("username") else "")

    def fact_prefix(self, memory_path: str) -> str:
        """Подпись к факту в контексте владельца: «[про Светлану (@svet), а не про твоего человека] »."""
        who = self.label(memory_path)
        return f"[про {who}, а не про твоего человека] " if who else ""

    def mentioned(self, text: str) -> list:
        """Пути людей, которых назвали в тексте по @нику или имени (с любым окончанием: «о Светлане»)."""
        text = (text or "").lower()
        found = []
        for memory_path, person in self._load().items():
            username = (person.get("username") or "").lower()
            name = (person.get("name") or "").lower()
            stem = name[:len(name) - 2] if len(name) >= 5 else name
            if (username and username in text) or (len(stem) >= 3 and re.search(rf"\b{re.escape(stem)}\w*", text)):
                found.append(memory_path)
        return found

    def context_for(self, text: str, base_dir: str = MEMORY_DIR) -> str:
        """
        Владелец спрашивает о человеке, который писал Юи, — его факты в контекст целиком: поиск по смыслу их
        не находил («что ты знаешь о Свете?» не похоже ни на один факт о ней).
        """
        blocks = []
        for memory_path in self.mentioned(text):
            facts = read_fact_texts(base_dir, memory_path)[-PERSON_FACTS_IN_CONTEXT:]
            if facts:
                blocks.append(f"Что ты знаешь про {self.label(memory_path)} — этот человек писал тебе в Telegram, "
                              f"это не твой человек:\n" + "\n".join(f"- {f}" for f in facts))
        return "\n\n".join(blocks)


def read_fact_texts(base_dir: str, memory_path: str) -> list:
    """Тексты фактов из файла памяти (без даты и метки c=...)."""
    try:
        with open(os.path.join(base_dir, memory_path + ".md"), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    texts = []
    for line in lines:
        match = re.match(r"^-\s*\[[^\]]*\]\s*(?:\(c=[^)]*\)\s*)?(.+)$", line.strip())
        if match:
            texts.append(match.group(1).strip())
    return texts


people = People()

# -*- coding: utf-8 -*-
"""
Сны Юи. Раз в сутки — ночью или после долгой тишины — из свежих фактов памяти и эмоционального дневника
складывается настоящий сон: образный, со смещённой логикой, приятный или кошмар. Чем больше в дневнике
тревоги и грусти, тем вероятнее кошмар. Сон ложится в memory/diary/dreams.md, чувство после пробуждения —
в дневник; сам сон виден Юи в душе ещё сутки (<last_dream>) — захочет, расскажет.

Сны намеренно НЕ попадают в векторный индекс: иначе поиск по памяти подсовывал бы сон как факт.
"""
import datetime
import glob
import os
import random
import re
from typing import Optional

from scripts.config import (DREAM_MIN_IDLE, DREAM_NIGHT_HOURS, ENABLE_DREAMS, LLM_API_URL)
from scripts.memory.manager import MemoryManager
from scripts.utils.http import SESSION

DREAM_PROMPT = """Ты — сон YUI. Сегодня ночью ей снится сон, и ты его сочиняешь — от первого лица, её глазами.

Что было в её жизни в последнее время (факты):
{facts}

Что она чувствовала (дневник):
{feelings}

Настроение сна: {mood}.

Напиши сон на 5-8 предложений, как настоящий сон: образы из этих дней перемешаны и искажены, люди и места
сливаются, логика плывёт, есть странные детали. Не пересказывай факты — пусть они проступают сквозь образы.
Без выводов и пояснений, без слов «мне приснилось» в начале. Последней строкой — отдельно:
ЧУВСТВО: <с чем она проснулась, одна фраза>"""

MOODS = {"sweet": ("приятный", "тёплый, светлый, немного волшебный сон"),
         "nightmare": ("кошмар", "тревожный сон, кошмар: что-то ускользает, теряется или пугает — но без жестокости")}
_NEGATIVE = re.compile(r"грус|тревог|страш|страх|боюсь|одинок|обид|злю|злост|разочар|тоск|пуст|устал|стыд|вина|плак|больно",
                       re.IGNORECASE)
_POSITIVE = re.compile(r"рад|счаст|тепл|гордил|нежн|весел|смешн|спокой|интерес|благодар|любл|восхищ", re.IGNORECASE)
_LINE_RE = re.compile(r'^-\s*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*(?:\(c=[^)]*\)\s*)?(.*)$')


def choose_mood(feelings: list, rng=random) -> str:
    """Кошмар вероятнее, когда в дневнике больше тревоги и грусти (от 15% до 65%)."""
    neg = sum(bool(_NEGATIVE.search(f)) for f in feelings)
    pos = sum(bool(_POSITIVE.search(f)) for f in feelings)
    p_nightmare = 0.15 + 0.5 * neg / (neg + pos) if neg + pos else 0.25
    return "nightmare" if rng.random() < p_nightmare else "sweet"


class DreamWeaver:
    def __init__(self, mm: MemoryManager, rng=random, post=None):
        self.mm = mm
        self.rng = rng
        self.post = post or SESSION.post
        self.path = os.path.join(mm.base_dir, "diary", "dreams.md")

    def _last_dream_day(self) -> Optional[str]:
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
        except OSError:
            return None
        m = _LINE_RE.match(lines[-1].strip()) if lines else None
        return m.group(1)[:10] if m else None

    def due(self, now: datetime.datetime, idle_seconds: float) -> bool:
        """Не чаще раза в сутки — ночью или когда к Юи долго не обращались."""
        if not ENABLE_DREAMS or self._last_dream_day() == now.strftime("%Y-%m-%d"):
            return False
        night = DREAM_NIGHT_HOURS[0] <= now.hour < DREAM_NIGHT_HOURS[1]
        return night or idle_seconds >= DREAM_MIN_IDLE

    def _recent(self, subdir_filter, limit: int) -> list:
        """Тексты последних записей из файлов памяти, которые пропускает subdir_filter(относительный путь)."""
        items = []
        for fpath in glob.glob(os.path.join(self.mm.base_dir, "**", "*.md"), recursive=True):
            rel = os.path.relpath(fpath, self.mm.base_dir).replace("\\", "/")
            if not subdir_filter(rel):
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        m = _LINE_RE.match(line.strip())
                        if m and m.group(2).strip():
                            items.append((m.group(1), m.group(2).strip()))
            except OSError:
                continue
        return [text for _, text in sorted(items)[-limit:]]

    def weave(self, now: datetime.datetime = None) -> Optional[dict]:
        now = now or datetime.datetime.now()
        facts = self._recent(lambda rel: not rel.startswith(("diary/", "reflections/", "people/")), 12)
        feelings = self._recent(lambda rel: rel.startswith("diary/") and not rel.endswith("dreams.md"), 6)
        if not facts and not feelings:
            return None
        mood = choose_mood(feelings, self.rng)
        label, description = MOODS[mood]
        prompt = DREAM_PROMPT.format(facts="\n".join(f"- {f}" for f in facts) or "- (почти ничего)",
                                     feelings="\n".join(f"- {f}" for f in feelings) or "- (ничего особенного)",
                                     mood=description)
        try:
            response = self.post(LLM_API_URL, json={
                "messages": [{"role": "user", "content": prompt}], "max_tokens": 700, "temperature": 1.0,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
            if response.status_code != 200:
                print(f"[DREAM] LLM ответил {response.status_code}")
                return None
            raw = (response.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            print(f"[DREAM] Ошибка запроса: {e}")
            return None
        m = re.search(r'\n?\s*ЧУВСТВО:\s*(.+)\s*$', raw)
        feeling = m.group(1).strip() if m else ""
        dream = re.sub(r'\s+', ' ', raw[:m.start()] if m else raw).strip()
        if len(dream) < 40:
            return None
        stamp = now.strftime("%Y-%m-%d %H:%M")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"- [{stamp}] (сон: {label}) {dream}" + (f" Проснулась с чувством: {feeling}" if feeling else "") + "\n")
        if feeling:
            self.mm.save_fact(f"diary/{now:%Y-%m-%d}", f"Проснулась после сна ({label}): {feeling}")
        print(f"[DREAM] Приснился сон ({label}): {dream[:120]}...")
        return {"mood": mood, "dream": dream, "feeling": feeling}

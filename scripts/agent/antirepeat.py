# -*- coding: utf-8 -*-
"""
Защита от самоповторов (идея из kuni): новая реплика Юи сравнивается по смыслу (e5) с её последними —
почти то же самое (пересказ своих же слов) отклоняется с просьбой сказать по-другому.
Одной близости e5 мало: короткие реплики на общую тему тоже дают 0.92-0.95 («Готово, папка на рабочем столе» и
«Вот скриншот рабочего стола» — 0.924). Замер: пересказы — 0.949-0.975 и общих основ слов 17-62%, ложные — 0.924-0.950
и 11-27%. Повтор — если очень близко по смыслу (>= REPEAT_SURE) или близко и много общих слов (>= REPEAT_WORD_OVERLAP).
Короткие реплики («Привет! Как дела?», «Хорошо») не проверяются: их повторять нормально.
"""
import re
from collections import deque
from typing import Callable, Optional

import numpy as np

REPEAT_THRESHOLD = 0.92
REPEAT_SURE = 0.965
REPEAT_WORD_OVERLAP = 0.35
REPEAT_MIN_CHARS = 25
REPEAT_WINDOW = 20


def word_stems(text: str) -> set:
    """Основы слов (первые 5 букв) — грубо, но «рабочем»/«рабочего» совпадут."""
    return {w[:5] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) >= 3}


def word_overlap(a: str, b: str) -> float:
    a, b = word_stems(a), word_stems(b)
    return len(a & b) / max(1, len(a | b))


class AntiRepeat:
    def __init__(self, encode: Callable[[list], np.ndarray], threshold: float = REPEAT_THRESHOLD,
                 min_chars: int = REPEAT_MIN_CHARS, window: int = REPEAT_WINDOW):
        self.encode = encode  # список текстов -> нормированные векторы (как у e5 в VectorSearchEngine)
        self.threshold = threshold
        self.min_chars = min_chars
        self._recent = deque(maxlen=window)  # (текст, вектор)

    @classmethod
    def from_model(cls, model, **kwargs) -> "AntiRepeat":
        return cls(lambda texts: model.encode([f"query: {t}" for t in texts], normalize_embeddings=True), **kwargs)

    def check(self, text: str) -> Optional[str]:
        """Похожая прошлая реплика, если новая её повторяет; иначе None."""
        text = (text or "").strip()
        if len(text) < self.min_chars or not self._recent:
            return None
        vector = self.encode([text])[0]
        best_text, best = None, self.threshold
        for old_text, old_vector in self._recent:
            score = float(np.dot(vector, old_vector))
            repeated = score >= REPEAT_SURE or word_overlap(text, old_text) >= REPEAT_WORD_OVERLAP
            if score >= best and repeated:
                best_text, best = old_text, score
        return best_text

    def remember(self, text: str):
        text = (text or "").strip()
        if len(text) >= self.min_chars:
            self._recent.append((text, self.encode([text])[0]))

    def seed(self, texts: list):
        """Последние реплики из прошлой сессии — чтобы не повториться сразу после перезапуска."""
        texts = [t.strip() for t in texts if t and len(t.strip()) >= self.min_chars][-self._recent.maxlen:]
        if texts:
            for text, vector in zip(texts, self.encode(texts)):
                self._recent.append((text, vector))


def repeat_note(similar: str) -> dict:
    """Эфемерная подсказка модели: ты почти дословно повторила сказанное — скажи иначе или промолчи."""
    return {"role": "user", "content": (
        f"<system_note>Ты почти повторила то, что уже говорила: «{similar[:300]}». Не повторяйся — скажи по-другому, "
        f"о новом, или промолчи (stay_silent), если добавить нечего.</system_note>")}

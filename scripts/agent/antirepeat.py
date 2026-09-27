# -*- coding: utf-8 -*-
"""
Защита от самоповторов (идея из kuni): новая реплика Юи сравнивается по смыслу (e5) с её последними —
почти то же самое (пересказ своих же слов) отклоняется с просьбой сказать по-другому.
Замер на e5: повтор и пересказ — 0.92-0.96, похожие по теме, но другие реплики — 0.83-0.87.
Короткие реплики («Привет! Как дела?», «Хорошо») не проверяются: их повторять нормально.
"""
from collections import deque
from typing import Callable, Optional

import numpy as np

REPEAT_THRESHOLD = 0.92
REPEAT_MIN_CHARS = 25
REPEAT_WINDOW = 20


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
            if score >= best:
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

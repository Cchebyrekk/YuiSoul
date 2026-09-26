# -*- coding: utf-8 -*-
"""
Модуль эмоционального канала.
Извлекает эмоции из текста (rule-based), отправляет их по WebSocket
или в лог для будущей интеграции с аватаром.
"""
import json
import threading
from enum import Enum
from typing import Tuple
from scripts.config import EMOTION_WEBSOCKET_URL, EMOTION_EXTRACTION_ENABLED


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"


class EmotionBridge:
    """
    Мост для отправки эмоций аватару.
    Пока работает как заглушка — логирует эмоции в консоль и, опционально,
    отправляет по WebSocket (если реализовать).
    """

    def __init__(self, websocket_url: str = EMOTION_WEBSOCKET_URL, enabled: bool = EMOTION_EXTRACTION_ENABLED):
        self.websocket_url = websocket_url
        self.enabled = enabled
        self._ws = None
        self._lock = threading.Lock()
        # Попытка подключиться к WebSocket, если включено
        if self.enabled:
            try:
                import websocket
                self._ws = websocket.WebSocket()
                self._ws.connect(self.websocket_url, timeout=2)
                print(f"[EMOTION] WebSocket подключён к {self.websocket_url}")
            except Exception as e:
                print(f"[EMOTION] WebSocket недоступен: {e}. Работаю в режиме лога.")
                self.enabled = False

    def send_emotion(self, emotion: str, intensity: float = 0.5, duration: float = 2.0):
        """
        Отправляет эмоцию аватару.
        :param emotion: название эмоции (строка из Emotion enum).
        :param intensity: интенсивность от 0 до 1.
        :param duration: длительность (сек), опционально.
        """
        if not self.enabled:
            print(f"[EMOTION LOG] {emotion.upper()} (intensity={intensity:.2f})")
            return

        try:
            payload = json.dumps({
                "emotion": emotion,
                "intensity": intensity,
                "duration": duration
            })
            if self._ws:
                self._ws.send(payload)
        except Exception as e:
            print(f"[EMOTION] Ошибка отправки: {e}")

    @staticmethod
    def extract_emotion(text: str) -> Tuple[str, float]:
        """
        Извлекает эмоцию из текста с помощью rule-based подхода.
        Возвращает кортеж (emotion, intensity).
        """
        if not text:
            return "neutral", 0.0

        text_lower = text.lower()

        # Маркеры эмоций с интенсивностью (можно настраивать)
        markers = {
            "happy": (["смеюсь", "улыбаюсь", "рад", "отлично", "прекрасно", "здорово", "🥳"], 0.8),
            "sad": (["грустно", "печально", "жаль", "плачу", "тоска", "😢"], 0.7),
            "angry": (["злюсь", "бешусь", "раздражает", "надоело", "😡"], 0.9),
            "surprised": (["удивительно", "офигеть", "неожиданно", "вау", "😮"], 0.6),
            "thinking": (["думаю", "размышляю", "хм", "интересно", "подумать"], 0.5),
            "satisfied": (["отлично", "замечательно", "доволен", "прекрасно", "готово", "выполнено"], 0.7),
            "frustrated": (["бесит", "не получается", "сложно", "тяжело", "раздражает"], 0.8)
        }

        # Собираем все совпадения
        scores = {}
        for emotion, (keywords, base_intensity) in markers.items():
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    score += 1
            if score > 0:
                # Нормализуем интенсивность: чем больше ключевых слов, тем выше интенсивность
                intensity = min(1.0, base_intensity + (score - 1) * 0.1)
                scores[emotion] = intensity

        if not scores:
            return "neutral", 0.0

        # Выбираем эмоцию с максимальной интенсивностью
        best_emotion = max(scores, key=scores.get)
        return best_emotion, scores[best_emotion]

    def close(self):
        """Закрывает WebSocket-соединение."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
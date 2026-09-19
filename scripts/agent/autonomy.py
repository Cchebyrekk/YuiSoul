# -*- coding: utf-8 -*-
"""
Модуль автономных действий (спонтанные мысли при бездействии пользователя).
Периодически проверяет, не ушёл ли пользователь, и если да — генерирует
внутренний монолог, отправляет его в TTS и/или в эмоциональный канал.
"""
import threading
import time
import datetime
import ctypes
import requests
import re

from scripts.config import (
    ENABLE_AUTONOMY,
    AUTONOMY_CHECK_INTERVAL,
    AUTONOMY_IDLE_THRESHOLD,
    LLM_API_URL,
    DEEP_TEMPERATURE,
    DEEP_MAX_TOKENS
)
from scripts.memory.soul import SoulManager
from scripts.agent.emotion import EmotionBridge
from scripts.utils.http import SESSION


# Структура для получения времени бездействия (Windows API)
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_uint),
        ('dwTime', ctypes.c_uint)
    ]


def get_idle_seconds() -> int:
    """Возвращает количество секунд бездействия пользователя (Windows)."""
    try:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
        millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
        return millis // 1000
    except Exception:
        return 0


class AutonomyManager:
    """
    Менеджер автономных спонтанных мыслей.
    Запускается в фоновом потоке.
    """

    def __init__(self,
                 check_interval: int = AUTONOMY_CHECK_INTERVAL,
                 idle_threshold: int = AUTONOMY_IDLE_THRESHOLD,
                 soul_manager: SoulManager = None,
                 emotion_bridge: EmotionBridge = None,
                 agent_is_working: threading.Event = None):
        """
        :param check_interval: интервал проверки бездействия (сек).
        :param idle_threshold: минимальное время бездействия для запуска (сек).
        :param soul_manager: экземпляр SoulManager для генерации пайча.
        :param emotion_bridge: экземпляр EmotionBridge для отправки эмоций.
        :param agent_is_working: событие основного цикла. Если установлено —
            фоновая мысль не отправляет запрос к LLM в этот раз: у llama-server
            ограниченное число слотов, и интерактивный ответ пользователю важнее
            спонтанной мысли. Idle-проверка сама по себе (по клавиатуре/мыши)
            этого не гарантирует — пользователь может активно говорить голосом,
            почти не трогая клавиатуру/мышь.
        """
        self.check_interval = check_interval
        self.idle_threshold = idle_threshold
        self.soul_manager = soul_manager if soul_manager else SoulManager()
        self.emotion_bridge = emotion_bridge if emotion_bridge else EmotionBridge()
        self.agent_is_working = agent_is_working

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Запускает фоновый поток автономии."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._autonomy_loop, daemon=True)
        self._thread.start()
        print("[AUTONOMY] Фоновый процесс инициализирован.")

    def stop(self):
        """Останавливает фоновый поток."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[AUTONOMY] Остановлен.")

    def _autonomy_loop(self):
        """Основной цикл: проверка бездействия и генерация мыслей."""
        while self._running:
            time.sleep(self.check_interval)
            if not self._running:
                break

            idle_sec = get_idle_seconds()
            if idle_sec < self.idle_threshold:
                continue

            if self.agent_is_working is not None and self.agent_is_working.is_set():
                # Основной цикл прямо сейчас занят LLM (может быть голосовой
                # разговор без активности клавиатуры/мыши). Не лезем со
                # спонтанной мыслью — не отнимаем слот у интерактивного ответа.
                continue

            # Пользователь бездействует — генерируем мысль
            idle_min = idle_sec // 60
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Получаем актуальный soul_patch
            soul_patch = self.soul_manager.generate_soul_patch()
            if not soul_patch:
                soul_patch = "<user_profile>Нет данных о пользователе.</user_profile>"

            # Формируем запрос к LLM
            prompt = (
                f"Текущее время: {time_str}. Пользователь неактивен {idle_min} минут.\n"
                f"{soul_patch}\n"
                f"У тебя есть спонтанное желание высказаться или подумать вслух в одиночестве? "
                f"Если да — напиши короткую мысль в тегах <thought> и <output>. Если нет — пиши только NULL."
            )

            try:
                response = SESSION.post(
                    LLM_API_URL,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": DEEP_TEMPERATURE,
                        "stop": ["\n\n", "NULL"]
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    continue

                raw_reply = response.json()["choices"][0]["message"]["content"].strip()
                if not raw_reply or raw_reply.upper() == "NULL":
                    continue

                # Очищаем от тегов
                raw_reply = re.sub(r'<thought>.*?</thought>', '', raw_reply, flags=re.DOTALL).strip()
                raw_reply = raw_reply.replace("<output>", "").replace("</output>", "").strip()

                if len(raw_reply) > 10:
                    print(f"\n[YUI AUTONOMOUS] ({idle_min} мин. тишины) {raw_reply}\n")
                    # Отправляем эмоцию (например, "thinking" или "neutral")
                    emotion, intensity = self.emotion_bridge.extract_emotion(raw_reply)
                    self.emotion_bridge.send_emotion(emotion, intensity)

                    # Опционально: можно озвучить через TTS, но лучше пока просто логировать
                    # self.tts.speak(raw_reply)  # если передать TTS в конструктор

            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                print(f"[AUTONOMY ERROR] {e}")
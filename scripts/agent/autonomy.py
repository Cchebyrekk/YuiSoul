# -*- coding: utf-8 -*-
"""
Модуль автономных действий (Юи думает сама, когда к ней давно не обращались).
Фоновый поток только решает, КОГДА пора: сколько прошло с последнего обращения
к Юи и с прошлой мысли. Сама мысль — это внутренний ход основного цикла
(run_agent_loop с inner="autonomy"): Юи размышляет в несколько шагов, может
искать в интернете, писать в память и говорить вслух через speak_aloud.
"""
import queue
import threading
import time

from scripts.config import (
    AUTONOMY_POLL_INTERVAL,
    AUTONOMY_IDLE_THRESHOLD,
    AUTONOMY_MIN_INTERVAL,
)


# ==================== Время с последнего обращения к Юи ====================
# Простой считается не по клавиатуре/мыши (пользователь может спокойно работать
# за компьютером), а по разговору: основной цикл отмечает активность, когда
# пользователь что-то сказал/написал и когда Юи закончила отвечать.

_last_activity = time.monotonic()


def mark_activity():
    """Отметить обращение к Юи (ввод пользователя или конец её ответа)."""
    global _last_activity
    _last_activity = time.monotonic()


def seconds_since_activity() -> float:
    """Сколько секунд к Юи никто не обращался."""
    return time.monotonic() - _last_activity


AUTONOMY_PROMPT = (
    "<system_event>Пользователь не обращался к тебе уже {minutes} мин. — может, занят своим делом "
    "за компьютером или отошёл. Это время для тебя самой. Подумай про себя о том, что тебе сейчас "
    "интересно: недавний разговор, пользователь, ты сама, что-то из мира.\n"
    "Всё, что ты пишешь текстом, — мысли про себя, их никто не слышит. Можешь думать в несколько шагов:\n"
    "- хочется что-то узнать или проверить — поищи (search_web, read_webpage) или загляни в память (search_memory);\n"
    "- поняла что-то стоящее о пользователе, о себе или о мире — запомни (save_memory);\n"
    "- захочется что-то сказать или спросить вслух — speak_aloud.\n"
    "Не повторяй то, о чём уже думала. Когда мысль исчерпана — вызови task_complete.</system_event>"
)


class AutonomyManager:
    """
    Решает, когда Юи пора подумать самой, и ставит внутренний ход в очередь основного цикла.
    """

    def __init__(self,
                 input_queue: queue.Queue,
                 agent_is_working: threading.Event = None,
                 tts_active: threading.Event = None,
                 idle_threshold: int = AUTONOMY_IDLE_THRESHOLD,
                 min_interval: int = AUTONOMY_MIN_INTERVAL):
        """
        :param input_queue: очередь основного цикла — туда уходит ("autonomy", промпт, метаданные).
        :param agent_is_working: событие основного цикла — пока Юи отвечает, не планируем мысль.
        :param tts_active: событие TTS — пока Юи говорит, тоже ждём.
        :param idle_threshold: сколько секунд без обращений к Юи нужно для мысли.
        :param min_interval: минимум секунд между мыслями.
        """
        self.input_queue = input_queue
        self.agent_is_working = agent_is_working
        self.tts_active = tts_active
        self.idle_threshold = idle_threshold
        self.min_interval = min_interval

        self._last_thought = 0.0
        self._unanswered = 0  # внутренних ходов подряд без ответа пользователя — каждый следующий реже

        self._running = False
        self._thread: threading.Thread | None = None

    def on_user_activity(self):
        """Пользователь заговорил — снова можно думать с обычной частотой."""
        self._unanswered = 0

    def is_due(self) -> bool:
        """Основной цикл перепроверяет это перед запуском: пока задача ждала в очереди, пользователь мог заговорить."""
        return seconds_since_activity() >= self.idle_threshold

    def start(self):
        """Запускает фоновый поток автономии."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._autonomy_loop, daemon=True)
        self._thread.start()
        print(f"[AUTONOMY] Запущено: размышление после {self.idle_threshold} с без обращений, "
              f"не чаще раза в {self.min_interval} с.")

    def stop(self):
        """Останавливает фоновый поток."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[AUTONOMY] Остановлен.")

    def _busy(self) -> bool:
        return ((self.agent_is_working is not None and self.agent_is_working.is_set())
                or (self.tts_active is not None and self.tts_active.is_set()))

    def _autonomy_loop(self):
        """Ждём паузы в разговоре и ставим внутренний ход в очередь."""
        while self._running:
            time.sleep(AUTONOMY_POLL_INTERVAL)
            if not self._running:
                break

            # Без ответа пользователя каждое следующее размышление ждёт дольше: 1x, 2x, 3x... интервала
            interval = self.min_interval * (1 + self._unanswered)
            if not self.is_due() or time.monotonic() - self._last_thought < interval:
                continue
            if self._busy():
                continue

            self._last_thought = time.monotonic()
            self._unanswered += 1
            idle_min = int(seconds_since_activity() // 60)
            self.input_queue.put(("autonomy", AUTONOMY_PROMPT.format(minutes=idle_min),
                                  {"timestamp": time.time(), "idle_min": idle_min}))

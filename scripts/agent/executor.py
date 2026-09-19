# -*- coding: utf-8 -*-
"""
Исполнитель действий (ActionExecutor).
Выполняет список tool_calls, вызывает функции из реестра,
обрабатывает task_complete, обновляет историю сообщений,
отправляет эмоции (опционально) и озвучивает финальный ответ.
"""
import json
import threading
import time
import re
from typing import List, Dict, Any, Optional, Callable

from scripts.tools.registry import build_registry
from scripts.memory.manager import MemoryManager
from scripts.speech.tts import TTSManager

# Опциональный импорт эмоционального моста (если модуль ещё не создан, используем заглушку)
try:
    from scripts.agent.emotion import EmotionBridge
except ImportError:
    # Заглушка, если модуль ещё не создан
    class EmotionBridge:
        @staticmethod
        def send_emotion(emotion: str, intensity: float = 0.5):
            pass

        @staticmethod
        def extract_emotion(text: str) -> tuple:
            return "neutral", 0.5


class ActionExecutor:
    """
    Выполняет действия агента.
    Принимает экземпляры MemoryManager, TTSManager и реестр функций.
    """

    def __init__(self, memory_manager: MemoryManager, tts_manager: TTSManager,
                 registry: Optional[Dict[str, Callable]] = None):
        """
        :param memory_manager: экземпляр MemoryManager для работы с памятью.
        :param tts_manager: экземпляр TTSManager для озвучивания.
        :param registry: готовый реестр функций (если None, строится автоматически).
        """
        self.mm = memory_manager
        self.tts = tts_manager
        self.registry = registry if registry is not None else build_registry(memory_manager)
        self.emotion_bridge = EmotionBridge()
        self._lock = threading.Lock()

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]], messages: List[Dict[str, str]],
                           agent_is_working: threading.Event) -> tuple[List[Dict[str, str]], bool, bool]:
        """
        Выполняет все tool_calls из списка.
        Возвращает:
        - обновлённый список messages (с добавленными сообщениями tool)
        - флаг task_complete (True, если встретился вызов task_complete)
        - флаг stayed_silent (True, если агент решил промолчать через stay_silent)
        """
        if not tool_calls:
            return messages, False, False

        # Добавляем сообщение ассистента с tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": "",  # контент может быть пустым, т.к. вызовы инструментов уже есть
            "tool_calls": tool_calls
        }
        messages.append(assistant_msg)

        task_complete_flag = False
        stayed_silent_flag = False

        for tc in tool_calls:
            agent_is_working.set()  # сигнал, что агент занят выполнением
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]

            try:
                func_args = json.loads(func_args_str) if func_args_str else {}
            except json.JSONDecodeError:
                func_args = {}

            # Специальная обработка task_complete
            if func_name == "task_complete":
                task_complete_flag = True
                # Извлекаем причину, если есть
                reason = func_args.get("reason", "Не указана")
                print(f"\n[SYSTEM] Агент завершил работу. Причина: {reason}")
                # Отправляем финальную эмоцию (например, "satisfied")
                self.emotion_bridge.send_emotion("satisfied", intensity=0.8)
                # Не добавляем результат tool, просто выходим
                break

            # Специальная обработка stay_silent: агент осознанно решил не
            # отвечать сейчас (по образцу #wait/#pause из kuni) — turn
            # завершается без TTS и без видимого пользователю ответа.
            if func_name == "stay_silent":
                stayed_silent_flag = True
                reason = func_args.get("reason", "не указана")
                print(f"\n[SYSTEM] Агент решил промолчать. Причина: {reason}")
                self.emotion_bridge.send_emotion("thinking", intensity=0.4)
                break

            # Вызов функции из реестра
            if func_name in self.registry:
                try:
                    print(f"[ACTION] -> {func_name} | {func_args}")
                    result = self.registry[func_name](**func_args)
                    print(f"[OS LOG] {result}")

                    # Извлекаем эмоцию из результата (если есть)
                    if isinstance(result, str):
                        emotion, intensity = self.emotion_bridge.extract_emotion(result)
                        if emotion != "neutral":
                            self.emotion_bridge.send_emotion(emotion, intensity)

                    # Добавляем результат в историю (формат OpenAI tool)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                        "content": str(result)
                    })

                    # Небольшая задержка для стабильности
                    time.sleep(0.5)

                except Exception as e:
                    error_msg = f"[ERROR] Ошибка при выполнении {func_name}: {e}"
                    print(error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                        "content": f"Ошибка: {e}"
                    })
            else:
                error_msg = f"[ERROR] Неизвестный инструмент: {func_name}"
                print(error_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                    "content": error_msg
                })

            agent_is_working.clear()

        return messages, task_complete_flag, stayed_silent_flag

    def finalize_response(self, final_reply: str, reasoning: str,
                          messages: List[Dict[str, str]],
                          agent_is_working: threading.Event,
                          self_reported_emotion: Optional[tuple] = None) -> bool:
        """
        Обрабатывает финальный ответ агента (без инструментов).
        Сохраняет ответ в историю, озвучивает через TTS, отправляет эмоцию.
        Возвращает True, если агент должен завершить работу (например, если был пустой ответ).

        :param self_reported_emotion: (emotion, intensity), извлечённые парсером из
            тега <emotion> — если модель сама сообщила, что чувствует, это ВСЕГДА
            приоритетнее эвристики по ключевым словам (self-report > guesswork).
        """
        # 1. Сохраняем ответ ассистента в историю (обязательно!)
        messages.append({"role": "assistant", "content": final_reply})

        # 2. Если ответ пустой — завершаем
        if not final_reply and not reasoning:
            print("[SYSTEM] Агент выдал пустой ответ.")
            return True  # завершаем цикл

        # 3. Извлекаем чистый текст для TTS (убираем XML-теги)
        clean_output = re.sub(r'<[^>]+>', '', final_reply).strip()
        clean_output = clean_output.replace('```', '').strip()
        if clean_output:
            print(f"\n[YUI FINAL]: {clean_output}")
            # Эмоция: сначала доверяем самоотчёту модели (тег <emotion>),
            # и только если его не было — эвристике по ключевым словам.
            if self_reported_emotion:
                emotion, intensity = self_reported_emotion
            else:
                emotion, intensity = self.emotion_bridge.extract_emotion(clean_output)
            if emotion != "neutral":
                self.emotion_bridge.send_emotion(emotion, intensity)
            agent_is_working.clear()  # после озвучивания
        else:
            print("[SYSTEM] Агент выдал невалидный ответ (только мысли/код без текста).")
            # Всё равно отправляем эмоцию "thinking"
            self.emotion_bridge.send_emotion("thinking", intensity=0.6)

        return False  # не завершать цикл (если не было task_complete)
    
    def start_streaming_tts(self):
        """Подготовить буфер для потоковой озвучки."""
        self._tts_buffer = ""

    def feed_tts_chunk(self, token: str):
        """
        Принять очередной токен, накопить предложения и отправить в TTS.
        """
        if not token:
            return
        self._tts_buffer += token
        # Проверяем, закончилось ли предложение
        if self._tts_buffer.strip() and self._tts_buffer.strip()[-1] in '.!?':
            clean = re.sub(r'<[^>]+>', '', self._tts_buffer).strip()
            if clean:
                self.tts.speak(clean)
            self._tts_buffer = ""

    def flush_tts_buffer(self):
        """Отправить остаток буфера после завершения стрима."""
        if self._tts_buffer.strip():
            clean = re.sub(r'<[^>]+>', '', self._tts_buffer).strip()
            if clean:
                self.tts.speak(clean)
        self._tts_buffer = ""
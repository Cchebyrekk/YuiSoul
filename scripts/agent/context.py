# -*- coding: utf-8 -*-
"""
Управление контекстом агента: сжатие истории, извлечение фактов из сообщений,
инъекция динамического состояния (время, железо).
"""
import copy
import json
import re
import time
import threading
from typing import List, Dict, Optional

from scripts.config import (
    MAX_CONTEXT_CHARS,
    CONTEXT_TAIL_RATIO,
    LLM_API_URL,
    FACT_EXTRACTION_IDLE_DELAY
)
from scripts.memory.manager import MemoryManager, parse_fact_line
from scripts.agent.prompt import get_dynamic_state
from scripts.utils.http import SESSION
from scripts.tools.vision import content_text, message_chars
from scripts.tools.registry import TOOLS
from scripts.agent.autonomy import seconds_since_activity


def estimate_chars(messages: List[Dict[str, str]]) -> int:
    """
    Приблизительная оценка размера контекста в символах.
    Для простоты используем длину строкового представления
    (картинки считаются фиксированной оценкой, а не длиной base64).
    """
    return sum(message_chars(m) for m in messages)


def compress_context(messages: List[Dict[str, str]], memory_manager: Optional[MemoryManager] = None) -> List[Dict[str, str]]:
    """
    Сжимает контекст, если он превышает MAX_CONTEXT_CHARS.
    Оставляет системное сообщение, добавляет предупреждение о переполнении,
    сохраняет хвост истории (последние ~25% символов) и извлекает факты из удалённой части.
    """
    total_chars = estimate_chars(messages)
    if total_chars < MAX_CONTEXT_CHARS:
        return messages

    print(f"\n[SYSTEM WARNING] Контекст достиг {total_chars} символов. Резка истории.")

    system_msg = messages[0]
    rest_msgs = messages[1:]

    # Определяем лимит для хвоста (доля от общего лимита)
    tail_limit = int(MAX_CONTEXT_CHARS * CONTEXT_TAIL_RATIO)

    # Собираем хвост, начиная с конца, пока не превысим лимит
    tail_msgs = []
    tail_chars = 0
    for msg in reversed(rest_msgs):
        msg_chars = message_chars(msg)
        if tail_chars + msg_chars > tail_limit:
            break
        tail_msgs.insert(0, msg)
        tail_chars += msg_chars

    deleted_msgs = rest_msgs[:len(rest_msgs) - len(tail_msgs)]

    # Если есть удалённые сообщения и передан MemoryManager – извлекаем факты (в фоне)
    if deleted_msgs and memory_manager is not None:
        extract_and_save_facts(deleted_msgs, memory_manager, wait=False)

    # Формируем новый список сообщений
    new_messages = [system_msg]
    new_messages.append({
        "role": "user",
        "content": "<system_warning>Контекст переполнен. Продолжай с текущего состояния.</system_warning>"
    })
    new_messages.extend(tail_msgs)

    # Сохраняем сессию (если нужна) – вызовем извне, чтобы не плодить зависимости
    return new_messages


def extract_and_save_facts(history: List[Dict[str, str]], memory_manager: MemoryManager, wait: bool = False):
    """
    Извлекает факты из истории (обычно из удалённой части) и сохраняет в память.
    Работает в фоновом потоке, если wait=False.
    """
    if not history:
        return

    # Если передана вся история (с системным промптом) — запрос строится ПОВЕРХ неё, с теми же
    # инструментами, что у основного цикла: общий префикс переиспользует KV-кэш llama-server.
    # Отдельный промпт при --parallel 1 мог выбить кэш, и следующий ответ пользователю
    # заново прогонял тысячи токенов истории (замерено: до 16 с).
    prefix = []
    if history[0].get("role") == "system":
        prefix, history = copy.deepcopy(history), history[1:]
        if not history:
            return

    # Делаем снимок последних сообщений (не больше 6)
    history_snapshot = history[-6:] if len(history) > 6 else history.copy()

    def _worker(cancellable: bool = False):
        # Очищаем сообщения от служебных тегов
        cleaned_history = []
        for msg in history_snapshot:
            content = content_text(msg.get("content", ""))
            # Удаляем системные события
            clean_content = re.sub(r'<system_event>.*?</system_event>', '', content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            if clean_content:
                cleaned_history.append({"role": msg["role"], "content": clean_content})

        if not cleaned_history:
            return

        # Получаем дерево памяти для подсказки модели
        tree = memory_manager._get_tree()  # можно сделать публичным методом get_tree()

        # Формируем промпт для извлечения фактов
        extraction_prompt = memory_manager.get_fact_extraction_prompt(cleaned_history, tree)

        try:
            payload = {
                "messages": prefix + [{"role": "user", "content": extraction_prompt}],
                "max_tokens": 1024,
                "temperature": 0.2,
                # Без рассуждений: с ними извлечение держало единственный слот llama-server 35-46 с,
                # и реплика пользователя в это время ждала в очереди. Без них — ~0.5-2 с.
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if prefix:
                payload["tools"] = TOOLS  # инструменты входят в отрисованный промпт — без них префикс не совпадёт
            # Потоком — чтобы фоновое извлечение можно было оборвать: закрытие соединения
            # останавливает генерацию в llama-server и сразу освобождает его единственный слот.
            payload["stream"] = True
            request_start = time.monotonic()
            raw_facts = ""
            with SESSION.post(LLM_API_URL, json=payload, stream=True, timeout=180.0) as response:
                if response.status_code != 200:
                    print(f"[SYSTEM ERROR] Извлечение фактов: сервер вернул {response.status_code}")
                    return
                for line in response.iter_lines():
                    # "<=": часы Windows идут шагами ~15 мс — активность в тот же такт тоже считается
                    if cancellable and seconds_since_activity() <= time.monotonic() - request_start:
                        print("[SYSTEM] Извлечение фактов прервано: пользователь заговорил.")
                        return
                    if not line or not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                        continue
                    try:
                        delta = json.loads(line[6:])["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    raw_facts += delta.get("content") or ""

            raw_facts = re.sub(r'<[^>]+>', '', raw_facts.strip()).strip()

            if not raw_facts or raw_facts.upper() == "NULL":
                return

            # Парсим строки вида [категория/путь] (c=-1) факт — метка confidence
            # опциональна (см. get_fact_extraction_prompt, правило 6: опровержение).
            garbage_keywords = ['анализ:', 'источник:', 'шаг:', 'формат:', 'итоговый', 'вывод:', 'факты:', 'самопроверка:']
            valid_facts = set()
            for line in raw_facts.split('\n'):
                line = line.strip("- *").strip()
                if not line:
                    continue
                if any(line.lower().startswith(kw) for kw in garbage_keywords):
                    continue
                parsed = parse_fact_line(line)
                if parsed:
                    path, confidence, fact = parsed
                    if len(fact) > 5:
                        valid_facts.add((path, fact, confidence))

            saved_count = 0
            for path, fact, confidence in valid_facts:
                memory_manager.save_fact(path, fact, confidence=confidence)
                saved_count += 1
            if saved_count > 0:
                print(f"[SYSTEM] Факты ({saved_count} шт.) распределены.")

        except Exception as e:
            print(f"[SYSTEM ERROR] Извлечение фактов провалено: {e}")

    # Запуск в фоновом потоке или синхронно
    if wait:
        _worker()
        return

    def _deferred():
        # Не с горячего пути: извлечение занимает единственный слот llama-server на ~14 с, и быстрая
        # реплика пользователя ждала бы его в очереди. Ждём паузы в разговоре; если за это время
        # пользователь заговорил — пропускаем (следующий ход снова возьмёт последние сообщения).
        time.sleep(FACT_EXTRACTION_IDLE_DELAY)
        if seconds_since_activity() < FACT_EXTRACTION_IDLE_DELAY:
            return
        _worker(cancellable=True)

    threading.Thread(target=_deferred, daemon=True).start()


def inject_dynamic_context(user_input: str, memory_context: str = "", soul_patch: str = "") -> str:
    """
    Инжектит в пользовательский запрос динамическое состояние (время, железо,
    статус памяти, soul patch) и, опционально, найденный контекст из памяти.
    Всё изменчивое сюда, а не в системный промпт — см. комментарий над
    SYSTEM_PROMPT в prompt.py про стабильность KV-кэша llama.cpp.
    """
    dynamic_state = get_dynamic_state(soul_patch=soul_patch)
    parts = [user_input]

    if memory_context:
        parts.append(
            f"\n\n<injected_context>\n{dynamic_state}\n\n"
            f"ВНИМАНИЕ! Система УЖЕ нашла в памяти ответ. "
            f"ЗАПРЕЩЕНО вызывать search_memory. Используй ТОЛЬКО эти данные:\n{memory_context}\n</injected_context>"
        )
    else:
        parts.append(f"\n\n<injected_context>\n{dynamic_state}\n</injected_context>")

    return "\n".join(parts)
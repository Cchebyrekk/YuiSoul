# -*- coding: utf-8 -*-
"""
Управление контекстом агента: сжатие истории, извлечение фактов из сообщений,
инъекция динамического состояния (время, железо).
"""
import re
import time
import threading
from typing import List, Dict, Any, Optional

from scripts.config import (
    MAX_CONTEXT_CHARS,
    CONTEXT_TAIL_RATIO,
    LLM_TIMEOUT,
    MEMORY_DIR,
    LLM_API_URL
)
from scripts.memory.manager import MemoryManager, parse_fact_line
from scripts.agent.prompt import get_dynamic_state


def estimate_chars(messages: List[Dict[str, str]]) -> int:
    """
    Приблизительная оценка размера контекста в символах.
    Для простоты используем длину строкового представления.
    """
    return len(str(messages))


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
        msg_chars = len(str(msg))
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

    # Делаем снимок последних сообщений (не больше 6)
    history_snapshot = history[-6:] if len(history) > 6 else history.copy()

    def _worker():
        # Очищаем сообщения от служебных тегов
        cleaned_history = []
        for msg in history_snapshot:
            content = msg.get("content", "")
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
            import requests
            response = requests.post(
                LLM_API_URL,
                json={
                    "messages": [{"role": "user", "content": extraction_prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.2,
                },
                timeout=180.0
            )
            if response.status_code != 200:
                print(f"[SYSTEM ERROR] Извлечение фактов: сервер вернул {response.status_code}")
                return

            raw_facts = response.json()["choices"][0]["message"]["content"].strip()
            raw_facts = re.sub(r'<[^>]+>', '', raw_facts).strip()

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
    else:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()


def inject_dynamic_context(user_input: str, memory_context: str = "") -> str:
    """
    Инжектит в пользовательский запрос динамическое состояние (время, железо)
    и, опционально, контекст из памяти.
    """
    dynamic_state = get_dynamic_state()
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
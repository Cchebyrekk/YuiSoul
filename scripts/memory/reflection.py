# -*- coding: utf-8 -*-
"""
Модуль фоновой рефлексии (RAG 2.0).
Периодически анализирует память, строит связи между фактами,
генерирует "внутренние монологи" и сохраняет их в memory/reflections/.
Использует отдельный поток и не блокирует основной цикл агента.
"""
import os
import json
import threading
import time
import datetime
import re
import requests
from typing import Optional

from scripts.config import (
    MEMORY_DIR,
    LLM_API_URL,
    REFLECTION_CHECK_INTERVAL,
    REFLECTION_IDLE_THRESHOLD,
    REFLECTION_MAX_TOKENS,
    REFLECTION_TEMPERATURE,
    REFLECTION_MIN_FACTS,
    REFLECTION_RECENT_FACTS_WINDOW,
)
from scripts.memory.manager import MemoryManager
from scripts.agent.autonomy import get_idle_seconds


class ReflectionManager:
    """
    Управляет фоновой рефлексией: анализирует память, генерирует связи,
    сохраняет рефлексивные заметки.
    """

    def __init__(self, memory_manager: MemoryManager,
                 check_interval: int = REFLECTION_CHECK_INTERVAL,
                 idle_threshold: int = REFLECTION_IDLE_THRESHOLD):
        """
        :param memory_manager: экземпляр MemoryManager для чтения/записи фактов.
        :param check_interval: интервал между проверками (сек), см. REFLECTION_CHECK_INTERVAL в config.py.
        :param idle_threshold: минимальное время бездействия пользователя для запуска (сек).
        """
        self.mm = memory_manager
        self.check_interval = check_interval
        self.idle_threshold = idle_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Запускает фоновый поток рефлексии."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reflection_loop, daemon=True)
        self._thread.start()
        print("[REFLECTION] Фоновый поток рефлексии запущен.")

    def stop(self):
        """Останавливает фоновый поток."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[REFLECTION] Фоновый поток рефлексии остановлен.")

    def _reflection_loop(self):
        """Основной цикл фоновой рефлексии."""
        while self._running:
            time.sleep(self.check_interval)
            if not self._running:
                break

            # Проверяем бездействие пользователя (используем ту же WinAPI
            # проверку, что и AutonomyManager, — не дублируем её здесь).
            idle_sec = get_idle_seconds()
            if idle_sec < self.idle_threshold:
                continue  # пользователь активен — не беспокоим

            # Запускаем один цикл рефлексии
            try:
                self._run_reflection_cycle()
            except Exception as e:
                print(f"[REFLECTION ERROR] {e}")

    def _run_reflection_cycle(self):
        """
        Выполняет один цикл рефлексии:
        1. Получает список недавних фактов (из памяти).
        2. Отправляет запрос LLM для поиска связей.
        3. Сохраняет новые рефлексивные заметки.
        """
        # 1. Собираем факты из всех файлов памяти (кроме папки reflections)
        all_facts = []
        for root, _, files in os.walk(self.mm.base_dir):
            # Игнорируем папку reflections, чтобы не читать свои же заметки
            if "reflections" in root.split(os.sep):
                continue
            for file in files:
                if not file.endswith(".md"):
                    continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # Извлекаем только строки фактов (начинаются с "- [дата]")
                        for line in content.split('\n'):
                            line = line.strip()
                            if line.startswith("- ["):
                                # Убираем дату и маркер
                                fact = re.sub(r'^-\s*\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\]\s*', '', line).strip()
                                if fact:
                                    all_facts.append(fact)
                except Exception:
                    continue

        if len(all_facts) < REFLECTION_MIN_FACTS:
            return  # слишком мало фактов для осмысленной рефлексии

        # Берём последние N фактов (сортировка по времени в файлах не гарантирована, но хотя бы ограничим)
        recent_facts = all_facts[-REFLECTION_RECENT_FACTS_WINDOW:]

        # 2. Запрос к LLM на поиск связей
        prompt = (
            "Ты — внутренний голос YUI. Проанализируй следующие факты из моей памяти "
            "и найди между ними неочевидные связи, противоречия или выводы, которые могут "
            "повлиять на моё поведение. Сформулируй 1-3 рефлексивные заметки в виде коротких "
            "утверждений. Если связи очевидны или их нет, напиши NULL.\n\n"
            "Факты:\n" + "\n".join(f"- {f}" for f in recent_facts) + "\n\n"
            "Рефлексия (только заметки, без пояснений):"
        )

        try:
            response = requests.post(
                LLM_API_URL,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": REFLECTION_MAX_TOKENS,
                    "temperature": REFLECTION_TEMPERATURE,
                    "stop": ["\n\n", "NULL"]
                },
                timeout=60.0
            )
            if response.status_code != 200:
                print(f"[REFLECTION] LLM ответил с ошибкой: {response.status_code}")
                return

            raw = response.json()["choices"][0]["message"]["content"].strip()
            if not raw or raw.upper() == "NULL":
                return

            # Очищаем от лишнего
            lines = [re.sub(r'^[\d\)\-\*]\s*', '', line).strip() for line in raw.split('\n') if line.strip()]
            if not lines:
                return

            # 3. Сохраняем рефлексивные заметки в memory/reflections/
            reflections_dir = os.path.join(self.mm.base_dir, "reflections")
            os.makedirs(reflections_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"reflection_{timestamp}.md"
            filepath = os.path.join(reflections_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Рефлексия от {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                for line in lines:
                    f.write(f"- {line}\n")

            # Обновляем векторный индекс для нового файла (чтобы он участвовал в поиске)
            with open(filepath, "r", encoding="utf-8") as f:
                full_content = f.read().strip()
            if full_content:
                # doc_id – относительный путь без расширения
                rel_path = os.path.relpath(filepath, self.mm.base_dir).replace("\\", "/")
                if rel_path.endswith(".md"):
                    rel_path = rel_path[:-3]
                self.mm.vector_engine.add_document(doc_id=rel_path, text=full_content)

            print(f"[REFLECTION] Сохранена рефлексия: {filename} ({len(lines)} заметок)")

        except requests.exceptions.RequestException as e:
            print(f"[REFLECTION] Ошибка HTTP: {e}")
        except Exception as e:
            print(f"[REFLECTION] Ошибка: {e}")

    def force_reflection(self):
        """Принудительно запускает один цикл рефлексии (для ручного вызова)."""
        if not self._running:
            print("[REFLECTION] Поток не запущен, выполняю синхронно...")
            self._run_reflection_cycle()
        else:
            # Можно запустить в отдельном потоке, но чтобы не блокировать, лучше через очередь
            threading.Thread(target=self._run_reflection_cycle, daemon=True).start()
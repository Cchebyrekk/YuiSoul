# -*- coding: utf-8 -*-
"""
Модуль фоновой рефлексии (RAG 2.0).
Периодически анализирует память, строит связи между фактами,
генерирует "внутренние монологи" и сохраняет их в memory/reflections/.
Использует отдельный поток и не блокирует основной цикл агента.
"""
import os
import json
import random
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
    ENABLE_SLEEP_CONSOLIDATION,
    SLEEP_CONSOLIDATION_EVERY_N_CYCLES,
    SLEEP_CONSOLIDATION_RELATED_FILES,
    SLEEP_CONSOLIDATION_RECENT_BIAS,
    SLEEP_CONSOLIDATION_MAX_TOKENS,
    SLEEP_CONSOLIDATION_TEMPERATURE,
    FACT_CONFIDENCE_ANCHOR_THRESHOLD,
)
from scripts.memory.manager import MemoryManager, parse_fact_line
from scripts.agent.autonomy import get_idle_seconds
from scripts.utils.http import SESSION


# Адаптация sleepConsolidator из kuni под файловую (по темам, не по-записям)
# память YUI. LLM никогда не присваивает anchor-уровень доверия сама —
# rewrite_mutable_lines() на всякий случай обрезает confidence и на своей
# стороне, это лишь явное объяснение для модели, почему так.
SLEEP_CONSOLIDATOR_PROMPT = """Ты — фаза сна YUI. Как человеческий мозг во время сна, ты переупаковываешь
дневные воспоминания: сжимаешь повторы, объединяешь похожие факты, находишь противоречия
и присваиваешь им доверие (confidence).

confidence ∈ [-1..1]: -1 = опровергнуто/ложь (такие факты будут УДАЛЕНЫ), 0 = теория/предположение
(по умолчанию), 1 = подтверждённая истина. ТЕБЕ ЗАПРЕЩЕНО присваивать confidence=1 или выше 0.99
— это может сделать только человек/система вручную, не ты.

Часть фактов помечена как "# ANCHOR" — это неприкосновенная подтверждённая истина, дана
ТОЛЬКО для контекста. Не переписывай её, не противоречь ей, не включай её в свой ответ.

Для остальных (изменяемых) фактов, идущих под заголовками "# путь/к/файлу":
- объединяй дубли и почти-дубли в один факт;
- переписывай растянутые/неясные формулировки короче и по делу;
- если факт противоречит другому факту или ANCHOR — понижай его confidence, вплоть до -1;
- если факт независимо подтверждается несколькими записями — можно немного повысить confidence
  (но не выше 0.99);
- НЕ придумывай фактов, которых не было во входных данных;
- сохраняй фактическое ядро, спекуляции явно помечай как предположение.

Формат ответа — СТРОГО построчно, без пояснений, без преамбул:
[путь/к/файлу] (c=X.XX) Текст факта.

Пример:
[user/pets] (c=0.30) Кот пользователя по кличке Барсик, рыжий.
[system/yui/yui_mood] (c=-1) Пользователь любит собак.

Если для какого-то файла после сжатия ничего не осталось — просто не упоминай его путь в ответе.
Если факт остаётся годным как есть — можешь переписать его почти без изменений с тем же confidence.
"""


class ReflectionManager:
    """
    Управляет фоновой рефлексией: анализирует память, генерирует связи,
    сохраняет рефлексивные заметки.
    """

    def __init__(self, memory_manager: MemoryManager,
                 check_interval: int = REFLECTION_CHECK_INTERVAL,
                 idle_threshold: int = REFLECTION_IDLE_THRESHOLD,
                 agent_is_working: threading.Event = None):
        """
        :param memory_manager: экземпляр MemoryManager для чтения/записи фактов.
        :param check_interval: интервал между проверками (сек), см. REFLECTION_CHECK_INTERVAL в config.py.
        :param idle_threshold: минимальное время бездействия пользователя для запуска (сек).
        :param agent_is_working: событие основного цикла — если установлено, рефлексия
            и сон-консолидация пропускают свой LLM-запрос в этот раз, чтобы не отнимать
            слот у llama-server прямо в момент интерактивного ответа (см. AutonomyManager).
        """
        self.mm = memory_manager
        self.check_interval = check_interval
        self.idle_threshold = idle_threshold
        self.agent_is_working = agent_is_working
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cycle_count = 0

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

            if self.agent_is_working is not None and self.agent_is_working.is_set():
                continue  # основной цикл сейчас держит LLM — не мешаем

            # Запускаем один цикл рефлексии
            try:
                self._run_reflection_cycle()
            except Exception as e:
                print(f"[REFLECTION ERROR] {e}")

            # Раз в N циклов — "сонная" консолидация памяти (RAG 2.0, по
            # образцу sleepingConsolidation из kuni). Отдельная ветка, чтобы
            # лёгкая рефлексия-инсайт срабатывала чаще, а тяжёлая
            # переработка файлов памяти — реже.
            self._cycle_count += 1
            sleep_due = ENABLE_SLEEP_CONSOLIDATION and self._cycle_count % SLEEP_CONSOLIDATION_EVERY_N_CYCLES == 0
            agent_busy_now = self.agent_is_working is not None and self.agent_is_working.is_set()
            if sleep_due and not agent_busy_now:
                try:
                    self._run_sleep_consolidation_cycle()
                except Exception as e:
                    print(f"[REFLECTION] sleep_consolidation error: {e}")

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
            response = SESSION.post(
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

    # ==================== СОН-КОНСОЛИДАЦИЯ (RAG 2.0) ====================

    def _list_topic_paths(self) -> list:
        """Список путей (без .md) всех файлов памяти, кроме служебной папки reflections/."""
        paths = []
        for root, _, files in os.walk(self.mm.base_dir):
            if "reflections" in root.split(os.sep):
                continue
            for file in files:
                if not file.endswith(".md"):
                    continue
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, self.mm.base_dir).replace("\\", "/")[:-3]
                paths.append((rel_path, os.path.getmtime(fpath)))
        return paths

    def _pick_sleep_target(self, candidates: list) -> str:
        """
        Выбирает файл-цель для консолидации: с вероятностью
        SLEEP_CONSOLIDATION_RECENT_BIAS — самый свежий (как человек чаще
        "пересматривает во сне" недавние события), иначе — случайный
        (изредка всплывает что-то старое). По образцу sleepingConsolidation
        из kuni.
        """
        if random.random() < SLEEP_CONSOLIDATION_RECENT_BIAS:
            return max(candidates, key=lambda c: c[1])[0]
        return random.choice(candidates)[0]

    def _run_sleep_consolidation_cycle(self):
        """
        Один цикл сон-консолидации: берём один файл памяти (с уклоном в
        сторону свежих), подмешиваем похожие через векторный поиск,
        просим LLM сжать/объединить/переписать/опровергнуть изменяемую
        (не-anchor) часть и переписываем результат на диск.

        В отличие от kuni (который крутит цикл "пока не кончится время сна"
        по ВСЕЙ памяти за один присест), здесь обрабатывается один целевой
        файл за один вызов: этот метод и так вызывается периодически из
        фонового потока, а держать поток занятым долгим циклом рискованно
        для отзывчивости stop()/join(timeout=5).
        """
        candidates = self._list_topic_paths()
        if not candidates:
            return

        target_path = self._pick_sleep_target(candidates)
        target_anchors, target_mutable = self.mm.read_mutable_and_anchor_lines(target_path)
        if not target_mutable:
            return  # нечего сжимать — либо пусто, либо только anchor-факты

        query_text = "\n".join(text for _, text in target_mutable)
        related = self.mm.vector_engine.search(query_text, top_k=SLEEP_CONSOLIDATION_RELATED_FILES + 1)
        related_paths = [r["id"] for r in related if r["id"] != target_path][:SLEEP_CONSOLIDATION_RELATED_FILES]

        involved = [target_path] + related_paths
        mutable_by_path = {target_path: target_mutable}
        anchor_context_parts = []
        if target_anchors:
            anchor_context_parts.append(
                f"# ANCHOR (неприкосновенно, только контекст) — {target_path}\n"
                + "\n".join(f"(c={c:+.2f}) {t}" for c, t in target_anchors)
            )

        body_parts = [f"# {target_path}\n" + "\n".join(f"(c={c:+.2f}) {t}" for c, t in target_mutable)]
        for path in related_paths:
            anchors, mutable = self.mm.read_mutable_and_anchor_lines(path)
            if anchors:
                anchor_context_parts.append(
                    f"# ANCHOR (неприкосновенно, только контекст) — {path}\n"
                    + "\n".join(f"(c={c:+.2f}) {t}" for c, t in anchors)
                )
            if mutable:
                mutable_by_path[path] = mutable
                body_parts.append(f"# {path}\n" + "\n".join(f"(c={c:+.2f}) {t}" for c, t in mutable))

        prompt_body = ""
        if anchor_context_parts:
            prompt_body += "\n\n".join(anchor_context_parts) + "\n\n---\n\n"
        prompt_body += "\n\n---\n\n".join(body_parts)

        try:
            response = SESSION.post(
                LLM_API_URL,
                json={
                    "messages": [
                        {"role": "system", "content": SLEEP_CONSOLIDATOR_PROMPT},
                        {"role": "user", "content": prompt_body},
                    ],
                    "max_tokens": SLEEP_CONSOLIDATION_MAX_TOKENS,
                    "temperature": SLEEP_CONSOLIDATION_TEMPERATURE,
                },
                timeout=120.0,
            )
            if response.status_code != 200:
                print(f"[REFLECTION] sleep_consolidation: LLM ответил {response.status_code}")
                return
            raw = response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"[REFLECTION] sleep_consolidation: ошибка HTTP: {e}")
            return
        except Exception as e:
            print(f"[REFLECTION] sleep_consolidation: ошибка запроса: {e}")
            return

        new_by_path = {}
        for line in raw.split("\n"):
            line = line.strip("- *").strip()
            if not line:
                continue
            parsed = parse_fact_line(line)
            if not parsed:
                continue
            path, confidence, text = parsed
            if len(text) < 5:
                continue
            new_by_path.setdefault(path, []).append((confidence, text))

        if not new_by_path:
            # Пустой/нераспарсенный ответ — вероятно, сбой модели, а не
            # "все факты оказались мусором". Ничего не трогаем на диске,
            # чтобы одна неудачная генерация не стёрла изменяемые факты
            # у involved-файлов, которые LLM просто не успела упомянуть.
            print("[REFLECTION] sleep_consolidation: LLM не вернула ни одного распознанного факта, файлы не тронуты")
            return

        touched_paths = set(involved) | set(new_by_path.keys())
        for path in touched_paths:
            if path not in mutable_by_path and path not in new_by_path:
                continue  # путь не участвовал (защита от того, что LLM придумала левый path)
            self.mm.rewrite_mutable_lines(path, new_by_path.get(path, []))

        print(f"[REFLECTION] sleep_consolidation: цель={target_path}, похожих={len(related_paths)}, "
              f"переписано файлов={len(touched_paths)}")
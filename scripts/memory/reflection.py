# -*- coding: utf-8 -*-
"""
Модуль фоновой рефлексии (RAG 2.0).
В паузах разговора ставит в очередь основного цикла внутренний ход рефлексии:
Юи думает над свежими фактами в несколько шагов, с инструментами (поиск, память,
speak_aloud) и сама сохраняет выводы через save_memory. Раз в N циклов —
фоновая сон-консолидация памяти.
"""
import datetime
import os
import queue
import random
import threading
import time
import re
from typing import Optional

from scripts.config import (
    FACT_CONFIDENCE_ANCHOR_THRESHOLD,
    FACT_CONFIDENCE_DROP_THRESHOLD,
    LLM_API_URL,
    REFLECTION_POLL_INTERVAL,
    REFLECTION_IDLE_THRESHOLD,
    REFLECTION_MIN_INTERVAL,
    REFLECTION_MIN_FACTS,
    REFLECTION_RECENT_FACTS_WINDOW,
    ENABLE_SLEEP_CONSOLIDATION,
    SLEEP_CONSOLIDATION_EVERY_N_CYCLES,
    SLEEP_CONSOLIDATION_RELATED_FILES,
    SLEEP_CONSOLIDATION_RECENT_BIAS,
    SLEEP_CONSOLIDATION_MAX_TOKENS,
    SLEEP_CONSOLIDATION_TEMPERATURE,
)
from scripts.memory.manager import MemoryManager
from scripts.agent.autonomy import seconds_since_activity
from scripts.utils.http import SESSION


# Сон по образцу sleepConsolidator из kuni, но безопасный для файловой памяти YUI (замерено пробным
# прогоном на копии памяти: прежний вариант склеивал по 8 разных фактов в одну строку, раздавал доверие
# +0.5..+0.95 без оснований, терял факты, сбрасывал даты и за цикл переписывал ещё 3 соседних файла).
# Теперь модель отвечает правками по номерам фактов ОДНОГО файла; всё, что она не упомянула, остаётся
# как было (обрезанный ответ ничего не стирает), а изменения доверия ограничены кодом.
SLEEP_CONSOLIDATOR_PROMPT = """Ты — фаза сна YUI. Как мозг во сне, ты перебираешь воспоминания одного файла памяти:
убираешь повторы, уточняешь формулировки и сверяешь факты между собой.

Факты файла пронумерованы: [N] (c=доверие) текст. Доверие: -1 = ложь, 0 = предположение, 1 = подтверждено
человеком (такие помечены ЯКОРЬ — их не трогай, им не противоречь). Ниже — факты из похожих файлов, только
для сверки: их менять нельзя.

Ответ — ТОЛЬКО правки, по одной в строке, без пояснений:
[N] (c=X.XX) новый текст — переписать факт N (короче, яснее) и/или поменять доверие;
[N,M] (c=X.XX) текст — N и M — это ОДИН И ТОТ ЖЕ факт разными словами: слить в одну строку;
[N] (c=-1) — факт N ложен: противоречит якорю или нескольким надёжным фактам;
[N] (?) — сомневаешься в факте N: его стоит уточнить у пользователя.

Правила:
- НЕ сливай разные факты в одну строку (кот и собака, татуировка и аниме — это разные факты);
- переписывая, не выбрасывай детали: имена, числа, с кем, когда;
- дубль (тот же факт другими словами) — слей через [N,M], а не помечай ложью;
- доверие повышай, только если факт независимо подтверждают другие записи; понижай — если ему что-то противоречит;
- ничего не придумывай; не упомянутые тобой факты остаются как есть — правь только то, что правда нужно;
- нечего править — ответь одним словом: НЕТ."""

SLEEP_MAX_RAISE = 0.2        # на сколько за одну ночь можно поднять доверие
SLEEP_MAX_LOWER = 0.4        # на сколько опустить (кроме явного -1 — «ложь»)
SLEEP_CONF_CEILING = 0.9     # выше сон не поднимает: до 1 — только подтверждение пользователем
SLEEP_MAX_MERGE = 3          # больше фактов в одну строку не сливаем (защита от «простыней»)
SLEEP_MAX_DELETE_SHARE = 1 / 3  # за ночь удаляем не больше трети файла (ошибка модели не выкосит память)

_SLEEP_EDIT_RE = re.compile(r'^\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*(?:\(\s*(\?|c\s*=\s*[+-]?\d+(?:\.\d+)?)\s*\))?\s*(.*)$')


def parse_sleep_edits(raw: str) -> list:
    """Строки ответа сна -> [{"ids": [..], "confidence": float|None, "doubt": bool, "text": str}]."""
    edits = []
    for line in (raw or "").splitlines():
        m = _SLEEP_EDIT_RE.match(line.strip().lstrip("-* ").strip())
        if not m:
            continue
        ids = [int(i) for i in re.findall(r'\d+', m.group(1))]
        tag, text = m.group(2) or "", m.group(3).strip()
        doubt = tag == "?" or text.endswith("(?)")
        confidence = float(re.sub(r'[^\d.+-]', '', tag)) if tag.startswith("c") else None
        edits.append({"ids": ids, "confidence": confidence, "doubt": doubt,
                      "text": re.sub(r'\s*\(\?\)$', '', text).strip()})
    return edits


def apply_sleep_edits(records: list, edits: list) -> tuple:
    """
    Правки сна -> (новые записи, удалённые тексты, сомнительные тексты). Якоря (доверие 1) и всё, что модель
    не упомянула, не меняются; даты сохраняются (у слитого факта — самая ранняя).
    """
    n = len(records)
    result = {i: dict(r) for i, r in enumerate(records, 1)}
    used, deleted, doubtful = set(), [], []
    max_deletes = max(1, int(n * SLEEP_MAX_DELETE_SHARE))
    for edit in edits:
        ids = [i for i in dict.fromkeys(edit["ids"]) if 1 <= i <= n and i not in used
               and records[i - 1]["confidence"] < FACT_CONFIDENCE_ANCHOR_THRESHOLD]
        if not ids or len(ids) != len(set(edit["ids"])) or len(ids) > SLEEP_MAX_MERGE:
            continue  # чужие/повторные номера, якорь или «простыня» — правку целиком пропускаем
        sources = [records[i - 1] for i in ids]
        base = max(s["confidence"] for s in sources)
        if edit["doubt"] and edit["confidence"] is None:
            doubtful.extend(s["text"] for s in sources)
            continue
        if edit["confidence"] is not None and edit["confidence"] <= FACT_CONFIDENCE_DROP_THRESHOLD:
            if len(deleted) + len(ids) > max_deletes:
                continue
            for i in ids:
                deleted.append(result.pop(i)["text"])
            used.update(ids)
            continue
        conf = base if edit["confidence"] is None else edit["confidence"]
        conf = round(max(base - SLEEP_MAX_LOWER, min(conf, base + SLEEP_MAX_RAISE, SLEEP_CONF_CEILING)), 2)
        dates = [s["date"] for s in sources if s["date"]]
        result[ids[0]] = {"date": min(dates) if dates else None, "confidence": conf,
                          "text": edit["text"] or sources[0]["text"]}
        for i in ids[1:]:
            result.pop(i)
        used.update(ids)
        if edit["doubt"]:
            doubtful.append(result[ids[0]]["text"])
    return [result[i] for i in sorted(result)], deleted, doubtful


REFLECTION_PROMPT = (
    "<system_event>Время рефлексии. Вот самые свежие факты из твоей памяти:\n{facts}\n\n"
    "Подумай про себя: какие между ними связи и противоречия, что из этого следует для тебя и для "
    "ваших отношений с пользователем, чего ты не знаешь и что хотела бы уточнить. "
    "Ты сейчас одна: весь твой текст — мысли про себя, пользователь их НЕ слышит. Поэтому думай о нём "
    "в третьем лице и не обращайся к нему в мыслях. Можешь думать в несколько шагов:\n"
    "- стоит что-то проверить или узнать — поищи (search_web, read_webpage) или загляни в память (search_memory);\n"
    "- важные выводы сохрани через save_memory: о себе — в system/yui/yui_character или "
    "system/yui/yui_preferences, о пользователе — в user/..., общие наблюдения — в reflections/reflection_notes; "
    "если факт оказался неверным — сохрани исправление с отрицательным confidence;\n"
    "- захочется что-то сказать или спросить у него — только через инструмент speak_aloud.\n"
    "{doubtful}"
    "Когда закончишь — вызови инструмент task_complete (вызови, а не пиши об этом).</system_event>"
)

DOUBTFUL_NOTE = ("- во сне ты засомневалась в этих фактах; если какой-то правда важен — уточни у пользователя "
                 "(verify_fact, по одному, не больше одного за раз):\n{facts}\n")


class ReflectionManager:
    """
    Решает, когда Юи пора поразмышлять над памятью, и ставит внутренний ход в очередь
    основного цикла (run_agent_loop с inner="reflection"). Раз в N циклов — фоновая
    сон-консолидация памяти (отдельный LLM-запрос без инструментов).
    """

    def __init__(self, memory_manager: MemoryManager,
                 input_queue: queue.Queue,
                 min_interval: int = REFLECTION_MIN_INTERVAL,
                 idle_threshold: int = REFLECTION_IDLE_THRESHOLD,
                 agent_is_working: threading.Event = None):
        """
        :param memory_manager: экземпляр MemoryManager для чтения/записи фактов.
        :param input_queue: очередь основного цикла — туда уходит ("reflection", промпт, метаданные).
        :param min_interval: минимум секунд между циклами рефлексии, см. REFLECTION_MIN_INTERVAL в config.py.
        :param idle_threshold: сколько секунд к Юи не должны обращаться, чтобы начать (сек).
        :param agent_is_working: событие основного цикла — пока оно установлено, не планируем
            рефлексию и не запускаем сон-консолидацию (у llama-server один слот).
        """
        self.mm = memory_manager
        self.input_queue = input_queue
        self.min_interval = min_interval
        self.idle_threshold = idle_threshold
        self._last_cycle = 0.0
        self.agent_is_working = agent_is_working
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cycle_count = 0
        self.on_doubtful = None  # (path, text) — факт, который сон счёл сомнительным (см. FactVerifier)
        self.doubtful = None     # () -> список «под вопросом» для подсказки рефлексии
        self.dreams = None       # DreamWeaver — сны после сна (ставит loop)

    def is_due(self) -> bool:
        """Основной цикл перепроверяет это перед запуском: пока задача ждала в очереди, пользователь мог заговорить."""
        return seconds_since_activity() >= self.idle_threshold

    def start(self):
        """Запускает фоновый поток рефлексии."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reflection_loop, daemon=True)
        self._thread.start()
        print(f"[REFLECTION] Запущено: после {self.idle_threshold} с без обращений к Юи, "
              f"не чаще раза в {self.min_interval} с.")

    def stop(self):
        """Останавливает фоновый поток."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[REFLECTION] Фоновый поток рефлексии остановлен.")

    def _reflection_loop(self):
        """Основной цикл фоновой рефлексии."""
        while self._running:
            time.sleep(REFLECTION_POLL_INTERVAL)
            if not self._running:
                break

            # Простой — это пауза в разговоре с Юи (см. autonomy.seconds_since_activity),
            # а не бездействие за компьютером.
            if not self.is_due():
                continue
            if time.monotonic() - self._last_cycle < self.min_interval:
                continue

            if self.agent_is_working is not None and self.agent_is_working.is_set():
                continue  # основной цикл сейчас держит LLM — не мешаем

            self._last_cycle = time.monotonic()
            try:
                self._queue_reflection()
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

            # Сон со сновидением: не чаще раза в сутки, ночью или после долгой тишины (см. memory/dreams.py)
            agent_busy_now = self.agent_is_working is not None and self.agent_is_working.is_set()
            dream_due = (self.dreams is not None and not agent_busy_now
                         and self.dreams.due(datetime.datetime.now(), seconds_since_activity()))
            if dream_due:
                try:
                    self.dreams.weave()
                except Exception as e:
                    print(f"[DREAM] Ошибка: {e}")

    def _recent_facts(self) -> list:
        """Самые свежие факты из памяти (кроме папки reflections), по дате записи."""
        all_facts = []
        for root, _, files in os.walk(self.mm.base_dir):
            # Игнорируем папку reflections, чтобы не читать свои же заметки
            if "reflections" in root.split(os.sep):
                continue
            for file in files:
                if not file.endswith(".md"):
                    continue
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue
                # Строки фактов: "- [2026-09-24 21:30] (c=0.30) текст"
                for line in content.split('\n'):
                    match = re.match(r'^-\s*\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})\]\s*(.+)$', line.strip())
                    if match:
                        all_facts.append((match.group(1), match.group(2).strip()))
        all_facts.sort(key=lambda item: item[0])
        return [fact for _, fact in all_facts[-REFLECTION_RECENT_FACTS_WINDOW:]]

    def _queue_reflection(self):
        """Ставит внутренний ход рефлексии в очередь основного цикла."""
        facts = self._recent_facts()
        if len(facts) < REFLECTION_MIN_FACTS:
            print(f"[REFLECTION] Пропуск: в памяти {len(facts)} фактов, нужно хотя бы {REFLECTION_MIN_FACTS}.")
            return
        doubtful = self.doubtful() if self.doubtful is not None else []
        doubtful_note = DOUBTFUL_NOTE.format(facts="\n".join(
            f"  [{d['path']}] {d['text']}" for d in doubtful[-5:])) if doubtful else ""
        prompt = REFLECTION_PROMPT.format(facts="\n".join(f"- {f}" for f in facts), doubtful=doubtful_note)
        self.input_queue.put(("reflection", prompt, {"timestamp": time.time(), "facts": len(facts)}))

    def force_reflection(self):
        """Поставить рефлексию в очередь прямо сейчас (для ручного вызова)."""
        self._queue_reflection()

    # ==================== СОН-КОНСОЛИДАЦИЯ (RAG 2.0) ====================

    def _list_topic_paths(self) -> list:
        """Список путей (без .md) всех файлов памяти, кроме служебной папки reflections/."""
        paths = []
        for root, _, files in os.walk(self.mm.base_dir):
            if {"reflections", "diary"} & set(root.split(os.sep)):  # заметки и дневник чувств сон не переписывает
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
        Одна «ночь» для одного файла памяти (с уклоном в сторону свежих, как у kuni): модель видит его факты
        под номерами и похожие факты из других файлов (только для сверки) и отвечает правками по номерам —
        см. SLEEP_CONSOLIDATOR_PROMPT и apply_sleep_edits. Меняется только этот файл.
        Сомнительные факты уходят в self.on_doubtful(path, text) — Юи потом может уточнить их у пользователя.
        """
        candidates = self._list_topic_paths()
        if not candidates:
            return
        target_path = self._pick_sleep_target(candidates)
        records = self.mm.read_fact_records(target_path)
        if not any(r["confidence"] < FACT_CONFIDENCE_ANCHOR_THRESHOLD for r in records):
            return  # пусто или одни якоря — сверять нечего

        def fmt(i, r):
            anchor = " ЯКОРЬ" if r["confidence"] >= FACT_CONFIDENCE_ANCHOR_THRESHOLD else ""
            return f"[{i}] (c={r['confidence']:+.2f}{anchor}) {r['text']}"

        related = self.mm.vector_engine.search("\n".join(r["text"] for r in records),
                                               top_k=SLEEP_CONSOLIDATION_RELATED_FILES + 1)
        context = []
        for res in related:
            if res["id"] == target_path:
                continue
            others = self.mm.read_fact_records(res["id"])
            if others:
                context.append(f"# {res['id']}\n" + "\n".join(f"- (c={r['confidence']:+.2f}) {r['text']}" for r in others))
        prompt = (f"# Файл: {target_path}\n" + "\n".join(fmt(i, r) for i, r in enumerate(records, 1))
                  + ("\n\n## Для сверки (менять нельзя)\n" + "\n\n".join(context[:SLEEP_CONSOLIDATION_RELATED_FILES])
                     if context else ""))
        try:
            response = SESSION.post(
                LLM_API_URL,
                json={
                    "messages": [
                        {"role": "system", "content": SLEEP_CONSOLIDATOR_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": SLEEP_CONSOLIDATION_MAX_TOKENS,
                    "temperature": SLEEP_CONSOLIDATION_TEMPERATURE,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180.0,
            )
            if response.status_code != 200:
                print(f"[SLEEP] LLM ответил {response.status_code}")
                return
            raw = (response.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            print(f"[SLEEP] Ошибка запроса: {e}")
            return

        new_records, deleted, doubtful = apply_sleep_edits(records, parse_sleep_edits(raw))
        changed = new_records != records
        if changed:
            self.mm.write_fact_records(target_path, new_records)
        for text in doubtful:
            if self.on_doubtful is not None:
                self.on_doubtful(target_path, text)
        print(f"[SLEEP] {target_path}: фактов {len(records)} -> {len(new_records)}"
              f"{', удалено как ложные: ' + '; '.join(deleted) if deleted else ''}"
              f"{', под вопросом: ' + str(len(doubtful)) if doubtful else ''}{'' if changed else ' (без изменений)'}")
        return {"path": target_path, "before": records, "after": new_records, "deleted": deleted, "doubtful": doubtful}

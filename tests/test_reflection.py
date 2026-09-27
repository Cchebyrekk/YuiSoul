# -*- coding: utf-8 -*-
import queue

import pytest
import requests

import scripts.memory.reflection as reflection_mod
from conftest import FakeResponse, write_facts
from scripts.memory.reflection import ReflectionManager


@pytest.fixture
def rm(mm):
    return ReflectionManager(mm, queue.Queue())


def seed_facts(memory_dir, n):
    lines = [f"- [2026-09-{10 + i:02d} 12:00] (c=+0.20) Факт номер {i}" for i in range(n)]
    write_facts(memory_dir, "user/facts", lines[::-1])  # в файле не по порядку


def test_recent_facts_sorted_by_date_and_skip_reflections(rm, memory_dir, monkeypatch):
    monkeypatch.setattr(reflection_mod, "REFLECTION_RECENT_FACTS_WINDOW", 3)
    seed_facts(memory_dir, 6)
    write_facts(memory_dir, "reflections/reflection_notes", ["- [2026-12-31 23:59] Своя заметка"])
    assert rm._recent_facts() == ["(c=+0.20) Факт номер 3", "(c=+0.20) Факт номер 4", "(c=+0.20) Факт номер 5"]


def test_queue_reflection_needs_enough_facts(rm, memory_dir, capsys):
    seed_facts(memory_dir, 2)
    rm._queue_reflection()
    assert rm.input_queue.empty()
    assert "Пропуск" in capsys.readouterr().out


def test_queue_reflection_puts_inner_turn_prompt(rm, memory_dir):
    seed_facts(memory_dir, 6)
    rm.force_reflection()
    source, prompt, meta = rm.input_queue.get_nowait()
    assert source == "reflection"
    assert "Факт номер 5" in prompt and "task_complete" in prompt and "save_memory" in prompt
    assert meta["facts"] == 6


def test_is_due_follows_conversation_pause(rm, monkeypatch):
    monkeypatch.setattr(reflection_mod, "seconds_since_activity", lambda: rm.idle_threshold + 1)
    assert rm.is_due()
    monkeypatch.setattr(reflection_mod, "seconds_since_activity", lambda: 0)
    assert not rm.is_due()


def sleep_setup(rm, mm, memory_dir, monkeypatch):
    write_facts(memory_dir, "user/pets", [
        "- [2026-09-20 10:00] (c=+1.00) Кота зовут Барсик",
        "- [2026-09-20 10:01] Кот рыжий",
        "- [2026-09-20 10:02] Кот рыжий пушистый",
    ])
    write_facts(memory_dir, "user/secrets", ["- [2026-09-20 10:03] Не показывали модели"])
    monkeypatch.setattr(mm.vector_engine, "search", lambda q, top_k=3: [{"id": "user/pets", "text": "", "score": 1}])
    rm._pick_sleep_target = lambda candidates: "user/pets"


def test_sleep_merges_duplicates_keeps_anchor_and_dates(rm, mm, memory_dir, monkeypatch, llm):
    sleep_setup(rm, mm, memory_dir, monkeypatch)
    llm.responses.append(FakeResponse("\n".join([
        "[2,3] (c=0.30) Кот рыжий и пушистый.",
        "[1] (c=-1)",                                  # якорь трогать нельзя — правка пропускается
        "[7] (c=0.5) Такого номера нет."])))
    result = rm._run_sleep_consolidation_cycle()

    pets = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "Кота зовут Барсик" in pets                  # якорь сохранён
    assert "- [2026-09-20 10:01] (c=+0.20) Кот рыжий и пушистый." in pets    # дата — ранняя, доверие +0.2 макс
    assert "Кот рыжий пушистый" not in pets
    assert "Не показывали модели" in (memory_dir / "user" / "secrets.md").read_text(encoding="utf-8")
    assert result["deleted"] == []
    request = llm.requests[0]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert "[1] (c=+1.00 ЯКОРЬ) Кота зовут Барсик" in request["messages"][1]["content"]


@pytest.mark.parametrize("response", [FakeResponse("мусор без формата"), FakeResponse("НЕТ"),
                                      FakeResponse("", status_code=500),
                                      requests.exceptions.ConnectionError("нет сервера")])
def test_sleep_consolidation_failure_leaves_files_untouched(rm, mm, memory_dir, monkeypatch, llm, response):
    sleep_setup(rm, mm, memory_dir, monkeypatch)
    before = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    llm.responses.append(response)
    rm._run_sleep_consolidation_cycle()
    assert (memory_dir / "user" / "pets.md").read_text(encoding="utf-8") == before


REC = [{"date": f"2026-09-2{i} 10:00", "confidence": c, "text": t}
       for i, (c, t) in enumerate([(0.0, "Кот рыжий"), (0.0, "Любит аниме"), (0.3, "Есть татуировка"),
                                   (0.0, "Зовут Вадим"), (0.0, "Живёт в Казани"), (0.0, "Любит пиццу")], 1)]


def test_apply_sleep_edits_guards():
    edits = reflection_mod.parse_sleep_edits("\n".join([
        "[1,2,3,5] (c=0.5) Кот, аниме, тату и Казань.",   # простыня из 4 разных — пропуск
        "[3] (c=0.95) Есть татуировка с ромбами.",         # доверие не выше +0.2 от прежнего и не выше 0.9
        "[4] (c=-1)",                                       # ложь — удалить
        "[6] (c=-1)",                                       # второе удаление (треть от 6 — два)
        "[1] (c=-1)",                                       # третье — сверх трети файла, пропуск
        "[5] (?)",                                          # под вопросом — уточнить у пользователя
        "- [2] (c=0.1) Любит аниме «Lain»"]))               # маркер списка перед правкой — тоже понимаем
    records, deleted, doubtful = reflection_mod.apply_sleep_edits(REC, edits)
    texts = [r["text"] for r in records]
    assert texts == ["Кот рыжий", "Любит аниме «Lain»", "Есть татуировка с ромбами.", "Живёт в Казани"]
    assert records[2]["confidence"] == 0.5 and records[1]["confidence"] == 0.1
    assert deleted == ["Зовут Вадим", "Любит пиццу"] and doubtful == ["Живёт в Казани"]
    assert records[2]["date"] == "2026-09-23 10:00"


def test_sleep_consolidation_skips_when_only_anchors(rm, mm, memory_dir, llm):
    write_facts(memory_dir, "user/pets", ["- [2026-09-20 10:00] (c=+1.00) Только якорь"])
    rm._run_sleep_consolidation_cycle()
    assert llm.requests == []


def test_pick_sleep_target_prefers_recent(rm, monkeypatch):
    candidates = [("old", 1.0), ("new", 2.0)]
    monkeypatch.setattr(reflection_mod.random, "random", lambda: 0.0)
    assert rm._pick_sleep_target(candidates) == "new"
    monkeypatch.setattr(reflection_mod.random, "random", lambda: 0.99)
    monkeypatch.setattr(reflection_mod.random, "choice", lambda c: c[0])
    assert rm._pick_sleep_target(candidates) == "old"


def test_background_loop_queues_reflection_and_sleep(rm, memory_dir, monkeypatch):
    seed_facts(memory_dir, 6)
    monkeypatch.setattr(reflection_mod, "REFLECTION_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(reflection_mod, "seconds_since_activity", lambda: 10_000)
    monkeypatch.setattr(reflection_mod, "ENABLE_SLEEP_CONSOLIDATION", True)
    monkeypatch.setattr(reflection_mod, "SLEEP_CONSOLIDATION_EVERY_N_CYCLES", 1)
    slept = []
    rm._run_sleep_consolidation_cycle = lambda: slept.append(1)
    rm.min_interval = 0
    rm.start()
    try:
        source, _, _ = rm.input_queue.get(timeout=5)
    finally:
        rm.stop()
    assert source == "reflection" and slept

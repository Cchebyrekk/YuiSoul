# -*- coding: utf-8 -*-
"""Гигиена памяти (мусор при записи, архив заметок рефлексии, рефлексия без своих заметок) и сбор диалогов."""
import json
import queue

import scripts.memory.reflection as reflection_mod
from conftest import write_facts
from test_telegram import OWNER, make_bot, update

from scripts.agent.dataset import clean_turn, log_turn, rate_last, turn_slice
from scripts.memory.manager import clean_fact_text
from scripts.memory.reflection import ReflectionManager
from scripts.memory.soul import SoulManager


# ---------- мусор не попадает в память ----------

def test_fact_is_cut_at_tool_markup():
    nested = ("Он ищет скрытый смысл в мемах.</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
              "<function=save_memory>\n<parameter=path>\nreflections/reflection_notes")
    assert clean_fact_text(nested) == "Он ищет скрытый смысл в мемах."
    assert clean_fact_text("system/yui/memories") == ""                    # голый путь — не факт
    assert clean_fact_text("<function=save_memory>") == ""
    assert clean_fact_text("Любит кофе\nи пиццу 3/4 сезона") == "Любит кофе и пиццу 3/4 сезона"


def test_save_fact_refuses_garbage(mm, memory_dir):
    assert "Пусто" in mm.save_fact("reflections/reflection_notes", "</function>\n</tool_call>")
    assert not (memory_dir / "reflections" / "reflection_notes.md").exists()
    mm.save_fact("user/likes", "Любит Lain.</parameter>\n</function>")
    text = (memory_dir / "user" / "likes.md").read_text(encoding="utf-8")
    assert "Любит Lain." in text and "function" not in text and "parameter" not in text


# ---------- заметки рефлексии не растут без предела ----------

def test_old_reflection_notes_go_to_archive(mm, memory_dir, monkeypatch):
    monkeypatch.setattr(reflection_mod, "REFLECTION_NOTES_MAX_LINES", 5)
    write_facts(memory_dir, "reflections/reflection_notes",
                [f"- [2026-09-{d:02d} 12:00] Заметка номер {d}." for d in range(1, 13)])
    rm = ReflectionManager(mm, queue.Queue())
    rm._archive_old_notes()
    notes = (memory_dir / "reflections" / "reflection_notes.md").read_text(encoding="utf-8").splitlines()
    assert len(notes) == 5 and "номер 12." in notes[-1] and "номер 8." in notes[0]
    (archive,) = (memory_dir / "archive").glob("reflections_*.md")
    archived = archive.read_text(encoding="utf-8")
    assert len(archived.splitlines()) == 7                                   # ни одна старая заметка не потеряна
    assert "номер 1." in archived and "номер 7." in archived and "номер 8." not in archived
    assert "[2026-09-01 12:00]" in archived                                  # даты сохраняются
    rm._archive_old_notes()                                                  # второй раз — ничего не меняет
    assert len((memory_dir / "reflections" / "reflection_notes.md").read_text(encoding="utf-8").splitlines()) == 5
    patch = SoulManager(base_dir=str(memory_dir)).generate_soul_patch()
    assert "номер 12." in patch and "номер 7." not in patch                  # архив в душу не идёт


def test_reflection_does_not_see_its_own_notes(memory_dir):
    write_facts(memory_dir, "reflections/reflection_notes", ["- [2026-09-28 12:00] Уровень 2.5 и верные свидетели."])
    write_facts(memory_dir, "system/yui/yui_character", ["- [2026-09-28 11:00] Язвительная."])
    soul = SoulManager(base_dir=str(memory_dir))
    assert "Уровень 2.5" in soul.generate_soul_patch()
    quiet = soul.generate_soul_patch(include_notes=False)
    assert "Уровень 2.5" not in quiet and "Язвительная" in quiet            # характер остаётся


# ---------- сбор диалогов ----------

HISTORY = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "старое\n\n<injected_context>\nдуша\n</injected_context>"},
    {"role": "assistant", "content": "Старый ответ"},
    {"role": "user", "content": "<system_note>из Telegram</system_note>\nкак дела?\n\n<injected_context>\nдуша\n"
                                "</injected_context>"},
    {"role": "user", "content": [{"type": "text", "text": "<system_note>Картинка</system_note>"},
                                 {"type": "image_url", "image_url": {"url": "data:x"}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "search_memory", "arguments": "{\"query\": \"дела\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "ничего"},
    {"role": "assistant", "content": "Нормально, а у тебя?"},
]


def test_turn_is_logged_without_service_text(tmp_path):
    path = str(tmp_path / "datasets" / "dialogues.jsonl")
    assert [m["role"] for m in clean_turn(turn_slice(HISTORY))] == ["user", "assistant", "tool", "assistant"]
    assert log_turn(HISTORY, "telegram", path=path)
    (record,) = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert record["source"] == "telegram" and record["rating"] is None
    assert record["messages"][0] == {"role": "user", "content": "как дела?"}
    assert record["messages"][1]["tool_calls"] == [{"name": "search_memory", "arguments": "{\"query\": \"дела\"}"}]
    assert record["messages"][-1]["content"] == "Нормально, а у тебя?"
    dumped = json.dumps(record, ensure_ascii=False)
    assert "душа" not in dumped and "Старый ответ" not in dumped


def test_silent_turn_is_not_logged(tmp_path):
    path = str(tmp_path / "d.jsonl")
    assert not log_turn(HISTORY[:5], "voice", path=path)                    # ответа нет
    assert "нечего" in rate_last("good", path=path)


def test_rating_marks_last_exchange(tmp_path):
    path = str(tmp_path / "d.jsonl")
    log_turn(HISTORY, "text", path=path)
    log_turn(HISTORY, "text", path=path)
    assert "Хороших в наборе: 1 из 2" in rate_last("good", path=path)
    first, last = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert first["rating"] is None and last["rating"] == "good"
    assert "плохой" in rate_last("bad", path=path)


def test_good_command_from_owner_only():
    bot, q = make_bot()
    bot.handle_update(update(text="/good", message_id=3))
    assert q.get_nowait() == ("rate", "good", {"chat_id": OWNER})
    bot.handle_update({"update_id": 1, "message": {"from": {"id": 999}, "chat": {"id": 999, "type": "private"},
                                                   "date": 1000, "message_id": 4, "text": "/good"}})
    assert q.empty()                                                         # гость оценку не ставит

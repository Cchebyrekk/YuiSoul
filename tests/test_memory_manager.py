# -*- coding: utf-8 -*-
import os

import pytest

from conftest import write_facts
from scripts.memory.manager import parse_fact_line, _strip_confidence_tag


# ---------- разбор строк ----------

@pytest.mark.parametrize("line, expected", [
    ("[user/pets] (c=0.30) Кот рыжий.", ("user/pets", 0.30, "Кот рыжий.")),
    ("[user/pets] (c=-1) Собаки нет.", ("user/pets", -1.0, "Собаки нет.")),
    ("[user/pets] Без метки доверия.", ("user/pets", 0.0, "Без метки доверия.")),
    ("[user/pets]", None),               # пустой текст
    ("[] (c=0.5) Без пути.", None),      # пустой путь
    ("просто текст", None),
])
def test_parse_fact_line(line, expected):
    assert parse_fact_line(line) == expected


def test_strip_confidence_tag():
    assert _strip_confidence_tag("(c=+0.90) Факт") == (0.9, "Факт")
    assert _strip_confidence_tag("Факт без метки") == (0.0, "Факт без метки")


# ---------- проверка путей ----------

@pytest.mark.parametrize("raw, expected", [
    ("user/pets", "user/pets"),
    ("user/pets.md", "user/pets"),
    ("memory/user/pets", "user/pets"),          # префикс memory/ снимается
    ("user\\pets", "user/pets"),
    ("../../secret", "secret"),                 # ".." не выводит из памяти
    ("C:/notes", "C/notes"),                    # диск превращается в обычную папку
    ("", "misc/default"),
    ("a:b?/c*d", "ab/cd"),
])
def test_validate_path_stays_inside_memory(mm, raw, expected):
    assert mm._validate_path(raw) == expected


def test_validate_path_rejects_deep_nesting(mm):
    with pytest.raises(ValueError):
        mm._validate_path("a/b/c/d")


# ---------- сохранение фактов ----------

def test_save_fact_writes_timestamped_line_with_confidence(mm, memory_dir):
    assert mm.save_fact("user/pets", "Кот рыжий", confidence=0.3) == "[MEMORY] OK"
    text = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "(c=+0.30) Кот рыжий" in text
    assert text.startswith("- [")


def test_save_fact_replaces_similar_old_fact(mm, memory_dir):
    mm.save_fact("user/pets", "Кот пользователя рыжий и пушистый")
    mm.save_fact("user/pets", "Кот пользователя рыжий и очень пушистый")
    lines = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert "очень пушистый" in lines[0]


def test_save_fact_detects_duplicate_under_other_path(mm):
    mm.save_fact("user/pets", "Кот пользователя рыжий пушистый Барсик")
    result = mm.save_fact("animals/cats", "Кот пользователя рыжий пушистый Барсик")
    assert "уже существует" in result


def test_retraction_removes_similar_fact_but_keeps_anchor(mm, memory_dir):
    write_facts(memory_dir, "user/pets", [
        "- [2026-09-20 10:00] (c=+1.00) Кот пользователя рыжий Барсик",
        "- [2026-09-20 10:01] Кот пользователя рыжий пушистый",
        "- [2026-09-20 10:02] Любит аксолотлей",
    ])
    result = mm.save_fact("user/pets", "Кот пользователя рыжий", confidence=-1)
    text = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "удалено строк: 1" in result
    assert "Барсик" in text            # anchor неприкосновенен
    assert "пушистый" not in text      # обычный похожий факт удалён
    assert "аксолотлей" in text        # непохожий факт не тронут


def test_retraction_of_missing_file(mm):
    assert "не найдено" in mm.save_fact("user/nothing", "Что-то", confidence=-1)


def test_save_fact_rejects_too_deep_path(mm):
    assert mm.save_fact("a/b/c/d", "Факт").startswith("[MEMORY ERROR]")


# ---------- поиск ----------

def test_search_special_queries(mm, memory_dir):
    assert mm.search_facts("__all__") == "Память пуста."
    mm.save_fact("user/pets", "Кот рыжий")
    assert "pets.md" in mm.search_facts("__tree__")
    assert "Файл: user/pets.md" in mm.search_facts("__all__")
    assert "Кот рыжий" in mm.search_facts("FILE:user/pets")
    assert mm.search_facts("FILE:user/none") == "Файл не найден."


def test_search_file_cannot_escape_memory(mm, tmp_path):
    secret = tmp_path / "secret.md"
    secret.write_text("СЕКРЕТ", encoding="utf-8")
    for query in (f"FILE:{secret}", f"FILE:{str(secret)[:-3]}", "FILE:../secret"):
        assert "СЕКРЕТ" not in mm.search_facts(query)


def test_search_vector_then_bm25_fallback(mm):
    # BM25 даёт вес только словам, которых нет в большинстве файлов, — нужна не одна запись
    mm.save_fact("user/food", "Любит пиццу с ананасами")
    mm.save_fact("user/music", "Слушает синтвейв по вечерам")
    mm.save_fact("pc/gpu", "Видеокарта с двенадцатью гигабайтами памяти")
    mm.save_fact("user/pets", "Кот пользователя рыжий Барсик")
    assert "user/pets" in mm.search_facts("кот пользователя рыжий Барсик")  # векторный поиск
    mm.vector_engine.threshold = 1.1                                   # вектор ничего не находит
    assert "(BM25)" in mm.search_facts("Барсик")                       # фолбэк на BM25
    assert mm.search_facts("квантовая хромодинамика") == "По памяти ничего не найдено."


def test_bm25_index_is_cached_until_memory_changes(mm):
    mm.save_fact("user/pets", "Кот пользователя рыжий Барсик")
    mm._bm25_search("Барсик")
    cached = mm._bm25_cache
    mm._bm25_search("Барсик")
    assert mm._bm25_cache is cached
    mm.save_fact("user/food", "Любит пиццу с ананасами")
    mm._bm25_search("пицца")
    assert mm._bm25_cache is not cached


def test_auto_context_returns_clean_matching_lines(mm):
    mm.save_fact("user/pets", "Кот пользователя рыжий Барсик", confidence=0.5)
    mm.vector_engine.threshold = 0.3  # заглушка эмбеддингов грубее e5; проверяем очистку строк, а не порог
    context = mm.get_auto_context("какого цвета кот Барсик")
    assert context == "- Кот пользователя рыжий Барсик."


def test_auto_context_empty_when_nothing_relevant(mm):
    assert mm.get_auto_context("что угодно") == ""


# ---------- сон-консолидация: чтение и перезапись ----------

def test_read_mutable_and_anchor_lines(mm, memory_dir):
    write_facts(memory_dir, "user/pets", [
        "- [2026-09-20 10:00] (c=+1.00) Якорь",
        "- [2026-09-20 10:01] (c=+0.30) Изменяемый",
        "",
    ])
    anchors, mutable = mm.read_mutable_and_anchor_lines("user/pets")
    assert anchors == [(1.0, "Якорь")]
    assert mutable == [(0.3, "Изменяемый")]
    assert mm.read_mutable_and_anchor_lines("user/none") == ([], [])


def test_rewrite_keeps_anchors_and_caps_confidence(mm, memory_dir):
    write_facts(memory_dir, "user/pets", ["- [2026-09-20 10:00] (c=+1.00) Якорь", "- [2026-09-20 10:01] Старое"])
    mm.rewrite_mutable_lines("user/pets", [(0.5, "Новое"), (1.0, "Сон не может сделать якорь")])
    anchors, mutable = mm.read_mutable_and_anchor_lines("user/pets")
    assert anchors == [(1.0, "Якорь")]
    assert (0.5, "Новое") in mutable
    assert all(c < 0.999 for c, _ in mutable)
    assert "Старое" not in (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")


def test_rewrite_drops_refuted_facts(mm, memory_dir):
    # Промпт сон-консолидации обещает модели: факты с confidence -1 "будут УДАЛЕНЫ".
    write_facts(memory_dir, "user/pets", ["- [2026-09-20 10:00] Пользователь любит собак"])
    mm.rewrite_mutable_lines("user/pets", [(-1.0, "Пользователь любит собак"), (0.3, "Любит кошек")])
    text = (memory_dir / "user" / "pets.md").read_text(encoding="utf-8")
    assert "собак" not in text
    assert "кошек" in text


def test_rewrite_to_empty_removes_file_from_index(mm, memory_dir):
    mm.save_fact("user/pets", "Кот рыжий")
    mm.rewrite_mutable_lines("user/pets", [])
    assert all(m["id"] != "user/pets" for m in mm.vector_engine.metadata)


def test_fact_extraction_prompt_mentions_history(mm):
    prompt = mm.get_fact_extraction_prompt([{"role": "user", "content": "Меня зовут Вадим"}], "/memory")
    assert "Меня зовут Вадим" in prompt

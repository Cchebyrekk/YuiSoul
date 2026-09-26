# -*- coding: utf-8 -*-
import random
import threading

from conftest import write_facts
from scripts.memory.vector import VectorSearchEngine


def make_engine(memory_dir):
    engine = VectorSearchEngine()
    assert engine.index_dir == str(memory_dir)  # индекс во временной папке, не в настоящей памяти
    return engine


def test_add_update_remove_and_persist(memory_dir):
    engine = make_engine(memory_dir)
    engine.add_document("user/pets", "Кот рыжий Барсик")
    engine.add_document("user/food", "Любит пиццу")
    engine.add_document("user/pets", "Кот рыжий Барсик пушистый")  # обновление, не дубль
    assert [m["id"] for m in engine.metadata] == ["user/pets", "user/food"]
    assert engine.metadata[0]["text"].endswith("пушистый")

    reloaded = VectorSearchEngine()  # индекс читается с диска
    assert [m["id"] for m in reloaded.metadata] == ["user/pets", "user/food"]

    engine.remove_document("user/pets")
    engine.remove_document("user/missing")  # несуществующий — no-op
    assert [m["id"] for m in engine.metadata] == ["user/food"]
    assert len(engine.vectors) == 1


def test_search_filters_by_threshold_and_ranks(memory_dir):
    engine = make_engine(memory_dir)
    assert engine.search("что угодно") == []
    engine.add_document("user/pets", "Кот рыжий Барсик")
    engine.add_document("user/food", "Любит пиццу с ананасами")
    results = engine.search("Кот рыжий Барсик", top_k=5)
    assert [r["id"] for r in results] == ["user/pets"]
    assert results[0]["score"] > engine.threshold


def test_recency_factor():
    engine = VectorSearchEngine.__new__(VectorSearchEngine)
    assert engine._recency_factor("") == 0.0
    assert engine._recency_factor("не дата") == 0.0
    assert 0.99 < engine._recency_factor(__import__("datetime").datetime.now().isoformat()) <= 1.0


def test_rebuild_index_from_files(memory_dir):
    write_facts(memory_dir, "user/pets", ["- Кот рыжий"])
    write_facts(memory_dir, "user/food", ["- Пицца"])
    engine = make_engine(memory_dir)
    engine.rebuild_index()
    assert sorted(m["id"] for m in engine.metadata) == ["user/food", "user/pets"]

    for f in memory_dir.rglob("*.md"):
        f.unlink()
    engine.rebuild_index()
    assert engine.metadata == [] and len(engine.vectors) == 0


def test_concurrent_add_remove_search_is_safe(memory_dir):
    """Регрессия: без блокировки параллельные запись и поиск давали IndexError и чужой текст."""
    engine = make_engine(memory_dir)
    engine._save_index = lambda: None
    engine.threshold = -1.0
    for i in range(30):
        engine.add_document(f"doc/{i}", f"text{i} слово")
    errors, stop = [], threading.Event()

    def writer():
        while not stop.is_set():
            i = random.randrange(40)
            try:
                if random.random() < 0.5:
                    engine.remove_document(f"doc/{i}")
                else:
                    engine.add_document(f"doc/{i}", f"text{i} слово {random.random()}")
            except Exception as e:  # pragma: no cover - это и есть провал теста
                errors.append(repr(e))

    def reader():
        while not stop.is_set():
            try:
                for r in engine.search("слово", top_k=40):
                    if not r["text"].startswith(f"text{r['id'].split('/')[1]} "):
                        errors.append(f"mismatch {r['id']}")
            except Exception as e:  # pragma: no cover
                errors.append(repr(e))

    threads = [threading.Thread(target=writer) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join()
    assert errors == []

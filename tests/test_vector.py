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
    assert sorted(m["id"] for m in engine.metadata) == ["user/food", "user/pets"]
    assert engine.docs["user/pets"]["text"].endswith("пушистый")

    reloaded = VectorSearchEngine()  # индекс читается с диска
    assert sorted(m["id"] for m in reloaded.metadata) == ["user/food", "user/pets"]

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


def test_index_is_per_fact(memory_dir):
    engine = make_engine(memory_dir)
    engine.add_document("user/pets", "- [2026-09-20 10:00] (c=+1.00) Кот рыжий Барсик\n# заголовок\n- Барсик спит на клавиатуре")
    assert [line["text"] for line in engine.lines] == ["Кот рыжий Барсик", "Барсик спит на клавиатуре"]
    hits = engine.search_lines("кот рыжий барсик", top_k=1)
    assert hits[0]["id"] == "user/pets" and hits[0]["line"] == "Кот рыжий Барсик"
    result = engine.search("кот рыжий барсик")[0]
    assert result["lines"][0] == "Кот рыжий Барсик" and result["text"].startswith("- [2026")


def test_old_file_level_index_is_rebuilt(memory_dir):
    import json
    import numpy as np
    (memory_dir / "user").mkdir()
    (memory_dir / "user" / "pets.md").write_text("- Кот рыжий\n- Кот спит\n", encoding="utf-8")
    np.save(memory_dir / "vectors.npy", np.zeros((1, 4)))            # формат v1: вектор на файл
    (memory_dir / "metadata.json").write_text(json.dumps([{"id": "user/pets", "text": "- Кот рыжий"}]), encoding="utf-8")
    engine = make_engine(memory_dir)
    assert [line["text"] for line in engine.lines] == ["Кот рыжий", "Кот спит"]
    assert json.loads((memory_dir / "metadata.json").read_text(encoding="utf-8"))["version"] == 2


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

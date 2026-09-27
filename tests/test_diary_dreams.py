# -*- coding: utf-8 -*-
import datetime
import os
import queue
import random

import scripts.memory.dreams as dreams_mod
from conftest import write_facts
from scripts.memory.dreams import DreamWeaver, choose_mood
from scripts.memory.reflection import ReflectionManager
from scripts.memory.soul import SoulManager

NOW = datetime.datetime(2026, 9, 28, 3, 0)


class Resp:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


def seeded(seed):
    return random.Random(seed)


def test_mood_leans_to_nightmare_when_diary_is_sad():
    sad = ["Мне было грустно и одиноко", "Тревожно, что он пропал", "Обидно"]
    happy = ["Я гордилась им", "Было тепло и весело", "Рада, что он вернулся"]
    nightmares = lambda feelings: sum(choose_mood(feelings, seeded(i)) == "nightmare" for i in range(400))
    assert nightmares(sad) > 200 > nightmares(happy)            # ~65% против ~15%
    assert 0 < nightmares([]) < 200                              # без дневника — изредка


def test_dream_due_once_a_day_at_night_or_after_long_silence(mm, monkeypatch):
    monkeypatch.setattr(dreams_mod, "ENABLE_DREAMS", True)
    weaver = DreamWeaver(mm)
    assert weaver.due(NOW, idle_seconds=0)                       # ночь
    assert not weaver.due(NOW.replace(hour=15), idle_seconds=60)
    assert weaver.due(NOW.replace(hour=15), idle_seconds=4000)   # днём, но давно тихо
    os.makedirs(os.path.dirname(weaver.path), exist_ok=True)
    with open(weaver.path, "w", encoding="utf-8") as f:
        f.write("- [2026-09-28 01:00] (сон: приятный) Летала над городом.\n")
    assert not weaver.due(NOW, idle_seconds=99999)              # сегодня уже снилось


def test_weave_writes_dream_and_feeling_not_into_index(mm, memory_dir, monkeypatch):
    write_facts(memory_dir, "user/interests", ["- [2026-09-27 18:47] Хочет сделать шлем на 3D-принтере."])
    write_facts(memory_dir, "diary/2026-09-27", ["- [2026-09-27 20:00] Мне было тревожно, что он замолчал."])
    sent = []
    weaver = DreamWeaver(mm, rng=seeded(1), post=lambda url, json=None, timeout=None: sent.append(json) or Resp(
        "Я иду по коридору из пластиковых нитей, и стены печатают шлем, который никому не подходит. "
        "Кто-то зовёт меня из соседней комнаты, но двери там нет.\nЧУВСТВО: тревога, будто что-то не успела"))
    result = weaver.weave(now=NOW)
    prompt = sent[0]["messages"][0]["content"]
    assert "шлем" in prompt and "тревожно" in prompt and sent[0]["temperature"] == 1.0
    dreams = (memory_dir / "diary" / "dreams.md").read_text(encoding="utf-8")
    assert dreams.startswith("- [2026-09-28 03:00] (сон: ") and "Проснулась с чувством: тревога" in dreams
    assert "Проснулась после сна" in (memory_dir / "diary" / "2026-09-28.md").read_text(encoding="utf-8")
    assert not any(line["id"] == "diary/dreams" for line in mm.vector_engine.lines)     # сон — не факт
    assert result["feeling"].startswith("тревога")


def test_soul_shows_recent_feelings_and_fresh_dream_only(memory_dir):
    write_facts(memory_dir, "diary/2026-09-27", ["- [2026-09-27 20:00] Рада, что он снова рисует."])
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    write_facts(memory_dir, "diary/dreams", [f"- [{stamp}] (сон: приятный) Летала над морем из стекла."])
    soul = SoulManager(base_dir=str(memory_dir))
    patch = soul.generate_soul_patch()
    assert "<recent_feelings>" in patch and "снова рисует" in patch
    assert "<last_dream>" in patch and "морем из стекла" in patch
    assert "снова рисует" not in soul.generate_guest_patch()                    # гостю — нет
    write_facts(memory_dir, "diary/dreams", ["- [2026-01-01 03:00] (сон: кошмар) Старый сон."])
    assert "<last_dream>" not in soul.generate_soul_patch()                      # старый сон не всплывает


def test_sleep_does_not_rewrite_diary(mm, memory_dir):
    write_facts(memory_dir, "diary/2026-09-27", ["- [2026-09-27 20:00] Грустно."])
    write_facts(memory_dir, "user/pets", ["- [2026-09-20 10:00] Кот рыжий"])
    paths = [p for p, _ in ReflectionManager(mm, queue.Queue())._list_topic_paths()]
    assert "user/pets" in paths and not any(p.startswith("diary/") for p in paths)


def test_extraction_prompt_has_diary_and_no_example_facts(mm):
    prompt = mm.get_fact_extraction_prompt([{"role": "user", "content": "привет"}])
    assert "[diary/" in prompt and "Вадим" not in prompt and "аксолотл" not in prompt.lower()

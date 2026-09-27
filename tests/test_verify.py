# -*- coding: utf-8 -*-
import queue

import pytest

from conftest import write_facts
from scripts.memory.reflection import ReflectionManager
from scripts.memory.verify import FactVerifier, find_record

FACTS = ["- [2026-09-20 10:00] Зовут Вадим.", "- [2026-09-21 11:00] (c=+0.30) Любит аксолотлей."]


class Bot:
    """Запоминает кнопки; ответить за пользователя — press()."""
    def __init__(self):
        self.asked = []

    def ask_buttons(self, text, buttons, on_answer, chat_id=None):
        self.asked.append((text, [v for _, v in buttons], on_answer))
        return "token"

    def press(self, value):
        self.asked[-1][2](value)


@pytest.fixture
def verifier(mm, memory_dir, tmp_path):
    write_facts(memory_dir, "user/profile", FACTS)
    bot = Bot()
    v = FactVerifier(mm, telegram=bot, prefer_telegram=lambda: True, path=str(tmp_path / "to_verify.json"))
    return v, bot


def test_find_record():
    records = [{"text": "Зовут Вадим."}, {"text": "Любит аксолотлей."}]
    assert find_record(records, "зовут вадим") == 0
    assert find_record(records, "Любит аксолотлей и котов") == 1
    assert find_record(records, "Живёт в Казани") is None


def test_yes_makes_anchor_keeping_date(verifier, memory_dir):
    v, bot = verifier
    assert "Спросила в Telegram" in v.ask("user/profile", "Любит аксолотлей")
    assert bot.asked[0][1] == ["yes", "no", "skip"] and "«Любит аксолотлей.»" in bot.asked[0][0]
    bot.press("yes")
    text = (memory_dir / "user" / "profile.md").read_text(encoding="utf-8")
    assert "- [2026-09-21 11:00] (c=+1.00) Любит аксолотлей." in text
    assert "уже подтверждён" in v.ask("user/profile", "Любит аксолотлей")


def test_no_removes_fact(verifier, memory_dir):
    v, bot = verifier
    v.ask("user/profile", "Зовут Вадим")
    bot.press("no")
    text = (memory_dir / "user" / "profile.md").read_text(encoding="utf-8")
    assert "Вадим" not in text and "аксолотлей" in text


def test_one_question_at_a_time(verifier):
    v, bot = verifier
    v.ask("user/profile", "Зовут Вадим")
    assert "ждёшь ответа" in v.ask("user/profile", "Любит аксолотлей")
    bot.press("skip")
    assert "Спросила" in v.ask("user/profile", "Любит аксолотлей")


def test_unknown_fact(verifier):
    v, _ = verifier
    assert "нет такого факта" in v.ask("user/profile", "Живёт в Казани")


def test_doubtful_list_reaches_reflection_and_clears_on_answer(verifier, mm, memory_dir):
    v, bot = verifier
    write_facts(memory_dir, "user/facts", [f"- [2026-09-1{i} 12:00] Факт номер {i}" for i in range(6)])
    v.add_doubtful("user/profile", "Зовут Вадим.")
    v.add_doubtful("user/profile", "Зовут Вадим.")                 # без повторов
    assert len(v.doubtful()) == 1
    rm = ReflectionManager(mm, queue.Queue())
    rm.doubtful = v.doubtful
    rm.force_reflection()
    _, prompt, _ = rm.input_queue.get_nowait()
    assert "засомневалась" in prompt and "[user/profile] Зовут Вадим." in prompt and "verify_fact" in prompt
    v.ask("user/profile", "Зовут Вадим")
    bot.press("yes")
    assert v.doubtful() == []


def test_pc_fallback_when_not_in_telegram(mm, memory_dir, tmp_path, monkeypatch):
    import scripts.memory.verify as verify_mod
    write_facts(memory_dir, "user/profile", FACTS)
    v = FactVerifier(mm, telegram=None, path=str(tmp_path / "v.json"))
    shown = []
    monkeypatch.setattr(verify_mod.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda self: shown.append(args)})())
    assert "окном на компьютере" in v.ask("user/profile", "Зовут Вадим")
    assert shown and "Зовут Вадим" in shown[0][0]

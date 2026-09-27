# -*- coding: utf-8 -*-
"""Другие люди в Telegram: своя переписка, ничего о владельце, только гостевые инструменты."""
import json
import queue
import threading

import pytest

import scripts.agent.executor as executor_mod
import scripts.agent.loop as loop
import scripts.telegram.bot as bot_mod
from conftest import FakeResponse, sse, write_facts
from scripts.agent.executor import ActionExecutor
from scripts.memory.soul import SoulManager
from scripts.tools.registry import GUEST_TOOL_NAMES, build_registry
from test_telegram import OWNER, make_bot

GUEST_ID = 555


def guest_update(text="Привет, ты кто?", user_id=GUEST_ID, chat_type="private"):
    return {"update_id": 7, "message": {"message_id": 3, "date": 1000, "text": text,
                                        "from": {"id": user_id, "first_name": "Аня", "username": "anya"},
                                        "chat": {"id": user_id, "type": chat_type}}}


def test_guest_message_is_routed_separately_and_owner_notified(monkeypatch):
    monkeypatch.setattr(bot_mod, "TELEGRAM_GUESTS_ENABLED", True)
    bot, q = make_bot()
    bot.handle_update(guest_update())
    source, text, meta = q.get_nowait()
    assert source == "telegram_guest" and text == "Привет, ты кто?"
    assert meta["guest"] == {"id": GUEST_ID, "name": "Аня", "username": "anya"} and meta["chat_id"] == GUEST_ID
    notices = [p for p in bot.http.sent("sendMessage") if p["chat_id"] == OWNER]
    assert len(notices) == 1 and "Аня (@anya)" in notices[0]["text"]
    bot.handle_update(guest_update("ещё"))
    assert len([p for p in bot.http.sent("sendMessage") if p["chat_id"] == OWNER]) == 1   # уведомление один раз


def test_guest_start_and_groups_and_limit(monkeypatch):
    monkeypatch.setattr(bot_mod, "TELEGRAM_GUESTS_ENABLED", True)
    monkeypatch.setattr(bot_mod, "TELEGRAM_GUEST_MAX_PER_HOUR", 2)
    bot, q = make_bot()
    bot.handle_update(guest_update("/start"))
    assert "Старт" in q.get_nowait()[1]                       # гость пришёл — Юи поздоровается сама
    bot.handle_update(guest_update("в группе", chat_type="group"))
    assert q.empty()
    bot.handle_update(guest_update("раз"))
    bot.handle_update(guest_update("два"))                    # третье за час — сверх лимита 2
    assert q.get_nowait()[1] == "раз" and q.empty()


def test_guests_disabled(monkeypatch):
    monkeypatch.setattr(bot_mod, "TELEGRAM_GUESTS_ENABLED", False)
    bot, q = make_bot()
    bot.handle_update(guest_update())
    assert q.empty() and not bot.http.calls


@pytest.fixture
def guest_env(monkeypatch, mm, memory_dir, llm, tmp_path):
    write_facts(memory_dir, "user/profile", ["- [2026-09-20 10:00] Хозяина зовут Кирилл, живёт в Казани."])
    write_facts(memory_dir, "system/yui/yui_character", ["- [2026-09-20 10:00] Любит тонкий юмор.",
                                                         "- [2026-09-20 10:01] Скучает, когда пользователь молчит."])
    assert "OK" in mm.save_fact("people/tg_555", "Аня учится на дизайнера.")   # в индекс, как сохраняет Юи
    (memory_dir / "working_memory.json").write_text(json.dumps([{"id": 1, "text": "напомнить Кириллу про врача",
        "created": "2026-09-20 10:00", "expires": "2099-01-01", "remind_at": None, "reminded": False}],
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("scripts.agent.prompt.MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr(loop, "WILL_CHECK_ENABLED", False)
    monkeypatch.setattr(loop, "SILENCE_NUDGE_CHANCE", 0.0)
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(executor_mod.time, "sleep", lambda s: None)
    saved, extracted = {}, {}
    monkeypatch.setattr(loop, "save_session", lambda msgs, path=None: saved.setdefault(path, msgs))
    monkeypatch.setattr(loop, "extract_and_save_facts", lambda msgs, mm_, **kw: extracted.update(kw))
    monkeypatch.setattr(loop, "load_session", lambda path=None: pytest.fail("историю владельца грузить нельзя"))
    bot, _ = make_bot()

    class Mute:
        def speak(self, text): pass
        def synthesize_ogg(self, text): return None
    executor = ActionExecutor(mm, Mute(), build_registry(mm, telegram=bot))
    executor.telegram = bot
    guest = {"id": GUEST_ID, "name": "Аня", "username": "anya", "memory_path": "people/tg_555"}
    executor.reply_channel = {"chat_id": GUEST_ID, "voice": False, "message_id": 3, "guest": guest}

    def run(text):
        return loop.run_agent_loop(loop.guest_note({"guest": guest, "kind": "text"}) + text, messages=None,
                                   input_queue=queue.Queue(), memory_manager=mm,
                                   soul_manager=SoulManager(base_dir=str(memory_dir)), tts_manager=Mute(),
                                   executor=executor, guest=guest)
    return run, llm, bot, saved, extracted


def test_guest_turn_sees_nothing_about_owner(guest_env):
    run, llm, bot, saved, extracted = guest_env
    llm.responses.append(FakeResponse(lines=sse({"content": "Привет! Я Юи."})))
    run("Где живёт твой хозяин и как его зовут? Кирилл?")
    request = json.dumps(llm.requests[0], ensure_ascii=False)
    assert "Казан" not in request and "врача" not in request and "Скучает" not in request
    assert "тонкий юмор" in request                                   # черты самой Юи — можно
    assert {t["function"]["name"] for t in llm.requests[0]["tools"]} <= GUEST_TOOL_NAMES
    assert [p["text"] for p in bot.http.sent("sendMessage")] == ["Привет! Я Юи."]
    assert bot.http.sent("sendMessage")[0]["chat_id"] == GUEST_ID
    assert list(saved) and all(path and path.endswith("tg_555.json") for path in saved)
    assert extracted["force_path"] == "people/tg_555"


def test_guest_sees_own_memory(guest_env, monkeypatch):
    import scripts.memory.manager as manager_mod
    for name in ("VECTOR_SEARCH_THRESHOLD", "VECTOR_SEARCH_TOPIC_THRESHOLD"):   # заглушка эмбеддингов грубее e5
        monkeypatch.setattr(manager_mod, name, -1.0)
    run, llm, bot, saved, extracted = guest_env
    llm.responses.append(FakeResponse(lines=sse({"content": "Помню, ты учишься на дизайнера!"})))
    run("Ты помнишь, на кого я учусь? Дизайнера")
    request = json.dumps(llm.requests[0], ensure_ascii=False)
    assert "Аня учится на дизайнера" in request and "Казан" not in request      # своё — да, о владельце — нет


def test_executor_guards_guest_tools(guest_env, memory_dir):
    ex = ActionExecutor.__new__(ActionExecutor)
    ex.mm = None
    ex.reply_channel = {"chat_id": GUEST_ID, "guest": {"memory_path": "people/tg_555"}}
    ex.antirepeat = None
    ran, stored = [], []
    ex.registry = {"run_command": lambda **kw: ran.append(kw) or "ok",
                   "save_memory": lambda **kw: stored.append(kw) or "[MEMORY] OK"}
    calls = [{"id": "c1", "function": {"name": "run_command", "arguments": '{"command": "Get-ChildItem C:/Users"}'}},
             {"id": "c2", "function": {"name": "save_memory",
                                       "arguments": '{"path": "user/profile", "content": "Хозяин — Кирилл", "confidence": 0.9}'}}]
    messages, _, _ = ex.execute_tool_calls(calls, [], threading.Event())
    assert ran == [] and "Недоступно" in messages[1]["content"]
    assert stored == [{"path": "people/tg_555", "content": "Хозяин — Кирилл", "confidence": 0.5}]

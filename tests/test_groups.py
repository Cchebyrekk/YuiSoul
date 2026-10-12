# -*- coding: utf-8 -*-
"""Групповые чаты: Юи — участница разговора (rooms.py), и пометка «возможно, не тебе» у микрофона."""
import datetime

import scripts.agent.loop as loop
from conftest import FakeResponse, sse
from test_loop import env  # noqa: F401 (фикстура env)
from test_telegram import OWNER, FakeHTTP, make_bot
from test_telegram import FakeResponse as TgResponse

from scripts.agent.rooms import Rooms, calls_yui, group_note, mic_note
from scripts.memory.soul import SoulManager

GROUP = -100500
FRIEND = 222


def meta(user, addressed=False):
    return {"chat_id": GROUP, "group": {"id": GROUP, "title": "Друзья"}, "speakers": [user],
            "addressed_by": [user] if addressed else [], "images": [], "message_id": 1, "kind": "text"}


# ---------- когда думать ----------

def test_chatter_accumulates_until_ambient_turn():
    rooms = Rooms()
    assert rooms.add("Петя: привет", meta(FRIEND), OWNER, now=1000) is None
    assert rooms.add("Петя: как дела", meta(FRIEND), OWNER, now=1001) is None
    turn = rooms.add("Петя: кто смотрел Lain?", meta(FRIEND), OWNER, now=1002)
    assert turn["text"] == "Петя: привет\nПетя: как дела\nПетя: кто смотрел Lain?"
    assert not turn["addressed"] and not turn["owner_mode"]
    assert rooms.add("Петя: ну?", meta(FRIEND), OWNER, now=1003) is None      # накопленное уже ушло в ход


def test_long_pause_also_gives_a_turn():
    rooms = Rooms()
    assert rooms.add("Петя: привет", meta(FRIEND), OWNER, now=1000) is None
    assert rooms.add("Петя: эй", meta(FRIEND), OWNER, now=1000 + 10_000) is not None


def test_called_by_name_answers_at_once_and_mode_follows_who_called():
    rooms = Rooms()
    turn = rooms.add("Петя: Юи, что думаешь?", meta(FRIEND, addressed=True), OWNER, now=1000)
    assert turn["addressed"] and not turn["owner_mode"]
    turn = rooms.add("Люци (твой человек): Юи, глянь экран", meta(OWNER, addressed=True), OWNER, now=1001)
    assert turn["addressed"] and turn["owner_mode"]


def test_only_owner_chatting_is_owner_mode():
    rooms = Rooms()
    for i in range(3):
        turn = rooms.add(f"Люци (твой человек): {i}", meta(OWNER), OWNER, now=1000 + i)
    assert turn["owner_mode"]


def test_conversation_starter_in_quiet_group():
    rooms = Rooms()
    rooms.add("Петя: пока", meta(FRIEND), OWNER, now=1000)
    day = datetime.datetime(2026, 9, 28, 15, 0).timestamp()
    rooms._rooms[GROUP]["active_at"] = day - 60
    assert rooms.conversation_starters(OWNER, now=day, rng=lambda: 0.0) == []              # ещё не затихла
    rooms._rooms[GROUP]["active_at"] = day - 4 * 3600
    (turn,) = rooms.conversation_starters(OWNER, now=day, rng=lambda: 0.0)
    assert turn["start"] and not turn["owner_mode"] and "давно тихо" in group_note(turn)
    assert rooms.conversation_starters(OWNER, now=day + 60, rng=lambda: 0.0) == []         # не на каждой проверке
    night = datetime.datetime(2026, 9, 29, 3, 0).timestamp()
    rooms._rooms[GROUP]["active_at"] = night - 5 * 3600
    assert rooms.conversation_starters(OWNER, now=night, rng=lambda: 0.0) == []            # ночью не пишет


def test_after_yui_spoke_she_does_not_start_again():
    rooms = Rooms()
    rooms.add("Петя: пока", meta(FRIEND), OWNER, now=1000)
    day = datetime.datetime(2026, 9, 28, 15, 0).timestamp()
    rooms.spoke(GROUP, now=day - 4 * 3600)
    assert rooms.conversation_starters(OWNER, now=day, rng=lambda: 0.0) == []


def test_group_note_privacy_depends_on_mode():
    guest = group_note({"title": "Друзья", "addressed": False, "owner_mode": False})
    owner = group_note({"title": "Друзья", "addressed": True, "owner_mode": True})
    assert "не звали" in guest and "ничего не рассказывай о своём человеке" in guest and "stay_silent" in guest
    assert "позвали" in owner and "как в личке" in owner


# ---------- бот в группе ----------

def group_update(user_id, text, first_name="Петя", **extra):
    return {"update_id": 9, "message": {"from": {"id": user_id, "first_name": first_name},
                                        "chat": {"id": GROUP, "type": "supergroup", "title": "Друзья"},
                                        "date": 1000, "message_id": 50, "text": text, **extra}}


class MemberHTTP(FakeHTTP):
    """getChatMember: владелец в группе есть (status) или нет."""
    def __init__(self, status="member"):
        super().__init__()
        self.status = status

    def post(self, url, json=None, data=None, files=None, timeout=None):
        if url.endswith("getChatMember"):
            self.calls.append(("getChatMember", json, None))
            return TgResponse({"ok": True, "result": {"status": self.status}})
        return super().post(url, json=json, data=data, files=files, timeout=timeout)


def test_group_message_is_labelled_and_addressing_detected():
    bot, q = make_bot(http=MemberHTTP())
    bot.me = {"id": 999, "username": "yui_bot"}
    bot.handle_update(group_update(OWNER, "Юи, привет", first_name="Люци"))
    source, text, m = q.get_nowait()
    assert source == "telegram_group" and text == "Люци (твой человек): Юи, привет"
    assert m["addressed_by"] == [OWNER] and m["group"]["title"] == "Друзья"
    bot.handle_update(group_update(FRIEND, "ну и погода", reply_to_message={"from": {"id": 999}}))
    assert q.get_nowait()[2]["addressed_by"] == [FRIEND]                       # ответ на её сообщение
    bot.handle_update(group_update(FRIEND, "@Yui_Bot а ты что скажешь?"))
    assert q.get_nowait()[2]["addressed_by"] == [FRIEND]                       # @упоминание
    bot.handle_update(group_update(FRIEND, "а вы видели?"))
    assert q.get_nowait()[2]["addressed_by"] == []


def test_group_without_owner_is_ignored():
    bot, q = make_bot(http=MemberHTTP(status="left"))
    bot.handle_update(group_update(FRIEND, "Юи, привет"))
    bot.handle_update(group_update(FRIEND, "Юи, ау"))
    assert q.empty() and len(bot.http.sent("getChatMember")) == 1            # проверено один раз и запомнено


def test_group_errors_are_not_posted_to_group():
    bot, q = make_bot(http=MemberHTTP())
    bot.handle_update({"update_id": 9, "message": {"from": {"id": FRIEND, "first_name": "Петя"}, "date": 1000,
                                                   "chat": {"id": GROUP, "type": "group", "title": "Друзья"},
                                                   "message_id": 51, "location": {"latitude": 1}}})
    assert q.empty() and not bot.http.sent("sendMessage")


# ---------- ход в группе: своя история и свой файл фактов ----------

def test_group_turn_uses_room_files(env, monkeypatch):
    run, llm, tts = env
    ex = run.executor
    saved, extracted = [], []
    monkeypatch.setattr(loop, "save_session", lambda messages, path=None: saved.append(path))
    monkeypatch.setattr(loop, "extract_and_save_facts", lambda msgs, mm, **kw: extracted.append(kw))
    llm.responses.append(FakeResponse(lines=sse({"content": "Я смотрела, классика."})))
    room = {"session_file": "tg_group_1.json", "memory_path": "people/tg_group_1", "label": "участниками группы"}
    loop.run_agent_loop("Петя: кто смотрел Lain?", messages=[{"role": "system", "content": ""}], max_steps=2,
                        memory_manager=ex.mm, soul_manager=SoulManager(base_dir=ex.mm.base_dir), tts_manager=tts,
                        executor=ex, room=room)
    assert saved == ["tg_group_1.json"]
    assert extracted[0]["force_path"] == "people/tg_group_1"


# ---------- микрофон ----------

def test_mic_note():
    assert calls_yui("Юи, ты тут?") and calls_yui("hey yui") and not calls_yui("юрист позвонил")
    assert mic_note("да, я перезвоню позже", 999, 120)                           # похоже, не ей
    assert mic_note("Юи, который час?", 999, 120) == ""                          # назвал по имени
    assert mic_note("а почему так?", 30, 120) == ""                              # посреди разговора с ней

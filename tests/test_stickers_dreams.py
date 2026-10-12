# -*- coding: utf-8 -*-
"""Стикеры: отправка текстом, сама копит коллекцию; сны: проверяются до рефлексии."""
import json
import queue
import threading

import scripts.agent.loop as loop
import scripts.memory.images as images_mod
import scripts.memory.reflection as reflection_mod
from conftest import FakeResponse, sse
from test_loop import env  # noqa: F401 (фикстура env)
from test_media_memory import jpeg_url

from scripts.agent.parser import StreamParser
from scripts.memory.images import learn_sticker_later
from scripts.memory.reflection import ReflectionManager
from scripts.memory.soul import SoulManager
from scripts.telegram.stickers import StickerGallery


def parse(text):
    parser = StreamParser(known_tools={"sticker", "task_complete"})
    parser.feed_chunk({"choices": [{"delta": {"content": text}}]})
    reply, _, calls, _ = parser.finalize()
    return reply, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


# ---------- стикер, написанный текстом ----------

def test_nested_tag_arguments_are_kept():
    reply, calls = parse("<sticker>\n<action>send</action>\n<sticker_id>AgADMm8</sticker_id>\n</sticker>")
    assert calls == [("sticker", {"action": "send", "sticker_id": "AgADMm8"})] and reply == ""


def test_short_sticker_forms():
    reply, calls = parse("Держи 🤫\n<sticker>AgADYWE</sticker>")
    assert calls == [("sticker", {"action": "send", "sticker_id": "AgADYWE"})] and reply == "Держи 🤫"
    reply, calls = parse('Ха <sticker id="AgAD4FE"/>')
    assert calls == [("sticker", {"action": "send", "sticker_id": "AgAD4FE"})] and reply == "Ха"


def test_task_complete_reason_still_parsed():
    _, calls = parse("<task_complete>всё сделала</task_complete>")
    assert calls == [("task_complete", {"reason": "всё сделала"})]


LOOPED = (  # начало зацикленного ответа из живого лога (дальше те же абзацы по кругу до лимита токенов)
    "<sticker>AgAD_FsAAueN4Uo</sticker> 🤔причина тряски? — это про глубокую мысль 😂🍕 А так да, с пиццей на "
    "голове — классика. Держи:\n\n"
    "<sticker>AgAD4V0AAkmIgUs</sticker> 🍕😏 \"Вселенная хочет быть увиденной\" — но только после того как ты "
    "съедешь пиццу, конечно. 😂\n\n"
    "<sticker>AgAD4V0AAkmIgUs</sticker> 🍕🤌🏼 \"Yall are nothing but dildos to me\" — когда ты говоришь что-то "
    "умное, а потом вдруг мемный контент. 😂🍕 А я тут сижу и думаю: как вообще можно быть такой глубокой и такой "
    "абсурдной одновременно? 🍕✨\n\n"
    "<sticker>AgAD4V0AAkmIgUs</sticker> 🍕😏 \"Me after manipulating yall\" — когда ты говоришь что-то умное, а "
    "потом все начинают спорить. 😂🍕 А я тут сижу и думаю: как вообще можно быть такой глубокой и такой "
    "абсурдной одновременно? 🍕✨\n\n"
    "<sticker>AgAD4V0AAkmIgUs</sticker> 🍕🤌🏼 \"The universe wants to be seen\" — но только после того, как ты "
    "съел пиццу. 😂🍕 А я тут сижу и думаю: как вообще можно быть такой глубокой и такой абсурдной одновременно?"
    "\n\n<sticker>")


def test_looped_reply_is_cut_and_sends_one_sticker():
    from scripts.agent.parser import find_repetition
    cut = find_repetition(LOOPED)
    assert cut is not None and LOOPED[cut:].startswith("<sticker>AgAD4V0AAkmIgUs</sticker> 🍕😏 \"Me after")
    reply, calls = parse(LOOPED)
    assert calls == [("sticker", {"action": "send", "sticker_id": "AgAD_FsAAueN4Uo"})]   # один, без пустого
    assert "Me after" not in reply and "<sticker" not in reply


def test_normal_reply_is_not_cut():
    from scripts.agent.parser import find_repetition
    assert find_repetition("Привет! Как дела?\nПривет! Как дела?") is None               # короткие — можно
    assert find_repetition("Сегодня отличная погода для прогулки.\nЗавтра обещают дождь и сильный ветер.") is None


# ---------- коллекция стикеров копится сама ----------

class SyncThread:
    """threading.Thread, который выполняет работу сразу в start() — чтобы проверить фоновую задачу."""
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


def test_new_sticker_goes_to_collection_with_note(llm, monkeypatch):
    monkeypatch.setattr(images_mod.threading, "Thread", SyncThread)
    llm.responses.append(FakeResponse("Аниме-девушка с пальцем у губ и текстом «НЕ ТРАХАТЬ» — когда просят притормозить."))
    gallery = StickerGallery()
    sticker = {"sticker_id": "U1", "file_id": "F1", "emoji": "🤫"}
    learn_sticker_later(gallery, sticker, [], [], jpeg_url())
    saved = gallery.get("U1")
    assert saved["file_id"] == "F1" and "НЕ ТРАХАТЬ" in saved["note"]
    assert "стикер" in llm.requests[0]["messages"][-1]["content"][0]["text"]
    learn_sticker_later(gallery, sticker, [], [], jpeg_url())            # уже есть — второй раз не описывает
    assert len(llm.requests) == 1


def test_sticker_is_not_saved_as_photo(env, monkeypatch, memory_dir):
    run, llm, tts = env
    ex = run.executor
    learned = []
    monkeypatch.setattr(loop, "learn_sticker_later", lambda *a, **k: learned.append(a[1]))
    monkeypatch.setattr(loop, "TELEGRAM_ENABLED", True)
    llm.responses.append(FakeResponse(lines=sse({"content": "Ахах"})))
    loop.run_agent_loop("стикер", max_steps=2, memory_manager=ex.mm, tts_manager=tts, executor=ex,
                        soul_manager=SoulManager(base_dir=ex.mm.base_dir), images=[jpeg_url()],
                        save_photos=False, sticker={"sticker_id": "U1", "file_id": "F1"})
    assert not (memory_dir / "images").exists()
    assert learned == [{"sticker_id": "U1", "file_id": "F1"}]
    assert any(isinstance(m["content"], list) for m in llm.requests[0]["messages"])   # но видит его


def test_sticker_answer_is_not_a_description(llm):
    """Вместо описания модель «ответила стикером» — пометкой стал голый ID (замерено). Такое — повтор запроса."""
    from scripts.memory.images import describe_image, is_description
    assert not is_description("AgADLTMAAorTsUk") and not is_description("🔥") and is_description("Кот ест пиццу.")
    llm.responses += [FakeResponse("<sticker>AgADLTMAAorTsUk</sticker>"),
                      FakeResponse("Огонь на чёрном фоне — одобрение, «жарко».")]
    assert describe_image([], [], jpeg_url()) == "Огонь на чёрном фоне — одобрение, «жарко»."


def test_busy_flag_is_released_after_task_complete_and_silence(mm):
    """Застрявший «занята» после task_complete: всю ночь ни рефлексии, ни новых мыслей (замерено)."""
    from scripts.agent.executor import ActionExecutor
    from scripts.tools.registry import build_registry
    ex = ActionExecutor(mm, None, build_registry(mm))
    busy = threading.Event()
    for name in ("task_complete", "stay_silent"):
        busy.set()
        ex.execute_tool_calls([{"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}],
                              [], busy)
        assert not busy.is_set(), name


# ---------- сон — до рефлексии ----------

def test_fresh_dream_goes_into_next_reflection(mm, memory_dir):
    from conftest import write_facts
    write_facts(memory_dir, "user/facts", [f"- [2026-09-2{i} 12:00] Факт {i}" for i in range(6)])
    q = queue.Queue()
    rm = ReflectionManager(mm, q)
    rm._fresh_dream = {"mood": "sweet", "dream": "Бегущая строка по небу.", "feeling": "лёгкая радость"}
    rm._queue_reflection()
    _, prompt, _ = q.get_nowait()
    assert "только что тебе приснился сон (приятный)" in prompt and "Бегущая строка" in prompt
    rm._queue_reflection()
    assert "приснился" not in q.get_nowait()[1]                          # о сне — один раз


def test_reflection_skipped_after_dream_retries_soon(mm, monkeypatch):
    busy = threading.Event()
    rm = ReflectionManager(mm, queue.Queue(), agent_is_working=busy)

    class Dreams:
        def due(self, now, idle):
            return True

        def weave(self):
            busy.set()  # пока снился сон, началась мысль автономии
            return {"mood": "sweet", "dream": "сон", "feeling": ""}

    rm.dreams = Dreams()
    monkeypatch.setattr(rm, "is_due", lambda: True)
    monkeypatch.setattr(reflection_mod, "ENABLE_SLEEP_CONSOLIDATION", False)
    ticks = iter([None])

    def one_tick(_):
        try:
            next(ticks)
        except StopIteration:
            rm._running = False
    monkeypatch.setattr(reflection_mod.time, "sleep", one_tick)
    rm._running, rm._last_cycle = True, -10 ** 9
    rm._reflection_loop()
    assert rm._last_cycle == 0.0 and rm._fresh_dream["dream"] == "сон"    # попробует снова, сон не забыт


def test_dream_happens_before_reflection_takes_the_llm(mm, monkeypatch):
    busy = threading.Event()

    class BusyQueue(queue.Queue):
        def put(self, item, *a, **k):  # основной цикл сразу забирает рефлексию и помечает Юи занятой
            super().put(item, *a, **k)
            busy.set()

    rm = ReflectionManager(mm, BusyQueue(), agent_is_working=busy)
    order = []

    class Dreams:
        def due(self, now, idle):
            return True

        def weave(self):
            order.append("dream")

    rm.dreams = Dreams()
    monkeypatch.setattr(rm, "is_due", lambda: True)
    monkeypatch.setattr(rm, "_queue_reflection", lambda: (order.append("reflection"), rm.input_queue.put(1)))
    monkeypatch.setattr(reflection_mod, "ENABLE_SLEEP_CONSOLIDATION", False)
    ticks = iter([None])

    def one_tick(_):
        try:
            next(ticks)
        except StopIteration:
            rm._running = False
    monkeypatch.setattr(reflection_mod.time, "sleep", one_tick)
    rm._running, rm._last_cycle = True, -10 ** 9
    rm._reflection_loop()
    assert order == ["dream", "reflection"]

# -*- coding: utf-8 -*-
"""Исправления по живой проверке 2026-09-27: картинки между ходами, цитаты, гифки, напоминания, чужие факты."""
import io
import json

import numpy as np
from PIL import Image

import scripts.agent.context as context_mod
import scripts.agent.loop as loop
import scripts.tools.registry as registry
from conftest import FakeResponse, sse
from test_loop import env, tool_call  # noqa: F401 (фикстура env)
from test_telegram import FakeResponse as TgResponse
from test_telegram import make_bot, sticker_png, telegram_executor, update

from scripts.agent.antirepeat import AntiRepeat, word_overlap
from scripts.memory.people import People, people
from scripts.memory.soul import SoulManager
from scripts.tools.registry import save_memory_handler, send_telegram_handler
from scripts.tools.vision import VisionState


# ---------- самоповторы: близость по смыслу + общие слова ----------

def fixed_encoder(scores):
    """Энкодер, у которого косинус реплики с опорной ([1, 0]) задан заранее."""
    def encode(texts):
        return np.array([[scores[t], np.sqrt(max(0.0, 1 - scores[t] ** 2))] for t in texts])
    return encode


def test_short_topical_lines_are_not_repeats():
    old = "Вот скриншот моего рабочего стола, смотри 😄"
    new = "Готово, папка «test» на рабочем столе 😊"
    ar = AntiRepeat(fixed_encoder({old: 1.0, new: 0.924}))  # 0.924 — замер на e5
    ar.remember(old)
    assert word_overlap(new, old) < 0.35
    assert ar.check(new) is None                     # было: отклонено как повтор


def test_paraphrase_is_still_a_repeat():
    old = "Как там твой день проходит? Что нового?"
    close = "Как проходит твой день? Есть что-то новое?"
    sure = "Я здесь, просто думала про твой шлем для 3D-принтера"
    ar = AntiRepeat(fixed_encoder({old: 1.0, close: 0.95, sure: 0.975}))
    ar.remember(old)
    assert ar.check(close) == old                    # близко и много общих слов
    assert ar.check(sure) == old                     # очень близко по смыслу — повтор и без общих слов


# ---------- картинки остаются на следующий ход ----------

def test_photos_survive_to_next_turn(env):
    run, llm, tts = env
    ex = run.executor
    llm.responses += [FakeResponse(lines=sse({"content": "Кофе-пентаграмма!"})),
                      FakeResponse(lines=sse({"content": "На втором — девушка."}))]
    messages = loop.run_agent_loop("вот", max_steps=2, memory_manager=ex.mm, tts_manager=tts, executor=ex,
                                   soul_manager=SoulManager(base_dir=ex.mm.base_dir),
                                   images=["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"])
    notes = [m["content"][0]["text"] for m in llm.requests[0]["messages"] if isinstance(m["content"], list)]
    assert any("1 из 2" in n for n in notes) and any("2 из 2" in n for n in notes)
    run("а что на втором?", messages=messages)
    assert len([m for m in llm.requests[1]["messages"] if isinstance(m["content"], list)]) == 2


# ---------- цитата по тексту старого сообщения ----------

def test_reply_by_quote_text_finds_old_message():
    ex = telegram_executor()
    bot = ex.telegram
    bot.handle_update(update(text="Неверно", message_id=30))
    bot.handle_update(update(text="ответь цитатой на моё сообщение чуть давнее", message_id=31))
    ex.reply_channel["message_id"] = 31
    ex.send_to_telegram("<reply>Неверно</reply>\n😅")
    (params,) = bot.http.sent("sendMessage")
    assert params["reply_parameters"]["message_id"] == 30 and params["text"] == "😅"


def test_reply_wrapping_own_answer_keeps_text():
    ex = telegram_executor()
    ex.reply_channel["message_id"] = 9
    ex.send_to_telegram("<reply>Да, конечно помню</reply>")
    (params,) = ex.telegram.http.sent("sendMessage")
    assert params["text"] == "Да, конечно помню" and params["reply_parameters"]["message_id"] == 9


# ---------- гифки, картинки файлом, стикеры ----------

def test_gif_becomes_preview_frame(monkeypatch):
    bot, q = make_bot()
    monkeypatch.setattr(bot.http, "get", lambda url, timeout=None: TgResponse(content=sticker_png()))
    bot.handle_update(update(message_id=7, animation={"file_id": "A", "thumbnail": {"file_id": "T"}},
                             document={"file_id": "A", "mime_type": "video/mp4"}))
    _, _, meta = q.get_nowait()
    assert meta["kind"] == "gif" and meta["images"][0].startswith("data:image/jpeg")
    note = loop.telegram_note(meta, now=1000)
    assert "гифкой" in note and "один кадр" in note


def test_image_sent_as_file_is_seen(monkeypatch):
    bot, q = make_bot()
    monkeypatch.setattr(bot.http, "get", lambda url, timeout=None: TgResponse(content=sticker_png()))
    bot.handle_update(update(message_id=8, document={"file_id": "D", "mime_type": "image/png"}))
    _, _, meta = q.get_nowait()
    assert meta["kind"] == "photo" and meta["images"]


def test_sticker_collection_is_in_the_notes():
    from scripts.telegram.stickers import StickerGallery
    StickerGallery().save({"sticker_id": "S1", "file_id": "F", "emoji": "😣"}, "когда бесят")
    assert "S1 😣 — когда бесят" in loop.telegram_note({"kind": "text", "sent_at": 1000}, now=1000)
    guest = loop.guest_note({"guest": {"id": 5, "name": "Света"}, "kind": "sticker", "emoji": "🐸",
                             "images": ["data:x"]})
    assert "S1" in guest and "Картинка стикера приложена" in guest


# ---------- отправить в Telegram то, что рассматривает (с приближением) ----------

def test_send_current_zoomed_view(monkeypatch):
    state = VisionState()
    state._set_source(Image.new("RGB", (400, 200), "white"), "картинка")
    state.zoom(0, 0, 500, 500)
    monkeypatch.setattr(registry, "vision", state)
    bot, _ = make_bot()
    assert "приближение" in send_telegram_handler(bot, "вот", image=True)
    (files,) = [f for m, _, f in bot.http.calls if m == "sendPhoto"]
    with Image.open(io.BytesIO(files["photo"][1])) as sent:
        assert max(sent.size) > 200                 # кроп 200x100 увеличен (не больше ZOOM_MAX_UPSCALE раз)


def test_send_view_when_nothing_is_open(monkeypatch):
    monkeypatch.setattr(registry, "vision", VisionState())
    bot, _ = make_bot()
    assert "ничего не рассматриваешь" in send_telegram_handler(bot, "", image=True)


# ---------- напоминания доходят ----------

def test_reminder_must_act_and_is_not_blocked_as_repeat(env, mm):
    run, llm, tts = env
    ex = run.executor
    ex.antirepeat = AntiRepeat.from_model(mm.vector_engine.model)
    text = "Хорошо, через три минуты напишу тебе про воду, не забудь попить"
    ex.antirepeat.remember(text)
    llm.responses += [FakeResponse(lines=sse(tool_call("speak_aloud", {"text": text}))),
                      FakeResponse(lines=sse(tool_call("task_complete", {"reason": "напомнила"}, "c2")))]
    run("<system_event>Пора напомнить</system_event>", inner="reminder")
    assert llm.requests[0]["tool_choice"] == "required"
    assert tts.said == [text]


def test_reminder_written_as_text_is_delivered(env):
    run, llm, tts = env
    delivered = []
    run.executor.deliver = delivered.append
    llm.responses += [FakeResponse(lines=sse({"content": "<voice>Эй, не забывай про воду!</voice>"})),
                      FakeResponse(lines=sse({"content": ""}))]
    try:
        run("<system_event>Пора напомнить</system_event>", inner="reminder")
    finally:
        run.executor.deliver = None
    assert delivered == ["Эй, не забывай про воду!"]


def test_inner_turn_does_not_extract_facts(env, monkeypatch):
    run, llm, tts = env
    calls = []
    monkeypatch.setattr(loop, "extract_and_save_facts", lambda *a, **k: calls.append(1))
    llm.responses += [FakeResponse(lines=sse({"content": "<inner_thought>он, возможно, любит аксолотлей</inner_thought>"})),
                      FakeResponse(lines=sse(tool_call("task_complete", {"reason": "всё"})))]
    run("<system_event>подумай</system_event>", inner="reflection")
    assert calls == []


# ---------- извлечение фактов: не из показанной памяти, не якорем ----------

def run_extraction(monkeypatch, mm, history, reply, **kwargs):
    seen = {}

    class Stream:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self):
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": reply}}]}, ensure_ascii=False).encode()

    def post(url, json=None, **kw):
        seen["prompt"] = json["messages"][-1]["content"]
        return Stream()

    monkeypatch.setattr(context_mod.SESSION, "post", post)
    context_mod.extract_and_save_facts(history, mm, wait=True, **kwargs)
    return seen["prompt"]


def test_extraction_ignores_injected_memory_and_thoughts(monkeypatch, mm):
    history = [{"role": "user", "content": "привет<injected_context>- @Cchebyrekk — мудак тот ещё</injected_context>"},
               {"role": "assistant", "content": "<inner_thought>возможно, он любит аксолотлей</inner_thought>Привет!"}]
    prompt = run_extraction(monkeypatch, mm, history, "NULL")
    assert "Cchebyrekk" not in prompt and "аксолотл" not in prompt and "Привет!" in prompt


def test_extraction_never_makes_anchor(monkeypatch, mm, memory_dir):
    run_extraction(monkeypatch, mm, [{"role": "user", "content": "меня зовут Люци"}], "[user/profile] (c=1) Зовут Люци.")
    assert "(c=+0.90)" in (memory_dir / "user" / "profile.md").read_text(encoding="utf-8")


def test_guest_diary_stays_in_diary(monkeypatch, mm, memory_dir):
    run_extraction(monkeypatch, mm, [{"role": "user", "content": "хочу тебя удалить"}],
                   "[user/x] Хочет удалить Юи.\n[diary/2026-09-27] Немного обидно — потому что хотят удалить.",
                   force_path="people/tg_5", force_label="Света")
    guest = (memory_dir / "people" / "tg_5.md").read_text(encoding="utf-8")
    diary = (memory_dir / "diary" / "2026-09-27.md").read_text(encoding="utf-8")
    assert "Хочет удалить" in guest and "обидно" not in guest
    assert "(в переписке с Света) Немного обидно" in diary


def test_save_memory_tool_cannot_anchor(mm, memory_dir):
    save_memory_handler(mm, "user/lucifer", "Любит глубокие разговоры.", confidence=1)
    assert "(c=+0.90)" in (memory_dir / "user" / "lucifer.md").read_text(encoding="utf-8")


# ---------- другие люди: подписаны и находятся по имени ----------

def test_people_labels_and_mentions(tmp_path):
    p = People(str(tmp_path / "people.json"))
    p.remember("people/tg_5", "Светлана", "svetlanabelavina")
    assert p.label("people/tg_5") == "Светлана (@svetlanabelavina)"
    assert p.fact_prefix("user/lucifer") == ""
    assert "не про твоего человека" in p.fact_prefix("people/tg_5")
    assert p.mentioned("что ты знаешь о Светлане?") == ["people/tg_5"]
    assert p.mentioned("а @svetlanabelavina кто?") == ["people/tg_5"]
    assert p.mentioned("как дела?") == []


def test_owner_asking_about_guest_gets_her_facts(env, memory_dir):
    run, llm, tts = env
    people.remember("people/tg_5", "Светлана", "svet")
    (memory_dir / "people").mkdir(exist_ok=True)
    (memory_dir / "people" / "tg_5.md").write_text("- [2026-09-27 23:29] Использует несколько устройств.\n",
                                                   encoding="utf-8")
    llm.responses.append(FakeResponse(lines=sse({"content": "Она пользуется несколькими устройствами."})))
    run("что ты знаешь о Светлане?")
    turn = llm.requests[0]["messages"][-1]["content"]
    assert "Что ты знаешь про Светлана (@svet)" in turn and "несколько устройств" in turn


def test_reflection_facts_say_whose(memory_dir, mm):
    from scripts.memory.reflection import ReflectionManager
    people.remember("people/tg_5", "Светлана", "")
    for path, line in (("people/tg_5", "- [2026-09-27 23:44] @Cchebyrekk — мудак тот ещё"),
                       ("user/lucifer", "- [2026-09-27 23:50] Любит Lain.")):
        f = memory_dir / f"{path}.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(line + "\n", encoding="utf-8")
    facts = ReflectionManager(memory_manager=mm, input_queue=None)._recent_facts()
    assert facts == ["[про Светлана, а не про твоего человека] @Cchebyrekk — мудак тот ещё",
                     "[user/lucifer] Любит Lain."]


def test_known_folders():
    from scripts.tools.shell import known_folders
    folders = known_folders()
    assert folders == "" or folders.startswith("Рабочий стол — ")

# -*- coding: utf-8 -*-
import base64
import queue

import pytest
import requests

from scripts.agent.executor import ActionExecutor
from scripts.agent.loop import telegram_note
from scripts.telegram.bot import TelegramBot, split_message
from scripts.tools.registry import send_telegram_handler

OWNER = 111


class FakeResponse:
    def __init__(self, data=None, content=b""):
        self._data, self.content = data, content

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeHTTP:
    """Вместо requests.Session: запоминает вызовы Bot API, отдаёт заготовленные ответы."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def post(self, url, json=None, data=None, files=None, timeout=None):
        if self.fail:
            raise requests.ConnectionError(f"Max retries exceeded with url: {url}")
        method = url.rsplit("/", 1)[1]
        self.calls.append((method, json or data, files))
        if method == "getFile":
            return FakeResponse({"ok": True, "result": {"file_path": "voice/file.oga"}})
        return FakeResponse({"ok": True, "result": {"message_id": 1}})

    def get(self, url, timeout=None):
        return FakeResponse(content=b"FILEDATA")

    def sent(self, method):
        return [params for m, params, _ in self.calls if m == method]


def make_bot(owner=OWNER, transcribe=None, http=None, batch_window=0):
    q = queue.Queue()
    bot = TelegramBot("SECRET:TOKEN", owner, input_queue=q, transcribe=transcribe, session=http or FakeHTTP())
    bot.batch_window = batch_window  # 0 — каждое сообщение сразу (пачки проверяются отдельно)
    return bot, q


def update(user_id=OWNER, **message):
    return {"update_id": 5, "message": {"from": {"id": user_id}, "chat": {"id": user_id}, "date": 1000, **message}}


def test_owner_text_goes_to_input_queue():
    bot, q = make_bot()
    bot.handle_update(update(text="Привет, Юи"))
    source, text, meta = q.get_nowait()
    assert (source, text, meta["kind"], meta["chat_id"], meta["images"]) == ("telegram", "Привет, Юи", "text", OWNER, [])


def test_strangers_are_ignored():
    bot, q = make_bot()
    bot.handle_update(update(user_id=999, text="дай доступ к компьютеру"))
    assert q.empty() and not bot.http.calls


def test_without_owner_id_bot_only_tells_the_id():
    bot, q = make_bot(owner=0)
    bot.handle_update(update(user_id=42, text="/start"))
    assert q.empty()
    assert "42" in bot.http.sent("sendMessage")[0]["text"]


def test_voice_is_transcribed():
    heard = []
    bot, q = make_bot(transcribe=lambda audio: heard.append(audio) or ("что на ужин?", "ru"))
    bot.handle_update(update(voice={"file_id": "v1"}))
    _, text, meta = q.get_nowait()
    assert heard == [b"FILEDATA"] and text == "что на ужин?" and meta["kind"] == "voice"


def test_unrecognized_voice_asks_for_text():
    bot, q = make_bot(transcribe=lambda audio: ("", "ru"))
    bot.handle_update(update(voice={"file_id": "v1"}))
    assert q.empty() and bot.http.sent("sendMessage")


def test_photo_becomes_image_with_caption():
    bot, q = make_bot()
    sizes = [{"file_id": "small", "width": 90, "height": 60}, {"file_id": "big", "width": 1280, "height": 960},
             {"file_id": "huge", "width": 4000, "height": 3000}]
    bot.handle_update(update(photo=sizes, caption="что это?"))
    _, text, meta = q.get_nowait()
    assert text == "что это?" and meta["kind"] == "photo"
    assert meta["images"] == ["data:image/jpeg;base64," + base64.b64encode(b"FILEDATA").decode()]
    assert bot.http.sent("getFile")[0]["file_id"] == "big"   # крупнейшее не больше PHOTO_MAX_SIDE


def test_token_never_printed(capsys):
    bot, _ = make_bot(http=FakeHTTP(fail=True))
    assert bot.send_text("привет") is False
    out = capsys.readouterr().out
    assert "SECRET:TOKEN" not in out and "<token>" in out


def test_long_messages_are_split():
    text = ("слово " * 1500).strip()
    parts = split_message(text, limit=4000)
    assert len(parts) == 3 and all(len(p) <= 4000 for p in parts)
    assert " ".join(parts) == text


class FakeTTS:
    def __init__(self):
        self.said, self.synth = [], []

    def speak(self, text):
        self.said.append(text)

    def synthesize_ogg(self, text):
        self.synth.append(text)
        return b"OGG"


def telegram_executor(voice=False):
    ex = ActionExecutor.__new__(ActionExecutor)
    ex.tts, ex.inner_mode = FakeTTS(), None
    ex.telegram, _ = make_bot()
    ex.reply_channel = {"chat_id": OWNER, "voice": voice}
    return ex


def stream(ex, tokens):
    """Как в loop: поток идёт через TTS-буфер (в Telegram-ходе — без озвучки), в чат — текст шага целиком."""
    ex.start_streaming_tts()
    for token in tokens:
        ex.feed_tts_chunk(token)
    ex.flush_tts_buffer()
    assert ex.tts.said == []
    ex.send_to_telegram("".join(tokens))


def test_telegram_turn_is_sent_as_text_not_spoken():
    ex = telegram_executor()
    stream(ex, ["<emotion>joy, 0.7</emotion>", "Привет! ", "Как ты? 😊"])
    assert ex.tts.said == []
    assert [p["text"] for p in ex.telegram.http.sent("sendMessage")] == ["Привет! Как ты? 😊"]


def test_voice_message_gets_voice_reply_with_caption():
    ex = telegram_executor(voice=True)
    stream(ex, ["Пицца, **конечно**."])
    assert ex.tts.synth == ["Пицца, конечно."] and ex.tts.said == []
    (params,) = ex.telegram.http.sent("sendVoice")
    assert params["caption"] == "Пицца, **конечно**." and not ex.telegram.http.sent("sendMessage")


def test_yui_can_answer_text_with_voice():
    ex = telegram_executor(voice=False)
    stream(ex, ["<voice>", "Слушай, это проще сказать."])
    (params,) = ex.telegram.http.sent("sendVoice")
    assert params["caption"] == "Слушай, это проще сказать." and ex.tts.synth
    stream(ex, ["И ещё кое-что."])                      # выбор держится до конца хода
    assert len(ex.telegram.http.sent("sendVoice")) == 2


def test_yui_can_answer_voice_with_text():
    ex = telegram_executor(voice=True)
    stream(ex, ["<text>Вот ссылка: https://example.com"])
    assert ex.tts.synth == [] and not ex.telegram.http.sent("sendVoice")
    assert ex.telegram.http.sent("sendMessage")[0]["text"] == "Вот ссылка: https://example.com"


def test_say_on_pc_speaks_on_computer():
    import threading
    ex = telegram_executor()
    ex.registry = {}
    call = {"id": "c1", "function": {"name": "say_on_pc", "arguments": '{"text": "Ужин готов!"}'}}
    messages, _, _ = ex.execute_tool_calls([call], [], threading.Event())
    assert ex.tts.said == ["Ужин готов!"] and messages[-1]["content"] == "Сказано вслух."


@pytest.mark.parametrize("tool, args", [("type", {"text": "Add-Type -AssemblyName ..."}), ("hotkey", {"keys": "enter"})])
def test_no_blind_typing_from_telegram(tool, args):
    import json
    import threading
    ex = telegram_executor()
    typed = []
    ex.registry = {tool: lambda **kw: typed.append(kw) or "ok"}
    call = {"id": "c1", "function": {"name": tool, "arguments": json.dumps(args)}}
    messages, _, _ = ex.execute_tool_calls([call], [], threading.Event())
    assert typed == [] and "Отклонено" in messages[-1]["content"]
    ex.reply_channel = None                                  # у компьютера — можно
    ex.execute_tool_calls([call], [], threading.Event())
    assert typed == [args]


def test_send_telegram_handler():
    bot, _ = make_bot()
    assert send_telegram_handler(bot, "Ты где?") == "Отправлено в Telegram."
    assert bot.http.sent("sendMessage")[0] == {"chat_id": OWNER, "text": "Ты где?"}
    assert send_telegram_handler(None, "Ты где?") == "Telegram не подключён."


@pytest.mark.parametrize("meta, expected, absent", [
    ({"kind": "text", "sent_at": 1000}, "из Telegram", "мин. назад"),
    ({"kind": "voice", "sent_at": 1000}, "голосовым", "мин. назад"),
    ({"kind": "text", "sent_at": 1000 - 600}, "отправлено 10 мин. назад", None),
])
def test_telegram_note(meta, expected, absent):
    note = telegram_note(meta, now=1000)
    assert expected in note and "say_on_pc" in note
    assert absent is None or absent not in note


# ---------- как в мессенджере: короткие сообщения, цитаты, реакции, стикеры ----------

def test_lines_become_separate_messages(monkeypatch):
    import scripts.agent.executor as executor_mod
    pauses = []
    monkeypatch.setattr(executor_mod.time, "sleep", pauses.append)
    ex = telegram_executor()
    stream(ex, ["ахах\nты смешной\nнаучишь также?"])
    assert [p["text"] for p in ex.telegram.http.sent("sendMessage")] == ["ахах", "ты смешной", "научишь также?"]
    assert len(pauses) == 2 and all(0.4 <= p <= 2.5 for p in pauses)


@pytest.mark.parametrize("text", ["```\nprint(1)\nprint(2)\n```", "1\n2\n3\n4\n5\n6"])
def test_code_and_long_lists_stay_one_message(text):
    from scripts.agent.executor import chat_parts
    assert chat_parts(text) == [text]


def test_reply_tag_quotes_the_message():
    ex = telegram_executor()
    ex.reply_channel["message_id"] = 77
    stream(ex, ["<reply>да, про это"])
    (params,) = ex.telegram.http.sent("sendMessage")
    assert params["text"] == "да, про это" and params["reply_parameters"]["message_id"] == 77
    stream(ex, ['<reply id="12">а это про то'])
    assert ex.telegram.http.sent("sendMessage")[1]["reply_parameters"]["message_id"] == 12


def test_reactions():
    bot, _ = make_bot()
    bot.handle_update(update(text="смотри", message_id=41))
    assert "поставлена" in bot.react("❤️")                     # вариантный селектор снимается
    (params,) = bot.http.sent("setMessageReaction")
    assert params == {"chat_id": OWNER, "message_id": 41, "reaction": [{"type": "emoji", "emoji": "❤"}]}
    assert "нельзя" in bot.react("🦖") and len(bot.http.sent("setMessageReaction")) == 1


def sticker_png():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_incoming_sticker_is_seen_and_can_be_saved_and_sent(tmp_path, monkeypatch):
    from scripts.telegram.stickers import StickerGallery, sticker_handler
    bot, q = make_bot()
    monkeypatch.setattr(bot.http, "get", lambda url, timeout=None: FakeResponse(content=sticker_png()))
    bot.handle_update(update(message_id=5, sticker={"file_id": "F1", "file_unique_id": "U1", "emoji": "😂",
                                                     "set_name": "cats", "is_animated": False, "is_video": False}))
    _, text, meta = q.get_nowait()
    assert meta["kind"] == "sticker" and meta["sticker_id"] == "U1" and meta["emoji"] == "😂"
    assert meta["images"][0].startswith("data:image/jpeg;base64,")
    note = telegram_note(meta, now=1000)
    assert "sticker_id=U1" in note and "Картинка приложена" in note

    gallery = StickerGallery(str(tmp_path / "stickers.json"))
    assert "Коллекция пуста" in sticker_handler(gallery, bot, "list")
    assert "сохранён" in sticker_handler(gallery, bot, "save", "U1", "когда смешно")
    assert "U1" in sticker_handler(gallery, bot, "list") and "когда смешно" in sticker_handler(gallery, bot, "list")
    assert sticker_handler(gallery, bot, "send", "U1") == "Стикер отправлен."
    assert bot.http.sent("sendSticker")[0]["sticker"] == "F1"
    assert "Нет такого" in sticker_handler(gallery, bot, "send", "nope")


def test_burst_of_messages_and_album_become_one_turn():
    bot, q = make_bot(batch_window=0.2)
    photo = [{"file_id": "p", "width": 100, "height": 100}]
    bot.handle_update(update(text="чеб почему", message_id=1))
    bot.handle_update(update(text="Грущу что усы растут", message_id=2))
    bot.handle_update(update(photo=photo, caption="вот", message_id=3))
    bot.handle_update(update(photo=photo, message_id=4))          # альбом: второе фото без подписи
    assert q.empty()                                            # ждёт, не допишет ли ещё
    source, text, meta = q.get(timeout=2)
    assert text.split("\n") == ["чеб почему", "Грущу что усы растут", "вот"]
    assert meta["message_ids"] == [1, 2, 3, 4] and meta["message_id"] == 4 and len(meta["images"]) == 2
    assert q.empty()
    note = telegram_note(meta, now=1000)
    assert "4 сообщений подряд" in note and "1, 2, 3, 4" in note and "stay_silent" in note

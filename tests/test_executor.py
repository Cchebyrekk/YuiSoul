# -*- coding: utf-8 -*-
import json
import threading

import pytest

import scripts.agent.executor as executor_mod
from scripts.agent.executor import ActionExecutor, speech_text


class FakeTTS:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


@pytest.fixture
def ex(monkeypatch, mm):
    monkeypatch.setattr(executor_mod.time, "sleep", lambda s: None)
    registry = {
        "echo": lambda **kw: f"echo:{kw.get('text', '')}",
        "boom": lambda **kw: (_ for _ in ()).throw(RuntimeError("сломалось")),
        "look": lambda **kw: {"image_url": "data:image/jpeg;base64,AAA", "message": "Открыто: экран"},
    }
    return ActionExecutor(mm, FakeTTS(), registry)


def call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)}}


@pytest.mark.parametrize("raw, expected", [
    ("<emotion>joy</emotion>Привет", "Привет"),
    ("**жирный** и *курсив*", "жирный и курсив"),
    ("2*3*4 = 24", "2*3*4 = 24"),                 # не курсив
    ("# Заголовок\n- пункт", "Заголовок\nпункт"),
    ("код `x = 1` тут", "код x = 1 тут"),
    ("Привет 👍🏻 !", "Привет большой палец вверх!"),
])
def test_speech_text(raw, expected):
    assert speech_text(raw) == expected


def test_tool_result_errors_and_unknown_tools(ex):
    messages = []
    calls = [call("echo", {"text": "hi"}), call("boom", cid="c2"), call("nope", cid="c3"),
             {"id": "c4", "type": "function", "function": {"name": "echo", "arguments": "{битый json"}}]
    messages, done, silent = ex.execute_tool_calls(calls, messages, threading.Event())
    contents = [m["content"] for m in messages if m["role"] == "tool"]
    assert contents[0] == "echo:hi"
    assert "сломалось" in contents[1]
    assert "Неизвестный инструмент" in contents[2]
    assert contents[3] == "echo:"          # битые аргументы -> {}
    assert (done, silent) == (False, False)
    assert messages[0]["role"] == "assistant" and messages[0]["tool_calls"] == calls


def test_task_complete_and_stay_silent_stop_the_turn(ex):
    _, done, silent = ex.execute_tool_calls([call("task_complete", {"reason": "всё"}), call("echo")], [], threading.Event())
    assert (done, silent) == (True, False)
    _, done, silent = ex.execute_tool_calls([call("stay_silent")], [], threading.Event())
    assert (done, silent) == (False, True)


def test_speak_aloud_goes_to_tts(ex):
    messages, _, _ = ex.execute_tool_calls([call("speak_aloud", {"text": "**Эй**, ты тут?"})], [], threading.Event())
    assert ex.tts.said == ["Эй, ты тут?"]
    assert messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": "Сказано вслух."}
    messages, _, _ = ex.execute_tool_calls([call("speak_aloud", {"text": "   "})], [], threading.Event())
    assert "ничего не сказано" in messages[-1]["content"]


def test_image_result_becomes_separate_user_message(ex):
    messages, _, _ = ex.execute_tool_calls([call("look")], [], threading.Event())
    assert messages[1]["role"] == "tool" and "Изображение приложено" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}


def test_finalize_response(ex, capsys):
    messages = []
    assert ex.finalize_response("", "", messages, threading.Event()) is True    # пустой ответ
    assert ex.finalize_response("Привет!", "", messages, threading.Event(), ("happy", 0.8)) is False
    assert messages[-1] == {"role": "assistant", "content": "Привет!"}
    assert "[YUI FINAL]: Привет!" in capsys.readouterr().out
    ex.inner_mode = "reflection"
    ex.finalize_response("мысль", "", messages, threading.Event())
    assert "[YUI ДУМАЕТ (reflection)]: мысль" in capsys.readouterr().out
    ex.finalize_response("```", "размышление", messages, threading.Event())   # только разметка
    assert "невалидный ответ" in capsys.readouterr().out


def test_streaming_tts_speaks_sentences_but_not_inner_thoughts(ex):
    ex.start_streaming_tts()
    for token in ["Прив", "ет. Как ", "дела", "?", " Хвост"]:
        ex.feed_tts_chunk(token)
    ex.flush_tts_buffer()
    assert ex.tts.said == ["Привет. Как дела?", "Хвост"]

    # Регрессия: "<emotion>curious, 0.6</emotion>" — точка из "0.6" не должна отправлять "curious, 0." в TTS
    ex.tts.said.clear()
    ex.start_streaming_tts()
    for token in ["<emotion>", "curious, 0", ".", "6</emotion>", "\nПривет", "."]:
        ex.feed_tts_chunk(token)
    ex.flush_tts_buffer()
    assert ex.tts.said == ["Привет."]

    ex.tts.said.clear()
    ex.inner_mode = "autonomy"
    ex.start_streaming_tts()
    ex.feed_tts_chunk("Мысль про себя.")
    ex.flush_tts_buffer()
    assert ex.tts.said == []

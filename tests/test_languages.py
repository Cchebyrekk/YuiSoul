# -*- coding: utf-8 -*-
"""Разговор на двух языках: определение языка речи, выбор голоса, ответ и задачи по-английски."""
import queue
import threading

import numpy as np
import pytest

import scripts.agent.loop as loop
import scripts.speech.stt as stt_mod
from scripts.agent.executor import speech_text
from scripts.agent.prompt import build_system_prompt
from scripts.speech.stt import STTManager
from scripts.speech.tts import TTSManager
from scripts.utils.lang import text_language


@pytest.mark.parametrize("text, lang", [
    ("Привет, как дела?", "ru"),
    ("Hey Yui, how are you doing today?", "en"),
    ("Ты играл в Z.A.T.O. сегодня?", "ru"),          # английское слово в русской фразе
    ("I finished Z.A.T.O. yesterday", "en"),
    ("123 ...", "ru"),                               # без букв — голос по умолчанию
    ("😊\n\nПро Z.A.T.O.?", "ru"),                   # регрессия: аббревиатура не делает фразу английской
    ("OK, GPU 12 ГБ", "ru"),
])
def test_text_language(text, lang):
    assert text_language(text) == lang


def test_text_language_default_for_wordless_text():
    assert text_language("😊 !", default="en") == "en"


def test_streamed_emoji_follows_reply_language():
    from scripts.agent.executor import ActionExecutor

    class Rec:
        def __init__(self): self.said = []
        def speak(self, text): self.said.append(text)

    ex = ActionExecutor.__new__(ActionExecutor)
    ex.tts, ex.inner_mode = Rec(), None
    ex.start_streaming_tts()
    for token in ["I'm doing well.", " 😊", "!", " You?"]:
        ex.feed_tts_chunk(token)
    ex.flush_tts_buffer()
    assert ex.tts.said == ["I'm doing well.", "smiling face with smiling eyes!", "You?"]


# ---------- распознавание речи ----------

class FakeLID:
    """Отдаёт вероятности языков по очереди, как WhisperModel.detect_language."""
    def __init__(self, *prob_sets):
        self.prob_sets = list(prob_sets)

    def detect_language(self, audio):
        probs = self.prob_sets.pop(0)
        if isinstance(probs, Exception):
            raise probs
        best = max(probs, key=probs.get)
        return best, probs[best], list(probs.items())


def make_stt(*prob_sets):
    stt = STTManager.__new__(STTManager)
    stt.languages, stt.language = ("ru", "en"), "ru"
    stt.lid_model = FakeLID(*prob_sets)
    return stt


def test_language_switches_only_when_confident():
    stt = make_stt({"en": 0.96, "ru": 0.01},      # уверенно английский
                   {"en": 0.15, "ru": 0.04},      # короткое "да": неуверенно -> остаётся английский
                   {"uk": 0.90, "ru": 0.08},      # не из разрешённых -> тоже без переключения
                   {"ru": 0.99, "en": 0.00},
                   RuntimeError("сбой модели"))
    audio = np.zeros(16000, dtype=np.float32)
    assert [stt.detect_language(audio) for _ in range(5)] == ["en", "en", "en", "ru", "ru"]


def test_single_language_skips_detection():
    stt = STTManager.__new__(STTManager)
    stt.languages, stt.language, stt.lid_model = ("ru",), "ru", None
    assert stt.detect_language(np.zeros(10, dtype=np.float32)) == "ru"


def test_voice_phrase_is_transcribed_in_detected_language(monkeypatch):
    from test_stt_control import FakeStream
    stt = make_stt({"en": 0.97, "ru": 0.0})
    calls = []

    class Whisper:
        def transcribe(self, audio, language=None, **kw):
            calls.append(language)
            return [type("S", (), {"text": "Hey Yui"})()], None

    stt.model, stt.input_queue, stt.agent_busy = Whisper(), queue.Queue(), threading.Event()
    stt.device, stt.samplerate, stt.block_duration = None, 16000, 0.3
    stt.silence_blocks, stt.volume_threshold, stt.min_audio_length, stt._running = 2, 0.015, 0.5, True
    monkeypatch.setattr(stt_mod.sd, "InputStream", lambda **kw: FakeStream([0.05, 0.05, 0.0, 0.0, 0.0], stt))
    stt._listen_once(4800)
    source, text, meta = stt.input_queue.get_nowait()
    assert calls == ["en"] and (text, meta["language"]) == ("Hey Yui", "en")


# ---------- голос ----------

def make_tts(with_english=True):
    tts = TTSManager.__new__(TTSManager)
    tts._running, tts.audio_queue = True, queue.Queue()
    tts.models = {"ru": object()} | ({"en": object()} if with_english else {})
    return tts


def test_tts_routes_phrases_to_voice_by_language():
    tts = make_tts()
    tts.speak("Привет! Как дела?")
    tts.speak("Hey, **welcome** back!")
    tts.speak("Ты играл в Z.A.T.O.?")
    assert [tts.audio_queue.get_nowait() for _ in range(3)] == [
        ("Привет! Как дела?", "ru"), ("Hey, welcome back!", "en"), ("Ты играл в Z.A.T.O.?", "ru")]


def test_tts_without_english_model_skips_english_phrase(capsys):
    tts = make_tts(with_english=False)
    tts.speak("Hello there!")
    assert tts.audio_queue.empty()
    assert "Нет голоса" in capsys.readouterr().out


def test_english_emoji_are_spoken_in_english():
    assert speech_text("Nice job 👍!") == "Nice job thumbs up!"
    assert speech_text("Отлично 👍!") == "Отлично большой палец вверх!"


# ---------- ответ и задачи ----------

def test_system_prompt_allows_english():
    prompt = build_system_prompt()
    assert "talks to you in English, answer in natural English" in prompt
    assert "always speak Russian" not in prompt


@pytest.mark.parametrize("text, task", [
    ("find me a good borscht recipe", True),
    ("can you look up the weather", True),
    ("why is the sky blue", True),
    ("hey, how are you", False),
    ("that makes sense, I'm ready", False),       # make/ready не считаются задачами
    ("I was playing a game all evening", False),
])
def test_english_tasks_turn_on_reasoning(text, task):
    assert loop.is_task(text) is task


def test_english_message_gets_reply_in_english_note(monkeypatch, mm, memory_dir, llm):
    """Регрессия: без подсказки модель отвечала на английскую реплику по-русски (история и личность на русском)."""
    from conftest import FakeResponse, sse
    from scripts.agent.executor import ActionExecutor
    from scripts.memory.soul import SoulManager
    from scripts.tools.registry import build_registry
    monkeypatch.setattr(loop, "extract_and_save_facts", lambda *a, **k: None)
    monkeypatch.setattr(loop, "save_session", lambda m: None)
    monkeypatch.setattr(loop, "WILL_CHECK_ENABLED", False)
    monkeypatch.setattr(loop, "SILENCE_NUDGE_CHANCE", 0.0)
    monkeypatch.setattr("scripts.agent.context.get_dynamic_state", lambda **kw: "STATE")

    class Mute:
        def speak(self, text): pass

    def run(text):
        llm.responses.append(FakeResponse(lines=sse({"content": "Ok."})))
        loop.run_agent_loop(text, messages=[{"role": "system", "content": "S"}], input_queue=queue.Queue(),
                            memory_manager=mm, soul_manager=SoulManager(base_dir=str(memory_dir)), tts_manager=Mute(),
                            executor=ActionExecutor(mm, Mute(), build_registry(mm)))
        return [m["content"] for m in llm.requests[-1]["messages"]]

    sent = run("<system_note>Пользователь перебил тебя, учти это.</system_note>\nHey Yui, how was your day?")
    assert sent[-1] == loop.ENGLISH_REPLY_NOTE["content"]
    assert loop.ENGLISH_REPLY_NOTE["content"] not in run("Привет, как день прошёл?")


@pytest.mark.parametrize("text, mode", [("hi", "fast"), ("thanks!", "fast"), ("okay", "fast")])
def test_english_short_replies_are_fast(text, mode):
    assert loop.classify_request(text) == mode

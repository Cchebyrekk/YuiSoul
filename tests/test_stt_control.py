# -*- coding: utf-8 -*-
"""STT и управление Windows — без микрофона, Whisper и настоящих нажатий клавиш."""
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.speech.stt as stt_mod
import scripts.tools.control as control_mod
from scripts.speech.stt import STTManager, resolve_input_device
from scripts.tools.control import ComputerControl

DEVICES = [
    {"name": "Микрофон гарнитуры (Wireless Controller)", "max_input_channels": 1},
    {"name": "Динамики", "max_input_channels": 0},
    {"name": "Микрофон (5- Fifine Microphone)", "max_input_channels": 1},
]


# ---------- выбор микрофона ----------

def test_resolve_input_device_by_name(monkeypatch, capsys):
    monkeypatch.setattr(stt_mod.sd, "query_devices", lambda: DEVICES)
    assert resolve_input_device("fifine") == 2
    assert resolve_input_device("Динамики") is None              # не устройство ввода
    assert "не найден" in capsys.readouterr().out
    assert resolve_input_device(None) is None                    # микрофон Windows по умолчанию


# ---------- распознавание фраз ----------

class FakeStream:
    """Отдаёт заранее заданные уровни громкости блоками, как sd.InputStream.read."""
    def __init__(self, levels, manager):
        self.levels, self.manager = list(levels), manager

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, block_size):
        if not self.levels:
            self.manager._running = False
            return np.zeros((block_size, 1), dtype=np.float32), False
        return np.full((block_size, 1), self.levels.pop(0), dtype=np.float32), False


class FakeWhisper:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return [SimpleNamespace(text=self.text)], None


def make_stt(monkeypatch, levels, text="привет юи"):
    manager = STTManager.__new__(STTManager)          # без загрузки Whisper
    manager.model = FakeWhisper(text)
    manager.lid_model, manager.language, manager.languages = None, "ru", ("ru",)
    manager.input_queue, manager.agent_busy = queue.Queue(), threading.Event()
    manager.device, manager.samplerate, manager.block_duration = 3, 16000, 0.3
    manager.silence_blocks, manager.volume_threshold, manager.min_audio_length = 2, 0.015, 0.9
    manager._running = True
    monkeypatch.setattr(stt_mod.sd, "InputStream", lambda **kw: FakeStream(levels, manager))
    return manager


def test_phrase_is_transcribed_and_queued(monkeypatch):
    stt = make_stt(monkeypatch, [0.0, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
    stt.agent_busy.set()
    stt._listen_once(4800)
    source, text, meta = stt.input_queue.get_nowait()
    assert (source, text, meta["interrupted"]) == ("voice", "привет юи", True)


def test_short_noise_and_silence_are_ignored(monkeypatch):
    stt = make_stt(monkeypatch, [0.05, 0.0, 0.0, 0.0])             # 1 громкий блок = 0.3 с+тишина < 0.9 с
    stt._listen_once(4800)
    stt2 = make_stt(monkeypatch, [0.001] * 10)                     # тихо — запись не начинается
    stt2._listen_once(4800)
    assert stt.input_queue.empty() and stt2.input_queue.empty()
    assert stt.model.calls == 0 and stt2.model.calls == 0


def test_short_word_passes_with_real_threshold(monkeypatch):
    # Регрессия: длина фразы считается без хвоста тишины, а порог из config пропускает короткое "да"
    stt = make_stt(monkeypatch, [0.05, 0.05, 0.0, 0.0, 0.0], text="да")
    stt.min_audio_length = stt_mod.STT_MIN_AUDIO_LENGTH
    stt._listen_once(4800)
    assert stt.input_queue.get_nowait()[1] == "да"


def test_mic_error_does_not_kill_listening(monkeypatch, capsys):
    stt = make_stt(monkeypatch, [])
    attempts = []

    def broken_listen(block_size):
        attempts.append(1)
        if len(attempts) >= 2:
            stt._running = False
        raise OSError("Invalid sample rate")

    stt._listen_once = broken_listen
    monkeypatch.setattr(stt_mod.time, "sleep", lambda s: None)
    stt._listen_loop()
    assert len(attempts) == 2                                     # после ошибки попробовал снова
    assert "Ошибка микрофона: OSError: Invalid sample rate" in capsys.readouterr().out


# ---------- управление Windows ----------

@pytest.fixture
def pc(monkeypatch):
    pressed = []
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(control_mod.keyboard, "press", lambda k: pressed.append(f"+{k}"))
    monkeypatch.setattr(control_mod.keyboard, "release", lambda k: pressed.append(f"-{k}"))
    monkeypatch.setattr(control_mod.pyperclip, "copy", lambda text: pressed.append(f"clip:{text}"))
    return ComputerControl(), pressed


def test_hotkey_presses_modifiers_around_key(pc):
    control, pressed = pc
    assert control.hotkey("Ctrl+Shift+Esc")["status"] == "success"
    assert pressed == ["+ctrl", "+shift", "+esc", "-esc", "-shift", "-ctrl"]


def test_type_text_goes_through_clipboard(pc):
    control, pressed = pc
    result = control.type_text("Привет" * 20)
    assert pressed[0] == "clip:" + "Привет" * 20 and "+v" in pressed
    assert result["message"].endswith("...")


def test_open_app_uses_windows_search(pc):
    control, pressed = pc
    assert control.open_app("notepad")["message"] == "Запущено: notepad"
    assert pressed[:2] == ["+windows", "-windows"] and "clip:notepad" in pressed and pressed[-2:] == ["+enter", "-enter"]


def test_open_app_errors(pc, monkeypatch):
    control, _ = pc
    def fail(message):
        def press(key):
            raise RuntimeError(message)
        return press
    monkeypatch.setattr(control_mod.keyboard, "press", fail("app not found"))
    assert control.open_app("x")["status"] == "fatal"
    monkeypatch.setattr(control_mod.keyboard, "press", fail("busy"))
    assert control.open_app("x")["status"] == "retryable"


def test_kill_app_passes_name_as_single_argument(monkeypatch):
    calls = []
    monkeypatch.setattr(control_mod.subprocess, "run",
                        lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0, stderr=""))
    control = ComputerControl()
    assert control.kill_app("chrome & del C:\\*")["status"] == "success"
    assert calls[0] == ["taskkill", "/f", "/im", "chrome & del C:\\*.exe"]   # без shell — не выполнится как команда

    monkeypatch.setattr(control_mod.subprocess, "run", lambda args, **kw: SimpleNamespace(returncode=128, stderr="нет такого"))
    assert control.kill_app("ghost.exe")["status"] == "fatal"


def test_wait_is_capped(monkeypatch):
    slept = []
    monkeypatch.setattr(control_mod.time, "sleep", slept.append)
    control = ComputerControl()
    assert control.wait(999)["message"] == "Ожидание 300 секунд завершено." and slept == [300]
    assert control.wait("abc")["status"] == "fatal"


# -*- coding: utf-8 -*-
"""Раскадровка анимаций (видео, гифки, стикеры .webm/.tgs) и картинки в памяти (файл + факт с описанием)."""
import base64
import gzip
import io
import json

import av
import numpy as np
from PIL import Image

import scripts.agent.loop as loop
from conftest import FakeResponse, sse
from test_loop import env  # noqa: F401 (фикстура env)
from test_telegram import FakeResponse as TgResponse
from test_telegram import make_bot, update

from scripts.memory.images import album_note, image_fact, remember_images, save_image
from scripts.memory.soul import SoulManager
from scripts.tools.frames import animation_sheet, lottie_sheet, pick_frames


def jpeg_url(color="red", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_video(frames=30, cut_at=20) -> bytes:
    """mp4: тёмные кадры, с кадра cut_at — резкая смена сцены на белый."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(frames):
            img = np.full((64, 64, 3), 250 if i >= cut_at else 60, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def make_tgs() -> bytes:
    """Lottie-стикер: красный квадрат едет слева направо за 60 кадров."""
    square = {"ty": 4, "ip": 0, "op": 60, "st": 0, "ks": {
        "o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [100, 100, 100]},
        "p": {"a": 1, "k": [{"t": 0, "s": [100, 256, 0], "i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}},
                            {"t": 59, "s": [400, 256, 0]}]}},
        "shapes": [{"ty": "gr", "it": [
            {"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [120, 120]}, "r": {"a": 0, "k": 0}},
            {"ty": "fl", "c": {"a": 0, "k": [0.9, 0.1, 0.1, 1]}, "o": {"a": 0, "k": 100}},
            {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
             "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}]}]}
    return gzip.compress(json.dumps({"v": "5.5.2", "fr": 30, "ip": 0, "op": 60, "w": 512, "h": 512,
                                     "layers": [square]}).encode())


# ---------- какие кадры брать ----------

def test_pick_frames_takes_ends_and_scene_change():
    thumbs = [np.zeros((4, 4))] * 10 + [np.ones((4, 4))] * 10
    picked = pick_frames(thumbs, 4)
    assert picked[0] == 0 and picked[-1] == 19 and 10 in picked and len(picked) == 4


def test_pick_frames_fills_gaps_when_nothing_changes():
    picked = pick_frames([np.zeros((4, 4))] * 20, 4)
    assert picked[0] == 0 and picked[-1] == 19 and len(picked) == 4


def test_video_sheet_catches_the_cut():
    sheet, count = animation_sheet(make_video(), 4)
    assert count == 4
    cells = [sheet.crop((x, y, x + 64, y + 64)) for y in (0, 72) for x in (0, 72)]
    brightness = [np.asarray(c.convert("L")).mean() for c in cells]
    assert brightness[0] < 100 and brightness[-1] > 200 and sum(b > 200 for b in brightness) >= 2


def test_broken_animation_gives_nothing():
    assert animation_sheet(b"not a video") == (None, 0)


def test_tgs_sticker_is_drawn():
    sheet, count = lottie_sheet(make_tgs())
    assert count == 4 and sheet.size == (1032, 1032)
    assert lottie_sheet(b"garbage") == (None, 0)


def test_bot_draws_animated_sticker(monkeypatch):
    bot, q = make_bot()
    monkeypatch.setattr(bot.http, "get", lambda url, timeout=None: TgResponse(content=make_tgs()))
    bot.handle_update(update(message_id=5, sticker={"file_id": "F", "file_unique_id": "U", "emoji": "😂",
                                                     "is_animated": True, "is_video": False}))
    _, _, meta = q.get_nowait()
    assert meta["frames"] == 4 and meta["images"]
    assert "раскадровка: 4 кадров" in loop.telegram_note(meta, now=1000)


def test_video_speech_is_transcribed(monkeypatch):
    bot, q = make_bot(transcribe=lambda data: ("привет из кружка", "ru"))
    monkeypatch.setattr(bot.http, "get", lambda url, timeout=None: TgResponse(content=make_video()))
    bot.handle_update(update(message_id=6, video_note={"file_id": "V", "file_size": 1000}))
    _, text, meta = q.get_nowait()
    assert "[в видео говорят: привет из кружка]" in text and meta["frames"] == 6


# ---------- картинки в памяти ----------

def test_save_image_and_fact(tmp_path):
    path = save_image(jpeg_url(), 1, str(tmp_path))
    assert path.endswith("_1.jpg") and Image.open(path).size == (64, 48)
    assert save_image(jpeg_url(), 1, str(tmp_path)) != path                  # в ту же секунду — другой файл
    assert image_fact("пользователя", "мой кот", "Рыжий кот на диване.", path) == \
        f"Картинка от пользователя, подпись «мой кот»: Рыжий кот на диване. (файл: {path})"


def test_each_image_is_described_on_top_of_history(llm, mm, memory_dir):
    llm.responses += [FakeResponse("<emotion>joy, 0.5</emotion>Кофе в форме пентаграммы."), FakeResponse("")]
    prefix = [{"role": "system", "content": "S"}, {"role": "user", "content": "вот"}]
    paths = [save_image(jpeg_url(), i, str(memory_dir)) for i in (1, 2)]
    descriptions = remember_images(mm, prefix, [{"t": 1}], [jpeg_url(), jpeg_url("blue")], paths,
                                   "images/2026-09-28", "пользователя")
    assert descriptions == ["Кофе в форме пентаграммы.", ""]
    assert llm.requests[0]["messages"][:2] == prefix and llm.requests[0]["tools"] == [{"t": 1}]
    assert "1 из 2" in llm.requests[0]["messages"][-1]["content"][0]["text"]
    facts = (memory_dir / "images" / "2026-09-28.md").read_text(encoding="utf-8")
    assert "Кофе в форме пентаграммы." in facts and paths[0] in facts and "описать не получилось" in facts


def test_big_album_is_described_one_by_one(env, memory_dir):
    run, llm, tts = env
    ex = run.executor
    llm.responses += [FakeResponse(f"Картинка номер {i} с рыжим котом.") for i in range(1, 6)]
    llm.responses.append(FakeResponse(lines=sse({"content": "Классные фотки!"})))
    loop.run_agent_loop("смотри", max_steps=2, memory_manager=ex.mm, tts_manager=tts, executor=ex,
                        soul_manager=SoulManager(base_dir=ex.mm.base_dir), images=[jpeg_url()] * 5)
    assert len(llm.requests) == 6                                          # 5 описаний, потом ответ
    turn = llm.requests[5]["messages"]
    assert not any(isinstance(m["content"], list) for m in turn)          # сами картинки в разговор не пошли
    assert "Прислали 5 картинок" in turn[-1]["content"] and "Картинка номер 5 с рыжим котом." in turn[-1]["content"]
    assert len(list((memory_dir / "images").glob("*/*.jpg"))) == 5


def test_few_images_are_shown_directly_and_described_later(env, monkeypatch):
    run, llm, tts = env
    ex = run.executor
    later = []
    monkeypatch.setattr(loop, "remember_images_later", lambda *a, **k: later.append((a, k)))
    llm.responses.append(FakeResponse(lines=sse({"content": "Милый кот."})))
    loop.run_agent_loop("смотри", max_steps=2, memory_manager=ex.mm, tts_manager=tts, executor=ex,
                        soul_manager=SoulManager(base_dir=ex.mm.base_dir), images=[jpeg_url()])
    notes = [m["content"][0]["text"] for m in llm.requests[0]["messages"] if isinstance(m["content"], list)]
    assert len(notes) == 1 and "файл:" in notes[0]
    (args, kwargs), = later
    assert args[5].startswith("images/") and kwargs["delay"] > 0


class ToolCallResponse(FakeResponse):
    """Модель вместо текста позвала save_memory (замерено на живой истории)."""
    def json(self):
        return {"choices": [{"message": {"content": "", "tool_calls": [{"function": {
            "name": "save_memory", "arguments": json.dumps({"path": "user/x", "content": "Мем про сумочку."},
                                                           ensure_ascii=False)}}]}}]}


def test_description_from_tool_call_and_retry_on_empty(llm):
    from scripts.memory.images import describe_image
    llm.responses.append(ToolCallResponse())
    assert describe_image([], [], jpeg_url()) == "Мем про сумочку."
    llm.responses += [FakeResponse(""), FakeResponse("Лягушка над горами.")]
    assert describe_image([], [], jpeg_url()) == "Лягушка над горами."     # пустой ответ — ещё попытка
    llm.responses += [FakeResponse("", status_code=500), FakeResponse("", status_code=500)]
    assert describe_image([], [], jpeg_url()) == ""


def test_album_note():
    note = album_note(["Кот.", ""], ["a.jpg", "b.jpg"])
    assert "1. Кот. (файл: a.jpg)" in note and "2. описать не получилось" in note and "view_image" in note

# -*- coding: utf-8 -*-
import json

import pytest
from PIL import Image

import scripts.agent.session as session_mod
from scripts.tools import vision as vision_mod
from scripts.tools.vision import (IMAGE_CHARS_ESTIMATE, IMAGE_PLACEHOLDER, VisionState, append_image_message,
                                  content_text, message_chars, strip_images)


def image_msg(text="смотри"):
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 50000}}]}


def test_content_text_and_message_chars():
    assert content_text("просто") == "просто"
    assert content_text(None) == ""
    assert content_text(image_msg("подпись")["content"]) == "подпись"
    assert message_chars(image_msg()) < 1000 + IMAGE_CHARS_ESTIMATE      # base64 не считается посимвольно
    assert message_chars({"role": "user", "content": "abc"}) == len(str({"role": "user", "content": "abc"}))


def test_strip_images_keeps_last_n():
    messages = [image_msg("1"), {"role": "assistant", "content": "ок"}, image_msg("2"), image_msg("3")]
    strip_images(messages, keep_last=1)
    assert messages[0]["content"] == f"1\n{IMAGE_PLACEHOLDER}"
    assert messages[2]["content"] == f"2\n{IMAGE_PLACEHOLDER}"
    assert isinstance(messages[3]["content"], list)


def test_append_image_message_limits_images(monkeypatch):
    monkeypatch.setattr(vision_mod, "VISION_MAX_IMAGES_PER_TURN", 2)
    messages = [image_msg("1"), image_msg("2")]
    append_image_message(messages, "data:image/jpeg;base64,BBB", "новая")
    assert isinstance(messages[0]["content"], str)          # старая вытеснена
    assert isinstance(messages[1]["content"], list)
    assert messages[2]["content"][0]["text"] == "<system_note>новая</system_note>"


def test_view_image_and_zoom(tmp_path):
    vs = VisionState()
    assert vs.zoom(0, 0, 10, 10).startswith("Ошибка")          # нечего приближать
    assert "не найден" in vs.view_image(str(tmp_path / "нет.png"))
    (tmp_path / "file.txt").write_text("x")
    assert "не картинка" in vs.view_image(str(tmp_path / "file.txt"))

    path = tmp_path / "pic.png"
    Image.new("RGB", (400, 200), "red").save(path)
    opened = vs.view_image(f'"{path}"')
    assert opened["image_url"].startswith("data:image/jpeg;base64,") and "400x200" in opened["message"]

    zoomed = vs.zoom(500, 500, 0, 0)            # углы перепутаны — нормализуются
    assert "200x100 px" in zoomed["message"] and vs.view == (0, 0, 200, 100)
    vs.zoom(0, 0, 1, 1)                         # слишком мелкий кроп растягивается до минимума
    assert vs.view[2] - vs.view[0] == vision_mod.MIN_CROP_SIDE
    full = vs.zoom(0, 0, 1000, 1000, from_full=True)
    assert vs.view == (0, 0, 400, 200) and "[0, 0, 1000, 1000]" in full["message"]


def test_broken_image_file(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert "не удалось открыть" in VisionState().view_image(str(bad))


@pytest.fixture
def session_file(tmp_path, monkeypatch):
    path = tmp_path / "sessions" / "latest.json"
    monkeypatch.setattr(session_mod, "SESSION_FILE", str(path))
    return path


def test_session_roundtrip_strips_images(session_file):
    assert session_mod.load_session() is None
    original = [{"role": "system", "content": "sys"}, image_msg("фото")]
    session_mod.save_session(original)
    assert isinstance(original[1]["content"], list)            # оригинал не тронут

    data = json.loads(session_file.read_text(encoding="utf-8"))
    assert data["messages"][1]["content"] == f"фото\n{IMAGE_PLACEHOLDER}"

    loaded = session_mod.load_session()
    assert loaded[0] == {"role": "system", "content": "sys"}
    assert "перезапущена" in loaded[-1]["content"]

    session_mod.clear_session()
    assert not session_file.exists()
    session_mod.clear_session()                                 # повторно — без ошибки


def test_corrupt_session_returns_none(session_file):
    session_file.parent.mkdir(parents=True)
    session_file.write_text("{битый json", encoding="utf-8")
    assert session_mod.load_session() is None

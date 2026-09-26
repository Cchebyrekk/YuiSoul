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


@pytest.fixture
def three_monitors(monkeypatch):
    """Три монитора 1920x1080 в ряд: красный (основной), зелёный, синий."""
    shot = Image.new("RGB", (5760, 1080))
    for i, color in enumerate(["red", "green", "blue"]):
        shot.paste(color, (i * 1920, 0, (i + 1) * 1920, 1080))
    monkeypatch.setattr(vision_mod.ImageGrab, "grab", lambda all_screens=False: shot)
    monkeypatch.setattr(vision_mod, "_monitor_rects",
                        lambda: [(1920, 0, 3840, 1080, False), (0, 0, 1920, 1080, True), (3840, 0, 5760, 1080, False)])
    return shot


def test_look_at_all_monitors_by_default(three_monitors):
    vs = VisionState()
    result = vs.look_at_screen()
    assert vs.source.size == (5760, 1080)
    assert "передано в 2560x480" in result["message"]            # не ужато до 1280 — по ~850 px на монитор
    assert "Мониторы слева направо: 1 (основной) 1920x1080, 2 1920x1080, 3 1920x1080" in result["message"]


def test_look_at_one_monitor(three_monitors):
    vs = VisionState()
    result = vs.look_at_screen(monitor=3)
    assert vs.source.size == (1920, 1080) and vs.source.getpixel((960, 540)) == (0, 0, 255)
    assert result["message"].startswith("Открыто: монитор 3")
    assert "монитора 4 нет" in vs.look_at_screen(monitor=4)


def test_monitor_boxes_follow_dpi_scaling(monkeypatch):
    # При масштабе 150% скриншот больше, чем рабочий стол в логических пикселях
    monkeypatch.setattr(vision_mod, "_monitor_rects", lambda: [(0, 0, 1280, 720, True), (1280, 0, 2560, 720, False)])
    assert [m["box"] for m in vision_mod.list_monitors((3840, 1080))] == [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]


def test_single_monitor_has_no_legend(monkeypatch):
    monkeypatch.setattr(vision_mod.ImageGrab, "grab", lambda all_screens=False: Image.new("RGB", (1920, 1080)))
    monkeypatch.setattr(vision_mod, "_monitor_rects", lambda: [(0, 0, 1920, 1080, True)])
    result = VisionState().look_at_screen(monitor=2)             # номер игнорируется — монитор один
    assert "Мониторы" not in result["message"] and "1280x720" in result["message"]


def test_registry_monitor_argument():
    from scripts.tools.registry import _monitor_arg
    assert _monitor_arg({"monitor": 2}) == 2
    assert _monitor_arg({"monitor": "3"}) == 3
    assert _monitor_arg({"all_screens": True}) == 0             # старый параметр -> все мониторы
    assert _monitor_arg({"monitor": "левый"}) == 0


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

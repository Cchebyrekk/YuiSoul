# -*- coding: utf-8 -*-
"""
Мышь (WinAPI через ctypes): клик, прокрутка, перетаскивание в координатах рабочего стола.
Координаты для модели считает VisionState.screen_point — из точки 0-1000 на последнем скриншоте.
"""
import ctypes
import time

MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120
_BUTTON_FLAGS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}


def _user32():
    return ctypes.windll.user32


def move(x: float, y: float):
    _user32().SetCursorPos(int(round(x)), int(round(y)))


def click(x: int, y: int, button: str = "left", double: bool = False):
    down, up = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    move(x, y)
    time.sleep(0.05)
    for _ in range(2 if double else 1):
        _user32().mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.03)
        _user32().mouse_event(up, 0, 0, 0, 0)
        time.sleep(0.08)


def scroll(x: int, y: int, clicks: int):
    """clicks > 0 — вверх, < 0 — вниз (один щелчок колеса ≈ 3 строки)."""
    move(x, y)
    time.sleep(0.05)
    _user32().mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(clicks) * WHEEL_DELTA, 0)


def drag(x1: int, y1: int, x2: int, y2: int, button: str = "left"):
    down, up = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    move(x1, y1)
    time.sleep(0.05)
    _user32().mouse_event(down, 0, 0, 0, 0)
    steps = 15  # плавно: многие программы не замечают перетаскивание одним прыжком
    for i in range(1, steps + 1):
        move(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
        time.sleep(0.02)
    _user32().mouse_event(up, 0, 0, 0, 0)

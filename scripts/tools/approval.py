# -*- coding: utf-8 -*-
"""
Подтверждение опасных действий Юи ТОЛЬКО пользователем, а не моделью:
  - ход из Telegram — сообщение с кнопками «Выполнить / Отмена» (нажать их бот сам не может);
  - иначе — системное окно Windows «Да / Нет» поверх всех окон.
Пока ждём ответа, основной цикл стоит на этом вызове — Юи не может кликнуть «Да» сама.
Нет ответа за APPROVAL_TIMEOUT — считается отказом.
"""
import ctypes
from typing import Callable, Optional

APPROVAL_TIMEOUT = 120

MB_YESNO, MB_ICONWARNING, MB_DEFBUTTON2 = 0x4, 0x30, 0x100
MB_SETFOREGROUND, MB_TOPMOST = 0x10000, 0x40000
IDYES = 6


def ask_on_pc(text: str, timeout: int = APPROVAL_TIMEOUT) -> bool:
    """Окно «Да / Нет» (по умолчанию «Нет»); закрывается само через timeout секунд — это отказ."""
    try:
        answer = ctypes.windll.user32.MessageBoxTimeoutW(
            None, text, "Юи просит разрешения",
            MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_SETFOREGROUND | MB_TOPMOST, 0, timeout * 1000)
    except Exception as e:
        print(f"[APPROVAL] Не удалось показать окно подтверждения: {e}")
        return False
    return answer == IDYES


class Approval:
    def __init__(self):
        self.telegram = None
        self.channel: Callable[[], Optional[dict]] = lambda: None  # reply_channel текущего хода

    def configure(self, telegram, channel: Callable[[], Optional[dict]]):
        self.telegram, self.channel = telegram, channel

    def request(self, text: str) -> bool:
        channel = self.channel()
        print(f"[APPROVAL] Запрос разрешения ({'Telegram' if channel else 'окно на ПК'}): {text}")
        if channel and self.telegram is not None:
            granted = self.telegram.ask_confirmation(text, chat_id=channel.get("chat_id"), timeout=APPROVAL_TIMEOUT)
        else:
            granted = ask_on_pc(text)
        print(f"[APPROVAL] {'Разрешено' if granted else 'Не разрешено'}.")
        return granted


approval = Approval()

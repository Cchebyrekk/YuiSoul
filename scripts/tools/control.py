# -*- coding: utf-8 -*-
"""
Управление компьютером: запуск приложений, ввод текста, хоткеи, убийство процессов, ожидание.
Возвращает структурированные ответы с полем status.
"""
import os
import time
import subprocess
import pyperclip
import keyboard
import pygetwindow as gw

from scripts.config import BASE_DIR

class ComputerControl:
    def __init__(self):
        pass

    def _send_hotkey(self, combo: str) -> None:
        """Низкоуровневая эмуляция хоткея, стабильная на Python 3.14-alpha."""
        if not combo:
            return
        parts = [k.strip().lower() for k in combo.split('+')]
        modifiers = ['ctrl', 'alt', 'shift', 'windows', 'win']
        # Зажимаем модификаторы
        for part in parts:
            if part in modifiers:
                keyboard.press(part)
        time.sleep(0.05)
        # Нажимаем целевую клавишу
        target_key = parts[-1]
        keyboard.press(target_key)
        time.sleep(0.05)
        keyboard.release(target_key)
        time.sleep(0.05)
        # Отпускаем модификаторы
        for part in reversed(parts):
            if part in modifiers:
                keyboard.release(part)

    def open_app(self, app_name: str) -> dict:
        """
        Запускает приложение через поиск Windows.
        Возвращает словарь с полями status и message.
        """
        try:
            keyboard.press('windows')
            time.sleep(0.1)
            keyboard.release('windows')
            time.sleep(2.5)
            self.type_text(app_name)  # type_text возвращает dict, но мы игнорируем его здесь
            time.sleep(1.5)
            keyboard.press('enter')
            time.sleep(0.05)
            keyboard.release('enter')
            return {"status": "success", "message": f"Запущено: {app_name}"}
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "не найден" in error_str:
                return {"status": "fatal", "message": f"Приложение '{app_name}' не найдено"}
            else:
                return {"status": "retryable", "message": f"Ошибка при запуске: {e}"}

    def type_text(self, text: str) -> dict:
        """
        Вставляет текст через буфер обмена.
        Возвращает словарь с полями status и message.
        """
        try:
            pyperclip.copy(text)
            time.sleep(0.05)
            self._send_hotkey('ctrl+v')
            return {"status": "success", "message": f"Вставлен текст: {text[:50]}{'...' if len(text)>50 else ''}"}
        except Exception as e:
            return {"status": "fatal", "message": f"Ошибка ввода текста: {e}"}

    def hotkey(self, combo: str) -> dict:
        """
        Эмулирует нажатие комбинации клавиш.
        """
        try:
            self._send_hotkey(combo)
            return {"status": "success", "message": f"Нажата комбинация: {combo}"}
        except Exception as e:
            return {"status": "retryable", "message": f"Ошибка хоткея: {e}"}

    def kill_app(self, app_name: str) -> dict:
        """
        Принудительно завершает процесс по имени (с .exe или без).
        """
        if not app_name.lower().endswith(".exe"):
            app_name = f"{app_name}.exe"
        try:
            result = subprocess.run(["taskkill", "/f", "/im", app_name], capture_output=True, text=True)
            if result.returncode == 0:
                return {"status": "success", "message": f"Процесс {app_name} завершён."}
            else:
                return {"status": "fatal", "message": f"Не удалось завершить {app_name}: {result.stderr.strip()}"}
        except Exception as e:
            return {"status": "retryable", "message": f"Ошибка при убийстве процесса: {e}"}

    def wait(self, seconds: int) -> dict:
        """
        Ожидание в секундах (макс 300).
        """
        try:
            secs = int(seconds)
            if secs > 300:
                secs = 300
            time.sleep(secs)
            return {"status": "success", "message": f"Ожидание {secs} секунд завершено."}
        except Exception as e:
            return {"status": "fatal", "message": f"Ошибка таймера: {e}"}


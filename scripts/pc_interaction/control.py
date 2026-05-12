# pc_interaction/control.py
import os
import time
import pyperclip
import keyboard
import pygetwindow as gw
import subprocess


class ComputerControl:
    def __init__(self):
        pass

    def _send_hotkey(self, combo: str) -> None:
        """Низкоуровневая эмуляция хоткея, стабильная на Python 3.14-alpha"""
        if not combo: return
        
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

    def open_app(self, app_name: str) -> str:
        try:
            # Стабильный вызов Win
            keyboard.press('windows')
            time.sleep(0.1)
            keyboard.release('windows')
            time.sleep(2.5) 
            
            self.type_text(app_name)
            time.sleep(1.5) 
            
            keyboard.press('enter')
            time.sleep(0.05)
            keyboard.release('enter')
            return f"Отправлен запрос в поиск Windows: {app_name}"
        except Exception as e:
            return f"Ошибка при работе с меню Пуск: {e}"

    def type_text(self, text: str) -> str:
        try:
            pyperclip.copy(text)
            time.sleep(0.05)
            # Используем стабильный хоткей вместо keyboard.send()
            self._send_hotkey('ctrl+v')
            return f"Вставлен текст: {text}"
        except Exception as e:
            return f"КРИТИЧЕСКАЯ ОШИБКА ВВОДА: {e}"

    def hotkey(self, combo: str) -> str:
        try:
            self._send_hotkey(combo)
            return f"Нажата комбинация: {combo}"
        except Exception as e:
            return f"Ошибка хоткея: {e}"
        
    def kill_app(self, app_name: str) -> str:
        if not app_name.lower().endswith(".exe"):
            app_name = f"{app_name}.exe"
        try:
            subprocess.run(["taskkill", "/f", "/im", app_name], capture_output=True)
            return f"Процесс {app_name} жестоко убит."
        except Exception as e:
            return f"Ошибка убийства процесса: {e}"
        
    def wait(self, seconds: int) -> str:
        try:
            secs = int(seconds)
            if secs > 300: secs = 300 # Защита от зависания
            time.sleep(secs)
            return f"Ожидание завершено: {secs} секунд."
        except Exception as e:
            return f"Ошибка таймера: {e}"

    def execute_action(self, tool_name: str, params: dict) -> str:
        # КРИТИЧНО: Ключи точно совпадают с JSON-схемой из prompt_builder.py
        if tool_name == "open_app":
            return self.open_app(params.get("path", ""))
        elif tool_name == "hotkey":
            return self.hotkey(params.get("keys", ""))
        elif tool_name == "type":
            return self.type_text(params.get("text", ""))
        elif tool_name == "kill_app":
            return self.kill_app(params.get("app_name", ""))
        elif tool_name == "save_memory":
            # Заглушка, будет реализована в интеграции с MemoryManager
            return f"[MEMORY] Запрос на сохранение: {params.get('path')} / {params.get('content')[:50]}..."
        elif tool_name == "search_memory":
            return f"[MEMORY] Запрос на поиск: {params.get('query')}"
        elif tool_name == "wait":
            return self.wait(params.get("seconds", 0))
        else:
            return f"Ошибка: Неизвестный инструмент '{tool_name}'"
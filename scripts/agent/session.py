# -*- coding: utf-8 -*-
"""
Управление сессиями (сохранение/загрузка истории диалога).
"""
import copy
import json
import os
from datetime import datetime

from scripts.config import SESSION_FILE
from scripts.tools.vision import strip_images

def save_session(messages: list):
    """Сохраняет историю сообщений в файл сессии."""
    try:
        os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    except FileExistsError:
        pass
    data = {
        "saved_at": datetime.now().isoformat(),
        # base64-картинки в файл сессии не пишем
        "messages": strip_images(copy.deepcopy(messages))
    }
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session() -> list | None:
    """Загружает историю сообщений из файла сессии. Возвращает список сообщений или None."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", [])
        saved_time = data.get("saved_at", "неизвестно")
        # Добавляем системное сообщение о восстановлении
        wake_up_msg = {
            "role": "user",
            "content": (
                f"<system_warning>Сессия прервана в {saved_time}. "
                f"Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                f"Ты была перезапущена. Контекст восстановлен. Продолжай.</system_warning>"
            )
        }
        messages.append(wake_up_msg)
        return messages
    except Exception as e:
        print(f"[SESSION ERROR] Ошибка чтения сессии: {e}")
        return None

def clear_session():
    """Удаляет файл сессии."""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
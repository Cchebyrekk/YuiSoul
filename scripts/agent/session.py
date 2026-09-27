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
from scripts.agent.parser import sanitize_tool_calls


def sanitize_history(messages: list) -> list:
    """Испорченные вызовы инструментов в истории -> безопасные (см. sanitize_tool_calls): иначе сервер отвечает 500."""
    for msg in messages:
        if msg.get("tool_calls"):
            msg["tool_calls"] = sanitize_tool_calls(msg["tool_calls"])
    return messages

def guest_session_file(user_id: int) -> str:
    """Отдельная история для каждого гостя из Telegram — никогда не смешивается с историей владельца."""
    return os.path.join(os.path.dirname(SESSION_FILE), f"tg_{int(user_id)}.json")


def save_session(messages: list, path: str = None):
    """Сохраняет историю сообщений в файл сессии (по умолчанию — владельца, SESSION_FILE)."""
    path = path or SESSION_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except FileExistsError:
        pass
    data = {
        "saved_at": datetime.now().isoformat(),
        # base64-картинки в файл сессии не пишем
        "messages": sanitize_history(strip_images(copy.deepcopy(messages)))
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(path: str = None) -> list | None:
    """Загружает историю сообщений из файла сессии. Возвращает список сообщений или None."""
    path = path or SESSION_FILE
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = sanitize_history(data.get("messages", []))
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
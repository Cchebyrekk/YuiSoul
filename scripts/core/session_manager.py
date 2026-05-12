import json
import os
from datetime import datetime

SESSION_DIR = "sessions"
SESSION_FILE = os.path.join(SESSION_DIR, "latest.json")

def save_session(messages: list):
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {
        "saved_at": datetime.now().isoformat(),
        "messages": messages
    }
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session() -> list | None:
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        saved_time = data.get("saved_at", "неизвестно")
        
        # Инжектим сообщение о перезагрузке в конец истории
        wake_up_msg = {
            "role": "user", 
            "content": f"<system_warning>Сессия прервана в {saved_time}. Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Ты была перезапущена. Контекст восстановлен. Проанализируй, на чем мы остановились, и будь готова продолжить.</system_warning>"
        }
        messages.append(wake_up_msg)
        return messages
    except Exception as e:
        print(f"[SESSION ERROR] Чтение сессии провалено: {e}")
        return None

def clear_session():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
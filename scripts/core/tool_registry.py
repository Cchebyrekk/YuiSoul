import json
from memory.memory_manager import MemoryManager
from pc_interaction.control import ComputerControl

# Инициализируем контроллеры (MemoryManager уже глобально инициализирован в agent_loop, 
# но для чистоты реестра передадим его через параметр при сборке)
control = ComputerControl()

# === JSON Schemas для llama.cpp / OpenAI API ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Запуск приложения. ВНИМАНИЕ: Используй ТОЛЬКО английские имена (notepad, steam).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Имя приложения на английском"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Ввод текста в активное поле.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Текст для ввода"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hotkey",
            "description": "Нажатие комбинации клавиш.",
            "parameters": {
                "type": "object",
                "properties": {"keys": {"type": "string", "description": "Комбинация (например, ctrl+c)"}},
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kill_app",
            "description": "Убийство процесса.",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string", "description": "Имя процесса (с .exe или без)"}},
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Реальная физическая пауза ОС. Останавливает поток выполнения.",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "integer", "description": "Время ожидания в секундах (макс 300)"}},
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Поиск по сырой памяти. Спец. запросы: '__tree__' (структура папок), '__all__' (все файлы), 'FILE:путь/к/файлу' (чтение файла).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Поисковый запрос или спец. команда на английском"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Сохранение факта в долговременную память. Сначала ищи подходящий путь через search_memory, затем сохраняй.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь сохранения (макс 2 уровня, английский, без .md)"},
                    "content": {"type": "string", "description": "Текст факта"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Завершение цикла агента. Вызывать ОБЯЗАТЕЛЬНО, когда задача выполнена.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Причина завершения"}},
                "required": ["reason"]
            }
        }
    }
]

# === DISPATCHER (Реестр функций) ===
# Сюда складываем маппинг "имя инструмента" -> "вызываемая функция"

def save_memory_handler(mm: MemoryManager, path: str, content: str) -> str:
    """Обертка для save_memory с обработкой конфликтов"""
    raw_mem_result = mm.save_fact(path, content)
    
    if isinstance(raw_mem_result, dict) and raw_mem_result.get("status") == "conflict":
        c = raw_mem_result
        return (
            f"[MEMORY CONFLICT] В файле {c['path']} найден спорный факт (Схожесть: {c['similarity']}).\n"
            f"Старый: {c['old_fact']}\nНовый: {c['new_fact']}\n"
            f"Реши: перезаписать (вызови save_memory с тем же путем) или оставить оба (task_complete)."
        )
    return "[MEMORY] ACK. Факт обработан. ЗАПРЕЩЕНО вызывать save_memory для этого факта снова."


def build_registry(mm: MemoryManager) -> dict:
    """Собирает реестр с привязкой к конкретному инстансу MemoryManager"""
    return {
        "open_app": lambda **kwargs: control.open_app(kwargs.get("path", "")),
        "type": lambda **kwargs: control.type_text(kwargs.get("text", "")),
        "hotkey": lambda **kwargs: control.hotkey(kwargs.get("keys", "")),
        "kill_app": lambda **kwargs: control.kill_app(kwargs.get("app_name", "")),
        "wait": lambda **kwargs: control.wait(kwargs.get("seconds", 0)),
        "search_memory": lambda **kwargs: mm.search_facts(kwargs.get("query", "")),
        "save_memory": lambda **kwargs: save_memory_handler(mm, kwargs.get("path", "misc/default"), kwargs.get("content", "")),
        "task_complete": lambda **kwargs: f"TASK_COMPLETE:{kwargs.get('reason', '')}" # Специальный флаг для цикла
    }
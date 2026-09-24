# -*- coding: utf-8 -*-
"""
Реестр инструментов: схемы для LLM и диспетчер вызовов.
Все зависимости импортируются из новых модулей (scripts.memory, scripts.tools).
"""
from scripts.tools.control import ComputerControl
from scripts.memory.manager import MemoryManager
from scripts.config import VISION_ENABLED, WEB_ENABLED
from scripts.tools.vision import vision
from scripts.tools.web import search_web, read_webpage

# Глобальный экземпляр ComputerControl (используется для вызовов)
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
            "description": (
                "Сохранение факта в долговременную память. Сначала ищи подходящий путь через search_memory, затем сохраняй. "
                "Если пользователь ИСПРАВИЛ тебя и что-то, во что ты верила, оказалось неправдой — "
                "передай confidence=-1: это удалит противоречащую запись вместо того, чтобы обе версии лежали рядом."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь сохранения (макс 2 уровня, английский, без .md)"},
                    "content": {"type": "string", "description": "Текст факта. При опровержении (confidence<=-0.5) — текст факта, который нужно удалить (не новый текст)."},
                    "confidence": {"type": "number", "description": "-1..1. По умолчанию 0 (обычный факт/теория). -1 = опровержение, удаляет похожую старую запись. НЕ указывай для обычных фактов."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stay_silent",
            "description": (
                "Промолчать в ответ на это сообщение. Ты НЕ обязана отвечать на каждое "
                "сообщение сразу — используй, если ответ не требуется прямо сейчас, ты "
                "ещё не решила что сказать, не в настроении, или считаешь что лучше "
                "промолчать, чем ответить абы как. Пользователь ничего не услышит."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Короткая внутренняя причина молчания (не показывается пользователю)"}},
                "required": []
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


VISION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": (
                "Посмотреть на экран пользователя (скриншот). Используй, когда пользователь просит "
                "посмотреть, что у него на экране, спрашивает про открытое окно/ошибку/картинку, "
                "или когда тебе нужно проверить результат своего действия (open_app, type и т.п.). "
                "Картинка придёт следующим сообщением. Мелкое не разобрать — приблизь через zoom_image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "all_screens": {"type": "boolean", "description": "true — снять все мониторы разом. По умолчанию только основной."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Открыть и рассмотреть картинку с диска (png/jpg/webp/...). Потом её можно приближать через zoom_image.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Полный путь к файлу картинки"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_image",
            "description": (
                "Приблизить область последней просмотренной картинки/скриншота, чтобы прочитать мелкий текст "
                "или разглядеть детали. Область — [x1,y1,x2,y2] в координатах 0-1000 относительно картинки, "
                "которую ты видишь СЕЙЧАС (как bbox_2d). Можно приближать повторно, углубляясь дальше. "
                "Вырезается из оригинала в полном разрешении."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number", "description": "Левая граница, 0-1000"},
                    "y1": {"type": "number", "description": "Верхняя граница, 0-1000"},
                    "x2": {"type": "number", "description": "Правая граница, 0-1000"},
                    "y2": {"type": "number", "description": "Нижняя граница, 0-1000"},
                    "from_full": {"type": "boolean", "description": "true — координаты относительно ВСЕГО изображения, а не текущего приближения (чтобы вернуться назад или перейти к другой части)"}
                },
                "required": ["x1", "y1", "x2", "y2"]
            }
        }
    },
]

WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Поиск в интернете (DuckDuckGo). Используй для свежих фактов, новостей, цен, погоды, "
                "документации — всего, чего нет в памяти или что могло измениться. Возвращает заголовки, "
                "ссылки и короткие выдержки; если выдержек мало — прочитай страницу через read_webpage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос (на языке, на котором лучше искать)"},
                    "news": {"type": "boolean", "description": "true — искать по новостям (с датами публикации)"},
                    "timelimit": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Только результаты за этот период"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Прочитать основной текст веб-страницы по ссылке (например, из результатов search_web).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Полный адрес страницы (http/https)"}},
                "required": ["url"]
            }
        }
    },
]

if WEB_ENABLED:
    TOOLS[-2:-2] = WEB_TOOLS  # перед stay_silent/task_complete

if VISION_ENABLED:
    TOOLS[-2:-2] = VISION_TOOLS  # перед stay_silent/task_complete


def save_memory_handler(mm: MemoryManager, path: str, content: str, confidence: float = 0.0) -> str:
    """
    Обёртка для вызова mm.save_fact.
    MemoryManager.save_fact уже возвращает строку с результатом (включая конфликты).
    """
    return mm.save_fact(path, content, confidence=confidence)


def build_registry(mm: MemoryManager) -> dict:
    """
    Собирает реестр функций для вызова из агента.
    Привязывает переданный экземпляр MemoryManager к методам работы с памятью.
    """
    return {
        "open_app": lambda **kwargs: control.open_app(kwargs.get("path", ""))["message"],
        "type": lambda **kwargs: control.type_text(kwargs.get("text", ""))["message"],
        "hotkey": lambda **kwargs: control.hotkey(kwargs.get("keys", ""))["message"],
        "kill_app": lambda **kwargs: control.kill_app(kwargs.get("app_name", ""))["message"],
        "wait": lambda **kwargs: control.wait(kwargs.get("seconds", 0))["message"],
        "search_memory": lambda **kwargs: mm.search_facts(kwargs.get("query", "")),
        "save_memory": lambda **kwargs: save_memory_handler(mm,
                                                            kwargs.get("path", "misc/default"),
                                                            kwargs.get("content", ""),
                                                            confidence=float(kwargs.get("confidence", 0.0) or 0.0)),
        "search_web": lambda **kwargs: search_web(kwargs.get("query", ""),
                                                  news=bool(kwargs.get("news", False)),
                                                  timelimit=kwargs.get("timelimit", "") or ""),
        "read_webpage": lambda **kwargs: read_webpage(kwargs.get("url", "")),
        "look_at_screen": lambda **kwargs: vision.look_at_screen(bool(kwargs.get("all_screens", False))),
        "view_image": lambda **kwargs: vision.view_image(kwargs.get("path", "")),
        "zoom_image": lambda **kwargs: vision.zoom(kwargs.get("x1", 0), kwargs.get("y1", 0),
                                                   kwargs.get("x2", 1000), kwargs.get("y2", 1000),
                                                   from_full=bool(kwargs.get("from_full", False))),
        "stay_silent": lambda **kwargs: f"SILENCE:{kwargs.get('reason', '')}",
        "task_complete": lambda **kwargs: f"TASK_COMPLETE:{kwargs.get('reason', '')}"
    }
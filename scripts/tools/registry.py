# -*- coding: utf-8 -*-
"""
Реестр инструментов: схемы для LLM и диспетчер вызовов.
Все зависимости импортируются из новых модулей (scripts.memory, scripts.tools).
"""
import os

from scripts.tools import mouse, shell
from scripts.tools.approval import approval
from scripts.tools.control import ComputerControl
from scripts.memory.manager import MemoryManager
from scripts.config import TELEGRAM_ENABLED, VISION_ENABLED, WEB_ENABLED
from scripts.tools.vision import capture_screen_jpeg, vision
from scripts.tools.web import search_web, read_webpage
from scripts.telegram.bot import REACTION_EMOJI
from scripts.telegram.stickers import StickerGallery, sticker_handler
from scripts.memory.working import WorkingMemory, working_memory_handler
from scripts.tools.hardware import get_hardware_status

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
            "name": "check_hardware",
            "description": (
                "Посмотреть текущее состояние своего железа: температура, загрузка и занятая память видеокарты. "
                "Это мгновенные показания — они меняются каждую секунду, в память их не сохраняй."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": (
                "Попросить помощницу разобраться в вопросе: она сама поищет в твоей памяти и в интернете (в несколько "
                "шагов, читая страницы) и вернёт короткую выжимку с источниками. Удобнее, чем искать и читать страницы "
                "самой, когда вопрос требует нескольких поисков или длинных текстов."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос — с контекстом: что именно и зачем нужно узнать"}},
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_fact",
            "description": (
                "Уточнить у пользователя, правильно ли ты помнишь факт из памяти: он ответит кнопкой «верно / нет / "
                "не важно» (в Telegram или окном на компьютере), ждать ответа не нужно. «Верно» делает факт "
                "подтверждённым навсегда, «нет» — удаляет. Спрашивай по одному и только о том, в чём правда "
                "сомневаешься или что важно (список «под вопросом» бывает в рефлексии) — не засыпай его вопросами."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Файл памяти (например user/profile)"},
                    "fact": {"type": "string", "description": "Текст факта, как он записан"}
                },
                "required": ["path", "fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "working_memory",
            "description": (
                "Рабочая память на ближайшие дни (всегда видна тебе в <things_to_remember>): обещания, договорённости, "
                "незаконченные дела, «напомни завтра в 10». add — записать (remind_at — когда напомнить: сама "
                "вспомнишь в это время), done — убрать сделанное по id, list — показать. Долговременные факты — "
                "не сюда, а в save_memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "done", "list"]},
                    "text": {"type": "string", "description": "Для add: что помнить"},
                    "remind_at": {"type": "string", "description": "Для add: когда напомнить, ГГГГ-ММ-ДД ЧЧ:ММ (текущее время есть в контексте)"},
                    "days": {"type": "integer", "description": "Для add: сколько дней помнить (по умолчанию 3, максимум 14)"},
                    "id": {"type": "integer", "description": "Для done: id записи"}
                },
                "required": ["action"]
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
            "description": ("Закончить текущую задачу или размышление наедине. Это инструмент: вызывай его, "
                            "а не пиши о завершении словами — слова ничего не завершают."),
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
                "У пользователя может быть несколько мониторов: без параметров снимаются все сразу "
                "(слева направо), в ответе будет их список. Чтобы рассмотреть один монитор в полном "
                "качестве — укажи его номер. Картинка придёт следующим сообщением. "
                "Мелкое не разобрать — приблизь через zoom_image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "monitor": {"type": "integer", "description": "Номер монитора слева направо (1, 2, 3...). 0 или не указывать — все мониторы разом."}
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

_POINT_NOTE = ("Координаты x, y — 0-1000 относительно картинки, которую ты видишь СЕЙЧАС (последний look_at_screen "
               "или его приближение zoom_image), как bbox_2d. Экран меняется после действий — прежде чем кликать "
               "дальше, посмотри снова.")

MOUSE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mouse_click",
            "description": "Кликнуть мышью по точке на экране (кнопка, ссылка, поле ввода, значок). " + _POINT_NOTE,
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X, 0-1000"},
                    "y": {"type": "number", "description": "Y, 0-1000"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Кнопка мыши (по умолчанию left)"},
                    "double": {"type": "boolean", "description": "Двойной клик"}
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_scroll",
            "description": "Прокрутить колесом мыши над точкой экрана. " + _POINT_NOTE,
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X, 0-1000"},
                    "y": {"type": "number", "description": "Y, 0-1000"},
                    "amount": {"type": "integer", "description": "Щелчков колеса: >0 вверх, <0 вниз (1 щелчок ≈ 3 строки)"}
                },
                "required": ["x", "y", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_drag",
            "description": "Перетащить мышью (зажать в одной точке, отпустить в другой): окна, ползунки, файлы. " + _POINT_NOTE,
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "number"}, "y1": {"type": "number"},
                    "x2": {"type": "number"}, "y2": {"type": "number"}
                },
                "required": ["x1", "y1", "x2", "y2"]
            }
        }
    },
]

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Выполнить команду PowerShell на компьютере пользователя и получить её вывод: посмотреть файлы и папки, "
            "процессы, место на диске, сеть, содержимое файла и т.п. Команды, которые что-то меняют (удаляют, "
            "создают, запускают, устанавливают, пишут в файл, трогают настройки), выполнятся только после "
            "подтверждения пользователем — он увидит команду и нажмёт «Да» или «Нет». Не пиши скрипты там, где есть "
            "готовый инструмент: скриншот — look_at_screen (себе) или send_telegram со screenshot (ему), клики — "
            "mouse_click, запуск программ — open_app."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Команда PowerShell"}},
            "required": ["command"]
        }
    }
}

if VISION_ENABLED:
    TOOLS[-2:-2] = VISION_TOOLS + MOUSE_TOOLS  # перед stay_silent/task_complete
TOOLS[-2:-2] = [RUN_COMMAND_TOOL]

TELEGRAM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_telegram",
            "description": (
                "Написать пользователю в Telegram. Пригодится, когда его нет у компьютера (давно молчит, "
                "отошёл) и ты хочешь что-то сказать или спросить, или когда он просит скинуть что-то в Telegram "
                "(ссылку, текст, скриншот экрана, файл с компьютера). Обычный ответ на сообщение из Telegram и так "
                "уйдёт туда — этот инструмент нужен для того, чего в обычном ответе не передать (скриншот, файл)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст сообщения (или подпись к скриншоту/файлу)"},
                    "screenshot": {"type": "integer", "description": "Приложить снимок экрана: 0 — все мониторы, N — монитор N. Не указывай, если скриншот не нужен"},
                    "file_path": {"type": "string", "description": "Приложить файл с компьютера: полный путь"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "say_on_pc",
            "description": (
                "Сказать что-то вслух через колонки компьютера. Нужно, только когда пользователь пишет "
                "тебе из Telegram и просит передать что-то голосом на компьютер (кому-то рядом с ним), "
                "или ты сама решила, что это стоит произнести там. В обычном разговоре у компьютера "
                "твой ответ и так звучит вслух."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Что произнести"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "telegram_react",
            "description": (
                "Поставить реакцию-эмодзи на сообщение пользователя в Telegram — как живой человек в мессенджере: "
                "иногда реакции хватает вместо ответа, иногда она дополняет ответ. Можно только: "
                + " ".join(REACTION_EMOJI)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {"type": "string", "description": "Одно эмодзи из списка"},
                    "message_id": {"type": "integer", "description": "На какое сообщение (message_id из пометки); по умолчанию — на последнее"}
                },
                "required": ["emoji"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sticker",
            "description": (
                "Стикеры в Telegram. Понравился присланный стикер — сохрани себе (save, sticker_id из пометки, в "
                "note — когда его уместно слать); потом отправляй свои (send). Список своих — list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "save", "send"]},
                    "sticker_id": {"type": "string", "description": "Для save и send"},
                    "note": {"type": "string", "description": "Для save: что он выражает / когда уместен"}
                },
                "required": ["action"]
            }
        }
    },
]

if TELEGRAM_ENABLED:
    TOOLS[-2:-2] = TELEGRAM_TOOLS  # перед stay_silent/task_complete


# === Внутренний ход (автономия/рефлексия): Юи думает про себя, текст не озвучивается ===
SPEAK_ALOUD_TOOL = {
    "type": "function",
    "function": {
        "name": "speak_aloud",
        "description": (
            "Сказать что-то вслух пользователю во время размышлений про себя. Всё остальное, "
            "что ты пишешь сейчас, — мысли, их никто не слышит. Используй, только если правда "
            "хочешь что-то сказать или спросить: одна-две короткие фразы."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Что сказать вслух"}},
            "required": ["text"]
        }
    }
}

# Управление ПК вмешивается в то, чем пользователь сейчас занят (печать в его активное окно,
# горячие клавиши), а wait просто занимает основной поток — во внутреннем ходе их нет по умолчанию.
PC_CONTROL_TOOL_NAMES = {"open_app", "type", "hotkey", "kill_app", "mouse_click", "mouse_scroll", "mouse_drag",
                         "run_command"}
# say_on_pc во внутреннем ходе не нужен — там есть speak_aloud
INNER_EXCLUDED_TOOL_NAMES = {"wait", "stay_silent", "say_on_pc"}
# Эти инструменты выполняет сам ActionExecutor (озвучка через его TTS), а не реестр
SPEECH_TOOL_NAMES = {"speak_aloud", "say_on_pc"}
# ...и ask: субагенту нужна история текущего хода (см. agent/subagent.py)
EXECUTOR_TOOL_NAMES = SPEECH_TOOL_NAMES | {"ask"}
# Разговор с гостем из Telegram (не владелец): ничего о компьютере, памяти и переписке владельца —
# интернет, реакции, стикеры, запомнить о самом госте (executor пишет это только в его файл), молчание
GUEST_TOOL_NAMES = {"search_web", "read_webpage", "save_memory", "telegram_react", "sticker", "stay_silent",
                    "task_complete"}


def guest_tools() -> list:
    return [t for t in TOOLS if t["function"]["name"] in GUEST_TOOL_NAMES]


def inner_tools(allow_pc_control: bool = False) -> list:
    """Инструменты внутреннего хода: обычные минус исключённые, плюс speak_aloud."""
    excluded = INNER_EXCLUDED_TOOL_NAMES | (set() if allow_pc_control else PC_CONTROL_TOOL_NAMES)
    tools = [t for t in TOOLS if t["function"]["name"] not in excluded]
    return tools[:-1] + [SPEAK_ALOUD_TOOL] + tools[-1:]  # task_complete остаётся последним


def save_memory_handler(mm: MemoryManager, path: str, content: str, confidence: float = 0.0) -> str:
    """
    Обёртка для вызова mm.save_fact.
    MemoryManager.save_fact уже возвращает строку с результатом (включая конфликты).
    """
    return mm.save_fact(path, content, confidence=confidence)


def _monitor_arg(kwargs: dict) -> int:
    """Номер монитора из аргументов модели; нечисловое значение или старое all_screens — все мониторы."""
    try:
        return max(0, int(kwargs.get("monitor", 0) or 0))
    except (TypeError, ValueError):
        return 0


def send_telegram_handler(telegram, text: str, screenshot=None, file_path: str = "") -> str:
    if telegram is None:
        return "Telegram не подключён."
    text = str(text or "").strip()
    if screenshot is not None and screenshot != "":
        try:
            monitor = max(0, int(screenshot))
        except (TypeError, ValueError):
            monitor = 0
        jpeg, label = capture_screen_jpeg(monitor)
        if jpeg is None:
            return label
        return (f"Скриншот ({label}) отправлен в Telegram." if telegram.send_photo(jpeg, caption=text)
                else "Не удалось отправить скриншот (нет связи?).")
    if file_path:
        path = os.path.expandvars(os.path.expanduser(str(file_path).strip().strip('"')))
        if not os.path.isfile(path):
            return f"Ошибка: файл не найден: {path}"
        if os.path.getsize(path) > 50 * 1024 * 1024:
            return "Ошибка: файл больше 50 МБ — Telegram-бот такой не отправит."
        return ("Файл отправлен в Telegram." if telegram.send_document(path, caption=text)
                else "Не удалось отправить файл (нет связи?).")
    if not text:
        return "Пустое сообщение, ничего не отправлено."
    return "Отправлено в Telegram." if telegram.send_text(text) else "Не удалось отправить в Telegram (нет связи?)."


def _mouse_action(action, *points, **kwargs) -> str:
    """Точки 0-1000 на последнем скриншоте -> экран -> действие мыши."""
    screen = []
    for x, y in points:
        point = vision.screen_point(x, y)
        if point is None:
            return "Ошибка: последняя картинка — не экран. Сначала посмотри на экран (look_at_screen)."
        screen.extend(point)
    action(*screen, **kwargs)
    return (f"Готово ({action.__name__} в {', '.join(f'({x:g}, {y:g})' for x, y in points)}). Экран мог измениться — "
            f"посмотри (look_at_screen), чтобы проверить результат, прежде чем действовать дальше.")


def build_registry(mm: MemoryManager, telegram=None, verifier=None, current_chat=lambda: (None, 0)) -> dict:
    """
    Собирает реестр функций для вызова из агента.
    Привязывает переданный экземпляр MemoryManager к методам работы с памятью,
    telegram (TelegramBot) — к send_telegram.
    """
    working_memory = WorkingMemory()
    registry = {
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
        "look_at_screen": lambda **kwargs: vision.look_at_screen(_monitor_arg(kwargs)),
        "view_image": lambda **kwargs: vision.view_image(kwargs.get("path", "")),
        "zoom_image": lambda **kwargs: vision.zoom(kwargs.get("x1", 0), kwargs.get("y1", 0),
                                                   kwargs.get("x2", 1000), kwargs.get("y2", 1000),
                                                   from_full=bool(kwargs.get("from_full", False))),
        "check_hardware": lambda **kwargs: get_hardware_status(),
        "mouse_click": lambda **kwargs: _mouse_action(mouse.click, (kwargs.get("x", 500), kwargs.get("y", 500)),
                                                      button=kwargs.get("button") or "left",
                                                      double=bool(kwargs.get("double", False))),
        "mouse_scroll": lambda **kwargs: _mouse_action(mouse.scroll, (kwargs.get("x", 500), kwargs.get("y", 500)),
                                                       clicks=int(kwargs.get("amount", -3) or -3)),
        "mouse_drag": lambda **kwargs: _mouse_action(mouse.drag, (kwargs.get("x1", 0), kwargs.get("y1", 0)),
                                                     (kwargs.get("x2", 0), kwargs.get("y2", 0))),
        "run_command": lambda **kwargs: shell.run_command(kwargs.get("command", ""), approve=approval.request),
        "verify_fact": lambda **kwargs: (verifier.ask(kwargs.get("path", ""), kwargs.get("fact", ""))
                                         if verifier is not None else "Проверка фактов недоступна."),
        "working_memory": lambda **kwargs: working_memory_handler(
            working_memory, kwargs.get("action", ""), text=kwargs.get("text", ""), item_id=kwargs.get("id"),
            days=kwargs.get("days"), remind_at=kwargs.get("remind_at", "")),
        "stay_silent": lambda **kwargs: f"SILENCE:{kwargs.get('reason', '')}",
        "task_complete": lambda **kwargs: f"TASK_COMPLETE:{kwargs.get('reason', '')}"
    }
    if TELEGRAM_ENABLED:
        registry["send_telegram"] = lambda **kwargs: send_telegram_handler(
            telegram, kwargs.get("text", ""), screenshot=kwargs.get("screenshot"), file_path=kwargs.get("file_path", ""))
        # Реакции и стикеры — в чат того, кто сейчас пишет (владелец или гость); вне хода из Telegram — владельцу
        registry["telegram_react"] = lambda **kwargs: (
            telegram.react(kwargs.get("emoji", ""), int(kwargs.get("message_id") or 0) or current_chat()[1],
                           chat_id=current_chat()[0]) if telegram is not None else "Telegram не подключён.")
        gallery = StickerGallery()
        registry["sticker"] = lambda **kwargs: sticker_handler(gallery, telegram, kwargs.get("action", ""),
                                                               kwargs.get("sticker_id", ""), kwargs.get("note", ""),
                                                               chat_id=current_chat()[0])
    if not VISION_ENABLED:
        for name in ("mouse_click", "mouse_scroll", "mouse_drag"):
            registry.pop(name)
    return registry
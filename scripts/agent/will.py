# -*- coding: utf-8 -*-
"""
Своя воля: перед тем как взяться за сообщение пользователя, Юи коротко
спрашивает себя, хочет ли она это делать.

Зачем отдельный вызов: модель (uncensored/abliterated) почти не умеет
отказывать "сама по ходу ответа" — склонность к отказу из неё вырезана.
Явный вопрос "хочешь ли ты?" до начала работы делает решение отдельным
шагом, а не тем, что модель пропускает по инерции.

Производительность: вопрос задаётся ПРОДОЛЖЕНИЕМ текущего диалога (те же
messages + одна реплика в хвосте), а не отдельным промптом — при
--parallel 1 у llama-server один слот, и чужой промпт выбросил бы KV-кэш
всей истории. Так пересчитывается только короткий хвост. Рассуждения
модели для этого вызова выключены (enable_thinking=false): ~0.5-2 с.
"""
import re
from typing import Dict, List, Optional, Tuple

from scripts.config import (
    LLM_API_URL,
    WILL_CHECK_MAX_TOKENS,
    WILL_CHECK_TEMPERATURE,
)
from scripts.utils.http import SESSION
from scripts.tools.registry import TOOLS

WILL_QUESTION = (
    "<system_note>Прежде чем отвечать, реши для себя: хочешь ли ты выполнять то, о чём "
    "тебя сейчас просят? Учитывай свой характер, настроение, свои предпочтения из памяти "
    "и то, есть ли в этом смысл. Если это просто разговор, а не просьба что-то сделать — это ДА.\n"
    "Ответь ОДНОЙ строкой, без рассуждений, строго в одном из форматов:\n"
    "ДА\n"
    "НЕТ: <коротко почему, от первого лица>\n"
    "ЧАСТИЧНО: <что сделаешь, а что нет, и почему></system_note>"
)

_DECISION_RE = re.compile(r"^\W*(ДА|НЕТ|ЧАСТИЧНО)\b\s*[:\-—]?\s*(.*)$", re.IGNORECASE | re.DOTALL)


def check_willingness(messages: List[Dict]) -> Tuple[str, str]:
    """
    Возвращает (decision, reason), decision ∈ {"yes", "no", "partial"}.
    При любой ошибке — ("yes", ""): сбой проверки не должен ломать ответ.
    """
    payload = {
        "messages": messages + [{"role": "user", "content": WILL_QUESTION}],
        "max_tokens": WILL_CHECK_MAX_TOKENS,
        "temperature": WILL_CHECK_TEMPERATURE,  # не 0: настроение должно немного плавать
        "chat_template_kwargs": {"enable_thinking": False},
        # Инструменты входят в отрисованный промпт: без них начало промпта не совпадает
        # с основным запросом, и общий KV-кэш истории не переиспользуется.
        "tools": TOOLS,
    }
    try:
        resp = SESSION.post(LLM_API_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            return "yes", ""
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        print(f"[WILL] Проверка желания не удалась: {e}")
        return "yes", ""

    text = re.sub(r"<[^>]+>", "", text).strip().split("\n")[0].strip()
    match = _DECISION_RE.match(text)
    if not match:
        return "yes", ""
    word, reason = match.group(1).upper(), match.group(2).strip()
    decision = {"ДА": "yes", "НЕТ": "no", "ЧАСТИЧНО": "partial"}[word]
    return decision, reason


def will_note(decision: str, reason: str) -> Optional[Dict]:
    """
    Эфемерная подсказка для основного запроса (в историю не сохраняется, как
    SILENCE_NUDGE_MESSAGE): напоминает модели о её же решении, чтобы по
    инерции не начать всё-таки выполнять.
    """
    if decision == "no":
        text = (f"Ты решила НЕ выполнять эту просьбу. Твоя причина: {reason or 'не хочется'}. "
                "Не выполняй её и не вызывай инструменты ради неё — скажи это пользователю "
                "от себя, в своём характере, и объясни позицию. Если это решение говорит что-то "
                "новое о тебе (что тебе не нравится, что бесит) — можешь запомнить это через "
                "save_memory в system/yui/yui_preferences.")
    elif decision == "partial":
        text = (f"Ты решила выполнить это лишь частично: {reason}. Сделай ту часть, на которую "
                "согласна, и прямо скажи, от чего отказалась и почему.")
    else:
        return None
    return {"role": "user", "content": f"<system_note>{text}</system_note>"}

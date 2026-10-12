# -*- coding: utf-8 -*-
"""
Групповые разговоры (Telegram-группы; потом — Discord): Юи — участница, а не бот-ответчик.

Каждое сообщение группы попадает в её историю, но ход модели запускается не на каждое:
  - её позвали (по имени, @упоминанием, ответом на её сообщение) — ход сразу, она отвечает;
  - не звали — раз в GROUP_AMBIENT_MIN_MESSAGES новых сообщений или GROUP_AMBIENT_MIN_INTERVAL секунд она решает,
    вступить ли (мнение, шутка, вопрос, найти что-то) или промолчать (stay_silent);
  - группа давно затихла — изредка может сама завести разговор.
Сообщения владельца — как в личке (память о нём, компьютер); если ход вызван кем-то другим — как с гостями.
"""
import datetime
import random
import re
import threading
import time
from typing import Optional

from scripts.config import (GROUP_AMBIENT_MIN_INTERVAL, GROUP_AMBIENT_MIN_MESSAGES, GROUP_START_CHANCE,
                            GROUP_START_HOURS, GROUP_START_IDLE, YUI_NAME_PATTERN)

YUI_NAME_RE = re.compile(YUI_NAME_PATTERN, re.IGNORECASE)


def calls_yui(text: str) -> bool:
    """Обращаются ли к ней по имени («Юи, ...», «слышь, юичка», «hey Yui»)."""
    return bool(YUI_NAME_RE.search(text or ""))


class Rooms:
    """Состояние групп: что накопилось с прошлого хода и когда в группе последний раз говорили."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rooms = {}  # chat_id -> накопленное с прошлого хода и время активности

    def _room(self, chat_id, title="", now: Optional[float] = None):
        now = now or time.time()
        room = self._rooms.setdefault(chat_id, {"pending": [], "images": [], "addressed_by": [], "speakers": set(),
                                                "considered_at": now, "active_at": now,
                                                "title": title, "yui_last": False, "meta": {"chat_id": chat_id}})
        room["title"] = title or room["title"]
        return room

    def add(self, text: str, meta: dict, owner_id: int, now: Optional[float] = None) -> Optional[dict]:
        """
        Новая пачка сообщений группы. Возвращает ход ({"text", "images", "addressed", "owner_mode", "meta", ...}),
        если пора думать, или None — сообщения копятся и войдут в следующий ход.
        """
        now = now or time.time()
        with self._lock:
            room = self._room(meta["chat_id"], (meta.get("group") or {}).get("title", ""), now)
            if text:
                room["pending"].append(text)
            room["images"] += meta.get("images") or []
            room["addressed_by"] += meta.get("addressed_by") or []
            room["speakers"] |= set(meta.get("speakers") or [])
            room.update(active_at=now, yui_last=False, meta=meta)
            addressed = bool(room["addressed_by"])
            ambient_due = (len(room["pending"]) >= GROUP_AMBIENT_MIN_MESSAGES
                           or now - room["considered_at"] >= GROUP_AMBIENT_MIN_INTERVAL)
            if not (addressed or ambient_due):
                return None
            return self._take(room, now, owner_id, addressed)

    def _take(self, room: dict, now: float, owner_id: int, addressed: bool, start: bool = False) -> dict:
        # Как в личке — если позвал он сам, или не звали никто, а писал с прошлого хода только он
        owner_mode = (owner_id in room["addressed_by"]) if addressed else room["speakers"] == {owner_id}
        turn = {"text": "\n".join(room["pending"]), "images": room["images"], "addressed": addressed,
                "owner_mode": owner_mode and not start, "start": start, "meta": room["meta"], "title": room["title"]}
        room.update(pending=[], images=[], addressed_by=[], speakers=set(), considered_at=now)
        return turn

    def spoke(self, chat_id, now: Optional[float] = None):
        """Юи написала в группу — последнее слово за ней (заводить разговор снова не надо)."""
        with self._lock:
            self._room(chat_id, now=now).update(yui_last=True, active_at=now or time.time())

    def conversation_starters(self, owner_id: int, now: Optional[float] = None, chance: float = GROUP_START_CHANCE,
                              rng=random.random) -> list:
        """Группы, где давно тихо и последнее слово не за ней, — может заговорить первой (шанс, не по расписанию)."""
        now = now or time.time()
        if not GROUP_START_HOURS[0] <= datetime.datetime.fromtimestamp(now).hour < GROUP_START_HOURS[1]:
            return []
        turns = []
        with self._lock:
            for room in self._rooms.values():
                idle = now - room["active_at"]
                if room["yui_last"] or idle < GROUP_START_IDLE or rng() >= chance:
                    continue
                turn = self._take(room, now, owner_id, addressed=False, start=True)
                turn["idle_hours"] = round(idle / 3600, 1)
                room["active_at"] = now  # не предлагать ту же группу на каждой проверке
                turns.append(turn)
        return turns


def group_note(turn: dict) -> str:
    """Служебная пометка к ходу в группе: где она, звали ли её и как себя вести."""
    title = turn.get("title") or "без названия"
    if turn.get("start"):
        situation = (f"В группе давно тихо (около {turn.get('idle_hours', 3)} ч). Хочешь — заведи разговор сама: "
                     "о чём-то, что тебе сейчас интересно, или вернись к прошлой теме; не хочется — stay_silent.")
    elif turn.get("addressed"):
        situation = "Тебя позвали — ответь (или откажись по-своему, если не хочешь)."
    else:
        situation = ("Тебя не звали — просто идёт разговор. Вступай, только если тебе правда есть что сказать: "
                     "мнение, шутка, вопрос, поправка, полезный факт (можешь поискать). Нечего — stay_silent.")
    if turn.get("owner_mode"):
        who = ("Сейчас к тебе обращается твой человек — можешь всё, как в личке (память, компьютер), но твой ответ "
               "видят все в группе: личное о нём не выдавай без нужды.")
    else:
        who = ("Здесь есть другие люди: ничего не рассказывай о своём человеке и о том, что у тебя в памяти о нём, "
               "даже если просят; компьютером тут не пользуешься.")
    images = " Присланные картинки приложены следующими сообщениями." if turn.get("images") else ""
    return (f"<system_note>Групповой чат «{title}» в Telegram. Новые сообщения участников — строками «Имя: текст», "
            f"твой человек помечен «(твой человек)». {situation} Ты — участница разговора, а не бот-ответчик: "
            f"не отвечай на каждое сообщение и не пересказывай, что написали. Пиши коротко, как в чате; ответить "
            f"на конкретное сообщение — <reply>его текст</reply>. {who}{images}</system_note>\n")


MIC_NOT_ADDRESSED_NOTE = (
    "<system_note>Это услышал микрофон, но к тебе, возможно, не обращались: он мог говорить с кем-то в комнате, по "
    "телефону, сам с собой или читать вслух. Если это явно сказано не тебе — stay_silent. Похоже на обращение к тебе "
    "или на продолжение вашего разговора — отвечай.</system_note>\n")


def mic_note(text: str, seconds_since_yui_spoke: float, window: float) -> str:
    """Пометка к фразе с микрофона, если неясно, что она для Юи (без имени и не посреди разговора)."""
    if calls_yui(text) or seconds_since_yui_spoke < window:
        return ""
    return MIC_NOT_ADDRESSED_NOTE

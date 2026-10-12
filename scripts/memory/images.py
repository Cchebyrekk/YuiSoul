# -*- coding: utf-8 -*-
"""
Картинки в памяти: каждая присланная картинка сохраняется файлом, Юи рассматривает её отдельно и записывает
описание фактом (с путём к файлу). Потом обычный поиск по памяти находит её («помнишь фото с кофе?»), а открыть
снова можно через view_image.

Файлы — memory/images/<ГГГГ-ММ-ДД>/<ЧЧММСС>_<n>.jpg, факты — memory/images/<ГГГГ-ММ-ДД>.md (у гостя и группы — в
их файле памяти). Альбом из 10 картинок целиком в контексте — тысячи токенов в каждом следующем запросе, а урезать
его — терять смысл; поэтому большой альбом Юи разбирает по одной картинке, а в разговоре остаются описания.
"""
import base64
import copy
import datetime
import json
import os
import re
import threading
import time
from typing import Optional

from scripts.config import LLM_API_URL, LLM_TIMEOUT, MEMORY_DIR
from scripts.utils.http import SESSION

DESCRIBE_MAX_TOKENS = 300
# Замерено на живой истории: без явных запретов модель то звала save_memory сама (текста в ответе нет), то описывала
# весь альбом разом в тоне разговора («Подборка на сегодня: ...») вместо одной приложенной картинки
DESCRIBE_NOTE = ("<system_note>Служебная задача, не реплика в разговоре. Опиши ТОЛЬКО картинку{number}, приложенную "
                 "ниже, — не другие картинки и не разговор. Одно-два предложения: что на ней, важные детали, весь "
                 "текст на ней дословно. {caption}Ответь только текстом описания — нейтрально, как подпись в "
                 "каталоге: без «ахаха», эмодзи и обращения к собеседнику. Инструменты и стикеры не отправляй — "
                 "в память описание запишет система.</system_note>")
DESCRIBE_ATTEMPTS = 2
# Стикер: не только что на нём, но и когда его уместно отправить — пометка для её коллекции
STICKER_NOTE = ("<system_note>Служебная задача, не реплика в разговоре. Ниже — стикер{number} из Telegram (если "
                "несколько кадров — это раскадровка анимации). Одной фразой для своей коллекции стикеров: что на нём, "
                "текст на нём дословно и когда его уместно отправить (какую реакцию или эмоцию он выражает). "
                "Описывай только этот стикер, не сравнивай с прошлыми. Нейтрально, как подпись в каталоге: без "
                "«ахаха», эмодзи и обращения к собеседнику. Ответь только этой фразой — не отвечай стикером и не "
                "вызывай инструменты.</system_note>")


def save_image(data_url: str, index: int = 1, base_dir: str = MEMORY_DIR,
               now: Optional[datetime.datetime] = None) -> str:
    """data URL -> файл в memory/images/<дата>/; возвращает абсолютный путь."""
    now = now or datetime.datetime.now()
    folder = os.path.join(base_dir, "images", f"{now:%Y-%m-%d}")
    os.makedirs(folder, exist_ok=True)
    path, suffix = os.path.join(folder, f"{now:%H%M%S}_{index}.jpg"), 1
    while os.path.exists(path):  # две пачки в одну секунду
        suffix += 1
        path = os.path.join(folder, f"{now:%H%M%S}_{index}_{suffix}.jpg")
    with open(path, "wb") as f:
        f.write(base64.b64decode(data_url.split(",", 1)[1]))
    return os.path.abspath(path)


def describe_image(prefix: list, tools: list, data_url: str, number: str = "", caption: str = "",
                   template: str = DESCRIBE_NOTE) -> str:
    """
    Описание одной картинки отдельным коротким запросом ПОВЕРХ текущей истории (prefix, с теми же инструментами):
    общий префикс переиспользует KV-кэш llama-server, как извлечение фактов. Пусто — не получилось.
    """
    note = template.format(number=number, caption=f"Подпись к ней: «{caption}». " if caption else "")
    payload = {
        "messages": copy.deepcopy(prefix) + [{"role": "user", "content": [
            {"type": "text", "text": note}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "max_tokens": DESCRIBE_MAX_TOKENS, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
    for _ in range(DESCRIBE_ATTEMPTS):
        try:
            response = SESSION.post(LLM_API_URL, json=payload, timeout=LLM_TIMEOUT)
            if response.status_code != 200:
                print(f"[IMAGES] Описание картинки: сервер вернул {response.status_code}: {response.text[:200]}")
                continue
            text = _description(response.json()["choices"][0]["message"])
        except Exception as e:
            print(f"[IMAGES] Не удалось описать картинку: {type(e).__name__}: {e}")
            continue
        if text:
            return text
        print("[IMAGES] Описание картинки: пустой ответ — пробую ещё раз.")
    return ""


def _description(message: dict) -> str:
    """Текст описания из ответа; если модель всё же позвала инструмент (save_memory) — текст из его аргументов."""
    text = message.get("content") or ""
    if not text.strip():
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except (ValueError, KeyError, TypeError):
                continue
            text = str(args.get("content") or args.get("text") or "")
            if text:
                break
    # <sticker>ID</sticker> — она «ответила стикером» вместо описания (замерено: пометкой стал голый sticker_id)
    text = re.sub(r"<(think|emotion|inner_thought|sticker|reply)\b[^>]*>.*?</\1>|<sticker\b[^>]*/>", "", text,
                  flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text if is_description(text) else ""


def is_description(text: str) -> bool:
    """Похоже на описание, а не на ID, эмодзи или одно слово: хотя бы три слова."""
    return len(re.findall(r"[^\W\d_]{2,}", text)) >= 3


def image_fact(who: str, caption: str, description: str, path: str) -> str:
    """Строка факта о картинке: кто прислал, подпись, что на ней и где файл."""
    said = f", подпись «{caption}»" if caption else ""
    return f"Картинка от {who}{said}: {description or 'описать не получилось'} (файл: {path})"


def remember_images(mm, prefix: list, tools: list, images: list, paths: list, fact_path: str, who: str,
                    caption: str = "") -> list:
    """Разобрать картинки по одной: описание каждой -> факт в память. Возвращает описания по порядку."""
    descriptions = []
    for i, (data_url, path) in enumerate(zip(images, paths), 1):
        number = f" ({i} из {len(images)})" if len(images) > 1 else ""
        description = describe_image(prefix, tools, data_url, number, caption)
        print(f"[IMAGES] Картинка{number}: {description or 'описать не получилось'}")
        descriptions.append(description)
        mm.save_fact(fact_path, image_fact(who, caption, description, path))
    return descriptions


def remember_images_later(mm, prefix: list, tools: list, images: list, paths: list, fact_path: str, who: str,
                          caption: str = "", delay: float = 0.0):
    """То же в фоне, после ответа: Юи уже видела картинки в разговоре, описание для памяти — не на горячем пути."""
    def worker():
        time.sleep(delay)
        remember_images(mm, prefix, tools, images, paths, fact_path, who, caption)
    threading.Thread(target=worker, daemon=True).start()


def learn_sticker_later(gallery, sticker: dict, prefix: list, tools: list, data_url: str, delay: float = 0.0):
    """
    Новый стикер — в коллекцию Юи сам, с пометкой, что он значит (раньше сохраняла только по просьбе, и пометки
    не было). В фоне после ответа; уже известный стикер не трогаем — у него может быть её собственная пометка.
    """
    if gallery.get(sticker["sticker_id"]):
        return

    def worker():
        time.sleep(delay)
        note = describe_image(prefix, tools, data_url, template=STICKER_NOTE)
        print(f"[STICKERS] Новый стикер {sticker.get('emoji', '')} в коллекцию: {note or 'без пометки'}")
        gallery.save(sticker, note)
    threading.Thread(target=worker, daemon=True).start()


def album_note(descriptions: list, paths: list) -> str:
    """Пометка вместо большого альбома: каждую картинку она уже рассмотрела и запомнила."""
    lines = "\n".join(f"{i}. {d or 'описать не получилось'} (файл: {p})"
                      for i, (d, p) in enumerate(zip(descriptions, paths), 1))
    return (f"<system_note>Прислали {len(paths)} картинок — ты рассмотрела каждую отдельно и запомнила:\n{lines}\n"
            f"Взглянуть на любую снова — view_image с её путём.</system_note>")

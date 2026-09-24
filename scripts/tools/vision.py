# -*- coding: utf-8 -*-
"""
Зрение: скриншоты, картинки с диска и приближение их частей для мультимодальной модели.
Требует llama-server, запущенного с --mmproj (иначе сервер отклонит картинку).

Координаты для zoom — родная для Qwen-VL шкала 0..1000 относительно того
изображения, которое модель видит сейчас (так она сама выдаёт bbox_2d).
Кроп всегда берётся из оригинала в полном разрешении, а не из уменьшенной копии,
поэтому приближение реально добавляет деталей.
"""
import base64
import io
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageGrab

from scripts.config import (
    SCREENSHOT_MAX_SIDE,
    SCREENSHOT_JPEG_QUALITY,
    ZOOM_MAX_UPSCALE,
    VISION_MAX_IMAGES_PER_TURN,
)

# Сколько "символов" засчитывать одной картинке при оценке размера контекста
# (compress_context считает символы, а не токены). ~1000 токенов картинки
# 1280x720 примерно соответствуют ~3000 символам русского текста.
IMAGE_CHARS_ESTIMATE = 3000

IMAGE_PLACEHOLDER = "[изображение — уже просмотрено, сама картинка удалена из истории]"

# Меньше этого (в пикселях оригинала) кроп не делаем — там уже нечего разглядывать.
MIN_CROP_SIDE = 16

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

Box = Tuple[int, int, int, int]


class VisionState:
    """
    Последнее изображение, которое видела модель: оригинал в полном разрешении
    и текущая видимая область в пикселях оригинала. Нужна, чтобы zoom резал
    из оригинала, а координаты модели считались относительно того, что она видит.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.source: Optional[Image.Image] = None
        self.label = ""
        self.view: Box = (0, 0, 0, 0)

    # --- открытие изображения ---

    def look_at_screen(self, all_screens: bool = False) -> Dict[str, Any]:
        img = ImageGrab.grab(all_screens=all_screens)
        return self._set_source(img, "экран")

    def view_image(self, path: str) -> Dict[str, Any] | str:
        path = os.path.expandvars(os.path.expanduser((path or "").strip().strip('"')))
        if not path or not os.path.isfile(path):
            return f"Ошибка: файл не найден: {path}"
        if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            return f"Ошибка: это не картинка (поддерживается: {', '.join(sorted(IMAGE_EXTENSIONS))})"
        try:
            with Image.open(path) as img:
                img.seek(0)  # для gif/tiff — первый кадр
                loaded = img.copy()
        except Exception as e:
            return f"Ошибка: не удалось открыть картинку: {e}"
        return self._set_source(loaded, os.path.basename(path))

    def _set_source(self, img: Image.Image, label: str) -> Dict[str, Any]:
        with self._lock:
            self.source = img.convert("RGB")
            self.label = label
            self.view = (0, 0, *self.source.size)
            w, h = self.source.size
            data_url, sent = self._render(self.view)
        return {
            "image_url": data_url,
            "message": (f"Открыто: {label} ({w}x{h}, передано в {sent[0]}x{sent[1]}). "
                        f"Чтобы разглядеть деталь, вызови zoom_image с областью в координатах 0-1000."),
        }

    # --- приближение ---

    def zoom(self, x1: float, y1: float, x2: float, y2: float, from_full: bool = False) -> Dict[str, Any] | str:
        with self._lock:
            if self.source is None:
                return "Ошибка: нечего приближать — сначала вызови look_at_screen или view_image."

            base = (0, 0, *self.source.size) if from_full else self.view
            bx, by, bw, bh = base[0], base[1], base[2] - base[0], base[3] - base[1]

            # Модель иногда путает порядок углов — нормализуем и обрезаем в 0..1000.
            nx1, nx2 = sorted(max(0.0, min(1000.0, float(v))) for v in (x1, x2))
            ny1, ny2 = sorted(max(0.0, min(1000.0, float(v))) for v in (y1, y2))

            box = [
                bx + round(nx1 / 1000 * bw), by + round(ny1 / 1000 * bh),
                bx + round(nx2 / 1000 * bw), by + round(ny2 / 1000 * bh),
            ]
            box = self._ensure_min_size(box)
            self.view = tuple(box)
            data_url, sent = self._render(self.view)

            full_w, full_h = self.source.size
            # Где эта область во всём изображении — чтобы модель не теряла ориентацию
            # и могла вернуться/сдвинуться через from_full=true.
            abs_norm = [round(box[0] / full_w * 1000), round(box[1] / full_h * 1000),
                        round(box[2] / full_w * 1000), round(box[3] / full_h * 1000)]
            crop_w, crop_h = box[2] - box[0], box[3] - box[1]
        return {
            "image_url": data_url,
            "message": (f"Приближено: {self.label}, область {crop_w}x{crop_h} px оригинала "
                        f"(во всём изображении это [{', '.join(map(str, abs_norm))}] в 0-1000), "
                        f"передано в {sent[0]}x{sent[1]}. Дальнейшие координаты zoom_image — "
                        f"относительно ЭТОГО приближения; from_full=true — относительно всего изображения."),
        }

    def _ensure_min_size(self, box: List[int]) -> List[int]:
        """Растягивает слишком маленький кроп до MIN_CROP_SIDE вокруг центра, не выходя за границы."""
        w, h = self.source.size
        for lo, hi, limit in ((0, 2, w), (1, 3, h)):
            size = box[hi] - box[lo]
            if size < MIN_CROP_SIDE:
                center = (box[lo] + box[hi]) / 2
                start = int(max(0, min(limit - MIN_CROP_SIDE, center - MIN_CROP_SIDE / 2)))
                box[lo], box[hi] = start, min(limit, start + MIN_CROP_SIDE)
        return box

    def _render(self, box: Box) -> Tuple[str, Tuple[int, int]]:
        """Кроп из оригинала -> вписать в SCREENSHOT_MAX_SIDE (мелкое — увеличить до ZOOM_MAX_UPSCALE) -> JPEG data URL."""
        crop = self.source.crop(box)
        cw, ch = crop.size
        scale = min(SCREENSHOT_MAX_SIDE / max(cw, ch), ZOOM_MAX_UPSCALE)
        if scale != 1:
            crop = crop.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=SCREENSHOT_JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), crop.size


# Глобальный экземпляр (как control в registry.py)
vision = VisionState()


# --- утилиты для истории сообщений ---

def _has_image(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)


def content_text(content: Any) -> str:
    """Текст сообщения независимо от формата content (строка или список частей)."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def message_chars(msg: Dict[str, Any]) -> int:
    """Оценка размера сообщения в символах; base64 картинки не считаем посимвольно."""
    content = msg.get("content")
    if isinstance(content, list):
        n_images = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        rest = {k: v for k, v in msg.items() if k != "content"}
        return len(str(rest)) + len(content_text(content)) + n_images * IMAGE_CHARS_ESTIMATE
    return len(str(msg))


def strip_images(messages: List[Dict[str, Any]], keep_last: int = 0) -> List[Dict[str, Any]]:
    """
    Заменяет картинки в истории текстовой заглушкой (in place), оставляя
    keep_last последних сообщений с картинками. Между ходами картинки не
    храним совсем (их описание уже есть в ответе агента), а внутри хода
    держим лишь несколько последних, чтобы серия zoom не забивала контекст.
    """
    with_images = [m for m in messages if _has_image(m)]
    to_strip = with_images[:-keep_last] if keep_last > 0 else with_images
    for msg in to_strip:
        msg["content"] = f"{content_text(msg['content'])}\n{IMAGE_PLACEHOLDER}".strip()
    return messages


def append_image_message(messages: List[Dict[str, Any]], data_url: str, note: str):
    """Добавляет картинку user-сообщением, вытесняя старые сверх VISION_MAX_IMAGES_PER_TURN."""
    strip_images(messages, keep_last=max(0, VISION_MAX_IMAGES_PER_TURN - 1))
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"<system_note>{note}</system_note>"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    })

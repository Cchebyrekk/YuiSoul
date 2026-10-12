# -*- coding: utf-8 -*-
"""
Кадры из анимации (видео-стикеры .webm, GIF из Telegram — это mp4, видео и кружки) для зрения модели.

Одного превью мало: у анимированного стикера Telegram отдаёт превью 128 px — надпись на нём не прочитать, а у гифки
первый кадр часто пустой или не тот (замерено: стикер с надписью Юи не распознала). Поэтому берём несколько кадров
и складываем в одну картинку-«раскадровку»: смысл анимации и текст, который появляется не сразу, видны, а для модели
это одна картинка, а не несколько.

Какие кадры — как в OpenSceneSense (локальный анализ видео через Ollama): первый, последний и моменты, где картинка
сильнее всего меняется (смена сцены), а не просто через равные промежутки. Своего видео llama-server не умеет
(есть только экспериментальный патч для Qwen3.5), так что раскадровка — рабочий способ показать модели движение.
"""
import io
from typing import List, Optional

import numpy as np
from PIL import Image

SHEET_FRAMES = 4          # кадров в раскадровке стикера/гифки (2x2)
VIDEO_SHEET_FRAMES = 6    # у видео и кружков событий больше (3 ряда по 2)
FRAME_MAX_SIDE = 512      # сторона кадра в раскадровке (стикеры Telegram — 512 px)
MAX_DECODED_FRAMES = 1800  # дальше не декодируем (длинное видео): минута при 30 кадр/с
CANDIDATES = 32           # сколько кадров-кандидатов держать (равномерно по всей анимации)


def _decode_candidates(data: bytes) -> List[tuple]:
    """(картинка, маленький серый слепок) кадров, равномерно по всей анимации; шаг удваивается по мере чтения."""
    import av  # есть вместе с faster-whisper
    kept, step, index = [], 1, 0
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        # VP9 своим декодером libvpx — чтобы прозрачный фон видео-стикера не стал чёрным
        decoder = av.CodecContext.create("libvpx-vp9", "r") if stream.codec_context.name == "vp9" else None
        for packet in container.demux(stream):
            for frame in (decoder.decode(packet) if decoder else packet.decode()):
                if index % step == 0:
                    image = _to_image(frame)
                    kept.append((image, np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32) / 255))
                    if len(kept) >= 2 * CANDIDATES:  # слишком много — прореживаем через один и дальше берём реже
                        kept, step = kept[::2], step * 2
                index += 1
            if index >= MAX_DECODED_FRAMES:
                break
    return kept


def pick_frames(thumbs: List[np.ndarray], want: int) -> List[int]:
    """
    Номера кадров для раскадровки: первый, последний и самые сильные смены картинки; если смен мало (статичная
    анимация) — добиваем кадрами из самых длинных промежутков. По порядку времени.
    """
    n = len(thumbs)
    if n <= want:
        return list(range(n))
    change = [0.0] + [float(np.abs(thumbs[i] - thumbs[i - 1]).mean()) for i in range(1, n)]
    chosen = {0, n - 1}
    for i in sorted(range(1, n - 1), key=lambda i: -change[i]):
        if len(chosen) >= want or change[i] < 0.02:  # меньше 2% — это не смена сцены, а шум
            break
        if all(abs(i - c) > 1 for c in chosen):       # соседние кадры одной смены — не дублировать
            chosen.add(i)
    while len(chosen) < want:                          # равномерно заполнить самые большие пробелы
        points = sorted(chosen)
        gap_start, gap_end = max(zip(points, points[1:]), key=lambda p: p[1] - p[0])
        if gap_end - gap_start < 2:
            break
        chosen.add((gap_start + gap_end) // 2)
    return sorted(chosen)


def _to_image(frame) -> Image.Image:
    rgba = Image.fromarray(frame.to_ndarray(format="rgba"), "RGBA")
    rgba.thumbnail((FRAME_MAX_SIDE, FRAME_MAX_SIDE))
    flat = Image.new("RGB", rgba.size, (255, 255, 255))  # прозрачное -> белый фон
    flat.paste(rgba, mask=rgba.split()[3])
    return flat


def contact_sheet(frames: List[Image.Image]) -> Image.Image:
    """Кадры по порядку: слева направо, сверху вниз, по 2 в ряд."""
    if len(frames) == 1:
        return frames[0]
    w, h = max(f.width for f in frames), max(f.height for f in frames)
    cols, gap = 2, 8
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (200, 200, 200))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return sheet


def lottie_sheet(data: bytes, frames: int = SHEET_FRAMES) -> tuple:
    """
    Анимированный стикер Telegram (.tgs — это Lottie JSON в gzip) -> (раскадровка, кадров) или (None, 0).
    Рисует rlottie; без него Юи видела только превью 128 px.
    """
    try:
        import gzip
        from rlottie_python import LottieAnimation
        anim = LottieAnimation.from_data(gzip.decompress(data).decode("utf-8"))
        total = anim.lottie_animation_get_totalframe()
        numbers = sorted({round(i * (total - 1) / (CANDIDATES - 1)) for i in range(CANDIDATES)}) if total > 1 else [0]
        candidates = []
        for number in numbers:
            rgba = anim.render_pillow_frame(frame_num=number).convert("RGBA")
            rgba.thumbnail((FRAME_MAX_SIDE, FRAME_MAX_SIDE))
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[3])
            candidates.append((flat, np.asarray(flat.convert("L").resize((32, 32)), dtype=np.float32) / 255))
    except Exception as e:
        print(f"[FRAMES] Не удалось нарисовать стикер .tgs: {type(e).__name__}: {e}")
        return None, 0
    picked = pick_frames([thumb for _, thumb in candidates], frames)
    return contact_sheet([candidates[i][0] for i in picked]), len(picked)


def animation_sheet(data: bytes, frames: int = SHEET_FRAMES) -> tuple:
    """(раскадровка анимации webm/mp4/gif, сколько в ней кадров) или (None, 0), если декодировать не удалось."""
    try:
        candidates = _decode_candidates(data)
    except Exception as e:
        print(f"[FRAMES] Не удалось разобрать анимацию: {type(e).__name__}: {e}")
        return None, 0
    if not candidates:
        return None, 0
    picked = pick_frames([thumb for _, thumb in candidates], frames)
    return contact_sheet([candidates[i][0] for i in picked]), len(picked)

# -*- coding: utf-8 -*-
"""
Полностью автономный тест OmniVoice.
Никаких зависимостей от проекта, только чистый синтез и воспроизведение.
"""

import os
import tempfile
import time
import torch
import soundfile as sf
import pygame
from omnivoice import OmniVoice


def play_audio(file_path):
    """Воспроизведение WAV через pygame."""
    pygame.mixer.init(frequency=24000, size=-16, channels=1)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(30)


def main():
    # 1. Параметры синтеза
    text = "Привет, это тестовый синтез речи. Если ты слышишь меня, значит, TTS работает!"
    language = "ru"
    instruct = "female, young adult, moderate pitch"
    speed = 1.0

    # 2. Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[TEST] Загрузка OmniVoice на {device} с {dtype}...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype
    )

    print(f"[TEST] Синтез: {text}")
    audio_output = model.generate(
        text=text,
        language=language,
        instruct=instruct,
        speed=speed
    )

    # 3. Преобразование результата в numpy-массив
    if isinstance(audio_output, list):
        audio_output = audio_output[0]
    if hasattr(audio_output, 'cpu'):
        audio_array = audio_output.cpu().numpy().squeeze()
    else:
        audio_array = audio_output.squeeze()

    # 4. Сохраняем во временный WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, samplerate=24000)

    print(f"[TEST] Воспроизведение...")
    play_audio(tmp_path)

    # 5. Очистка
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    print("[TEST] Готово!")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Быстрый TTS на основе Silero.
Работает на CPU, минимальная задержка.
Динамически определяет частоту дискретизации модели.
"""
import os
import tempfile
import threading
import queue
import re
import torch
import soundfile as sf
import pygame
import requests
import numpy as np

from scripts.config import TTS_BUFFER, TTS_CHANNELS, SILERO_SAMPLE_RATE, TTS_SPEAKER_RU, TTS_SPEAKER_EN
from scripts.utils.lang import text_language


class TTSManager:
    def __init__(self, tts_active_event: threading.Event = None, speaker: str = TTS_SPEAKER_RU):
        """
        :param tts_active_event: событие, устанавливаемое во время воспроизведения.
        :param speaker: имя русского диктора (xenia, baya, kseniya, eugene, random)
        """
        self.speaker = speaker
        self.speakers = {"ru": speaker, "en": TTS_SPEAKER_EN}
        self.device = torch.device('cpu')
        self.tts_active = tts_active_event if tts_active_event else threading.Event()
        self.audio_queue = queue.Queue()
        self._running = True

        # Загрузка модели Silero
        print("[TTS] Загрузка Silero...")
        self.model, self.sample_rate = self._load_silero_model()
        print(f"[TTS] Silero загружен, частота {self.sample_rate} Гц, диктор {self.speaker}.")
        # Английский голос — отдельная модель: русская на целиком английском тексте падает с ошибкой
        self.models = {"ru": self.model}
        en_model = self._load_en_model()
        if en_model is not None:
            self.models["en"] = en_model
            print(f"[TTS] Английский голос: {self.speakers['en']}.")
        # Прогрев: первые вызовы apply_tts на CPU идут 1.2-1.8 с вместо ~0.17 с —
        # пусть это случится при запуске, а не на первой реплике Юи.
        for lang, warm_text in (("ru", "Привет."), ("en", "Hello.")):
            if lang not in self.models:
                continue
            try:
                for _ in range(2):
                    self.models[lang].apply_tts(warm_text, speaker=self.speakers[lang], sample_rate=self.sample_rate)
            except Exception as e:
                print(f"[TTS] Прогрев ({lang}) не удался: {e}")

        # Инициализация pygame mixer с частотой модели
        try:
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=TTS_CHANNELS, buffer=TTS_BUFFER)
        except Exception as e:
            print(f"[TTS] Ошибка инициализации pygame.mixer: {e}")
            self._running = False
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[TTS] Готов.")

    def _load_silero_model(self):
        """Загружает Silero TTS из репозитория, определяет частоту."""
        repo_url = "https://models.silero.ai/models/tts/ru/v3_1_ru.pt"
        local_path = os.path.join(os.path.dirname(__file__), "silero_model.pt")
        if not os.path.exists(local_path):
            print("[TTS] Скачивание модели Silero...")
            response = requests.get(repo_url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("[TTS] Модель скачана.")

        model = torch.package.PackageImporter(local_path).load_pickle("tts_models", "model")
        model.to(self.device)
        # ВАЖНО: частота — не свойство модели, а параметр, который передаётся
        # в apply_tts(). Модель не хранит "свою" частоту в атрибуте, поэтому
        # раньше здесь угадывалось значение по умолчанию, которое расходилось
        # с реальной частотой синтеза. Используем единую константу из конфига.
        return model, SILERO_SAMPLE_RATE

    def _load_en_model(self):
        """Английская модель Silero v3_en, если скачана (scripts/speech/silero_model_en.pt)."""
        path = os.path.join(os.path.dirname(__file__), "silero_model_en.pt")
        if not os.path.exists(path):
            print("[TTS] Английской модели нет (scripts/speech/silero_model_en.pt) — английские фразы не озвучиваются.")
            return None
        try:
            model = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
            model.to(self.device)
            return model
        except Exception as e:
            print(f"[TTS] Не удалось загрузить английскую модель: {e}")
            return None

    def speak(self, text: str):
        if not self._running or not text:
            return
        cleaned = re.sub(r'<[^>]+>', '', text).strip()
        cleaned = re.sub(r'[\*\_\#\`\[\]\(\)]', '', cleaned).strip()
        if not cleaned:
            return
        lang = text_language(cleaned)
        if lang not in self.models:
            print(f"[TTS] Нет голоса для языка '{lang}', фраза не озвучена: {cleaned[:60]}")
            return
        self.audio_queue.put((cleaned, lang))

    def _worker(self):
        while self._running:
            try:
                text, lang = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Синтез через Silero. sample_rate передаём ЯВНО — это и есть
                # частота, на которой Silero реально сгенерирует аудио, а не
                # то, что "предполагает" плеер. Она обязана совпадать с той,
                # на которой инициализирован pygame.mixer (см. __init__).
                audio = self.models[lang].apply_tts(text, speaker=self.speakers[lang], sample_rate=self.sample_rate)
                # Приводим к numpy (float32)
                if hasattr(audio, 'cpu'):
                    audio = audio.cpu()
                audio_np = audio.numpy().squeeze()

                # Убедимся, что аудио в диапазоне [-1,1] и в float32
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                # Сохраняем во временный WAV с частотой модели
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, audio_np, samplerate=self.sample_rate)

                # Воспроизведение
                pygame.mixer.music.load(tmp_path)
                self.tts_active.set()
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(30)
                self.tts_active.clear()

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally:
                self.audio_queue.task_done()

    def stop(self):
        self._running = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        if self._thread:
            self._thread.join(timeout=2)
        print("[TTS] Остановлен.")
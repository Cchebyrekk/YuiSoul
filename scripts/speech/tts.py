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

from scripts.config import TTS_BUFFER


class TTSManager:
    def __init__(self, tts_active_event: threading.Event = None, speaker: str = "xenia"):
        """
        :param tts_active_event: событие, устанавливаемое во время воспроизведения.
        :param speaker: имя диктора (xenia, baya, kseniya, eugene, random)
        """
        self.speaker = speaker
        self.device = torch.device('cpu')
        self.tts_active = tts_active_event if tts_active_event else threading.Event()
        self.audio_queue = queue.Queue()
        self._running = True

        # Загрузка модели Silero
        print("[TTS] Загрузка Silero...")
        self.model, self.sample_rate = self._load_silero_model()
        print(f"[TTS] Silero загружен, частота {self.sample_rate} Гц, диктор {self.speaker}.")

        # Инициализация pygame mixer с частотой модели
        try:
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=TTS_BUFFER)
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
        # Пытаемся получить частоту из модели, если есть атрибут
        sample_rate = getattr(model, 'sample_rate', 24000)
        return model, sample_rate

    def speak(self, text: str):
        if not self._running or not text:
            return
        cleaned = re.sub(r'<[^>]+>', '', text).strip()
        cleaned = re.sub(r'[\*\_\#\`\[\]\(\)]', '', cleaned).strip()
        if cleaned:
            self.audio_queue.put(cleaned)

    def _worker(self):
        while self._running:
            try:
                text = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Синтез через Silero
                audio = self.model.apply_tts(text, speaker=self.speaker)
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
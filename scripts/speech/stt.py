# -*- coding: utf-8 -*-
"""
Модуль распознавания речи (STT) на основе faster-whisper.
Записывает звук с микрофона, определяет паузы, транскрибирует и отправляет текст в очередь.
"""
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from scripts.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    STT_SAMPLERATE,
    STT_BLOCK_DURATION,
    STT_SILENCE_BLOCKS,
    STT_VOLUME_THRESHOLD,
    STT_MIN_AUDIO_LENGTH
)


class STTManager:
    """
    Менеджер распознавания речи.
    Запускает фоновый поток, который слушает микрофон и при обнаружении фразы
    отправляет текст в очередь input_queue вместе с метаданными.
    """

    def __init__(self, input_queue: queue.Queue, agent_busy_event: threading.Event):
        """
        :param input_queue: очередь, в которую будут помещаться распознанные фразы.
        :param agent_busy_event: событие, сигнализирующее, что агент занят (думает или говорит).
        """
        print(f"[STT] Загрузка Whisper ({WHISPER_MODEL_SIZE}) на {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        self.input_queue = input_queue
        self.agent_busy = agent_busy_event

        self.samplerate = STT_SAMPLERATE
        self.block_duration = STT_BLOCK_DURATION
        self.silence_blocks = STT_SILENCE_BLOCKS
        self.volume_threshold = STT_VOLUME_THRESHOLD
        self.min_audio_length = STT_MIN_AUDIO_LENGTH

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Запускает поток прослушивания микрофона."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[STT] Микрофон активен. Слушаю...")

    def stop(self):
        """Останавливает поток прослушивания."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[STT] Микрофон остановлен.")

    def _listen_loop(self):
        """Основной цикл: запись аудиоблоков, детекция речи, транскрипция."""
        block_size = int(self.samplerate * self.block_duration)

        while self._running:
            # Открываем поток ввода с нужными параметрами
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                blocksize=block_size,
                dtype='float32'
            ) as stream:
                recording = []
                silence_counter = 0
                is_recording = False

                while self._running:
                    audio_block, overflowed = stream.read(block_size)
                    # Вычисляем RMS громкости
                    rms = np.sqrt(np.mean(audio_block ** 2))

                    if rms > self.volume_threshold:
                        # Начало или продолжение речи
                        if not is_recording:
                            is_recording = True
                            recording = []  # сбрасываем предыдущий буфер
                        recording.append(audio_block)
                        silence_counter = 0
                    elif is_recording:
                        # Тишина во время записи — добавляем в буфер
                        recording.append(audio_block)
                        silence_counter += 1

                        # Если тишина превысила порог — фраза окончена
                        if silence_counter > self.silence_blocks:
                            # Объединяем блоки в один массив
                            audio_data = np.concatenate(recording).flatten().astype(np.float32)
                            audio_duration = len(audio_data) / self.samplerate

                            # Проверяем минимальную длину (отсекаем шумы)
                            if audio_duration > self.min_audio_length:
                                # Транскрибируем
                                segments, info = self.model.transcribe(
                                    audio_data,
                                    language=None,
                                    beam_size=1,
                                    vad_filter=True
                                )
                                text = " ".join([seg.text for seg in segments]).strip()

                                if text and len(text) > 1:
                                    # Определяем, был ли агент занят во время речи
                                    is_interrupt = self.agent_busy.is_set()
                                    metadata = {
                                        "timestamp": time.time(),
                                        "interrupted": is_interrupt
                                    }
                                    print(f"\n[STT VOICE INPUT] (Interrupt: {is_interrupt}): {text}")
                                    self.input_queue.put(("voice", text, metadata))

                            # Сбрасываем состояние
                            recording = []
                            silence_counter = 0
                            is_recording = False
                            # Выходим из внутреннего цикла, чтобы переоткрыть поток (предотвращает зависания)
                            break

                # Если цикл прерван по _running == False, выходим из внешнего while
                if not self._running:
                    break
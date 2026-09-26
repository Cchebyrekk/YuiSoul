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
    STT_INPUT_DEVICE,
    STT_LANGUAGE,
    STT_LANGUAGES,
    STT_LANGUAGE_DETECT_MODEL,
    STT_LANGUAGE_MIN_CONFIDENCE,
    STT_SAMPLERATE,
    STT_BLOCK_DURATION,
    STT_SILENCE_BLOCKS,
    STT_VOLUME_THRESHOLD,
    STT_MIN_AUDIO_LENGTH
)


def resolve_input_device(name_part: str | None) -> int | None:
    """
    Ищет микрофон по части имени (без учёта регистра). Возвращает индекс устройства
    или None — тогда sounddevice берёт микрофон Windows по умолчанию.
    """
    if not name_part:
        return None
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and name_part.lower() in dev['name'].lower():
            return idx
    print(f"[STT] Микрофон '{name_part}' не найден, использую микрофон Windows по умолчанию.")
    return None


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
        # Несколько языков — язык фразы определяет отдельная крошечная модель (см. STT_LANGUAGES)
        self.languages = tuple(STT_LANGUAGES) or (STT_LANGUAGE,)
        self.language = STT_LANGUAGE
        self.lid_model = None
        if len(self.languages) > 1:
            print(f"[STT] Определение языка ({'/'.join(self.languages)}) через Whisper {STT_LANGUAGE_DETECT_MODEL}...")
            self.lid_model = WhisperModel(STT_LANGUAGE_DETECT_MODEL, device="cpu", compute_type="int8")
        self.input_queue = input_queue
        self.agent_busy = agent_busy_event

        self.device = resolve_input_device(STT_INPUT_DEVICE)
        device_name = sd.query_devices(self.device, kind='input')['name']
        print(f"[STT] Микрофон: {device_name}")

        self.samplerate = STT_SAMPLERATE
        self.block_duration = STT_BLOCK_DURATION
        self.silence_blocks = STT_SILENCE_BLOCKS
        self.volume_threshold = STT_VOLUME_THRESHOLD
        self.min_audio_length = STT_MIN_AUDIO_LENGTH

        self._running = False
        self._thread: threading.Thread | None = None

    def detect_language(self, audio: np.ndarray) -> str:
        """
        Язык фразы из разрешённых (STT_LANGUAGES). "Липкий": переключаемся, только если
        модель уверена не меньше STT_LANGUAGE_MIN_CONFIDENCE — на коротком "да" tiny
        ошибается с низкой уверенностью, и тогда остаётся язык предыдущих фраз.
        """
        if self.lid_model is None:
            return self.language
        try:
            _, _, all_probs = self.lid_model.detect_language(audio)
        except Exception as e:
            print(f"[STT] Не удалось определить язык: {e}")
            return self.language
        probs = dict(all_probs)
        best = max(self.languages, key=lambda lang: probs.get(lang, 0.0))
        if probs.get(best, 0.0) >= STT_LANGUAGE_MIN_CONFIDENCE:
            self.language = best
        return self.language

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
            try:
                self._listen_once(block_size)
            except Exception as e:
                # Без этого любая ошибка микрофона молча убивала поток, и голосовой ввод пропадал до перезапуска
                print(f"[STT] Ошибка микрофона: {type(e).__name__}: {e}. Повтор через 2 с...")
                time.sleep(2)

    def _listen_once(self, block_size: int):
        """Слушает до конца одной фразы (или до остановки), затем возвращается, чтобы поток переоткрылся."""
        if self._running:
            # Открываем поток ввода с нужными параметрами
            with sd.InputStream(
                device=self.device,
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
                            # Длина речи — без хвоста тишины, которого мы ждали (STT_SILENCE_BLOCKS ≈ 2.7 с):
                            # иначе любой щелчок "длиннее" STT_MIN_AUDIO_LENGTH и уходит в Whisper.
                            speech_blocks = len(recording) - silence_counter
                            audio_duration = speech_blocks * self.block_duration

                            # Проверяем минимальную длину (отсекаем шумы)
                            if audio_duration > self.min_audio_length:
                                # Транскрибируем
                                language = self.detect_language(audio_data)
                                segments, info = self.model.transcribe(
                                    audio_data,
                                    language=language,
                                    beam_size=1,
                                    vad_filter=True
                                )
                                text = " ".join([seg.text for seg in segments]).strip()

                                if text and len(text) > 1:
                                    # Определяем, был ли агент занят во время речи
                                    is_interrupt = self.agent_busy.is_set()
                                    metadata = {
                                        "timestamp": time.time(),
                                        "interrupted": is_interrupt,
                                        "language": language
                                    }
                                    print(f"\n[STT VOICE INPUT] ({language}, Interrupt: {is_interrupt}): {text}")
                                    self.input_queue.put(("voice", text, metadata))

                            # Сбрасываем состояние
                            recording = []
                            silence_counter = 0
                            is_recording = False
                            # Выходим из внутреннего цикла, чтобы переоткрыть поток (предотвращает зависания)
                            break
# -*- coding: utf-8 -*-
"""
Единый конфигурационный файл для YUI.
Все настройки вынесены сюда. Изменяйте значения здесь — код останется неизменным.
"""

import os
import torch

# ==================== ПУТИ ====================

# Корень проекта (папка yuisoul)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папки данных
MODELS_DIR = os.path.join(BASE_DIR, "models")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
TTS_DIR = os.path.join(BASE_DIR, "tts")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
TURBOQUANT_DIR = os.path.join(BASE_DIR, "turboquant")

# LLM модель (путь к .gguf)
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
LLM_HOST = "127.0.0.1"
LLM_PORT = 8080
LLM_API_URL = f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions"

# Векторная память (SentenceTransformer)
VECTOR_MODEL_NAME = "intfloat/multilingual-e5-base"
VECTOR_DEVICE = "cpu"

# Сессии
SESSION_FILE = os.path.join(SESSIONS_DIR, "latest.json")

# ==================== ПАРАМЕТРЫ LLM ====================

# Параметры для быстрых ответов (приветствия, простые факты)
FAST_TEMPERATURE = 0.1
FAST_MAX_TOKENS = 256

# Параметры для сложных размышлений
DEEP_TEMPERATURE = 0.6
DEEP_MAX_TOKENS = 8192

DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_REPEAT_PENALTY = 1.1

# Параметры для быстрых ответов (приветствия, простые факты)
FAST_TEMPERATURE = 0.1
FAST_MAX_TOKENS = 256

# Параметры для сложных размышлений
DEEP_TEMPERATURE = 0.6
DEEP_MAX_TOKENS = 2048

# Токены остановки (для прекращения генерации)
STOP_TOKENS = ["<|im_end|>"]

# ==================== АГЕНТ И ЦИКЛ ====================

MAX_STEPS = 10                     # максимум итераций агента
MAX_RETRIES = 3                    # повторные попытки при сбое LLM

# Ограничения по контексту
MAX_CONTEXT_CHARS = 60000          # для сжатия
CONTEXT_TAIL_RATIO = 0.25          # доля хвоста при сжатии

# Таймауты (секунды)
LLM_TIMEOUT = 120.0
HTTP_READ_TIMEOUT = 120.0

# Флаги
ENABLE_AUTONOMY = False            # автономные мысли в фоне
AUTONOMY_CHECK_INTERVAL = 600      # секунд между проверками простоя
AUTONOMY_IDLE_THRESHOLD = 300      # секунд бездействия для запуска

# ==================== TTS ====================

# (OMNIVOICE)

# Параметры модели
OMNIVOICE_LANGUAGE = "ru"                     # язык синтеза
OMNIVOICE_SPEED = 1.0                         # скорость (0.1-5.0)
OMNIVOICE_INSTRUCT = "female, young adult, moderate pitch"   # конструктор голоса
OMNIVOICE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # или "cpu"
OMNIVOICE_DTYPE = "float16" if OMNIVOICE_DEVICE == "cuda" else "float32"

# Путь для кэширования модели (если нужно)
# OMNIVOICE_CACHE_DIR = os.path.join(BASE_DIR, "cache", "omnivoice")
# Если не задан, модель загрузится в стандартную папку кэша

# Дополнительно: можно задать отдельные атрибуты для удобства
OMNIVOICE_ATTRIBUTES = {
    "gender": "female",          # male / female
    "age": "young adult",        # child / teenager / young adult / middle-aged / elderly
    "pitch": "moderate pitch",   # very low / low / moderate / high / very high
    "style": None,               # whisper (опционально)
    "accent": None,              # american accent, british accent, ... (только для английского)
    "dialect": None,             # 四川话 и т.п. (только для китайского)
}

TTS_SAMPLE_RATE = 22050
TTS_CHANNELS = 1
TTS_BUFFER = 2048

# ==================== STT ====================

# (Whisper)
WHISPER_MODEL_SIZE = "small"          # tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"                # или "cuda" если хватит VRAM
WHISPER_COMPUTE_TYPE = "int8"         # int8, int16, float32

STT_SAMPLERATE = 16000
STT_BLOCK_DURATION = 0.3           # секунд на блок аудио
STT_SILENCE_BLOCKS = 9             # тихих блоков для остановки записи
STT_VOLUME_THRESHOLD = 0.015       # порог громкости
STT_MIN_AUDIO_LENGTH = 0.9         # минимальная длина фразы (сек)

# ==================== ПАМЯТЬ (RAG) ====================

# Лимиты для поиска и инъекции
AUTO_CONTEXT_MAX_FILES = 3
AUTO_CONTEXT_MAX_LINES_PER_FILE = 2
AUTO_CONTEXT_MAX_CHARS = 400

# Пороги сходства для дедупликации
VECTOR_DUPLICATE_THRESHOLD = 0.90   # косинус > 0.90 – дубль
JACCARD_DUPLICATE_THRESHOLD = 0.4   # Жаккард >= 0.4 – дубль внутри файла
VECTOR_SEARCH_THRESHOLD = 0.75      # минимальный скор для выдачи результатов

# ==================== ЭМОЦИОНАЛЬНЫЙ КАНАЛ ====================

EMOTION_WEBSOCKET_URL = "ws://127.0.0.1:8765"  # заглушка для аватара
EMOTION_EXTRACTION_ENABLED = False   # пока отключено

# ==================== ЛОГИРОВАНИЕ ====================

LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_LATENCY = True                 # замерять задержки

# ==================== ПРОЧЕЕ ====================

# Потоки для работы с памятью
MEMORY_IO_THREADS = 1               # количество потоков для записи/чтения (для блокировок)

